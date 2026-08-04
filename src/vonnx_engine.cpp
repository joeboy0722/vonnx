#include "vonnx/vonnx_engine.hpp"

namespace vonnx {

vONNXEngine::vONNXEngine(const EngineConfig& config)
    : config_(config) {
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    CUDA_CHECK(cudaStreamCreate(&exec_stream_));

    // 初始化 ONNX Runner 執行器
    runner_ = std::make_unique<ONNXRunner>(config_.model_path, config_.gpu_id);

    int target_w = runner_->get_input_width();
    int target_h = runner_->get_input_height();
    int num_classes = runner_->get_num_classes();
    int num_anchors = runner_->get_num_anchors();

    // 每一頁的大小 (可存放一張未壓縮的 4K RGBA 影像或更大)
    size_t page_size = 3840 * 2160 * 4;
    tensor_pool_ = std::make_unique<PagedTensorPool>(config_.max_gpu_pages, page_size);

    // 初始化動態動態動態 Dynamic Batch 排程器
    scheduler_ = std::make_unique<ContinuousScheduler>(config_.max_batch_size, config_.max_latency_ms);

    // 預先在 GPU 記憶體一次性配置常駐 Tensor 空間 (零執行期 cudaMalloc Overhead)
    size_t input_tensor_bytes = config_.max_batch_size * 3 * target_h * target_w * sizeof(float);
    size_t output_tensor_bytes = config_.max_batch_size * (4 + num_classes) * num_anchors * sizeof(float);
    size_t affine_meta_bytes = config_.max_batch_size * sizeof(AffineMatrix);
    size_t out_boxes_bytes = config_.max_batch_size * MAX_BOXES_PER_IMAGE * sizeof(DeviceBBox);
    size_t out_counts_bytes = config_.max_batch_size * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_batch_input_tensor_, input_tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_batch_output_tensor_, output_tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_batch_affine_meta_, affine_meta_bytes));
    CUDA_CHECK(cudaMalloc(&d_batch_out_boxes_, out_boxes_bytes));
    CUDA_CHECK(cudaMalloc(&d_batch_out_counts_, out_counts_bytes));

    start();
}

vONNXEngine::~vONNXEngine() {
    stop();

    if (exec_stream_) {
        cudaStreamDestroy(exec_stream_);
        exec_stream_ = nullptr;
    }

    if (d_batch_input_tensor_) cudaFree(d_batch_input_tensor_);
    if (d_batch_output_tensor_) cudaFree(d_batch_output_tensor_);
    if (d_batch_affine_meta_) cudaFree(d_batch_affine_meta_);
    if (d_batch_out_boxes_) cudaFree(d_batch_out_boxes_);
    if (d_batch_out_counts_) cudaFree(d_batch_out_counts_);
}

void vONNXEngine::start() {
    if (!is_running_) {
        is_running_ = true;
        worker_thread_ = std::thread(&vONNXEngine::worker_loop, this);
    }
}

void vONNXEngine::stop() {
    if (is_running_) {
        is_running_ = false;
        if (scheduler_) scheduler_->stop();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
}

std::future<std::vector<DetectionResult>> vONNXEngine::predict_async(
    const FrameTask& task,
    InferenceCallback callback
) {
    return scheduler_->submit_task(task, callback);
}

std::vector<DetectionResult> vONNXEngine::predict_sync(const FrameTask& task) {
    auto fut = predict_async(task);
    return fut.get();
}

void vONNXEngine::worker_loop() {
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));

    int target_w = runner_->get_input_width();
    int target_h = runner_->get_input_height();
    int num_classes = runner_->get_num_classes();
    int num_anchors = runner_->get_num_anchors();

    std::vector<DeviceBBox> host_boxes_buf(config_.max_batch_size * MAX_BOXES_PER_IMAGE);
    std::vector<int> host_counts_buf(config_.max_batch_size);

    while (is_running_) {
        BatchJob job;
        if (!scheduler_->pop_batch(job, 10)) {
            continue;
        }

        int curr_batch = static_cast<int>(job.tasks.size());
        if (curr_batch == 0) continue;

        std::vector<const void*> src_ptrs(curr_batch);
        std::vector<int> src_widths(curr_batch);
        std::vector<int> src_heights(curr_batch);
        std::vector<int> src_strides(curr_batch);
        std::vector<PixelFormat> formats(curr_batch);

        std::vector<int> temp_allocated_pages;

        // 整理傳入的 Frame 資料與 H2D 暫存頁
        for (int i = 0; i < curr_batch; ++i) {
            const auto& t = job.tasks[i];
            src_widths[i] = t.width;
            src_heights[i] = t.height;
            src_strides[i] = t.step;
            formats[i] = t.format;

            if (t.is_gpu_ptr) {
                src_ptrs[i] = t.data_ptr;
            } else {
                // 若輸入在 Host (CPU)，從 Paged Pool 配置一個 GPU 頁面並作非同步 H2D 拷貝
                int page_id = tensor_pool_->allocate_page();
                temp_allocated_pages.push_back(page_id);
                void* d_page = tensor_pool_->get_page_ptr(page_id);
                size_t img_bytes = t.height * t.step;

                CUDA_CHECK(cudaMemcpyAsync(d_page, t.data_ptr, img_bytes, cudaMemcpyHostToDevice, exec_stream_));
                src_ptrs[i] = d_page;
            }
        }

        // 1. 執行並列 Fused CUDA Preprocessing Kernel
        launch_batched_preprocess(
            src_ptrs,
            src_widths,
            src_heights,
            src_strides,
            formats,
            d_batch_input_tensor_,
            d_batch_affine_meta_,
            curr_batch,
            target_w,
            target_h,
            exec_stream_
        );

        // 2. 執行 ONNX Runtime Zero-Copy 推論 (Ort::IoBinding 指標直通)
        runner_->run_zero_copy(
            d_batch_input_tensor_,
            d_batch_output_tensor_,
            curr_batch,
            exec_stream_
        );

        // 3. 執行並列 Batched CUDA NMS Kernel
        launch_batched_yolo_nms(
            d_batch_output_tensor_,
            d_batch_affine_meta_,
            curr_batch,
            num_classes,
            num_anchors,
            config_.conf_threshold,
            config_.iou_threshold,
            d_batch_out_boxes_,
            d_batch_out_counts_,
            exec_stream_
        );

        // 4. 將最終 NMS 篩選後的邊界框複製回 Host 記憶體
        CUDA_CHECK(cudaMemcpyAsync(
            host_counts_buf.data(),
            d_batch_out_counts_,
            curr_batch * sizeof(int),
            cudaMemcpyDeviceToHost,
            exec_stream_
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            host_boxes_buf.data(),
            d_batch_out_boxes_,
            curr_batch * MAX_BOXES_PER_IMAGE * sizeof(DeviceBBox),
            cudaMemcpyDeviceToHost,
            exec_stream_
        ));

        // 等待當前 Stream 流水線完成
        CUDA_CHECK(cudaStreamSynchronize(exec_stream_));

        // 釋放臨時占用的 Paged GPU 頁面
        for (int p_id : temp_allocated_pages) {
            tensor_pool_->free_page(p_id);
        }

        // 5. 打包並派送 DetectionResult 給呼叫端
        std::vector<DetectionResult> results(curr_batch);
        for (int i = 0; i < curr_batch; ++i) {
            results[i].stream_id = job.tasks[i].stream_id;
            results[i].frame_id = job.tasks[i].frame_id;

            int count = std::min(host_counts_buf[i], MAX_BOXES_PER_IMAGE);
            const DeviceBBox* batch_boxes = host_boxes_buf.data() + i * MAX_BOXES_PER_IMAGE;

            for (int k = 0; k < count; ++k) {
                if (!batch_boxes[k].is_suppressed) {
                    BBox box;
                    box.x1 = batch_boxes[k].x1;
                    box.y1 = batch_boxes[k].y1;
                    box.x2 = batch_boxes[k].x2;
                    box.y2 = batch_boxes[k].y2;
                    box.score = batch_boxes[k].score;
                    box.class_id = batch_boxes[k].class_id;
                    results[i].boxes.push_back(box);
                }
            }
        }

        // 履行 Promise 並觸發非同步 Callback
        job.result_promise.set_value(results);
        if (job.async_callback) {
            job.async_callback(results);
        }
    }
}

} // namespace vonnx
