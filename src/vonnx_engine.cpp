#include "vonnx/vonnx_engine.hpp"
#include "vonnx/model_precision.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
namespace vonnx {
namespace {
thread_local vONNXEngine *active_engine = nullptr;
}
struct vONNXEngine::Slot {
    int id{};
    BatchJob job;
    std::exception_ptr upload_error;
    std::vector<PreprocessFrame> frames;
    std::vector<MaskPlan> plans;
    std::unique_ptr<PreprocessWorkspace> preprocess;
    std::unique_ptr<NMSWorkspace> nms;
    DeviceBuffer<uint8_t> raw, mask_data;
    DeviceBuffer<float> input, prediction, prototype, pose;
    DeviceBuffer<uint16_t> input16, prediction16, prototype16;
    DeviceBuffer<AffineMatrix> affine;
    DeviceBuffer<DeviceBBox> boxes;
    DeviceBuffer<int> counts;
    DeviceBuffer<MaskPlan> mask_plans;
    PinnedBuffer<uint8_t> legacy_upload, host_masks;
    PinnedBuffer<int> host_counts;
    PinnedBuffer<DeviceBBox> host_boxes;
    PinnedBuffer<float> host_pose;
    PinnedBuffer<MaskPlan> host_plans;
    CudaEvent begin, uploaded, compute_start, pre, infer, post, copied, masks_begin, masks_done, final_copy;
    std::vector<std::unique_ptr<CudaEvent>> producers;
    std::chrono::steady_clock::time_point compute_begin;
    Slot(bool timing, bool seg)
        : begin(timing, timing), uploaded(timing), compute_start(timing, timing), pre(timing, timing),
          infer(timing, timing), post(timing), copied(timing), masks_begin(timing, timing && seg),
          masks_done(timing, seg), final_copy(timing, seg) {}
    size_t bytes() const {
        return raw.bytes() + mask_data.bytes() + input.bytes() + prediction.bytes() + prototype.bytes() +
               pose.bytes() + input16.bytes() + prediction16.bytes() + prototype16.bytes() +
               affine.bytes() + boxes.bytes() + counts.bytes() + mask_plans.bytes() +
               preprocess->frames.bytes() + nms->scores.bytes() + nms->sorted_scores.bytes() +
               nms->indices.bytes() + nms->sorted_indices.bytes() + nms->scratch.bytes();
    }
};
vONNXEngine::vONNXEngine(const EngineConfig &config) : config_(config) {
    if(config_.precision!="auto" && config_.precision!="fp32" && config_.precision!="fp16")
        throw std::invalid_argument("precision must be auto, fp32 or fp16");
    const auto source_precision=inspect_precision(config_.model_path);
    if(config_.precision=="fp32" && source_precision!="fp32")
        throw std::invalid_argument("FP16/mixed source cannot be requested as FP32; original precision cannot be restored");
    const bool converted=config_.precision=="fp16" && source_precision!="fp16";
    if(config_.batch_shape_policy!="auto" && config_.batch_shape_policy!="dynamic" && config_.batch_shape_policy!="fixed")
        throw std::invalid_argument("batch_shape_policy must be auto, dynamic or fixed");
    if(config_.graph_specialization!="off" && config_.graph_specialization!="auto")
        throw std::invalid_argument("graph_specialization must be off or auto");
    if (config_.batch_mode != "throughput" && config_.batch_mode != "latency")
        throw std::invalid_argument("Invalid batch_mode");
    if (config_.max_latency_ms == -1)
        config_.max_latency_ms = config_.batch_mode == "latency" ? 0 : 2;
    if (!config_.pipeline_depth)
        config_.pipeline_depth = config_.batch_mode == "latency" ? 1 : 2;
    if (!config_.max_batch_size || config_.max_batch_size > 1024 || !config_.queue_capacity ||
        config_.max_input_width <= 0 || config_.max_input_height <= 0 || config_.max_det <= 0 ||
        config_.max_det > MAX_BOXES_PER_IMAGE || !std::isfinite(config_.conf_threshold) ||
        config_.conf_threshold < 0 || config_.conf_threshold > 1 || !std::isfinite(config_.iou_threshold) ||
        config_.iou_threshold < 0 || config_.iou_threshold > 1 || !std::isfinite(config_.max_latency_ms) ||
        config_.max_latency_ms < 0 || config_.pipeline_depth < 1 || config_.pipeline_depth > 2 ||
        !config_.gpu_memory_budget_mb || config_.concurrency > 65536)
        throw std::invalid_argument("Invalid engine configuration");
    if (size_t(config_.max_input_width) >
            std::numeric_limits<size_t>::max() / size_t(config_.max_input_height) / 4 ||
        config_.gpu_memory_budget_mb > std::numeric_limits<size_t>::max() / (1024 * 1024))
        throw std::invalid_argument("Size overflow");
    if (config_.max_gpu_pages && config_.max_gpu_pages < config_.max_batch_size)
        throw std::invalid_argument("max_gpu_pages must cover max_batch_size");
    const bool fixed=config_.batch_shape_policy=="fixed" || (config_.batch_shape_policy=="auto" &&
        (converted || source_precision!="fp32") && config_.batch_mode=="throughput");
    const bool specialize=config_.graph_specialization=="auto";
    const int physical_batch=(fixed || config_.max_batch_size==1)?int(config_.max_batch_size):0;
    if(config_.cuda_graph && (!specialize || !physical_batch))
        throw std::invalid_argument("cuda_graph requires graph_specialization=auto and a fixed execution batch");
    if(converted || specialize) config_.model_path=convert_precision(config_.model_path,config_.model_cache_dir,
        config_.precision_python,config_.precision_tool,specialize?config_.input_height:0,
        specialize?config_.input_width:0,specialize?physical_batch:0,converted);
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    size_t total{};
    CUDA_CHECK(cudaMemGetInfo(&baseline_free_, &total));
    stream_.create();
    upload_stream_.create();
    download_stream_.create();
    runner_ = std::make_unique<ONNXRunner>(config_.model_path, config_.gpu_id, config_.input_width,
                                           config_.input_height, int(config_.max_batch_size), stream_.get(),
                                           config_.gpu_memory_budget_mb*1024*1024, converted || source_precision!="fp32",
                                           physical_batch,config_.cuda_graph);
    cached_spec_ = runner_->spec();
    cached_spec_.source_precision=source_precision;
    cached_spec_.effective_precision=converted?inspect_precision(config_.model_path):source_precision;
    cached_spec_.converted=converted;
    if(config_.batch_shape_policy=="fixed" || (config_.batch_shape_policy=="auto" &&
       cached_spec_.effective_precision!="fp32" && config_.batch_mode=="throughput"))
        cached_spec_.fixed_batch_size=int(config_.max_batch_size);
    cached_spec_.prepared_model_path=config_.model_path;
    const auto &s = cached_spec_;
    const size_t b = config_.max_batch_size, budget = config_.gpu_memory_budget_mb * 1024 * 1024;
    page_size_ = size_t(config_.max_input_width) * config_.max_input_height * 4;
    if (b > std::numeric_limits<size_t>::max() / page_size_ ||
        (config_.concurrency && config_.concurrency > std::numeric_limits<size_t>::max() / page_size_))
        throw std::invalid_argument("Staging size overflow");
    if (config_.concurrency)
        staging_.allocate(config_.concurrency * page_size_);
    scheduler_ = std::make_unique<ContinuousScheduler>(b, config_.max_latency_ms, config_.queue_capacity,
                                                       config_.concurrency, staging_.get(), page_size_);
    uint64_t before = thread_device_allocations;
    for (int id = 0; id < config_.pipeline_depth; ++id) {
        auto p = std::make_unique<Slot>(config_.collect_timings, s.task == "segment");
        p->id = id;
        p->frames.reserve(b);
        p->plans.reserve(b * MAX_BOXES_PER_IMAGE);
        p->job.tasks.reserve(b);
        p->job.requests.reserve(b);
        p->preprocess = std::make_unique<PreprocessWorkspace>(b);
        p->nms = std::make_unique<NMSWorkspace>(int(b), s.anchors, stream_.get());
        p->raw.allocate(b * page_size_);
        if (!config_.concurrency)
            p->legacy_upload.allocate(b * page_size_);
        p->input.allocate(b * 3 * s.width * s.height);
        p->prediction.allocate(b * s.channels * s.anchors);
        if(s.input_dtype==10) p->input16.allocate(p->input.size());
        if(s.output_dtypes[0]==10) p->prediction16.allocate(p->prediction.size());
        p->affine.allocate(b);
        p->boxes.allocate(b * MAX_BOXES_PER_IMAGE);
        p->counts.allocate(b);
        p->host_counts.allocate(b);
        p->host_boxes.allocate(b * MAX_BOXES_PER_IMAGE);
        if (s.task == "pose") {
            p->pose.allocate(b * MAX_BOXES_PER_IMAGE * s.keypoints * 3);
            p->host_pose.allocate(p->pose.size());
        }
        if (s.task == "segment") {
            p->prototype.allocate(b * s.mask_channels * s.mask_height * s.mask_width);
            if(s.output_dtypes[1]==10) p->prototype16.allocate(p->prototype.size());
            p->mask_data.allocate(config_.max_mask_bytes);
            p->mask_plans.allocate(b * MAX_BOXES_PER_IMAGE);
            p->host_masks.allocate(config_.max_mask_bytes);
            p->host_plans.allocate(b * MAX_BOXES_PER_IMAGE);
        }
        for (size_t i = 0; i < b; ++i)
            p->producers.push_back(std::make_unique<CudaEvent>());
        reserved_bytes_ += p->bytes();
        slots_.push_back(std::move(p));
        free_slots_.push_back(id);
        if (reserved_bytes_ + 16 * 1024 * 1024 >= budget)
            throw std::runtime_error("Explicit buffers exceed GPU memory budget");
    }
    initialization_allocations_ = thread_device_allocations - before;
    arena_limit_ = budget - reserved_bytes_ - 16 * 1024 * 1024;
    runner_->initialize_cuda(config_.gpu_id, stream_.get(), arena_limit_);
    for (auto &p : slots_) {
        std::vector<void *> outputs{p->prediction16.get()?static_cast<void*>(p->prediction16.get()):p->prediction.get()};
        if (s.task == "segment")
            outputs.push_back(p->prototype16.get()?static_cast<void*>(p->prototype16.get()):p->prototype.get());
        runner_->prepare_bindings(p->id, p->input16.get()?static_cast<void*>(p->input16.get()):p->input.get(), outputs, int(b));
        CUDA_CHECK(cudaMemsetAsync(p->input.get(), 0, p->input.bytes(), stream_.get()));
        if(p->input16.get()) CUDA_CHECK(cudaMemsetAsync(p->input16.get(),0,p->input16.bytes(),stream_.get()));
    }
    size_t available{};
    CUDA_CHECK(cudaMemGetInfo(&available, &total));
    initialized_bytes_ = baseline_free_ > available ? baseline_free_ - available : 0;
    start();
}
vONNXEngine::~vONNXEngine() {
    try {
        close();
    } catch (...) {
    }
}
void vONNXEngine::start() {
    std::lock_guard<std::mutex> lock(lifecycle_);
    if (terminal_ || closed_)
        throw std::runtime_error("Stopped/closed engines cannot restart");
    if (started_) {
        if (paused_)
            throw std::runtime_error("Use resume for paused engines");
        return;
    }
    running_ = true;
    started_ = true;
    uploader_done_ = false;
    auto ready = worker_ready_.get_future();
    worker_ = std::thread(&vONNXEngine::worker_loop, this);
    try {
        ready.get();
        uploader_ = std::thread(&vONNXEngine::upload_loop, this);
    } catch (...) {
        {
            std::lock_guard<std::mutex> guard(slots_mutex_);
            uploader_done_ = true;
        }
        slots_cv_.notify_all();
        worker_.join();
        throw;
    }
}
void vONNXEngine::finish_threads(bool terminal) {
    if (active_engine == this)
        throw std::runtime_error("Cannot change engine lifecycle from its callback");
    if (terminal)
        scheduler_->stop();
    else
        scheduler_->pause();
    running_ = false;
    slots_cv_.notify_all();
    if (uploader_.joinable())
        uploader_.join();
    if (worker_.joinable())
        worker_.join();
}
void vONNXEngine::pause() {
    if (active_engine == this)
        throw std::runtime_error("Cannot change engine lifecycle from its callback");
    std::unique_lock<std::mutex> lock(lifecycle_);
    lifecycle_cv_.wait(lock, [&] { return !transitioning_; });
    if (closed_ || terminal_)
        throw std::runtime_error("Stopped/closed engines cannot pause");
    if (paused_)
        return;
    paused_ = true;
    transitioning_ = true;
    lifecycle_cv_.wait(lock, [&] {return submitting_==0;});
    lock.unlock();
    scheduler_->pause();
    paused_dispatch_ = true;
    slots_cv_.notify_all();
    {
        std::unique_lock<std::mutex> slots_lock(slots_mutex_);
        slots_cv_.wait(slots_lock, [&] {
            return scheduler_->pending_count() == 0 && free_slots_.size() == slots_.size();
        });
    }
    lock.lock();
    transitioning_ = false;
    lifecycle_cv_.notify_all();
}
void vONNXEngine::resume() {
    if (active_engine == this)
        throw std::runtime_error("Cannot change engine lifecycle from its callback");
    std::unique_lock<std::mutex> lock(lifecycle_);
    lifecycle_cv_.wait(lock, [&] { return !transitioning_; });
    if (closed_ || terminal_)
        throw std::runtime_error("Stopped/closed engines cannot resume");
    if (!paused_)
        return;
    scheduler_->resume();
    paused_ = false;
    paused_dispatch_ = false;
    slots_cv_.notify_all();
}
void vONNXEngine::stop() {
    if (active_engine == this)
        throw std::runtime_error("Cannot change engine lifecycle from its callback");
    std::unique_lock<std::mutex> lock(lifecycle_);
    lifecycle_cv_.wait(lock, [&] { return !transitioning_; });
    if (closed_ || terminal_)
        return;
    terminal_ = true;
    paused_ = false;
    transitioning_ = true;
    lifecycle_cv_.wait(lock, [&] {return submitting_==0;});
    lock.unlock();
    finish_threads(true);
    lock.lock();
    transitioning_ = false;
    lifecycle_cv_.notify_all();
}
void vONNXEngine::close() {
    stop();
    std::lock_guard<std::mutex> lock(lifecycle_);
    if (closed_)
        return;
    terminal_ = true;
    paused_ = false;
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    CUDA_CHECK(cudaStreamSynchronize(upload_stream_.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    CUDA_CHECK(cudaStreamSynchronize(download_stream_.get()));
    runner_.reset();
    slots_.clear();
    scheduler_.reset();
    staging_.reset();
    upload_stream_.reset();
    download_stream_.reset();
    stream_.reset();
    closed_ = true;
}
vONNXEngine::ResourceStats vONNXEngine::resource_stats() const {
    std::lock_guard<std::mutex> lock(lifecycle_);
    ResourceStats s;
    s.explicit_device_allocations = initialization_allocations_;
    s.hot_path_device_allocations = hot_allocations_;
    s.reserved_device_bytes = reserved_bytes_;
    s.arena_limit_bytes = arena_limit_;
    s.initialized_device_bytes = initialized_bytes_;
    s.pipeline_depth = config_.pipeline_depth;
    s.timer_backend = scheduler_ ? scheduler_->timer_backend() : "closed";
    s.state = closed_ ? "closed" : terminal_ ? "stopped" : paused_ ? "paused" : "running";
    if (!closed_) {
        int previous{};
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(config_.gpu_id));
        size_t free{}, total{};
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        s.current_device_bytes = baseline_free_ > free ? baseline_free_ - free : 0;
        CUDA_CHECK(cudaSetDevice(previous));
    }
    return s;
}
size_t vONNXEngine::validate_input(const FrameTask &t) const {
    if (!t.data_ptr || t.width <= 0 || t.height <= 0 || t.width > config_.max_input_width ||
        t.height > config_.max_input_height || t.step <= 0)
        throw std::invalid_argument("Invalid image pointer/dimensions/stride or "
                                    "input exceeds configured limit");
    int channels = 3;
    switch (t.format) {
    case PixelFormat::RGBA32:
        channels = 4;
        break;
    case PixelFormat::NV12:
        channels = 1;
        if (t.width % 2 || t.height % 2)
            throw std::invalid_argument("NV12 requires even dimensions");
        break;
    case PixelFormat::RGB24:
    case PixelFormat::BGR24:
        break;
    default:
        throw std::invalid_argument("Invalid pixel format");
    }
    if (size_t(t.step) < size_t(t.width) * channels)
        throw std::invalid_argument("Image stride too small");
    size_t bytes = size_t(t.step) * t.height;
    if (t.format == PixelFormat::NV12)
        bytes += size_t(t.step) * (t.height / 2);
    if (bytes > page_size_)
        throw std::invalid_argument("Image exceeds pool page capacity");
    if (t.is_gpu_ptr) {
        cudaPointerAttributes a{};
        CUDA_CHECK(cudaPointerGetAttributes(&a, t.data_ptr));
        if ((a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged) || a.device != config_.gpu_id)
            throw std::invalid_argument("GPU pointer is not on the configured device");
    }
    if (t.data_size && t.data_size < bytes)
        throw std::invalid_argument("Input buffer too small");
    return bytes;
}
std::future<std::vector<DetectionResult>> vONNXEngine::predict_async(const FrameTask &task,
                                                                     InferenceCallback cb) {
    const auto entered=config_.collect_timings?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    {
        std::lock_guard<std::mutex> lock(lifecycle_);
        if (closed_ || terminal_ || paused_)
            throw std::runtime_error("Engine stopped/paused/closed; new requests rejected");
        ++submitting_;
    }
    auto finished=[&] {
        std::lock_guard<std::mutex> lock(lifecycle_);
        --submitting_;
        lifecycle_cv_.notify_all();
    };
    try {
        auto copy=task;
        copy.data_size=validate_input(task);
        auto future=scheduler_->submit_task(copy,std::move(cb),entered);
        finished();
        return future;
    } catch(...) {finished();throw;}
}
std::vector<DetectionResult> vONNXEngine::predict_sync(const FrameTask &task) {
    return predict_async(task).get();
}
void vONNXEngine::upload_loop() {
    active_engine = this;
    try {
        CUDA_CHECK(cudaSetDevice(config_.gpu_id));
        while (running_ || scheduler_->pending_count()) {
            int id;
            {
                std::unique_lock<std::mutex> lock(slots_mutex_);
                slots_cv_.wait(lock, [&] {
                    const bool pending = scheduler_->pending_count() != 0;
                    return (!running_ && !pending) ||
                           (!free_slots_.empty() && (!paused_dispatch_ || pending));
                });
                if (!running_ && scheduler_->pending_count() == 0)
                    break;
                id = free_slots_.front();
                free_slots_.pop_front();
            }
            auto &slot = *slots_[id];
            if (!scheduler_->pop_batch(slot.job, 10)) {
                std::lock_guard<std::mutex> lock(slots_mutex_);
                free_slots_.push_back(id);
                slots_cv_.notify_all();
                continue;
            }
            slot.upload_error = nullptr;
            try {
                slot.frames.resize(slot.job.tasks.size());
                slot.begin.record(upload_stream_.get());
                for (size_t i = 0; i < slot.job.tasks.size(); ++i) {
                    const auto &t = slot.job.tasks[i];
                    auto &f = slot.frames[i];
                    f.width = t.width;
                    f.height = t.height;
                    f.stride = t.step;
                    f.format = t.format;
                    if (t.is_gpu_ptr) {
                        f.ptr = static_cast<const uint8_t *>(t.data_ptr);
                        // Also bridge producer stream 0 explicitly: our streams
                        // are non-blocking and no longer implicitly wait for it.
                        slot.producers[i]->record(t.user_stream);
                        CUDA_CHECK(cudaStreamWaitEvent(upload_stream_.get(), slot.producers[i]->get(), 0));
                    } else {
                        const void *source = t.data_ptr;
                        if (slot.legacy_upload.get()) {
                            uint8_t *staging = slot.legacy_upload.get() + i * page_size_;
                            std::memcpy(staging, source, t.data_size);
                            source = staging;
                        }
                        f.ptr = slot.raw.get() + i * page_size_;
                        CUDA_CHECK(cudaMemcpyAsync(const_cast<uint8_t *>(f.ptr), source, t.data_size,
                                                   cudaMemcpyHostToDevice, upload_stream_.get()));
                    }
                    const auto &s = cached_spec_;
                    double r = std::min(double(s.width) / t.width, double(s.height) / t.height);
                    int rw = int(std::nearbyint(t.width * r)), rh = int(std::nearbyint(t.height * r));
                    f.affine = {float(r),
                                float(std::nearbyint((s.width - rw) / 2.0 - .1)),
                                float(std::nearbyint((s.height - rh) / 2.0 - .1)),
                                t.width,
                                t.height,
                                rw,
                                rh};
                }
                std::copy(slot.frames.begin(), slot.frames.end(), slot.preprocess->host_frames.get());
                CUDA_CHECK(cudaMemcpyAsync(slot.preprocess->frames.get(), slot.preprocess->host_frames.get(),
                                           slot.frames.size() * sizeof(PreprocessFrame),
                                           cudaMemcpyHostToDevice, upload_stream_.get()));
                slot.uploaded.record(upload_stream_.get());
            } catch (...) {
                slot.upload_error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(slots_mutex_);
                if(config_.collect_timings) {
                    const auto ready=std::chrono::steady_clock::now();
                    for(auto &request:slot.job.requests) request->trace[PendingRequest::SlotReady]=ready;
                }
                ready_slots_.push_back(id);
            }
            slots_cv_.notify_all();
        }
    } catch (...) {
        auto error = std::current_exception();
        scheduler_->stop();
        running_ = false;
        BatchJob job;
        while (scheduler_->pop_batch(job, 0))
            job.fail(error);
    }
    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        uploader_done_ = true;
    }
    slots_cv_.notify_all();
}
void vONNXEngine::worker_loop() {
    active_engine = this;
    try {
        CUDA_CHECK(cudaSetDevice(config_.gpu_id));
        for (auto &slot : slots_)
            for (size_t batch = cached_spec_.fixed_batch_size?config_.max_batch_size:1; batch <= config_.max_batch_size; ++batch) {
                for(int repeat=0;repeat<(config_.cuda_graph?3:1);++repeat)
                    runner_->run(slot->id, int(batch));
                CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
                size_t available{}, total{};
                CUDA_CHECK(cudaMemGetInfo(&available, &total));
                initialized_bytes_ = baseline_free_ > available ? baseline_free_ - available : 0;
                if (initialized_bytes_ > config_.gpu_memory_budget_mb * 1024 * 1024)
                    throw std::runtime_error("Initialization/warmup exceeds GPU memory budget");
            }
        worker_ready_.set_value();
    } catch (...) {
        worker_ready_.set_exception(std::current_exception());
        return;
    }
    while (true) {
        int id;
        {
            std::unique_lock<std::mutex> lock(slots_mutex_);
            slots_cv_.wait(lock, [&] { return !ready_slots_.empty() || uploader_done_; });
            if (ready_slots_.empty())
                break;
            id = ready_slots_.front();
            ready_slots_.pop_front();
        }
        auto &slot = *slots_[id];
        uint64_t before = thread_device_allocations;
        try {
            if (slot.upload_error)
                std::rethrow_exception(slot.upload_error);
            process_slot(slot);
        } catch (...) {
            auto error = std::current_exception();
            cudaStreamSynchronize(upload_stream_.get());
            cudaStreamSynchronize(stream_.get());
            cudaStreamSynchronize(download_stream_.get());
            slot.job.fail(error);
        }
        hot_allocations_ += thread_device_allocations - before;
        slot.job.tasks.clear();
        slot.job.requests.clear();
        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            free_slots_.push_back(id);
        }
        slots_cv_.notify_all();
    }
}

void vONNXEngine::process_slot(Slot &slot) {
    const auto &s = cached_spec_;
    auto &job = slot.job;
    const int batch = int(job.tasks.size());
    const auto stream = stream_.get(), download = download_stream_.get();
    auto &frames = slot.frames;
    std::vector<DetectionResult> results(batch);
    auto &begin = slot.begin;
    auto &h2d = slot.uploaded;
    auto &pre = slot.pre;
    auto &infer = slot.infer;
    auto &post = slot.post;
    auto &copied = slot.copied;
    auto &masks_begin = slot.masks_begin;
    auto &masks_done = slot.masks_done;
    auto &final_copy = slot.final_copy;
    slot.compute_begin = std::chrono::steady_clock::now();
    for (int b = 0; b < batch; ++b) {
        auto &r = results[b];
        r.stream_id = job.tasks[b].stream_id;
        r.frame_id = job.tasks[b].frame_id;
        r.batch_size = batch;
        r.physical_batch_size=s.fixed_batch_size?s.fixed_batch_size:batch;
        if(config_.collect_timings) job.requests[b]->trace[PendingRequest::ComputeBegin]=slot.compute_begin;
        r.queue_ms =
            std::chrono::duration<double, std::milli>(job.dispatched - job.requests[b]->submitted).count();
        r.batch_wait_ms = std::min(r.queue_ms, job.batch_wait_ms);
        r.queue_ms -= r.batch_wait_ms;
        r.dispatch_ms =
            std::chrono::duration<double, std::milli>(slot.compute_begin - job.dispatched).count();
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, slot.uploaded.get(), 0));
    slot.compute_start.record(stream);
    launch_prepared_preprocess(batch, *slot.preprocess, slot.input.get(), slot.affine.get(), s.width,
                               s.height, stream);
    const int physical=s.fixed_batch_size?s.fixed_batch_size:batch;
    if(physical>batch) CUDA_CHECK(cudaMemsetAsync(slot.input.get()+size_t(batch)*3*s.width*s.height,0,
        size_t(physical-batch)*3*s.width*s.height*sizeof(float),stream));
    if(slot.input16.get()) launch_to_half(slot.input.get(),slot.input16.get(),size_t(physical)*3*s.width*s.height,stream);
    pre.record(stream);
    float *outputs[] = {slot.prediction.get(), slot.prototype.get()};
    size_t output_count = s.task == "segment" ? 2 : 1;
    if(config_.collect_timings) {
        const auto now=std::chrono::steady_clock::now();
        for(auto &request:job.requests) request->trace[PendingRequest::OrtEnter]=now;
    }
    runner_->run(slot.id, physical);
    if(config_.collect_timings) {
        const auto now=std::chrono::steady_clock::now();
        for(auto &request:job.requests) request->trace[PendingRequest::OrtExit]=now;
    }
    if(slot.prediction16.get()) launch_from_half(slot.prediction16.get(),slot.prediction.get(),size_t(batch)*s.channels*s.anchors,stream);
    if(slot.prototype16.get()) launch_from_half(slot.prototype16.get(),slot.prototype.get(),size_t(batch)*s.mask_channels*s.mask_height*s.mask_width,stream);
    infer.record(stream);
    launch_batched_yolo_nms(slot.prediction.get(), slot.affine.get(), batch, s.classes, s.channels, s.anchors,
                            config_.conf_threshold, config_.iou_threshold, config_.max_det, *slot.nms,
                            slot.boxes.get(), slot.counts.get(), stream);
    if (s.task == "pose")
        launch_pose(slot.prediction.get(), slot.boxes.get(), slot.counts.get(), slot.affine.get(), batch,
                    s.classes, s.channels, s.anchors, s.keypoints, s.keypoint_dims, slot.pose.get(), stream);
    post.record(stream);
    CUDA_CHECK(cudaStreamWaitEvent(download, post.get(), 0));
    int *counts = slot.host_counts.get();
    DeviceBBox *boxes = slot.host_boxes.get();
    float *keypoints = slot.host_pose.get();
    CUDA_CHECK(cudaMemcpyAsync(counts, slot.counts.get(), size_t(batch) * sizeof(int), cudaMemcpyDeviceToHost,
                               download));
    CUDA_CHECK(cudaMemcpyAsync(boxes, slot.boxes.get(),
                               (size_t(batch) * MAX_BOXES_PER_IMAGE) * sizeof(DeviceBBox),
                               cudaMemcpyDeviceToHost, download));
    if (s.task == "pose") {

        CUDA_CHECK(cudaMemcpyAsync(keypoints, slot.pose.get(),
                                   (size_t(batch) * MAX_BOXES_PER_IMAGE * s.keypoints * 3) * sizeof(float),
                                   cudaMemcpyDeviceToHost, download));
    }
    copied.record(download);
    CUDA_CHECK(cudaEventSynchronize(copied.get()));
    if(config_.collect_timings) {
        const auto now=std::chrono::steady_clock::now();
        for(auto &request:job.requests) request->trace[PendingRequest::D2HReady]=now;
    }
    auto &plans = slot.plans;
    plans.clear();
    size_t mask_bytes = 0;
    for (int b = 0; b < batch; ++b)
        for (int k = 0; k < counts[b]; ++k) {
            auto box = boxes[b * MAX_BOXES_PER_IMAGE + k];
            results[b].boxes.push_back({box.x1, box.y1, box.x2, box.y2, box.score, box.class_id});
            if (s.task == "pose") {
                std::vector<DetectionResult::Keypoint> kp(s.keypoints);
                for (int j = 0; j < s.keypoints; ++j) {
                    const float *v = keypoints + ((b * MAX_BOXES_PER_IMAGE + k) * s.keypoints + j) * 3;
                    kp[j] = {v[0], v[1], v[2]};
                }
                results[b].keypoints.push_back(std::move(kp));
            }
            if (s.task == "segment") {
                auto a = frames[b].affine;
                int x = config_.full_masks ? 0 : int(std::ceil(box.x1)),
                    y = config_.full_masks ? 0 : int(std::ceil(box.y1));
                int right = config_.full_masks ? a.src_w : int(std::ceil(box.x2)),
                    bottom = config_.full_masks ? a.src_h : int(std::ceil(box.y2));
                int w = std::max(0, right - x), h = std::max(0, bottom - y);
                double gain = std::min(double(s.mask_height) / a.src_h, double(s.mask_width) / a.src_w);
                double px = (s.mask_width - std::nearbyint(a.src_w * gain)) / 2.0,
                       py = (s.mask_height - std::nearbyint(a.src_h * gain)) / 2.0;
                int left = int(std::nearbyint(px - .1)), top = int(std::nearbyint(py - .1)),
                    pr = s.mask_width - int(std::nearbyint(px + .1)),
                    pb = s.mask_height - int(std::nearbyint(py + .1));
                if (mask_bytes + size_t(w) * h > config_.max_mask_bytes)
                    throw std::runtime_error("Mask output exceeds max_mask_bytes");
                plans.push_back({b, k, x, y, w, h, left, top, pr, pb, mask_bytes});
                mask_bytes += size_t(w) * h;
                DetectionResult::Mask mask;
                mask.x = x;
                mask.y = y;
                mask.width = w;
                mask.height = h;
                results[b].masks.push_back(std::move(mask));
            }
        }
    uint8_t *host_masks = slot.host_masks.get();
    if (mask_bytes) {
        masks_begin.record(stream);
        if (!plans.empty()) {
            std::copy(plans.begin(), plans.end(), slot.host_plans.get());
            CUDA_CHECK(cudaMemcpyAsync(slot.mask_plans.get(), slot.host_plans.get(),
                                       plans.size() * sizeof(MaskPlan), cudaMemcpyHostToDevice, stream));
            launch_masks(slot.prediction.get(), slot.prototype.get(), slot.boxes.get(), slot.affine.get(),
                         slot.mask_plans.get(), int(plans.size()), s.classes, s.channels, s.anchors,
                         s.mask_channels, s.mask_height, s.mask_width, slot.mask_data.get(), stream);
        }
        masks_done.record(stream);
        CUDA_CHECK(cudaStreamWaitEvent(download, masks_done.get(), 0));
        if (mask_bytes)
            CUDA_CHECK(cudaMemcpyAsync(host_masks, slot.mask_data.get(), mask_bytes, cudaMemcpyDeviceToHost,
                                       download));
        final_copy.record(download);
        CUDA_CHECK(cudaEventSynchronize(final_copy.get()));
    }
    for (auto &p : plans) {
        auto &mask = results[p.batch].masks[p.box];
        if (p.width && p.height)
            mask.data.assign(host_masks + p.offset, host_masks + p.offset + size_t(p.width) * p.height);
    }
    if (config_.collect_timings)
        for (auto &result : results) {
            result.h2d_ms = h2d.since(begin);
            result.preprocess_ms = pre.since(slot.compute_start);
            result.inference_ms = infer.since(pre);
            result.postprocess_ms = post.since(infer) + (mask_bytes ? masks_done.since(masks_begin) : 0);
            result.d2h_ms = copied.since(post) + (mask_bytes ? final_copy.since(masks_done) : 0);
        }
    if (config_.capture_debug) {
        for (int b = 0; b < batch; ++b) {
            auto &result = results[b];
            size_t input_n = size_t(3) * s.width * s.height;
            result.debug_input.resize(input_n);
            CUDA_CHECK(cudaMemcpy(result.debug_input.data(), slot.input.get() + b * input_n,
                                  input_n * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < output_count; ++j) {
                size_t n = 1;
                for (size_t dim = 1; dim < s.output_shapes[j].size(); ++dim)
                    n *= size_t(s.output_shapes[j][dim]);
                result.debug_outputs.emplace_back(n);
                CUDA_CHECK(cudaMemcpy(result.debug_outputs.back().data(), outputs[j] + b * n,
                                      n * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
    }

    job.complete(std::move(results));
}
} // namespace vonnx
