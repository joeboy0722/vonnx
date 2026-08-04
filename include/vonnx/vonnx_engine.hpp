#pragma once

#include "vonnx/common.hpp"
#include "vonnx/paged_tensor_pool.hpp"
#include "vonnx/cuda_preprocess.cuh"
#include "vonnx/cuda_nms.cuh"
#include "vonnx/onnx_runner.hpp"
#include "vonnx/continuous_scheduler.hpp"

namespace vonnx {

// vONNX Engine 設定參數
struct EngineConfig {
    std::string model_path;          // ONNX 模型路徑 (例如 yolov8n.onnx / yolov11n.onnx)
    size_t max_batch_size{16};       // 最大動態批次尺寸
    double max_latency_ms{2.0};      // 等待組 Batch 的最大超時延遲 (毫秒)
    float conf_threshold{0.25f};     // 信心分數門檻
    float iou_threshold{0.45f};      // NMS IoU 重疊門檻
    size_t max_gpu_pages{64};        // 預配置 GPU Paged Buffer 頁數
    int gpu_id{0};                   // 使用的 GPU 裝置編號
};

// 通用型 vONNX 高併發非同步視覺推論引擎
class vONNXEngine {
public:
    vONNXEngine(const EngineConfig& config);
    ~vONNXEngine();

    // 非同步提交單幀影像 tasks，傳回未來控制物件 std::future<std::vector<DetectionResult>>
    std::future<std::vector<DetectionResult>> predict_async(
        const FrameTask& task,
        InferenceCallback callback = nullptr
    );

    // 同步推論介面 (方便單幀或批量呼叫)
    std::vector<DetectionResult> predict_sync(const FrameTask& task);

    // 啟動背景動態排程與 GPU 流水線執行緒
    void start();

    // 停止引擎
    void stop();

private:
    // 背景 worker 執行緒：不斷拉取動態 Batch 並在 GPU CUDA Stream 上並行處理
    void worker_loop();

private:
    EngineConfig config_;
    std::unique_ptr<ONNXRunner> runner_;
    std::unique_ptr<PagedTensorPool> tensor_pool_;
    std::unique_ptr<ContinuousScheduler> scheduler_;

    cudaStream_t exec_stream_{nullptr};

    // 預先在 GPU 上的全域指標，避免任何動態 cudaMalloc
    float* d_batch_input_tensor_{nullptr};   // [MaxB, 3, H, W]
    float* d_batch_output_tensor_{nullptr};  // [MaxB, 4+C, N]
    AffineMatrix* d_batch_affine_meta_{nullptr}; // [MaxB]
    DeviceBBox* d_batch_out_boxes_{nullptr};     // [MaxB * MAX_BOXES_PER_IMAGE]
    int* d_batch_out_counts_{nullptr};          // [MaxB]

    std::thread worker_thread_;
    std::atomic<bool> is_running_{false};
};

} // namespace vonnx
