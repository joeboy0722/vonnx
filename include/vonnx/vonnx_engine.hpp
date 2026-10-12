#pragma once
#include "vonnx/common.hpp"
#include "vonnx/continuous_scheduler.hpp"
#include "vonnx/cuda_nms.cuh"
#include "vonnx/cuda_preprocess.cuh"
#include "vonnx/onnx_runner.hpp"
#include "vonnx/paged_tensor_pool.hpp"
namespace vonnx {
struct EngineConfig {
    std::string model_path;
    size_t max_batch_size{8};
    double max_latency_ms{-1.0};
    std::string batch_mode{"throughput"};
    size_t concurrency{32}, gpu_memory_budget_mb{6144};
    int pipeline_depth{0};
    bool collect_timings{false};
    float conf_threshold{0.25f}, iou_threshold{0.7f};
    size_t max_gpu_pages{0};
    int gpu_id{0};
    int input_width{640}, input_height{640}, max_input_width{1920}, max_input_height{1080};
    size_t queue_capacity{1024};
    int max_det{300};
    size_t max_mask_bytes{64 * 1024 * 1024};
    bool full_masks{false}, capture_debug{false};
};
class vONNXEngine {
  public:
    explicit vONNXEngine(const EngineConfig &);
    ~vONNXEngine();
    std::future<std::vector<DetectionResult>> predict_async(const FrameTask &, InferenceCallback = nullptr);
    std::vector<DetectionResult> predict_sync(const FrameTask &);
    const ModelSpec &model_spec() const { return cached_spec_; }
    void start();
    void stop();
    void pause();
    void resume();
    void close();
    struct ResourceStats {
        uint64_t initialization_count{1}, explicit_device_allocations{}, hot_path_device_allocations{};
        size_t reserved_device_bytes{}, arena_limit_bytes{}, initialized_device_bytes{},
            current_device_bytes{};
        int pipeline_depth{};
        std::string timer_backend, state;
    };
    ResourceStats resource_stats() const;

  private:
    void worker_loop();
    void upload_loop();
    void finish_threads(bool terminal);
    size_t validate_input(const FrameTask &) const;
    EngineConfig config_;
    CudaStream stream_, upload_stream_, download_stream_;
    std::unique_ptr<ONNXRunner> runner_;
    std::unique_ptr<ContinuousScheduler> scheduler_;
    struct Slot;
    void process_slot(Slot &);
    ModelSpec cached_spec_;
    std::vector<std::unique_ptr<Slot>> slots_;
    PinnedBuffer<uint8_t> staging_;
    size_t page_size_{}, baseline_free_{}, reserved_bytes_{}, arena_limit_{}, initialized_bytes_{};
    uint64_t initialization_allocations_{};
    std::atomic<uint64_t> hot_allocations_{0};
    std::thread worker_, uploader_;
    std::promise<void> worker_ready_;
    std::mutex slots_mutex_;
    std::condition_variable slots_cv_;
    std::deque<int> free_slots_, ready_slots_;
    bool uploader_done_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_dispatch_{false};
    bool started_{false};
    bool paused_{false}, terminal_{false}, closed_{false}, transitioning_{false};
    std::condition_variable lifecycle_cv_;
    mutable std::mutex lifecycle_;
};
} // namespace vonnx
