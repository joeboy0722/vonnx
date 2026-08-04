#pragma once

#include "vonnx/common.hpp"
#include "vonnx/paged_tensor_pool.hpp"
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>

namespace vonnx {

// 完成推論後的回呼函數型別
using InferenceCallback = std::function<void(const std::vector<DetectionResult>&)>;

// 排程打包後的一批任務批次包
struct BatchJob {
    std::vector<FrameTask> tasks;
    std::promise<std::vector<DetectionResult>> result_promise;
    InferenceCallback async_callback;
};

// vONNX 微秒級 Continuous Dynamic Batch 排程器
class ContinuousScheduler {
public:
    ContinuousScheduler(size_t max_batch_size, double max_latency_ms);
    ~ContinuousScheduler();

    // 生產端 (Producer): 外部相機/RTSP 串流發送 FrameTask 進入佇列
    std::future<std::vector<DetectionResult>> submit_task(const FrameTask& task, InferenceCallback callback = nullptr);

    // 消費端 (Consumer): 嘗試拉取一批湊好的動態 BatchJob (若超時或達 Max Batch 立即傳回)
    bool pop_batch(BatchJob& out_job, int timeout_ms);

    // 取得佇列中等待累積的任務數
    size_t pending_count();

    // 停止排程器
    void stop();

private:
    size_t max_batch_size_;
    double max_latency_ms_;
    std::queue<std::pair<FrameTask, std::shared_ptr<BatchJob>>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> is_running_{true};
};

} // namespace vonnx
