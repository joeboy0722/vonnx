#include "vonnx/continuous_scheduler.hpp"
#include <chrono>

namespace vonnx {

ContinuousScheduler::ContinuousScheduler(size_t max_batch_size, double max_latency_ms)
    : max_batch_size_(max_batch_size), max_latency_ms_(max_latency_ms) {}

ContinuousScheduler::~ContinuousScheduler() {
    stop();
}

void ContinuousScheduler::stop() {
    is_running_ = false;
    queue_cv_.notify_all();
}

std::future<std::vector<DetectionResult>> ContinuousScheduler::submit_task(
    const FrameTask& task,
    InferenceCallback callback
) {
    auto job = std::make_shared<BatchJob>();
    job->async_callback = callback;
    std::future<std::vector<DetectionResult>> fut = job->result_promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push({task, job});
    }

    queue_cv_.notify_one();
    return fut;
}

size_t ContinuousScheduler::pending_count() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

bool ContinuousScheduler::pop_batch(BatchJob& out_job, int timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // 等待直到佇列有資料或收到停止訊號
    if (task_queue_.empty()) {
        queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
            return !task_queue_.empty() || !is_running_;
        });
    }

    if (!is_running_ && task_queue_.empty()) return false;
    if (task_queue_.empty()) return false;

    auto start_time = std::chrono::high_resolution_clock::now();
    out_job.tasks.clear();

    // 動態湊 Batch 循環
    while (is_running_ && !task_queue_.empty() && out_job.tasks.size() < max_batch_size_) {
        auto item = task_queue_.front();
        task_queue_.pop();

        out_job.tasks.push_back(item.first);

        // 如果是第一個 Task，綁定其 Promise & Callback
        if (out_job.tasks.size() == 1) {
            out_job.async_callback = item.second->async_callback;
            out_job.result_promise = std::move(item.second->result_promise);
        }

        // 檢查是否已達到最大 Latency 限制，達到則立即使該批次發出
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
        if (elapsed_ms >= max_latency_ms_) break;
    }

    return !out_job.tasks.empty();
}

} // namespace vonnx
