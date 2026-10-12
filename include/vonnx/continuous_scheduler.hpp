#pragma once
#include "vonnx/common.hpp"
#include "vonnx/deadline_waiter.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
namespace vonnx {
using InferenceCallback = std::function<void(const std::vector<DetectionResult> &)>;
struct PendingRequest {
    FrameTask task{};
    std::vector<uint8_t> owned_input;
    std::promise<std::vector<DetectionResult>> promise;
    InferenceCallback callback;
    std::chrono::steady_clock::time_point submitted;
    std::function<void()> release_input;
    void release() {
        if (release_input) {
            auto fn = std::move(release_input);
            fn();
        }
    }
};
struct BatchJob {
    std::vector<FrameTask> tasks;
    std::vector<std::shared_ptr<PendingRequest>> requests;
    std::chrono::steady_clock::time_point dispatched;
    double batch_wait_ms{};
    void complete(std::vector<DetectionResult> results);
    void fail(std::exception_ptr error);
};
class ContinuousScheduler {
  public:
    ContinuousScheduler(size_t max_batch_size, double max_latency_ms, size_t capacity = 1024,
                        size_t concurrency = 0, uint8_t *staging = nullptr, size_t page_size = 0);
    ~ContinuousScheduler();
    std::future<std::vector<DetectionResult>> submit_task(const FrameTask &, InferenceCallback = nullptr);
    bool pop_batch(BatchJob &, int timeout_ms);
    size_t pending_count();
    void stop();
    void pause();
    void resume();
    std::string timer_backend() const { return waiter_.backend(); }

  private:
    size_t max_batch_size_, capacity_;
    double max_latency_ms_;
    std::deque<std::shared_ptr<PendingRequest>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool accepting_{true};
    bool terminal_{false};
    size_t concurrency_{}, inflight_{}, page_size_{};
    uint8_t *staging_{};
    std::vector<size_t> free_inputs_;
    DeadlineWaiter waiter_;
};
} // namespace vonnx
