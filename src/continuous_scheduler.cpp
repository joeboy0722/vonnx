#include "vonnx/continuous_scheduler.hpp"
#include <cmath>
#include <cstring>
namespace vonnx {
void BatchJob::complete(std::vector<DetectionResult> results) {
    if (results.size() != requests.size())
        throw std::logic_error("Result/request count mismatch");
    for (size_t i = 0; i < requests.size(); ++i) {
        std::vector<DetectionResult> own;
        own.push_back(std::move(results[i]));
        std::vector<DetectionResult> callback_value;
        if (requests[i]->callback)
            callback_value = own;
        requests[i]->release();
        requests[i]->promise.set_value(std::move(own));
        if (requests[i]->callback) {
            try {
                requests[i]->callback(callback_value);
            } catch (...) {
                std::cerr << "vONNX: callback threw; request result remains available\n";
            }
        }
    }
}
void BatchJob::fail(std::exception_ptr error) {
    for (auto &request : requests) {
        request->release();
        try {
            request->promise.set_exception(error);
        } catch (const std::future_error &) { /* Already fulfilled before a later allocation failed. */
        }
    }
}
ContinuousScheduler::ContinuousScheduler(size_t batch, double latency, size_t capacity, size_t concurrency,
                                         uint8_t *staging, size_t page_size)
    : max_batch_size_(batch), capacity_(capacity), max_latency_ms_(latency), concurrency_(concurrency),
      page_size_(page_size), staging_(staging) {
    if (!batch || !capacity || !std::isfinite(latency) || latency < 0)
        throw std::invalid_argument("Invalid scheduler configuration");
    for (size_t i = 0; i < concurrency; ++i)
        free_inputs_.push_back(i);
}
ContinuousScheduler::~ContinuousScheduler() { stop(); }
void ContinuousScheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = false;
        terminal_ = true;
    }
    cv_.notify_all();
    waiter_.notify();
}
void ContinuousScheduler::pause() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = false;
    }
    cv_.notify_all();
    waiter_.notify();
}
void ContinuousScheduler::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (terminal_)
        throw std::runtime_error("Stopped scheduler cannot resume");
    accepting_ = true;
}
size_t ContinuousScheduler::pending_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}
std::future<std::vector<DetectionResult>> ContinuousScheduler::submit_task(const FrameTask &task,
                                                                           InferenceCallback cb) {
    auto request = std::make_shared<PendingRequest>();
    request->task = task;
    request->callback = std::move(cb);
    auto future = request->promise.get_future();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_)
        throw std::runtime_error("Engine stopped; new requests rejected");
    if (queue_.size() >= capacity_)
        throw std::runtime_error("Request queue full");
    if (concurrency_ && inflight_ >= concurrency_)
        throw std::runtime_error("Concurrency capacity full");
    size_t page = size_t(-1);
    if (!task.is_gpu_ptr && task.data_ptr) {
        size_t bytes = task.data_size ? task.data_size : size_t(task.height) * task.step;
        if (!task.data_size && task.format == PixelFormat::NV12)
            bytes += size_t(task.height / 2) * task.step;
        if (staging_) {
            if (bytes > page_size_)
                throw std::invalid_argument("Input exceeds staging capacity");
            page = free_inputs_.back();
            free_inputs_.pop_back();
            request->task.data_ptr = staging_ + page * page_size_;
        } else {
            request->owned_input.resize(bytes);
            request->task.data_ptr = request->owned_input.data();
        }
        std::memcpy(const_cast<void *>(request->task.data_ptr), task.data_ptr, bytes);
        request->task.data_size = bytes;
    }
    request->submitted = std::chrono::steady_clock::now();
    try {
        request->release_input = [this, page] {
            std::lock_guard<std::mutex> lock(mutex_);
            --inflight_;
            if (page != size_t(-1))
                free_inputs_.push_back(page);
        };
        queue_.push_back(request);
    } catch (...) {
        if (page != size_t(-1))
            free_inputs_.push_back(page);
        throw;
    }
    ++inflight_;
    cv_.notify_one();
    waiter_.notify();
    return future;
}
bool ContinuousScheduler::pop_batch(BatchJob &out, int timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, std::chrono::milliseconds(timeout), [&] { return !queue_.empty() || !accepting_; });
    if (queue_.empty())
        return false;
    const auto deadline =
        queue_.front()->submitted + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<double, std::milli>(max_latency_ms_));
    auto wait_begin = std::chrono::steady_clock::now();
    while (queue_.size() < max_batch_size_ && accepting_) {
        double remaining = std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0)
            break;
        lock.unlock();
        waiter_.wait(remaining);
        lock.lock();
    }
    out.dispatched = std::chrono::steady_clock::now();
    out.batch_wait_ms = std::chrono::duration<double, std::milli>(out.dispatched - wait_begin).count();
    out.tasks.clear();
    out.requests.clear();
    while (!queue_.empty() && out.tasks.size() < max_batch_size_) {
        out.requests.push_back(queue_.front());
        out.tasks.push_back(queue_.front()->task);
        queue_.pop_front();
    }
    return true;
}
} // namespace vonnx
