#include "vonnx/continuous_scheduler.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
namespace vonnx {
void BatchJob::complete(std::vector<DetectionResult> results) {
    if (results.size() != requests.size())
        throw std::logic_error("Result/request count mismatch");
    // Prepare every throwing allocation before publishing any result. Callbacks
    // execute only after all promises in this batch have been fulfilled.
    std::vector<std::vector<DetectionResult>> own(requests.size()), callback_values(requests.size());
    static const char *names[] = {"api_entry", "admitted", "copy_done", "enqueued", "dequeued",
        "slot_ready", "compute_begin", "ort_enter", "ort_exit", "d2h_ready", "cpu_ready", "future_publish"};
    for (size_t i = 0; i < requests.size(); ++i) {
        own[i].push_back(std::move(results[i]));
        auto &request=*requests[i];
        if(request.trace_enabled) {
            request.trace[PendingRequest::CpuReady]=std::chrono::steady_clock::now();
            for(int j=0;j<PendingRequest::FuturePublish;++j)
                own[i][0].diagnostic_trace[names[j]]=std::chrono::duration<double,std::milli>(request.trace[j]-request.trace[0]).count();
            own[i][0].diagnostic_trace["future_publish"]=0;
        }
        if(request.callback) callback_values[i]=own[i];
    }
    for (size_t i = 0; i < requests.size(); ++i) {
        requests[i]->release();
        if(requests[i]->trace_enabled) {
            requests[i]->trace[PendingRequest::FuturePublish]=std::chrono::steady_clock::now();
            const double published=std::chrono::duration<double,std::milli>(
                requests[i]->trace[PendingRequest::FuturePublish]-requests[i]->trace[0]).count();
            own[i][0].diagnostic_trace["future_publish"]=published;
            if(requests[i]->callback) callback_values[i][0].diagnostic_trace["future_publish"]=published;
        }
        requests[i]->promise.set_value(std::move(own[i]));
    }
    for (size_t i = 0; i < requests.size(); ++i) {
        if (requests[i]->callback) {
            try {
                requests[i]->callback(callback_values[i]);
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
        uint8_t *staging, size_t page_size, std::function<void(void*,const void*,size_t)> input_copier)
    : max_batch_size_(batch), capacity_(capacity), max_latency_ms_(latency), concurrency_(concurrency),
      page_size_(page_size), staging_(staging), input_copier_(std::move(input_copier)) {
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
        InferenceCallback cb, std::chrono::steady_clock::time_point api_entry) {
    auto request = std::make_shared<PendingRequest>();
    request->task = task;
    request->callback = std::move(cb);
    auto future = request->promise.get_future();
    request->trace_enabled=api_entry!=std::chrono::steady_clock::time_point{};
    request->trace[PendingRequest::ApiEntry]=api_entry;
    std::unique_lock<std::mutex> lock(mutex_);
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
        }
        request->task.data_size = bytes;
    }
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
    if(request->trace_enabled) request->trace[PendingRequest::Admitted]=std::chrono::steady_clock::now();
    lock.unlock();
    try {
        if(!task.is_gpu_ptr && task.data_ptr) {
            if(!staging_) {
                request->owned_input.resize(request->task.data_size);
                request->task.data_ptr=request->owned_input.data();
            }
            auto *target=const_cast<void*>(request->task.data_ptr);
            if(input_copier_) input_copier_(target,task.data_ptr,request->task.data_size);
            else std::memcpy(target,task.data_ptr,request->task.data_size);
        }
        if(request->trace_enabled) request->trace[PendingRequest::CopyDone]=std::chrono::steady_clock::now();
    } catch(...) {
        lock.lock();
        auto it=std::find(queue_.begin(),queue_.end(),request);
        if(it!=queue_.end()) queue_.erase(it);
        --inflight_;
        if(page!=size_t(-1)) free_inputs_.push_back(page);
        request->release_input={};
        lock.unlock();cv_.notify_all();waiter_.notify();throw;
    }
    lock.lock();
    request->submitted=std::chrono::steady_clock::now();
    if(request->trace_enabled) request->trace[PendingRequest::Enqueued]=request->submitted;
    request->ready=true;
    lock.unlock();
    cv_.notify_one();
    waiter_.notify();
    return future;
}
bool ContinuousScheduler::pop_batch(BatchJob &out, int timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, std::chrono::milliseconds(timeout), [&] {
        return (!queue_.empty() && queue_.front()->ready) || (queue_.empty() && !accepting_); });
    if (queue_.empty() || !queue_.front()->ready)
        return false;
    const auto deadline =
        queue_.front()->submitted + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<double, std::milli>(max_latency_ms_));
    auto wait_begin = std::chrono::steady_clock::now();
    auto ready_count=[&] {size_t n=0; for(auto &r:queue_) {if(!r->ready || n==max_batch_size_) break; ++n;} return n;};
    while (ready_count() < max_batch_size_ && accepting_) {
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
    while (!queue_.empty() && queue_.front()->ready && out.tasks.size() < max_batch_size_) {
        if(queue_.front()->trace_enabled) queue_.front()->trace[PendingRequest::Dequeued]=out.dispatched;
        out.requests.push_back(queue_.front());
        out.tasks.push_back(queue_.front()->task);
        queue_.pop_front();
    }
    return true;
}
} // namespace vonnx
