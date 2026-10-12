#pragma once
#include "vonnx/common.hpp"
#include <atomic>
namespace vonnx {
inline std::atomic<uint64_t> explicit_device_allocations{0};
inline thread_local uint64_t thread_device_allocations = 0;
template <class T> class DeviceBuffer {
    T *ptr_{nullptr};
    size_t count_{0};
    int device_{0};

  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void allocate(size_t count) {
        reset();
        if (!count)
            return;
        CUDA_CHECK(cudaGetDevice(&device_));
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        ++explicit_device_allocations;
        ++thread_device_allocations;
        count_ = count;
    }
    void reset() noexcept {
        if (ptr_) {
            int old = 0;
            cudaGetDevice(&old);
            cudaSetDevice(device_);
            cudaFree(ptr_);
            cudaSetDevice(old);
            ptr_ = nullptr;
            count_ = 0;
        }
    }
    T *get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
};
template <class T> class PinnedBuffer {
    T *ptr_{};
    size_t count_{};

  public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { reset(); }
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete;
    void allocate(size_t count) {
        reset();
        if (!count)
            return;
        CUDA_CHECK(cudaHostAlloc(&ptr_, count * sizeof(T), cudaHostAllocDefault));
        count_ = count;
    }
    void reset() noexcept {
        if (ptr_)
            cudaFreeHost(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }
    T *get() const { return ptr_; }
    size_t size() const { return count_; }
};
class CudaStream {
    cudaStream_t stream_{nullptr};
    int device_{0};

  public:
    CudaStream() = default;
    ~CudaStream() { reset(); }
    void reset() noexcept {
        if (stream_) {
            int old = 0;
            cudaGetDevice(&old);
            cudaSetDevice(device_);
            cudaStreamDestroy(stream_);
            cudaSetDevice(old);
            stream_ = nullptr;
        }
    }
    void create() {
        reset();
        CUDA_CHECK(cudaGetDevice(&device_));
        // Model streams must not inherit legacy-default-stream dependencies
        // from another session or application. Dependencies are explicit events.
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    cudaStream_t get() const { return stream_; }
};
class CudaEvent {
    cudaEvent_t event_{};
    int device_{};

  public:
    explicit CudaEvent(bool timing = false, bool enabled = true) {
        if (!enabled)
            return;
        CUDA_CHECK(cudaGetDevice(&device_));
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, timing ? cudaEventDefault : cudaEventDisableTiming));
    }
    ~CudaEvent() {
        if (event_) {
            int old{};
            cudaGetDevice(&old);
            cudaSetDevice(device_);
            cudaEventDestroy(event_);
            cudaSetDevice(old);
        }
    }
    void record(cudaStream_t s) {
        if (event_)
            CUDA_CHECK(cudaEventRecord(event_, s));
    }
    cudaEvent_t get() const { return event_; }
    float since(const CudaEvent &start) const {
        if (!event_ || !start.event_)
            return 0;
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }
};
} // namespace vonnx
