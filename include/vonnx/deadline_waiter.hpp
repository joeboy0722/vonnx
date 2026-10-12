#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace vonnx {
// One notification event and one timer, retained for the owner's lifetime.
class DeadlineWaiter {
#ifdef _WIN32
    HANDLE notification_{}, timer_{};
    bool high_resolution_{};
#else
    std::mutex mutex_;
    std::condition_variable cv_;
    bool notified_{};
#endif
  public:
    DeadlineWaiter() {
#ifdef _WIN32
        notification_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0x2, TIMER_ALL_ACCESS);
        high_resolution_ = timer_ != nullptr;
        if (!timer_)
            timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
        if (!notification_ || !timer_) {
            if (notification_)
                CloseHandle(notification_);
            if (timer_)
                CloseHandle(timer_);
            throw std::runtime_error("Unable to initialize deadline timer/event");
        }
#endif
    }
    DeadlineWaiter(const DeadlineWaiter &) = delete;
    DeadlineWaiter &operator=(const DeadlineWaiter &) = delete;
    ~DeadlineWaiter() {
#ifdef _WIN32
        CloseHandle(timer_);
        CloseHandle(notification_);
#endif
    }
    std::string backend() const {
#ifdef _WIN32
        return high_resolution_ ? "windows_high_resolution_timer" : "windows_standard_timer";
#else
        return "condition_variable";
#endif
    }
    void notify() {
#ifdef _WIN32
        SetEvent(notification_);
#else
        {
            std::lock_guard<std::mutex> lock(mutex_);
            notified_ = true;
        }
        cv_.notify_one();
#endif
    }
    void wait(double seconds) {
        if (seconds <= 0)
            return;
#ifdef _WIN32
        LARGE_INTEGER due{};
        due.QuadPart = -std::max<LONGLONG>(1, static_cast<LONGLONG>(seconds * 1e7));
        if (!SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE))
            throw std::runtime_error("Unable to arm deadline timer");
        HANDLE handles[] = {notification_, timer_};
        DWORD result = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
        CancelWaitableTimer(timer_);
        if (result == WAIT_FAILED)
            throw std::runtime_error("Deadline wait failed");
#else
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::duration<double>(seconds), [&] { return notified_; });
        notified_ = false;
#endif
    }
};
} // namespace vonnx
