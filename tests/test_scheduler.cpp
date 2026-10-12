#include "vonnx/continuous_scheduler.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
using namespace vonnx;
int main() {
    FrameTask task{};
    std::vector<uint8_t> image(12, 7);
    task.data_ptr = image.data();
    task.data_size = image.size();
    task.height = 2;
    task.step = 6;
    ContinuousScheduler s(4, 15, 4);
    int calls = 0;
    auto first = s.submit_task(task, [&](auto &r) {
        assert(r.size() == 1);
        ++calls;
        throw std::runtime_error("callback test");
    });
    task.frame_id = 2;
    auto second = s.submit_task(task, [&](auto &r) {
        assert(r[0].frame_id == 2);
        ++calls;
    });
    image[0] = 99;
    BatchJob b;
    const auto begin = std::chrono::steady_clock::now();
    assert(s.pop_batch(b, 0));
    assert(std::chrono::steady_clock::now() - begin >= std::chrono::milliseconds(10));
    assert(b.tasks.size() == 2 && static_cast<const uint8_t *>(b.tasks[0].data_ptr)[0] == 7);
    std::vector<DetectionResult> results(2);
    results[1].frame_id = 2;
    b.complete(results);
    assert(first.get().size() == 1 && second.get()[0].frame_id == 2 && calls == 2);
    auto error = s.submit_task(task);
    assert(s.pop_batch(b, 0));
    b.fail(std::make_exception_ptr(std::runtime_error("worker test")));
    try {
        error.get();
        assert(false);
    } catch (const std::runtime_error &) {
    }
    auto drain = s.submit_task(task);
    s.stop();
    assert(s.pop_batch(b, 0));
    b.complete({DetectionResult{}});
    drain.get();
    assert(!s.pop_batch(b, 0));
    try {
        s.submit_task(task);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    ContinuousScheduler bounded(1, 0, 1);
    auto queued = bounded.submit_task(task);
    try {
        bounded.submit_task(task);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    bounded.stop();
    assert(bounded.pop_batch(b, 0));
    b.complete({DetectionResult{}});
    queued.get();
    std::cout << "scheduler regression PASS\n";
}
