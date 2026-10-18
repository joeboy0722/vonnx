#include "vonnx/continuous_scheduler.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <cstring>
#include <atomic>
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
    // Copy is outside the scheduler lock, but reservation order is FIFO and
    // capacity includes copies which have not yet been published.
    std::promise<void> copying, release_copy;
    auto release_signal=release_copy.get_future().share();
    std::atomic<int> copies{0};
    std::vector<uint8_t> pinned(24);
    ContinuousScheduler concurrent(2,0,2,2,pinned.data(),12,[&](void *dst,const void *src,size_t n) {
        if(copies.fetch_add(1)==0) {copying.set_value();release_signal.wait();}
        std::memcpy(dst,src,n);
    });
    task.frame_id=10;
    auto submit_one=std::async(std::launch::async,[&]{return concurrent.submit_task(task);});
    copying.get_future().wait();
    FrameTask next=task;next.frame_id=11;
    auto submit_two=std::async(std::launch::async,[&]{return concurrent.submit_task(next);});
    assert(submit_two.wait_for(std::chrono::seconds(2))==std::future_status::ready);
    auto two=submit_two.get();
    assert(concurrent.pending_count()==2);
    assert(!concurrent.pop_batch(b,0));
    try {concurrent.submit_task(next);assert(false);} catch(const std::runtime_error&) {}
    concurrent.pause();
    release_copy.set_value();
    auto one=submit_one.get();
    assert(concurrent.pop_batch(b,0));
    assert(b.tasks.size()==2 && b.tasks[0].frame_id==10 && b.tasks[1].frame_id==11);
    b.complete({DetectionResult{},DetectionResult{}});one.get();two.get();
    concurrent.resume();
    auto again=concurrent.submit_task(next);concurrent.stop();
    assert(concurrent.pop_batch(b,0));b.complete({DetectionResult{}});again.get();
    // A failed copy releases both the page and admission capacity, with no FIFO hole.
    bool fail_copy=true;
    ContinuousScheduler rollback(1,0,1,1,pinned.data(),12,[&](void *dst,const void *src,size_t n) {
        if(fail_copy) throw std::runtime_error("injected copy failure");
        std::memcpy(dst,src,n);
    });
    try {rollback.submit_task(task);assert(false);} catch(const std::runtime_error&) {}
    assert(rollback.pending_count()==0);
    fail_copy=false;
    auto recovered=rollback.submit_task(task);
    assert(rollback.pop_batch(b,0));b.complete({DetectionResult{}});recovered.get();
    // The first callback must see the entire batch's futures already fulfilled.
    ContinuousScheduler callbacks(2,0,2);
    std::future<std::vector<DetectionResult>> callback_second;
    std::promise<void> callback_running,callback_release;
    auto callback_signal=callback_release.get_future().share();
    auto callback_first=callbacks.submit_task(task,[&](auto&) {
        assert(callback_second.wait_for(std::chrono::milliseconds(0))==std::future_status::ready);
        callback_running.set_value();
        callback_signal.wait();
        throw std::runtime_error("isolated callback");
    });
    callback_second=callbacks.submit_task(task);
    assert(callbacks.pop_batch(b,0));
    auto complete=std::async(std::launch::async,[&]{b.complete({DetectionResult{},DetectionResult{}});});
    assert(callback_running.get_future().wait_for(std::chrono::seconds(2))==std::future_status::ready);
    assert(callback_first.wait_for(std::chrono::milliseconds(0))==std::future_status::ready);
    assert(callback_second.wait_for(std::chrono::milliseconds(0))==std::future_status::ready);
    // A callback blocked on model X must not block an unrelated scheduler N.
    ContinuousScheduler independent(1,0,1);
    auto independent_result=independent.submit_task(task);
    BatchJob independent_batch;
    assert(independent.pop_batch(independent_batch,0));
    independent_batch.complete({DetectionResult{}});
    assert(independent_result.wait_for(std::chrono::milliseconds(0))==std::future_status::ready);
    independent_result.get();
    callback_release.set_value();complete.get();
    callback_first.get();callback_second.get();
    std::cout << "scheduler regression PASS\n";
}
