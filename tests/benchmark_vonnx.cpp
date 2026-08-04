#include "vonnx/vonnx_engine.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <numeric>

using namespace vonnx;

int main(int argc, char** argv) {
    std::cout << "========================================================" << std::endl;
    std::cout << "  vONNX Engine: YOLOv8 / YOLOv11 高併發非同步效能測試  " << std::endl;
    std::cout << "========================================================" << std::endl;

    std::string model_path = "yolov8n.onnx";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "載入模型檔: " << model_path << std::endl;

    EngineConfig config;
    config.model_path = model_path;
    config.max_batch_size = 16;
    config.max_latency_ms = 2.0;
    config.conf_threshold = 0.25f;
    config.iou_threshold = 0.45f;
    config.max_gpu_pages = 64;
    config.gpu_id = 0;

    try {
        vONNXEngine engine(config);

        // 模擬 16 路非同步 RTSP / 相機碼流
        int num_streams = 16;
        int frames_per_stream = 50;
        std::atomic<int> completed_frames{0};
        std::vector<double> latencies_ms;
        std::mutex lat_mutex;

        // 建立 1920x1080 假影像 Buffer (BGR24)
        int img_w = 1920;
        int img_h = 1080;
        std::vector<uint8_t> dummy_bgr(img_h * img_w * 3, 128);

        std::cout << "啟動 " << num_streams << " 路非同步串流，總計測試 "
                  << (num_streams * frames_per_stream) << " 幀..." << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> workers;
        for (int s = 0; s < num_streams; ++s) {
            workers.emplace_back([&, s]() {
                for (int f = 0; f < frames_per_stream; ++f) {
                    FrameTask task;
                    task.stream_id = s;
                    task.frame_id = f;
                    task.data_ptr = dummy_bgr.data();
                    task.is_gpu_ptr = false;
                    task.width = img_w;
                    task.height = img_h;
                    task.step = img_w * 3;
                    task.format = PixelFormat::BGR24;

                    auto frame_t0 = std::chrono::high_resolution_clock::now();

                    // 非同步提交推論任務
                    auto fut = engine.predict_async(task, [&](const std::vector<DetectionResult>& results) {
                        auto frame_t1 = std::chrono::high_resolution_clock::now();
                        double ms = std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();
                        
                        {
                            std::lock_guard<std::mutex> lock(lat_mutex);
                            latencies_ms.push_back(ms);
                        }
                        completed_frames++;
                    });

                    // 模擬 30 FPS 的幀率間隔 (約 33 ms)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            });
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }

        // 等待所有非同步任務完成
        while (completed_frames < num_streams * frames_per_stream) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();

        double avg_latency = 0.0;
        if (!latencies_ms.empty()) {
            avg_latency = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
        }

        std::cout << "\n------------------ 測試結果 ------------------" << std::endl;
        std::cout << "總處理時間: " << total_sec << " 秒" << std::endl;
        std::cout << "總處理幀數: " << completed_frames << " 幀" << std::endl;
        std::cout << "系統整體吞吐量: " << (completed_frames / total_sec) << " FPS" << std::endl;
        std::cout << "平均單幀延遲: " << avg_latency << " ms" << std::endl;
        std::cout << "----------------------------------------------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "測試過程發生例外: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
