// Native C++ client: independent CPU producers, no Python runtime or GIL.
#include "vonnx_c_api.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
struct Api {
    decltype(&vonnx_config_init_v2) init;
    decltype(&vonnx_create_engine_v2) create;
    decltype(&vonnx_predict_v1) predict;
    decltype(&vonnx_result_free) free_result;
    decltype(&vonnx_destroy_engine) destroy;
    decltype(&vonnx_last_error) error;
};
struct Row {
    int model, lane;
    uint64_t frame;
    double ms;
};
int wmain(int argc, wchar_t **argv) {
    if (argc != 3)
        return 2;
    auto root = std::filesystem::path(argv[1]);
    auto output = std::filesystem::path(argv[2]);
    std::filesystem::create_directories(output);
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    AddDllDirectory((root / "vonnx_standalone_sdk").c_str());
    AddDllDirectory(L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64");
    auto library_folder = root / "vonnx_standalone_sdk";
    HMODULE dll =
        LoadLibraryExW((library_folder / "vonnx.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!dll) {
        std::cerr << "LoadLibrary error " << GetLastError() << std::endl;
        return 3;
    }
    Api api{};
#define LOAD(field, name)                                                                                    \
    api.field = reinterpret_cast<decltype(api.field)>(GetProcAddress(dll, #name));                           \
    if (!api.field)                                                                                          \
        return 4;
    LOAD(init, vonnx_config_init_v2);
    LOAD(create, vonnx_create_engine_v2);
    LOAD(predict, vonnx_predict_v1);
    LOAD(free_result, vonnx_result_free);
    LOAD(destroy, vonnx_destroy_engine);
    LOAD(error, vonnx_last_error);
#undef LOAD
    const char *labels[] = {"detect_n_1", "detect_n_2", "detect_s", "pose_l"};
    const char *models[] = {"yolov8n", "yolov8n", "yolov8s", "yolov8l-pose"};
    std::vector<std::vector<uint8_t>> images[2];
    for (int task = 0; task < 2; ++task)
        for (int i = 0; i < 8; ++i) {
            std::ifstream file(root / "runs/stream_isolation/inputs" /
                                   ((task ? "pose_" : "detect_") + std::to_string(i) + ".bin"),
                               std::ios::binary);
            images[task].emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            if (images[task].back().size() != 1920 * 1080 * 3)
                return 5;
        }
    std::vector<vonnx_engine_handle_t> engines;
    std::ofstream json(output / "result.json"), rows(output / "requests.jsonl");
    try {
        for (int i = 0; i < 4; ++i) {
            std::cout << "Initialize " << labels[i] << std::endl;
            vonnx_config_v2 c{};
            api.init(&c);
            auto path = (root / "models" / models[i] / (std::string(models[i]) + ".onnx")).u8string();
            c.base.model_path = path.c_str();
            c.base.input_height = c.base.input_width = 1920;
            c.base.max_input_width = 1920;
            c.base.max_input_height = 1080;
            c.base.max_batch_size = 1;
            c.base.max_latency_ms = 0;
            c.batch_mode = 1;
            c.pipeline_depth = 1;
            c.collect_timings = 0;
            c.concurrency = 32;
            auto handle = api.create(&c);
            if (!handle)
                throw std::runtime_error(api.error());
            engines.push_back(handle);
        }
        auto predict = [&](int model, int lane, uint64_t frame) {
            auto &pixels = images[model == 3][frame % 8];
            vonnx_image_v1 input{};
            input.version = 1;
            input.struct_size = sizeof(input);
            input.data = pixels.data();
            input.data_size = pixels.size();
            input.width = 1920;
            input.height = 1080;
            input.stride = 1920 * 3;
            input.pixel_format = 1;
            input.stream_id = lane;
            input.frame_id = frame;
            vonnx_result_v1 *r = nullptr;
            if (api.predict(engines[model], &input, &r) != 0)
                throw std::runtime_error(api.error());
            bool correct = r && r->frame_id == frame && r->stream_id == uint64_t(lane);
            api.free_result(r);
            if (!correct)
                throw std::runtime_error("Request ID mismatch");
        };
        for (int i = 0; i < 4; ++i)
            for (int n = 0; n < 5; ++n)
                predict(i, 0, n);
        json << "{\"status\":\"pending_cleanup\",\"client\":\"C++ C API "
                "v2\",\"imgsz\":1920,\"batch\":1,\"depth\":1,\"scenarios\":[";
        bool comma = false;
        auto measure = [&](int only, int lanes, double seconds) {
            std::atomic<int> ready{0};
            std::atomic<bool> go{false}, stop{false};
            std::mutex mutex;
            std::vector<Row> data;
            std::exception_ptr error;
            std::vector<std::thread> threads;
            for (int model = 0; model < 4; ++model)
                if (only < 0 || only == model)
                    for (int lane = 0; lane < lanes; ++lane)
                        threads.emplace_back([&, model, lane] {
                            ++ready;
                            while (!go)
                                std::this_thread::sleep_for(100us);
                            try {
                                for (uint64_t frame = lane; !stop; frame += lanes) {
                                    auto begin = Clock::now();
                                    predict(model, lane, frame);
                                    Row row{model, lane, frame,
                                            std::chrono::duration<double, std::milli>(Clock::now() - begin)
                                                .count()};
                                    std::lock_guard<std::mutex> lock(mutex);
                                    data.push_back(row);
                                }
                            } catch (...) {
                                std::lock_guard<std::mutex> lock(mutex);
                                error = std::current_exception();
                                stop = true;
                            }
                        });
            while (ready != int(threads.size()))
                std::this_thread::sleep_for(100us);
            auto begin = Clock::now();
            go = true;
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
            stop = true;
            for (auto &thread : threads)
                thread.join();
            if (error)
                std::rethrow_exception(error);
            double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
            if (comma)
                json << ",";
            comma = true;
            auto name =
                only < 0 ? "concurrent_" + std::to_string(lanes) : "idle_" + std::string(labels[only]);
            json << "{\"name\":\"" << name << "\",\"lanes\":" << lanes << ",\"seconds\":" << elapsed
                 << ",\"total_fps\":" << data.size() / elapsed << ",\"per_model\":[";
            bool item = false;
            for (int model = 0; model < 4; ++model)
                if (only < 0 || only == model) {
                    std::vector<double> latencies;
                    for (auto &r : data)
                        if (r.model == model)
                            latencies.push_back(r.ms);
                    double sum = 0;
                    for (double v : latencies)
                        sum += v;
                    std::sort(latencies.begin(), latencies.end());
                    if (latencies.empty())
                        throw std::runtime_error("No completed requests");
                    if (item)
                        json << ",";
                    item = true;
                    json << "{\"model\":\"" << labels[model] << "\",\"success\":" << latencies.size()
                         << ",\"fps\":" << latencies.size() / elapsed
                         << ",\"mean_ms\":" << sum / latencies.size()
                         << ",\"p95_ms\":" << latencies[size_t(.95 * (latencies.size() - 1))] << "}";
                }
            json << "]}";
            json.flush();
            for (auto &r : data)
                rows << "{\"scenario\":\"" << name << "\",\"model\":\"" << labels[r.model]
                     << "\",\"lane\":" << r.lane << ",\"frame\":" << r.frame << ",\"latency_ms\":" << r.ms
                     << "}\n";
            std::cout << name << " total_fps=" << data.size() / elapsed << std::endl;
        };
        for (int i = 0; i < 4; ++i)
            measure(i, 1, 3);
        measure(-1, 1, 8);
        measure(-1, 4, 8);
        json << "]}\n";
        json.close();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        for (auto e : engines)
            api.destroy(e);
        return 1;
    }
    for (auto e : engines)
        api.destroy(e);
    FreeLibrary(dll);
    std::ifstream completed(output / "result.json");
    std::string text((std::istreambuf_iterator<char>(completed)), std::istreambuf_iterator<char>());
    completed.close();
    std::string pending = "\"status\":\"pending_cleanup\"";
    text.replace(text.find(pending), pending.size(), "\"status\":\"passed\"");
    std::ofstream verified(output / "result.json");
    verified << text;
}
