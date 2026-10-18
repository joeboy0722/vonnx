#include "vonnx_c_api.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <windows.h>
std::string environment_utf8(const wchar_t* name) {
    DWORD n=GetEnvironmentVariableW(name,nullptr,0);
    if(!n)return {};
    std::wstring value(n,L'\0');
    GetEnvironmentVariableW(name,value.data(),n);
    int bytes=WideCharToMultiByte(CP_UTF8,0,value.c_str(),-1,nullptr,0,nullptr,nullptr);
    std::string text(size_t(bytes),'\0');
    WideCharToMultiByte(CP_UTF8,0,value.c_str(),-1,text.data(),bytes,nullptr,nullptr);
    text.pop_back();return text;
}
// Input is tightly packed RGB24, not a JPEG/PNG file. Model path is UTF-8.
int run(int argc, char** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    // Compile with /DELAYLOAD:vonnx.dll so DLL directories are set before loading.
    wchar_t dirs[32768];
    DWORD length = GetEnvironmentVariableW(L"VONNX_DLL_DIRS", dirs, 32768);
    if (length && length < 32768) {
        std::wstring paths(dirs);
        size_t start = 0;
        while (start < paths.size()) {
            size_t end = paths.find(L';', start);
            auto path = paths.substr(start, end - start);
            if (!path.empty()) AddDllDirectory(path.c_str());
            if (end == std::wstring::npos) break;
            start = end + 1;
        }
    }
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: c_api.exe model.onnx image.rgb width height [auto|fp32|fp16]\n";
        return 2;
    }
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192) return 2;
    std::ifstream input(std::filesystem::u8path(argv[2]), std::ios::binary);
    std::vector<unsigned char> pixels((std::istreambuf_iterator<char>(input)), {});
    if (pixels.size() != size_t(width) * height * 3) return 2;
    vonnx_config_v3 config{};
    vonnx_config_init_v3(&config);
    std::string precision=argc==6?argv[5]:"auto";
    if(precision!="auto" && precision!="fp32" && precision!="fp16") return 2;
    config.precision=precision=="fp16"?2:precision=="fp32"?1:0;
    auto python=environment_utf8(L"VONNX_PYTHON");
    auto cache=environment_utf8(L"VONNX_MODEL_CACHE_DIR");
    config.precision_python=python.empty()?nullptr:python.c_str();
    config.model_cache_dir=cache.empty()?nullptr:cache.c_str();
    config.base.base.model_path = argv[1];
    config.base.base.input_height = config.base.base.input_width = 640;
    config.base.base.max_input_height = height;
    config.base.base.max_input_width = width;
    // This example submits a single image. For bulk requests use throughput,
    // max_batch_size=8 and concurrency=32 with independent caller threads.
    config.base.base.max_batch_size = 1;
    config.base.base.max_latency_ms = 0;
    config.base.batch_mode = 1;
    config.base.pipeline_depth = 1;
    config.base.concurrency = 1;
    auto engine = vonnx_create_engine_v3(&config);
    if (!engine) { std::cerr << vonnx_last_error() << '\n'; return 1; }
    vonnx_image_v1 image{};
    image.version = 1; image.struct_size = sizeof(image);
    image.data = pixels.data(); image.data_size = pixels.size();
    image.width = width; image.height = height; image.stride = width * 3;
    image.pixel_format = 1; // RGB24
    image.stream_id = 0; image.frame_id = 42;
    vonnx_result_v1* result = nullptr;
    const int status = vonnx_predict_v1(engine, &image, &result);
    if (status != 0) std::cerr << vonnx_last_error() << '\n';
    else {
        std::cout << "frame=" << result->frame_id << " boxes=" << result->count
                  << " task=" << result->task << " physical_batch="
                  << vonnx_result_physical_batch_v3(result) << '\n';
        // keypoints: count * keypoints_per_box; masks: count entries when present.
        vonnx_result_free(result);
    }
    vonnx_destroy_engine(engine);
    return status == 0 ? 0 : 1;
}
int wmain(int argc,wchar_t** wide) {
    std::vector<std::string> strings;strings.reserve(argc);
    std::vector<char*> args;args.reserve(argc);
    for(int i=0;i<argc;++i) {
        int n=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
        if(n<=0)return 2;
        std::string value(size_t(n),'\0');
        WideCharToMultiByte(CP_UTF8,0,wide[i],-1,value.data(),n,nullptr,nullptr);
        value.pop_back();strings.push_back(std::move(value));
    }
    for(auto &value:strings)args.push_back(value.data());
    return run(argc,args.data());
}
