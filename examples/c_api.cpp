#include "vonnx_c_api.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <windows.h>
// Input is tightly packed RGB24, not a JPEG/PNG file. Model path is UTF-8.
int main(int argc, char** argv) {
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
    if (argc != 5) {
        std::cerr << "Usage: c_api.exe model.onnx image.rgb width height\n";
        return 2;
    }
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192) return 2;
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<unsigned char> pixels((std::istreambuf_iterator<char>(input)), {});
    if (pixels.size() != size_t(width) * height * 3) return 2;
    vonnx_config_v2 config{};
    vonnx_config_init_v2(&config);
    config.base.model_path = argv[1];
    config.base.input_height = config.base.input_width = 640;
    config.base.max_input_height = height;
    config.base.max_input_width = width;
    config.base.max_batch_size = 8;
    config.concurrency = 32;
    auto engine = vonnx_create_engine_v2(&config);
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
                  << " task=" << result->task << '\n';
        // keypoints: count * keypoints_per_box; masks: count entries when present.
        vonnx_result_free(result);
    }
    vonnx_destroy_engine(engine);
    return status == 0 ? 0 : 1;
}
