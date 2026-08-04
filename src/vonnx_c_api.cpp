#define VONNX_EXPORTS
#include "vonnx_c_api.h"
#include "vonnx/vonnx_engine.hpp"
#include <cstring>
#include <vector>

using namespace vonnx;

VONNX_API vonnx_engine_handle_t vonnx_create_engine(
    const char* model_path,
    int max_batch_size,
    float max_latency_ms,
    float conf_threshold,
    float iou_threshold,
    int max_gpu_pages,
    int gpu_id
) {
    try {
        EngineConfig config;
        config.model_path = model_path ? model_path : "yolov8n.onnx";
        config.max_batch_size = max_batch_size > 0 ? max_batch_size : 16;
        config.max_latency_ms = max_latency_ms > 0.0f ? max_latency_ms : 2.0f;
        config.conf_threshold = conf_threshold > 0.0f ? conf_threshold : 0.25f;
        config.iou_threshold = iou_threshold > 0.0f ? iou_threshold : 0.45f;
        config.max_gpu_pages = max_gpu_pages > 0 ? max_gpu_pages : 64;
        config.gpu_id = gpu_id >= 0 ? gpu_id : 0;

        auto* engine = new vONNXEngine(config);
        return reinterpret_cast<vonnx_engine_handle_t>(engine);
    } catch (const std::exception& ex) {
        printf("[vONNX C API Error] %s\n", ex.what());
        return nullptr;
    } catch (...) {
        printf("[vONNX C API Error] Unknown exception in vonnx_create_engine\n");
        return nullptr;
    }
}

VONNX_API void vonnx_destroy_engine(vonnx_engine_handle_t handle) {
    if (!handle) return;
    auto* engine = reinterpret_cast<vONNXEngine*>(handle);
    delete engine;
}

VONNX_API int vonnx_predict(
    vonnx_engine_handle_t handle,
    uint64_t stream_id,
    uint64_t frame_id,
    const unsigned char* image_data,
    int width,
    int height,
    int pixel_format,
    vonnx_bbox_t* out_boxes,
    int max_out_boxes
) {
    if (!handle || !image_data || !out_boxes || max_out_boxes <= 0) {
        return -1;
    }

    try {
        auto* engine = reinterpret_cast<vONNXEngine*>(handle);

        FrameTask task;
        task.stream_id = stream_id;
        task.frame_id = frame_id;
        task.data_ptr = const_cast<unsigned char*>(image_data);
        task.is_gpu_ptr = false;
        task.width = width;
        task.height = height;

        if (pixel_format == 0) {
            task.format = PixelFormat::BGR24;
            task.step = width * 3;
        } else if (pixel_format == 1) {
            task.format = PixelFormat::RGB24;
            task.step = width * 3;
        } else if (pixel_format == 2) {
            task.format = PixelFormat::RGBA32;
            task.step = width * 4;
        } else if (pixel_format == 3) {
            task.format = PixelFormat::NV12;
            task.step = width;
        } else {
            task.format = PixelFormat::BGR24;
            task.step = width * 3;
        }

        std::vector<DetectionResult> results = engine->predict_sync(task);
        if (results.empty()) return 0;

        const auto& boxes = results[0].boxes;
        int copy_count = std::min(static_cast<int>(boxes.size()), max_out_boxes);

        for (int i = 0; i < copy_count; ++i) {
            out_boxes[i].x1 = boxes[i].x1;
            out_boxes[i].y1 = boxes[i].y1;
            out_boxes[i].x2 = boxes[i].x2;
            out_boxes[i].y2 = boxes[i].y2;
            out_boxes[i].score = boxes[i].score;
            out_boxes[i].class_id = boxes[i].class_id;
        }

        return copy_count;
    } catch (...) {
        return -1;
    }
}
