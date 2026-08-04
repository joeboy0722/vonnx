#ifndef VONNX_C_API_H
#define VONNX_C_API_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef VONNX_EXPORTS
    #define VONNX_API __declspec(dllexport)
  #else
    #define VONNX_API __declspec(dllimport)
  #endif
#else
  #define VONNX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 邊界框結構體 (與 C/C++/C#/Python ctypes 相容)
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} vonnx_bbox_t;

// 引擎控制句柄指標
typedef void* vonnx_engine_handle_t;

// 1. 建立 vONNX GPU 推論引擎
VONNX_API vonnx_engine_handle_t vonnx_create_engine(
    const char* model_path,
    int max_batch_size,
    float max_latency_ms,
    float conf_threshold,
    float iou_threshold,
    int max_gpu_pages,
    int gpu_id
);

// 2. 銷毀 vONNX 引擎
VONNX_API void vonnx_destroy_engine(vonnx_engine_handle_t handle);

// 3. 執行推論 (像素格式: BGR24=0, RGB24=1, RGBA32=2, NV12=3)
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
);

#ifdef __cplusplus
}
#endif

#endif // VONNX_C_API_H
