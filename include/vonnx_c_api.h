#ifndef VONNX_C_API_H
#define VONNX_C_API_H
#include <stddef.h>
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
typedef struct {
    float x1, y1, x2, y2, score;
    int class_id;
} vonnx_bbox_t;
typedef void *vonnx_engine_handle_t;
VONNX_API vonnx_engine_handle_t vonnx_create_engine(const char *, int, float, float, float, int, int);
VONNX_API void vonnx_destroy_engine(vonnx_engine_handle_t);
VONNX_API int vonnx_predict(vonnx_engine_handle_t, uint64_t, uint64_t, const unsigned char *, int, int, int,
                            vonnx_bbox_t *, int);
/* v1 structures are initialized with vonnx_config_init. imgsz is (height,width).
   Host buffers may be released after predict returns. GPU buffers must remain
   valid until completion. NV12 is a contiguous Y+UV buffer, full range BT.601.
   Do not destroy an engine concurrently with API calls. Error text is thread-local. */
typedef struct {
    uint32_t version, struct_size;
    const char *model_path;
    int input_height, input_width, max_input_height, max_input_width;
    int max_batch_size, gpu_id, max_det, full_masks;
    float max_latency_ms, conf_threshold, iou_threshold;
    size_t queue_capacity, max_mask_bytes;
} vonnx_config_v1;
typedef struct {
    uint32_t version, struct_size;
    const void *data;
    size_t data_size;
    int width, height, stride, pixel_format, is_gpu;
    uintptr_t user_stream;
    uint64_t stream_id, frame_id;
} vonnx_image_v1;
typedef struct {
    float x, y, confidence;
} vonnx_keypoint_t;
typedef struct {
    int x, y, width, height;
    const uint8_t *data;
    size_t data_size;
} vonnx_mask_t;
typedef struct {
    uint64_t stream_id, frame_id;
    size_t count, keypoints_per_box;
    const vonnx_bbox_t *boxes;
    const vonnx_keypoint_t *keypoints;
    const vonnx_mask_t *masks;
    int task;     /* 0=detect, 1=pose, 2=segment */
    void *_owner; /* Internal; do not modify. */
} vonnx_result_v1;
VONNX_API void vonnx_config_init(vonnx_config_v1 *);
VONNX_API const char *vonnx_last_error(void);
VONNX_API vonnx_engine_handle_t vonnx_create_engine_v1(const vonnx_config_v1 *);
VONNX_API int vonnx_predict_v1(vonnx_engine_handle_t, const vonnx_image_v1 *, vonnx_result_v1 **);
VONNX_API void vonnx_result_free(vonnx_result_v1 *);
/* v2 retains all v1 layouts; v1 image/result descriptors work with v2 handles.
   batch_mode: 0=throughput, 1=latency. max_latency_ms=-1 selects mode default.
   pause drains and retains resources; stop is terminal; close releases resources.
   Destroy the handle after close. Lifecycle operations must not run in callbacks. */
typedef struct {
    uint32_t version, struct_size;
    vonnx_config_v1 base;
    size_t concurrency, gpu_memory_budget_mb;
    int batch_mode, pipeline_depth, collect_timings;
} vonnx_config_v2;
typedef struct {
    uint32_t version, struct_size;
    uint64_t initialization_count, explicit_device_allocations, hot_path_device_allocations;
    size_t reserved_device_bytes, arena_limit_bytes, initialized_device_bytes, current_device_bytes;
    int pipeline_depth, state, high_resolution_timer; /* state: 0=running 1=paused 2=stopped 3=closed */
} vonnx_resources_v2;
VONNX_API void vonnx_config_init_v2(vonnx_config_v2 *);
VONNX_API vonnx_engine_handle_t vonnx_create_engine_v2(const vonnx_config_v2 *);
VONNX_API int vonnx_pause_v2(vonnx_engine_handle_t);
VONNX_API int vonnx_resume_v2(vonnx_engine_handle_t);
VONNX_API int vonnx_stop_v2(vonnx_engine_handle_t);
VONNX_API int vonnx_close_v2(vonnx_engine_handle_t);
VONNX_API int vonnx_resources_get_v2(vonnx_engine_handle_t, vonnx_resources_v2 *);
#ifdef __cplusplus
}
#endif
#endif
