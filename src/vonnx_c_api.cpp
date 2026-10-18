#define VONNX_EXPORTS
#include "vonnx_c_api.h"
#include "vonnx/vonnx_engine.hpp"
#include <algorithm>
#include <cstring>
using namespace vonnx;
namespace {
thread_local std::string error_text;
void error() {
    try {
        throw;
    } catch (const std::exception &e) {
        error_text = e.what();
    } catch (...) {
        error_text = "Unknown native error";
    }
}
struct OwnedResult {
    vonnx_result_v1 view{};
    DetectionResult result;
    std::vector<vonnx_bbox_t> boxes;
    std::vector<vonnx_keypoint_t> keypoints;
    std::vector<vonnx_mask_t> masks;
};
EngineConfig from_v1(const vonnx_config_v1 *c) {
    if (!c || c->version != 1 || c->struct_size != sizeof(*c) || !c->model_path)
        throw std::invalid_argument("Invalid config version/size/model path");
    EngineConfig cfg;
    cfg.model_path = c->model_path;
    cfg.precision = "fp32";
    cfg.input_height = c->input_height;
    cfg.input_width = c->input_width;
    cfg.max_input_height = c->max_input_height;
    cfg.max_input_width = c->max_input_width;
    cfg.max_batch_size = c->max_batch_size;
    cfg.gpu_id = c->gpu_id;
    cfg.max_det = c->max_det;
    cfg.full_masks = c->full_masks != 0;
    cfg.max_latency_ms = c->max_latency_ms;
    cfg.conf_threshold = c->conf_threshold;
    cfg.iou_threshold = c->iou_threshold;
    cfg.queue_capacity = c->queue_capacity;
    cfg.max_mask_bytes = c->max_mask_bytes;
    cfg.pipeline_depth = 1;
    cfg.concurrency = 0;
    return cfg;
}
int lifecycle(vonnx_engine_handle_t handle, void (vONNXEngine::*method)()) {
    error_text.clear();
    try {
        if (!handle)
            throw std::invalid_argument("Null engine handle");
        (static_cast<vONNXEngine *>(handle)->*method)();
        return 0;
    } catch (...) {
        error();
        return -1;
    }
}
} // namespace
VONNX_API const char *vonnx_last_error() { return error_text.c_str(); }
VONNX_API void vonnx_config_init(vonnx_config_v1 *c) {
    if (!c)
        return;
    *c = {};
    c->version = 1;
    c->struct_size = sizeof(*c);
    c->input_height = c->input_width = 640;
    c->max_input_height = 1080;
    c->max_input_width = 1920;
    c->max_batch_size = 16;
    c->max_det = 300;
    c->max_latency_ms = 2;
    c->conf_threshold = .25f;
    c->iou_threshold = .7f;
    c->queue_capacity = 1024;
    c->max_mask_bytes = 64 * 1024 * 1024;
}
VONNX_API vonnx_engine_handle_t vonnx_create_engine_v1(const vonnx_config_v1 *c) {
    error_text.clear();
    try {
        return new vONNXEngine(from_v1(c));
    } catch (...) {
        error();
        return nullptr;
    }
}
VONNX_API vonnx_engine_handle_t vonnx_create_engine(const char *path, int batch, float latency, float conf,
                                                    float iou, int pages, int gpu) {
    error_text.clear();
    try {
        EngineConfig cfg;
        cfg.model_path = path ? path : "yolov8n.onnx";
        cfg.precision = "fp32";
        cfg.max_batch_size = batch > 0 ? batch : 16;
        cfg.max_latency_ms = latency > 0 ? latency : 2;
        cfg.conf_threshold = conf >= 0 ? conf : .25f;
        cfg.iou_threshold = iou >= 0 ? iou : .45f;
        cfg.max_gpu_pages = pages > 0 ? pages : 0;
        cfg.gpu_id = gpu >= 0 ? gpu : 0;
        cfg.pipeline_depth = 1;
        cfg.concurrency = 0;
        return new vONNXEngine(cfg);
    } catch (...) {
        error();
        return nullptr;
    }
}
VONNX_API void vonnx_config_init_v2(vonnx_config_v2 *c) {
    if (!c)
        return;
    *c = {};
    c->version = 2;
    c->struct_size = sizeof(*c);
    vonnx_config_init(&c->base);
    c->base.max_batch_size = 8;
    c->base.max_latency_ms = -1;
    c->concurrency = 32;
    c->gpu_memory_budget_mb = 6144;
}
VONNX_API vonnx_engine_handle_t vonnx_create_engine_v2(const vonnx_config_v2 *c) {
    error_text.clear();
    try {
        if (!c || c->version != 2 || c->struct_size != sizeof(*c) || c->batch_mode < 0 || c->batch_mode > 1)
            throw std::invalid_argument("Invalid v2 configuration");
        auto cfg = from_v1(&c->base);
        if (!c->concurrency)
            throw std::invalid_argument("v2 concurrency must be positive");
        cfg.concurrency = c->concurrency;
        cfg.gpu_memory_budget_mb = c->gpu_memory_budget_mb;
        cfg.batch_mode = c->batch_mode ? "latency" : "throughput";
        cfg.pipeline_depth = c->pipeline_depth;
        cfg.collect_timings = c->collect_timings != 0;
        return new vONNXEngine(cfg);
    } catch (...) {
        error();
        return nullptr;
    }
}
VONNX_API int vonnx_pause_v2(vonnx_engine_handle_t h) { return lifecycle(h, &vONNXEngine::pause); }
VONNX_API void vonnx_config_init_v3(vonnx_config_v3 *c) {
    if(!c) return;
    *c={}; c->version=3;c->struct_size=sizeof(*c);vonnx_config_init_v2(&c->base);
}
static vonnx_engine_handle_t create_v3(const vonnx_config_v3 *c, bool specialize=false,bool graph=false) {
    error_text.clear();
    try {
        if(!c || c->version!=3 || c->struct_size!=sizeof(*c) || c->precision<0 || c->precision>2 ||
           c->batch_shape_policy<0 || c->batch_shape_policy>2 ||
           c->base.version!=2 || c->base.struct_size!=sizeof(c->base) ||
           c->base.batch_mode<0 || c->base.batch_mode>1 || !c->base.concurrency)
            throw std::invalid_argument("Invalid v3 configuration");
        auto cfg=from_v1(&c->base.base);
        cfg.graph_specialization=specialize?"auto":"off";
        cfg.cuda_graph=graph;
        cfg.concurrency=c->base.concurrency;
        cfg.gpu_memory_budget_mb=c->base.gpu_memory_budget_mb;
        cfg.batch_mode=c->base.batch_mode?"latency":"throughput";
        cfg.pipeline_depth=c->base.pipeline_depth;
        cfg.collect_timings=c->base.collect_timings!=0;
        cfg.precision=c->precision==2?"fp16":c->precision==1?"fp32":"auto";
        cfg.batch_shape_policy=c->batch_shape_policy==2?"fixed":c->batch_shape_policy==1?"dynamic":"auto";
        if(c->model_cache_dir) cfg.model_cache_dir=c->model_cache_dir;
        if(c->precision_python) cfg.precision_python=c->precision_python;
        if(c->precision_tool) cfg.precision_tool=c->precision_tool;
        return new vONNXEngine(cfg);
    } catch(...) {error();return nullptr;}
}
VONNX_API vonnx_engine_handle_t vonnx_create_engine_v3(const vonnx_config_v3 *c) {return create_v3(c);}
VONNX_API void vonnx_config_init_v4(vonnx_config_v4 *c) {
    if(!c)return;
    *c={};c->version=4;c->struct_size=sizeof(*c);vonnx_config_init_v3(&c->base);
}
VONNX_API vonnx_engine_handle_t vonnx_create_engine_v4(const vonnx_config_v4 *c) {
    error_text.clear();
    if(!c || c->version!=4 || c->struct_size!=sizeof(*c) ||
       c->graph_specialization<0 || c->graph_specialization>1 || c->cuda_graph<0 || c->cuda_graph>1) {
        error_text="Invalid v4 configuration";return nullptr;
    }
    return create_v3(&c->base,c->graph_specialization==1,c->cuda_graph==1);
}
VONNX_API int vonnx_model_info_get_v4(vonnx_engine_handle_t h,vonnx_model_info_v4 *out) {
    error_text.clear();
    try {
        if(!h || !out || out->version!=4 || out->struct_size!=sizeof(*out))
            throw std::invalid_argument("Invalid v4 model descriptor");
        const auto &s=static_cast<vONNXEngine*>(h)->model_spec();
        out->graph_specialized=s.graph_specialized;
        out->cuda_graph=s.cuda_graph;
        for(int i=0;i<4;++i) {
            out->source_input_shape[i]=s.source_input_shape.at(i);
            out->execution_input_shape[i]=s.execution_input_shape.at(i);
        }
        if(s.source_model_sha256.size()>64 || s.prepared_model_sha256.size()>64)
            throw std::runtime_error("Invalid model fingerprint");
        std::memset(out->source_model_sha256,0,sizeof(out->source_model_sha256));
        std::memset(out->prepared_model_sha256,0,sizeof(out->prepared_model_sha256));
        std::memcpy(out->source_model_sha256,s.source_model_sha256.data(),s.source_model_sha256.size());
        std::memcpy(out->prepared_model_sha256,s.prepared_model_sha256.data(),s.prepared_model_sha256.size());
        return 0;
    }catch(...){error();return -1;}
}
VONNX_API int vonnx_model_info_get_v3(vonnx_engine_handle_t h,vonnx_model_info_v3 *out) {
    error_text.clear();
    try {
        if(!h || !out || out->version!=3 || out->struct_size!=sizeof(*out))
            throw std::invalid_argument("Invalid v3 model descriptor");
        const auto &s=static_cast<vONNXEngine*>(h)->model_spec();
        if(s.prepared_model_path.size()>=sizeof(out->prepared_model_path))
            throw std::runtime_error("Prepared model path exceeds C API descriptor capacity");
        auto code=[](const std::string& p){return p=="fp32"?1:p=="fp16"?2:3;};
        out->source_precision=code(s.source_precision);out->effective_precision=code(s.effective_precision);
        out->converted=s.converted;out->input_dtype=s.input_dtype;
        out->output_count=int(s.output_dtypes.size());
        out->fixed_batch_size=s.fixed_batch_size;
        for(int i=0;i<2;++i) out->output_dtypes[i]=i<out->output_count?s.output_dtypes[i]:0;
        std::memcpy(out->prepared_model_path,s.prepared_model_path.c_str(),s.prepared_model_path.size()+1);
        return 0;
    }catch(...){error();return -1;}
}
VONNX_API int vonnx_resume_v2(vonnx_engine_handle_t h) { return lifecycle(h, &vONNXEngine::resume); }
VONNX_API int vonnx_stop_v2(vonnx_engine_handle_t h) { return lifecycle(h, &vONNXEngine::stop); }
VONNX_API int vonnx_close_v2(vonnx_engine_handle_t h) { return lifecycle(h, &vONNXEngine::close); }
VONNX_API int vonnx_resources_get_v2(vonnx_engine_handle_t h, vonnx_resources_v2 *out) {
    error_text.clear();
    try {
        if (!h || !out || out->version != 2 || out->struct_size != sizeof(*out))
            throw std::invalid_argument("Invalid v2 resource descriptor");
        auto s = static_cast<vONNXEngine *>(h)->resource_stats();
        out->initialization_count = s.initialization_count;
        out->explicit_device_allocations = s.explicit_device_allocations;
        out->hot_path_device_allocations = s.hot_path_device_allocations;
        out->reserved_device_bytes = s.reserved_device_bytes;
        out->arena_limit_bytes = s.arena_limit_bytes;
        out->initialized_device_bytes = s.initialized_device_bytes;
        out->current_device_bytes = s.current_device_bytes;
        out->pipeline_depth = s.pipeline_depth;
        out->state = s.state == "closed" ? 3 : s.state == "stopped" ? 2 : s.state == "paused" ? 1 : 0;
        out->high_resolution_timer = s.timer_backend == "windows_high_resolution_timer";
        return 0;
    } catch (...) {
        error();
        return -1;
    }
}
VONNX_API void vonnx_destroy_engine(vonnx_engine_handle_t handle) {
    error_text.clear();
    try {
        delete static_cast<vONNXEngine *>(handle);
    } catch (...) {
        error();
    }
}
VONNX_API int vonnx_predict_v1(vonnx_engine_handle_t handle, const vonnx_image_v1 *image,
                               vonnx_result_v1 **output) {
    error_text.clear();
    if (output)
        *output = nullptr;
    try {
        if (!handle || !image || !output || image->version != 1 || image->struct_size != sizeof(*image) ||
            !image->data_size || image->pixel_format < 0 || image->pixel_format > 3)
            throw std::invalid_argument("Invalid image descriptor");
        auto *engine = static_cast<vONNXEngine *>(handle);
        FrameTask task{};
        task.data_ptr = image->data;
        task.data_size = image->data_size;
        task.width = image->width;
        task.height = image->height;
        task.step = image->stride;
        task.format = static_cast<PixelFormat>(image->pixel_format);
        task.is_gpu_ptr = image->is_gpu != 0;
        task.user_stream = reinterpret_cast<cudaStream_t>(image->user_stream);
        task.stream_id = image->stream_id;
        task.frame_id = image->frame_id;
        auto owned = std::make_unique<OwnedResult>();
        owned->result = engine->predict_sync(task).at(0);
        auto &r = owned->result;
        for (auto &b : r.boxes)
            owned->boxes.push_back({b.x1, b.y1, b.x2, b.y2, b.score, b.class_id});
        for (auto &kp : r.keypoints)
            for (auto &p : kp)
                owned->keypoints.push_back({p.x, p.y, p.confidence});
        for (auto &mask : r.masks)
            owned->masks.push_back(
                {mask.x, mask.y, mask.width, mask.height, mask.data.data(), mask.data.size()});
        auto &v = owned->view;
        v.stream_id = r.stream_id;
        v.frame_id = r.frame_id;
        v.count = r.boxes.size();
        v.keypoints_per_box = engine->model_spec().keypoints;
        v.boxes = owned->boxes.data();
        v.keypoints = owned->keypoints.data();
        v.masks = owned->masks.empty() ? nullptr : owned->masks.data();
        v.task = engine->model_spec().task == "detect" ? 0 : engine->model_spec().task == "pose" ? 1 : 2;
        v._owner = owned.get();
        *output = &owned.release()->view;
        return 0;
    } catch (...) {
        error();
        return -1;
    }
}
VONNX_API void vonnx_result_free(vonnx_result_v1 *result) {
    if (result)
        delete static_cast<OwnedResult *>(result->_owner);
}
VONNX_API int vonnx_result_physical_batch_v3(const vonnx_result_v1 *result) {
    error_text.clear();
    if(!result || !result->_owner) {error_text="Invalid result";return -1;}
    return static_cast<const OwnedResult*>(result->_owner)->result.physical_batch_size;
}
VONNX_API int vonnx_predict(vonnx_engine_handle_t handle, uint64_t stream, uint64_t frame,
                            const unsigned char *data, int w, int h, int fmt, vonnx_bbox_t *boxes,
                            int capacity) {
    error_text.clear();
    try {
        if (!handle || !data || !boxes || capacity <= 0 || fmt < 0 || fmt > 3 || w <= 0 || h <= 0)
            throw std::invalid_argument("Invalid legacy input");
        auto *engine = static_cast<vONNXEngine *>(handle);
        if (engine->model_spec().task != "detect")
            throw std::invalid_argument("Legacy bbox API only supports Detect; use vonnx_predict_v1");
        FrameTask t{};
        t.stream_id = stream;
        t.frame_id = frame;
        t.data_ptr = data;
        t.width = w;
        t.height = h;
        t.step = w * (fmt == 2 ? 4 : fmt == 3 ? 1 : 3);
        t.format = static_cast<PixelFormat>(fmt);
        t.is_gpu_ptr = false;
        auto r = engine->predict_sync(t).at(0);
        int count = std::min(int(r.boxes.size()), capacity);
        for (int i = 0; i < count; ++i) {
            auto &b = r.boxes[i];
            boxes[i] = {b.x1, b.y1, b.x2, b.y2, b.score, b.class_id};
        }
        return count;
    } catch (...) {
        error();
        return -1;
    }
}
