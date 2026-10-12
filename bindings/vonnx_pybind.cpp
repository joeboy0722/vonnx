#include "vonnx/vonnx_engine.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
namespace py = pybind11;
using namespace vonnx;
struct PredictionFuture {
    std::future<std::vector<DetectionResult>> future;
};
static FrameTask numpy_task(uint64_t stream, uint64_t frame, py::array_t<uint8_t, 0> array, PixelFormat fmt) {
    auto info = array.request();
    int h, w, channels = fmt == PixelFormat::RGBA32 ? 4 : 3;
    if (fmt == PixelFormat::NV12) {
        if (info.ndim != 2 || info.shape[0] % 3 || info.strides[1] != 1)
            throw std::invalid_argument("NV12 expects contiguous [H*3/2,W]");
        h = int(info.shape[0] * 2 / 3);
        w = int(info.shape[1]);
    } else {
        if (info.ndim != 3 || info.shape[2] != channels || info.strides[2] != 1 ||
            info.strides[1] != channels)
            throw std::invalid_argument("Expected interleaved uint8 HWC array with matching channels");
        h = int(info.shape[0]);
        w = int(info.shape[1]);
    }
    if (info.strides[0] <= 0 || info.strides[0] != (fmt == PixelFormat::NV12 ? w : w * channels))
        throw std::invalid_argument("Python arrays must be C-contiguous; use numpy.ascontiguousarray");
    FrameTask task{};
    task.stream_id = stream;
    task.frame_id = frame;
    task.data_ptr = info.ptr;
    task.width = w;
    task.height = h;
    task.step = int(info.strides[0]);
    task.format = fmt;
    task.data_size = size_t(info.size);
    return task;
}
PYBIND11_MODULE(vonnx_py, m) {
    m.def("load_image", [](const std::string &path) {
        int w, h, c;
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open image: " + path);
        std::vector<uint8_t> encoded((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
        if (encoded.size() > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Image file too large");
        auto *data = stbi_load_from_memory(encoded.data(), int(encoded.size()), &w, &h, &c, 3);
        if (!data)
            throw std::runtime_error("Cannot load image: " + path);
        py::capsule owner(data, [](void *p) { stbi_image_free(p); });
        return py::array_t<uint8_t>({h, w, 3}, {w * 3, 3, 1}, data, owner);
    });
    py::enum_<PixelFormat>(m, "PixelFormat")
        .value("BGR24", PixelFormat::BGR24)
        .value("RGB24", PixelFormat::RGB24)
        .value("RGBA32", PixelFormat::RGBA32)
        .value("NV12", PixelFormat::NV12)
        .export_values();
    py::class_<BBox>(m, "BBox")
        .def_readonly("x1", &BBox::x1)
        .def_readonly("y1", &BBox::y1)
        .def_readonly("x2", &BBox::x2)
        .def_readonly("y2", &BBox::y2)
        .def_readonly("score", &BBox::score)
        .def_readonly("class_id", &BBox::class_id);
    py::class_<DetectionResult::Keypoint>(m, "Keypoint")
        .def_readonly("x", &DetectionResult::Keypoint::x)
        .def_readonly("y", &DetectionResult::Keypoint::y)
        .def_readonly("confidence", &DetectionResult::Keypoint::confidence);
    py::class_<DetectionResult::Mask>(m, "Mask")
        .def_readonly("x", &DetectionResult::Mask::x)
        .def_readonly("y", &DetectionResult::Mask::y)
        .def_readonly("width", &DetectionResult::Mask::width)
        .def_readonly("height", &DetectionResult::Mask::height)
        .def_property_readonly("data", [](const DetectionResult::Mask &mask) {
            py::array_t<uint8_t> a({mask.height, mask.width});
            std::copy(mask.data.begin(), mask.data.end(), a.mutable_data());
            return a;
        });
    py::class_<DetectionResult>(m, "DetectionResult")
        .def_readonly("stream_id", &DetectionResult::stream_id)
        .def_readonly("frame_id", &DetectionResult::frame_id)
        .def_readonly("boxes", &DetectionResult::boxes)
        .def_readonly("keypoints", &DetectionResult::keypoints)
        .def_readonly("masks", &DetectionResult::masks)
        .def_readonly("batch_size", &DetectionResult::batch_size)
        .def_readonly("debug_input", &DetectionResult::debug_input)
        .def_readonly("debug_outputs", &DetectionResult::debug_outputs)
        .def_readonly("queue_ms", &DetectionResult::queue_ms)
        .def_readonly("batch_wait_ms", &DetectionResult::batch_wait_ms)
        .def_readonly("dispatch_ms", &DetectionResult::dispatch_ms)
        .def_readonly("h2d_ms", &DetectionResult::h2d_ms)
        .def_readonly("preprocess_ms", &DetectionResult::preprocess_ms)
        .def_readonly("inference_ms", &DetectionResult::inference_ms)
        .def_readonly("postprocess_ms", &DetectionResult::postprocess_ms)
        .def_readonly("d2h_ms", &DetectionResult::d2h_ms);
    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &EngineConfig::model_path)
        .def_readwrite("max_batch_size", &EngineConfig::max_batch_size)
        .def_readwrite("max_latency_ms", &EngineConfig::max_latency_ms)
        .def_readwrite("batch_mode", &EngineConfig::batch_mode)
        .def_readwrite("concurrency", &EngineConfig::concurrency)
        .def_readwrite("pipeline_depth", &EngineConfig::pipeline_depth)
        .def_readwrite("gpu_memory_budget_mb", &EngineConfig::gpu_memory_budget_mb)
        .def_readwrite("collect_timings", &EngineConfig::collect_timings)
        .def_readwrite("conf_threshold", &EngineConfig::conf_threshold)
        .def_readwrite("iou_threshold", &EngineConfig::iou_threshold)
        .def_readwrite("max_gpu_pages", &EngineConfig::max_gpu_pages)
        .def_readwrite("gpu_id", &EngineConfig::gpu_id)
        .def_readwrite("input_width", &EngineConfig::input_width)
        .def_readwrite("input_height", &EngineConfig::input_height)
        .def_readwrite("max_input_width", &EngineConfig::max_input_width)
        .def_readwrite("max_input_height", &EngineConfig::max_input_height)
        .def_readwrite("queue_capacity", &EngineConfig::queue_capacity)
        .def_readwrite("max_det", &EngineConfig::max_det)
        .def_readwrite("capture_debug", &EngineConfig::capture_debug)
        .def_readwrite("full_masks", &EngineConfig::full_masks)
        .def_readwrite("max_mask_bytes", &EngineConfig::max_mask_bytes)
        .def_property(
            "imgsz", [](const EngineConfig &c) { return py::make_tuple(c.input_height, c.input_width); },
            [](EngineConfig &c, py::object size) {
                if (py::isinstance<py::int_>(size)) {
                    c.input_width = c.input_height = size.cast<int>();
                } else {
                    auto v = size.cast<std::vector<int>>();
                    if (v.size() != 2)
                        throw std::invalid_argument("imgsz expects int or (height,width)");
                    c.input_height = v[0];
                    c.input_width = v[1];
                }
            });
    py::class_<ModelSpec>(m, "ModelSpec")
        .def_readonly("task", &ModelSpec::task)
        .def_readonly("width", &ModelSpec::width)
        .def_readonly("height", &ModelSpec::height)
        .def_readonly("classes", &ModelSpec::classes)
        .def_readonly("anchors", &ModelSpec::anchors)
        .def_readonly("keypoints", &ModelSpec::keypoints)
        .def_readonly("keypoint_dims", &ModelSpec::keypoint_dims)
        .def_readonly("output_shapes", &ModelSpec::output_shapes)
        .def_readonly("output_names", &ModelSpec::output_names);
    py::class_<PredictionFuture>(m, "PredictionFuture")
        .def(
            "result",
            [](PredictionFuture &f, double timeout) {
                py::gil_scoped_release release;
                if (timeout >= 0 &&
                    f.future.wait_for(std::chrono::duration<double>(timeout)) != std::future_status::ready)
                    throw std::runtime_error("Prediction timeout");
                return f.future.get();
            },
            py::arg("timeout") = -1.0);
    py::class_<DeadlineWaiter>(m, "DeadlineWaiter")
        .def(py::init<>())
        .def("notify", &DeadlineWaiter::notify)
        .def("wait", &DeadlineWaiter::wait, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("backend", &DeadlineWaiter::backend);
    using Stats = vONNXEngine::ResourceStats;
    py::class_<Stats>(m, "ResourceStats")
        .def_readonly("initialization_count", &Stats::initialization_count)
        .def_readonly("explicit_device_allocations", &Stats::explicit_device_allocations)
        .def_readonly("hot_path_device_allocations", &Stats::hot_path_device_allocations)
        .def_readonly("reserved_device_bytes", &Stats::reserved_device_bytes)
        .def_readonly("arena_limit_bytes", &Stats::arena_limit_bytes)
        .def_readonly("initialized_device_bytes", &Stats::initialized_device_bytes)
        .def_readonly("current_device_bytes", &Stats::current_device_bytes)
        .def_readonly("pipeline_depth", &Stats::pipeline_depth)
        .def_readonly("timer_backend", &Stats::timer_backend)
        .def_readonly("state", &Stats::state);
    py::class_<vONNXEngine>(m, "vONNXEngine")
        .def(py::init([](const EngineConfig &c) {
            py::gil_scoped_release release;
            return std::make_unique<vONNXEngine>(c);
        }))
        .def("start", &vONNXEngine::start)
        .def("stop", &vONNXEngine::stop, py::call_guard<py::gil_scoped_release>())
        .def("pause", &vONNXEngine::pause, py::call_guard<py::gil_scoped_release>())
        .def("resume", &vONNXEngine::resume, py::call_guard<py::gil_scoped_release>())
        .def("close", &vONNXEngine::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("resource_stats",
                               [](const vONNXEngine &e) {
                                   py::gil_scoped_release release;
                                   return e.resource_stats();
                               })
        .def_property_readonly("model_spec", &vONNXEngine::model_spec,
                               py::return_value_policy::reference_internal)
        .def(
            "predict_numpy",
            [](vONNXEngine &self, uint64_t stream, uint64_t frame, py::array_t<uint8_t, 0> array,
               PixelFormat fmt) {
                auto task = numpy_task(stream, frame, array, fmt);
                py::gil_scoped_release release;
                return self.predict_sync(task);
            },
            py::arg("stream_id"), py::arg("frame_id"), py::arg("array").noconvert(),
            py::arg("format") = PixelFormat::BGR24)
        .def(
            "submit_numpy",
            [](vONNXEngine &self, uint64_t stream, uint64_t frame, py::array_t<uint8_t, 0> array,
               PixelFormat fmt) {
                auto task = numpy_task(stream, frame, array, fmt);
                py::gil_scoped_release release;
                return PredictionFuture{self.predict_async(task)};
            },
            py::arg("stream_id"), py::arg("frame_id"), py::arg("array").noconvert(),
            py::arg("format") = PixelFormat::BGR24);
}
