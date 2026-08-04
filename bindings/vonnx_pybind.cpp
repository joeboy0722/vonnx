#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "vonnx/vonnx_engine.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace py = pybind11;
using namespace vonnx;

PYBIND11_MODULE(vonnx_py, m) {
    m.doc() = "vONNX Engine: 高併發非同步 YOLOv8/v11 CUDA/ONNX 視覺推論引擎 Python 綁定";

    // 暴露輕量化 C++ 影像讀取工具 (無須 OpenCV / PIL 依賴)
    m.def("load_image", [](const std::string& filepath) {
        int w, h, channels;
        unsigned char* data = stbi_load(filepath.c_str(), &w, &h, &channels, 3);
        if (!data) {
            throw std::runtime_error("無法開啟並載入圖片檔案: " + filepath);
        }
        py::capsule free_when_done(data, [](void* f) { stbi_image_free(f); });
        return py::array_t<uint8_t>(
            {h, w, 3},
            {w * 3, 3, 1},
            data,
            free_when_done
        );
    }, py::arg("filepath"), "輕量化讀取 JPG/PNG 圖片為 Numpy RGB24 陣列");

    // 暴露 PixelFormat 列舉
    py::enum_<PixelFormat>(m, "PixelFormat")
        .value("BGR24", PixelFormat::BGR24)
        .value("RGB24", PixelFormat::RGB24)
        .value("RGBA32", PixelFormat::RGBA32)
        .value("NV12", PixelFormat::NV12)
        .export_values();

    // 暴露 BBox 結構
    py::class_<BBox>(m, "BBox")
        .def_readwrite("x1", &BBox::x1)
        .def_readwrite("y1", &BBox::y1)
        .def_readwrite("x2", &BBox::x2)
        .def_readwrite("y2", &BBox::y2)
        .def_readwrite("score", &BBox::score)
        .def_readwrite("class_id", &BBox::class_id)
        .def("__repr__", [](const BBox& b) {
            return "<BBox class=" + std::to_string(b.class_id) +
                   " score=" + std::to_string(b.score) +
                   " [" + std::to_string(b.x1) + ", " + std::to_string(b.y1) +
                   ", " + std::to_string(b.x2) + ", " + std::to_string(b.y2) + "]>";
        });

    // 暴露 DetectionResult 結構
    py::class_<DetectionResult>(m, "DetectionResult")
        .def_readwrite("stream_id", &DetectionResult::stream_id)
        .def_readwrite("frame_id", &DetectionResult::frame_id)
        .def_readwrite("boxes", &DetectionResult::boxes);

    // 暴露 EngineConfig 結構
    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &EngineConfig::model_path)
        .def_readwrite("max_batch_size", &EngineConfig::max_batch_size)
        .def_readwrite("max_latency_ms", &EngineConfig::max_latency_ms)
        .def_readwrite("conf_threshold", &EngineConfig::conf_threshold)
        .def_readwrite("iou_threshold", &EngineConfig::iou_threshold)
        .def_readwrite("max_gpu_pages", &EngineConfig::max_gpu_pages)
        .def_readwrite("gpu_id", &EngineConfig::gpu_id);

    // 暴露 vONNXEngine 主類別
    py::class_<vONNXEngine>(m, "vONNXEngine")
        .def(py::init<const EngineConfig&>(), py::arg("config"))
        .def("start", &vONNXEngine::start)
        .def("stop", &vONNXEngine::stop)
        .def("predict_numpy", [](vONNXEngine& self, uint64_t stream_id, uint64_t frame_id, py::array_t<uint8_t> array, PixelFormat fmt) {
            py::buffer_info info = array.request();
            if (info.ndim < 2) {
                throw std::runtime_error("Numpy array 的維度必須至少為 2D (Height, Width)！");
            }
            int h = static_cast<int>(info.shape[0]);
            int w = static_cast<int>(info.shape[1]);
            int step = static_cast<int>(info.strides[0]);

            FrameTask task;
            task.stream_id = stream_id;
            task.frame_id = frame_id;
            task.data_ptr = info.ptr;
            task.is_gpu_ptr = false;
            task.width = w;
            task.height = h;
            task.step = step;
            task.format = fmt;
            task.user_stream = nullptr;

            return self.predict_sync(task);
        }, py::arg("stream_id"), py::arg("frame_id"), py::arg("array"), py::arg("format") = PixelFormat::BGR24);
}
