#pragma once

#include "vonnx/common.hpp"
#include <onnxruntime_cxx_api.h>

namespace vonnx {

// ONNX Runtime 零拷貝推論執行器 (使用 Ort::IoBinding)
class ONNXRunner {
public:
    ONNXRunner(const std::string& model_path, int use_gpu_id = 0);
    ~ONNXRunner() = default;

    // 取得模型導出的預設輸入高寬 (例如 640x640)
    int get_input_width() const { return input_w_; }
    int get_input_height() const { return input_h_; }
    int get_num_classes() const { return num_classes_; }
    int get_num_anchors() const { return num_anchors_; }

    // 使用 Ort::IoBinding 執行零拷貝 (GPU Pointer to GPU Pointer) 的 Forward 推論
    void run_zero_copy(
        const float* d_input_tensor,   // Preprocess 產生的 [B, 3, H, W] GPU 指標
        float* d_output_tensor,        // 預配置放 NMS 輸入的 [B, 4+C, N] GPU 指標
        int batch_size,
        cudaStream_t stream
    );

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "vONNXEngine"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_cuda_{"Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault};

    std::string input_name_;
    std::string output_name_;

    int input_w_{640};
    int input_h_{640};
    int num_classes_{80};
    int num_anchors_{8400};

    std::vector<int64_t> output_shape_pattern_;
};

} // namespace vonnx
