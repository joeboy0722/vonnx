#include "vonnx/onnx_runner.hpp"

namespace vonnx {

ONNXRunner::ONNXRunner(const std::string& model_path, int use_gpu_id) {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 嘗試啟用 CUDA Execution Provider
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = use_gpu_id;
    session_options_.AppendExecutionProvider_CUDA(cuda_options);

    // 實體化 ONNX Session
#ifdef _WIN32
    std::wstring w_model_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, w_model_path.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    // 解析 Input Tensor 名稱與維度
    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name_ptr.get();

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();

    if (input_dims.size() == 4) {
        if (input_dims[2] > 0) input_h_ = static_cast<int>(input_dims[2]);
        if (input_dims[3] > 0) input_w_ = static_cast<int>(input_dims[3]);
    }

    // 解析 Output Tensor 名稱與維度 (適配 YOLOv8/v11 3D 或傳統 4D)
    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = output_name_ptr.get();

    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();

    output_shape_pattern_.clear();

    if (output_dims.size() == 3) {
        int channels = (output_dims[1] > 0) ? static_cast<int>(output_dims[1]) : 84;
        num_classes_ = std::max(1, channels - 4);
        num_anchors_ = (output_dims[2] > 0) ? static_cast<int>(output_dims[2]) : 8400;
        output_shape_pattern_ = {channels, num_anchors_};
    } else if (output_dims.size() == 4) {
        int channels = (output_dims[1] > 0) ? static_cast<int>(output_dims[1]) : 84;
        num_classes_ = std::max(1, channels - 4);
        int h = (output_dims[2] > 0) ? static_cast<int>(output_dims[2]) : 1;
        int w = (output_dims[3] > 0) ? static_cast<int>(output_dims[3]) : 1;
        num_anchors_ = h * w;
        output_shape_pattern_ = {channels, h, w};
    } else {
        num_classes_ = 80;
        num_anchors_ = 8400;
        output_shape_pattern_ = {84, 8400};
    }
}

void ONNXRunner::run_zero_copy(
    const float* d_input_tensor,
    float* d_output_tensor,
    int batch_size,
    cudaStream_t stream
) {
    Ort::IoBinding io_binding(*session_);

    // 建立動態 Batch 輸入維度 [B, 3, H, W]
    std::vector<int64_t> input_shape = {batch_size, 3, input_h_, input_w_};
    size_t input_elements = batch_size * 3 * input_h_ * input_w_;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda_,
        const_cast<float*>(d_input_tensor),
        input_elements,
        input_shape.data(),
        input_shape.size()
    );

    // 建立動態 Batch 輸出維度 (精確對應模型 Shape Rank)
    std::vector<int64_t> output_shape;
    output_shape.push_back(batch_size);
    size_t output_elements = batch_size;
    for (auto dim : output_shape_pattern_) {
        output_shape.push_back(dim);
        output_elements *= dim;
    }

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda_,
        d_output_tensor,
        output_elements,
        output_shape.data(),
        output_shape.size()
    );

    // 綁定輸入與輸出 (GPU 記憶體零拷貝介面)
    io_binding.BindInput(input_name_.c_str(), input_tensor);
    io_binding.BindOutput(output_name_.c_str(), output_tensor);

    // 設定與 CUDA Stream 同步
    Ort::RunOptions run_options;
    run_options.AddConfigEntry("cuda_stream", std::to_string(reinterpret_cast<uintptr_t>(stream)).c_str());

    // 執行 ONNX 推論
    session_->Run(run_options, io_binding);
}

} // namespace vonnx
