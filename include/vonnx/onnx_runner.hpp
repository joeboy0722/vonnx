#pragma once
#include "vonnx/common.hpp"
#include "vonnx/cuda_resources.hpp"
#include <onnxruntime_cxx_api.h>
namespace vonnx {
struct ModelSpec {
    std::string task, input_name;
    int width{}, height{}, classes{}, anchors{}, channels{}, keypoints{}, keypoint_dims{}, mask_channels{},
        mask_height{}, mask_width{}, stride{32};
    bool dynamic_batch{};
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
};
class ONNXRunner {
  public:
    ONNXRunner(const std::string &, int gpu = 0, int width = 640, int height = 640, int max_batch = 16,
               cudaStream_t stream = nullptr);
    ~ONNXRunner();
    const ModelSpec &spec() const { return spec_; }
    int get_input_width() const { return spec_.width; }
    int get_input_height() const { return spec_.height; }
    int get_num_classes() const { return spec_.classes; }
    int get_num_anchors() const { return spec_.anchors; }
    void run_zero_copy(const float *, const std::vector<float *> &outputs, int batch, cudaStream_t);
    void initialize_cuda(int gpu, cudaStream_t stream, size_t arena_bytes);
    void prepare_bindings(int slot, float *input, const std::vector<float *> &outputs, int max_batch);
    void run(int slot, int batch);

  private:
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "vONNX"};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_;
    ModelSpec spec_;
    std::string path_;
    Ort::RunOptions run_options_;
    struct Binding {
        std::unique_ptr<Ort::IoBinding> io;
        std::vector<Ort::Value> tensors;
    };
    std::vector<std::vector<std::unique_ptr<Binding>>> bindings_;
};
} // namespace vonnx
