#pragma once
#include "vonnx/common.hpp"
#include "vonnx/cuda_resources.hpp"
#include <onnxruntime_cxx_api.h>
namespace vonnx {
struct ModelSpec {
    std::string task, input_name;
    std::string source_precision, effective_precision, prepared_model_path;
    std::string runtime_version;
    bool cuda_prefer_nhwc{false};
    int fixed_batch_size{0}; // 0=dynamic; padding never produces public results
    bool converted{false};
    bool graph_specialized{false};
    bool cuda_graph{false};
    std::vector<int64_t> source_input_shape, execution_input_shape;
    std::string source_model_sha256, prepared_model_sha256;
    int input_dtype{1};
    std::vector<int> output_dtypes;
    int width{}, height{}, classes{}, anchors{}, channels{}, keypoints{}, keypoint_dims{}, mask_channels{},
        mask_height{}, mask_width{}, stride{32};
    bool dynamic_batch{};
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
};
class ONNXRunner {
  public:
    ONNXRunner(const std::string &, int gpu = 0, int width = 640, int height = 640, int max_batch = 16,
               cudaStream_t stream = nullptr, size_t probe_budget = size_t(6144)*1024*1024,
               bool prefer_nhwc = false, int fixed_batch = 0, bool cuda_graph = false);
    ~ONNXRunner();
    const ModelSpec &spec() const { return spec_; }
    int get_input_width() const { return spec_.width; }
    int get_input_height() const { return spec_.height; }
    int get_num_classes() const { return spec_.classes; }
    int get_num_anchors() const { return spec_.anchors; }
    void run_zero_copy(const float *, const std::vector<float *> &outputs, int batch, cudaStream_t);
    void initialize_cuda(int gpu, cudaStream_t stream, size_t arena_bytes);
    void prepare_bindings(int slot, void *input, const std::vector<void *> &outputs, int max_batch);
    void run(int slot, int batch);

  private:
    // ORT CUDA providers may retain process-wide logging state. Keep the Env
    // alive across independent engine closes and Python ORT session creation.
    static Ort::Env &environment() {
        // Deliberately process-lifetime: Windows may unload Python's ORT/CUDA
        // providers before C++ static destructors. Destructing Env at that point
        // is unsafe. Sessions, arenas and engine CUDA resources still close.
        static Ort::Env *env=new Ort::Env{ORT_LOGGING_LEVEL_ERROR,"vONNX"};
        return *env;
    }
    Ort::Env &env_{environment()};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_;
    ModelSpec spec_;
    std::string path_;
    bool prefer_nhwc_{}, cuda_graph_{};
    Ort::RunOptions run_options_;
    struct Binding {
        std::unique_ptr<Ort::IoBinding> io;
        std::vector<Ort::Value> tensors;
        Ort::RunOptions options;
    };
    std::vector<std::vector<std::unique_ptr<Binding>>> bindings_;
};
} // namespace vonnx
