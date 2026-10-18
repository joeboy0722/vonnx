#include "vonnx/onnx_runner.hpp"
#include <algorithm>
#include <filesystem>
#include <regex>
#include <sstream>
#include <fstream>
namespace vonnx {
ONNXRunner::~ONNXRunner() {
    bindings_.clear();
    session_.reset();
}
ONNXRunner::ONNXRunner(const std::string &path, int gpu, int width, int height, int max_batch,
                       cudaStream_t stream, size_t probe_budget, bool prefer_nhwc, int fixed_batch, bool cuda_graph)
    : memory_("Cuda", OrtDeviceAllocator, gpu, OrtMemTypeDefault), path_(path), prefer_nhwc_(prefer_nhwc),cuda_graph_(cuda_graph) {
    spec_.cuda_prefer_nhwc=prefer_nhwc;
    spec_.cuda_graph=cuda_graph;
    spec_.runtime_version=OrtGetApiBase()->GetVersionString();
    options_.SetIntraOpNumThreads(1);
    options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // Metadata inspection never executes the graph on CPU. The initialization
    // probe below executes with CUDA, including models with FP16 operators.
    initialize_cuda(gpu,stream,probe_budget);
    Ort::AllocatorWithDefaultOptions allocator;
    auto metadata = session_->GetModelMetadata();
    auto read = [&](const char *key) {
        auto v = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        return v ? std::string(v.get()) : std::string();
    };
    spec_.task = read("task");
    if (spec_.task != "detect" && spec_.task != "pose" && spec_.task != "segment")
        throw std::invalid_argument("Model requires task metadata: detect/pose/segment");
    if (read("end2end") == "True" || read("args").find("'nms': True") != std::string::npos)
        throw std::invalid_argument("Only raw YOLOv8 outputs (nms=False) supported");
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != (spec_.task == "segment" ? 2 : 1))
        throw std::invalid_argument("Unexpected model input/output count");
    const std::string stride = read("stride");
    if (!stride.empty())
        spec_.stride = std::stoi(stride);
    if (width <= 0 || height <= 0 || spec_.stride <= 0 || width % spec_.stride || height % spec_.stride)
        throw std::invalid_argument("imgsz must be positive stride multiples");
    auto input = session_->GetInputTypeInfo(0);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto dims = input_info.GetShape();
    spec_.source_input_shape=dims;
    spec_.graph_specialized=read("vonnx.graph_specialized")=="1";
    if(spec_.graph_specialized) {
        spec_.source_input_shape.clear();
        std::stringstream values(read("vonnx.source_input_shape")); std::string value;
        while(std::getline(values,value,',')) spec_.source_input_shape.push_back(std::stoll(value));
        if(spec_.source_input_shape.size()!=4) throw std::invalid_argument("Invalid specialized source shape");
        spec_.source_model_sha256=read("vonnx.source_sha256");
        auto record_path=std::filesystem::u8path(path);record_path.replace_extension(".json");
        std::ifstream record(record_path);
        std::string text((std::istreambuf_iterator<char>(record)),{}); std::smatch hash;
        if(!std::regex_search(text,hash,std::regex("\"sha256\"\\s*:\\s*\"([a-f0-9]{64})\"")))
            throw std::invalid_argument("Missing specialized cache fingerprint");
        spec_.prepared_model_sha256=hash[1];
    }
    spec_.input_dtype = int(input_info.GetElementType());
    if ((spec_.input_dtype != 1 && spec_.input_dtype != 10) || dims.size() != 4 ||
        dims[1] != 3)
        throw std::invalid_argument("Only FP32/FP16 NCHW RGB models supported");
    if ((dims[2] > 0 && dims[2] != height) || (dims[3] > 0 && dims[3] != width))
        throw std::invalid_argument("imgsz does not match static ONNX shape");
    spec_.dynamic_batch = spec_.source_input_shape[0] < 0;
    if (!spec_.dynamic_batch && (spec_.source_input_shape[0] != 1 || max_batch != 1))
        throw std::invalid_argument("Static batch requires batch=1 and max_batch_size=1");
    const int probe_batch=spec_.graph_specialized && fixed_batch?fixed_batch:1;
    if(dims[0]>0 && dims[0]!=probe_batch)
        throw std::invalid_argument("Prepared model batch does not match execution batch");
    spec_.execution_input_shape={fixed_batch?fixed_batch:dims[0],3,height,width};
    spec_.width = width;
    spec_.height = height;
    spec_.input_name = session_->GetInputNameAllocated(0, allocator).get();
    const std::string names = read("names");
    std::regex class_key("[,{]\\s*[\"']?([0-9]+)[\"']?\\s*:");
    int expected = 0;
    for (std::sregex_iterator it(names.begin(), names.end(), class_key), end; it != end; ++it) {
        if (std::stoi((*it)[1]) != expected++)
            throw std::invalid_argument("Model names must contain contiguous class IDs");
    }
    spec_.classes = expected;
    if (!expected)
        throw std::invalid_argument("Missing class names metadata");
    if (spec_.task == "pose") {
        std::string shape = read("kpt_shape");
        std::regex digit("[0-9]+");
        std::vector<int> values;
        for (std::sregex_iterator it(shape.begin(), shape.end(), digit), end; it != end; ++it)
            values.push_back(std::stoi(it->str()));
        if (values.size() != 2 || values[0] <= 0 || (values[1] != 2 && values[1] != 3))
            throw std::invalid_argument("Invalid kpt_shape metadata");
        spec_.keypoints = values[0];
        spec_.keypoint_dims = values[1];
    }
    if(size_t(height)>SIZE_MAX/size_t(width)/3/size_t(probe_batch)/size_t(spec_.input_dtype==10?2:4))
        throw std::invalid_argument("Initialization probe size overflow");
    const size_t probe_bytes=size_t(probe_batch)*3*height*width*(spec_.input_dtype==10?2:4);
    if(probe_bytes>probe_budget) throw std::runtime_error("Initialization probe exceeds GPU memory budget");
    std::vector<uint8_t> probe(probe_bytes, 0);
    auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[4] = {probe_batch, 3, height, width};
    auto tensor = Ort::Value::CreateTensor(cpu, probe.data(), probe.size(), shape, 4,
                    ONNXTensorElementDataType(spec_.input_dtype));
    Ort::IoBinding binding(*session_);
    binding.BindInput(spec_.input_name.c_str(), tensor);
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        spec_.output_names.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
        auto type = session_->GetOutputTypeInfo(i);
        int dtype=int(type.GetTensorTypeAndShapeInfo().GetElementType());
        if (dtype != 1 && dtype != 10)
            throw std::invalid_argument("Only FP32/FP16 outputs supported");
        spec_.output_dtypes.push_back(dtype);
        binding.BindOutput(spec_.output_names.back().c_str(), cpu);
    }
    Ort::RunOptions probe_options;
    if(cuda_graph_)probe_options.AddConfigEntry("gpu_graph_id","-1");
    session_->Run(probe_options, binding);
    auto outputs = binding.GetOutputValues();
    for (auto &output : outputs) {
        auto sh = output.GetTensorTypeAndShapeInfo().GetShape();
        for (auto d : sh)
            if (d <= 0)
                throw std::runtime_error("Unresolved output shape");
        spec_.output_shapes.push_back(sh);
    }
    auto pred = spec_.output_shapes[0];
    if (pred.size() != 3 || pred[0] != probe_batch)
        throw std::invalid_argument("Expected prediction [B,C,N]");
    spec_.channels = int(pred[1]);
    spec_.anchors = int(pred[2]);
    if (spec_.task == "segment") {
        auto proto = spec_.output_shapes[1];
        if (proto.size() != 4 || proto[0] != probe_batch)
            throw std::invalid_argument("Expected prototype [B,M,H,W]");
        spec_.mask_channels = int(proto[1]);
        spec_.mask_height = int(proto[2]);
        spec_.mask_width = int(proto[3]);
    }
    for(auto &shape:spec_.output_shapes) shape[0]=1; // per-image capacity; bindings set physical batch
    if (spec_.channels != 4 + spec_.classes + spec_.keypoints * spec_.keypoint_dims + spec_.mask_channels)
        throw std::invalid_argument("Model channels do not match task metadata");
}
void ONNXRunner::initialize_cuda(int gpu, cudaStream_t stream, size_t arena_bytes) {
    session_.reset();
    options_ = Ort::SessionOptions{};
    options_.SetIntraOpNumThreads(1);
    options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    if(cuda_graph_)options_.AddConfigEntry("session.disable_cpu_ep_fallback","1");
    OrtCUDAProviderOptionsV2 *cuda = nullptr;
    auto &api = Ort::GetApi();
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
    std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2 *)> owner(
        cuda, api.ReleaseCUDAProviderOptions);
    const char *keys[] = {"device_id", "use_tf32", "cudnn_conv_algo_search", "gpu_mem_limit",
                          "arena_extend_strategy", "prefer_nhwc", "enable_cuda_graph"};
    std::string gpu_text = std::to_string(gpu), limit = std::to_string(arena_bytes);
    const char *values[] = {gpu_text.c_str(), "0", "HEURISTIC", limit.c_str(), "kSameAsRequested",prefer_nhwc_?"1":"0",cuda_graph_?"1":"0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, keys, values, 7));
    Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(cuda, "user_compute_stream", stream));
    options_.AppendExecutionProvider_CUDA_V2(*cuda);
    session_ = std::make_unique<Ort::Session>(env_, std::filesystem::u8path(path_).c_str(), options_);
    // Keep EP-owned host release callbacks completed before session teardown.
    // The bound GPU tensors remain resident; copy-stream overlap is independent.
}
void ONNXRunner::prepare_bindings(int slot, void *input, const std::vector<void *> &outputs,
                                  int max_batch) {
    if (bindings_.size() <= size_t(slot))
        bindings_.resize(slot + 1);
    auto &cache = bindings_[slot];
    cache.resize(max_batch + 1);
    for (int batch = spec_.graph_specialized && spec_.execution_input_shape[0]>0?int(spec_.execution_input_shape[0]):1; batch <= max_batch; ++batch) {
        auto record = std::make_unique<Binding>();
        if(cuda_graph_)record->options.AddConfigEntry("gpu_graph_id",std::to_string(slot+1).c_str());
        record->io = std::make_unique<Ort::IoBinding>(*session_);
        std::vector<int64_t> shape{batch, 3, spec_.height, spec_.width};
        record->tensors.push_back(Ort::Value::CreateTensor(
            memory_, input, size_t(batch) * 3 * spec_.height * spec_.width * (spec_.input_dtype==10?2:4),
            shape.data(), 4, ONNXTensorElementDataType(spec_.input_dtype)));
        record->io->BindInput(spec_.input_name.c_str(), record->tensors.back());
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto sh = spec_.output_shapes[i];
            sh[0] = batch;
            size_t count = 1;
            for (auto d : sh)
                count *= size_t(d);
            record->tensors.push_back(
                Ort::Value::CreateTensor(memory_, outputs[i], count*(spec_.output_dtypes[i]==10?2:4),
                    sh.data(), sh.size(), ONNXTensorElementDataType(spec_.output_dtypes[i])));
            record->io->BindOutput(spec_.output_names[i].c_str(), record->tensors.back());
        }
        cache[batch] = std::move(record);
    }
}
void ONNXRunner::run(int slot, int batch) {
    auto &record=*bindings_.at(slot).at(batch);
    session_->Run(record.options,*record.io);
}
void ONNXRunner::run_zero_copy(const float *input, const std::vector<float *> &outputs, int batch,
                               cudaStream_t) {
    if(spec_.input_dtype!=1 || std::any_of(spec_.output_dtypes.begin(),spec_.output_dtypes.end(),[](int t){return t!=1;}))
        throw std::invalid_argument("run_zero_copy accepts FP32 I/O only; use prepared typed bindings for FP16 I/O");
    std::vector<int64_t> shape{batch, 3, spec_.height, spec_.width};
    auto tensor = Ort::Value::CreateTensor<float>(
        memory_, const_cast<float *>(input), size_t(batch) * 3 * spec_.height * spec_.width, shape.data(), 4);
    Ort::IoBinding binding(*session_);
    binding.BindInput(spec_.input_name.c_str(), tensor);
    std::vector<Ort::Value> tensors;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto sh = spec_.output_shapes[i];
        sh[0] = batch;
        size_t n = 1;
        for (auto d : sh)
            n *= size_t(d);
        tensors.push_back(Ort::Value::CreateTensor<float>(memory_, outputs[i], n, sh.data(), sh.size()));
        binding.BindOutput(spec_.output_names[i].c_str(), tensors.back());
    }
    Ort::RunOptions options;
    if(cuda_graph_)options.AddConfigEntry("gpu_graph_id","-1");
    session_->Run(options, binding);
}
} // namespace vonnx
