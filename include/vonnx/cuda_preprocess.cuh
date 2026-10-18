#pragma once
#include "vonnx/common.hpp"
#include "vonnx/cuda_resources.hpp"
namespace vonnx {
void launch_to_half(const float*, uint16_t*, size_t, cudaStream_t);
void launch_from_half(const uint16_t*, float*, size_t, cudaStream_t);
struct AffineMatrix {
    float scale, pad_x, pad_y;
    int src_w, src_h, resized_w, resized_h;
};
struct PreprocessFrame {
    const uint8_t *ptr;
    int width, height, stride;
    PixelFormat format;
    AffineMatrix affine;
};
class PreprocessWorkspace {
  public:
    DeviceBuffer<PreprocessFrame> frames;
    PinnedBuffer<PreprocessFrame> host_frames;
    explicit PreprocessWorkspace(size_t batch) {
        frames.allocate(batch);
        host_frames.allocate(batch);
    }
};
void launch_batched_preprocess(const std::vector<PreprocessFrame> &, PreprocessWorkspace &, float *,
                               AffineMatrix *, int, int, cudaStream_t);
void launch_prepared_preprocess(int batch, PreprocessWorkspace &, float *, AffineMatrix *, int, int,
                                cudaStream_t);
} // namespace vonnx
