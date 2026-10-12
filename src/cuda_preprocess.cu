#include "vonnx/cuda_preprocess.cuh"
namespace vonnx {
__device__ int pixel(const PreprocessFrame &f, int x, int y, int c) {
    if (f.format == PixelFormat::NV12) {
        float yy = f.ptr[y * f.stride + x];
        int uv = f.stride * f.height + (y / 2) * f.stride + (x / 2) * 2;
        float u = f.ptr[uv] - 128.f, v = f.ptr[uv + 1] - 128.f;
        float value = c == 0 ? yy + 1.402f * v : c == 1 ? yy - .344136f * u - .714136f * v : yy + 1.772f * u;
        return min(255, max(0, __float2int_rn(value)));
    }
    int channels = f.format == PixelFormat::RGBA32 ? 4 : 3;
    int channel = f.format == PixelFormat::BGR24 ? 2 - c : c;
    return f.ptr[y * f.stride + x * channels + channel];
}
__global__ void letterbox(const PreprocessFrame *frames, float *dst, AffineMatrix *meta, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, b = blockIdx.z;
    if (x >= w || y >= h)
        return;
    const auto f = frames[b];
    const auto a = f.affine;
    if (x == 0 && y == 0)
        meta[b] = a;
    int rx = x - int(a.pad_x), ry = y - int(a.pad_y);
    for (int c = 0; c < 3; ++c) {
        int value = 114;
        if (rx >= 0 && ry >= 0 && rx < a.resized_w && ry < a.resized_h) {
            float fx = float((double(rx) + .5) * double(f.width) / a.resized_w - .5),
                  fy = float((double(ry) + .5) * double(f.height) / a.resized_h - .5);
            int ix = int(floorf(fx)), iy = int(floorf(fy));
            fx -= ix;
            fy -= iy;
            if (ix < 0) {
                ix = 0;
                fx = 0;
            }
            if (ix >= f.width - 1) {
                ix = f.width - 1;
                fx = 0;
            }
            int iy0 = max(0, min(iy, f.height - 1)), iy1 = max(0, min(iy + 1, f.height - 1));
            int ax0 = __float2int_rn((1 - fx) * 2048), ax1 = __float2int_rn(fx * 2048);
            int by0 = __float2int_rn((1 - fy) * 2048), by1 = __float2int_rn(fy * 2048);
            int s0 = pixel(f, ix, iy0, c) * ax0 + pixel(f, min(ix + 1, f.width - 1), iy0, c) * ax1;
            int s1 = pixel(f, ix, iy1, c) * ax0 + pixel(f, min(ix + 1, f.width - 1), iy1, c) * ax1;
            value = (((by0 * (s0 >> 4)) >> 16) + ((by1 * (s1 >> 4)) >> 16) + 2) >> 2;
        }
        dst[(b * 3 + c) * w * h + y * w + x] = float(value) / 255.f;
    }
}
void launch_batched_preprocess(const std::vector<PreprocessFrame> &frames, PreprocessWorkspace &workspace,
                               float *dst, AffineMatrix *meta, int w, int h, cudaStream_t stream) {
    std::copy(frames.begin(), frames.end(), workspace.host_frames.get());
    CUDA_CHECK(cudaMemcpyAsync(workspace.frames.get(), workspace.host_frames.get(),
                               frames.size() * sizeof(PreprocessFrame), cudaMemcpyHostToDevice, stream));
    launch_prepared_preprocess(int(frames.size()), workspace, dst, meta, w, h, stream);
}
void launch_prepared_preprocess(int batch, PreprocessWorkspace &workspace, float *dst, AffineMatrix *meta,
                                int w, int h, cudaStream_t stream) {
    letterbox<<<dim3((w + 15) / 16, (h + 15) / 16, batch), dim3(16, 16), 0, stream>>>(workspace.frames.get(),
                                                                                      dst, meta, w, h);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace vonnx
