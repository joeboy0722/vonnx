#include "vonnx/cuda_preprocess.cuh"

namespace vonnx {

// CUDA 核心函數：處理單張或批次圖像的 Letterbox 縮放與 Channel 排列
__global__ void preprocess_letterbox_kernel(
    const uint8_t* const* src_ptrs,
    const int* src_widths,
    const int* src_heights,
    const int* src_strides,
    const PixelFormat* formats,
    float* dst_tensor,
    AffineMatrix* affine_meta,
    int batch_size,
    int target_w,
    int target_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 目標圖片的 X 座標 [0, target_w)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 目標圖片的 Y 座標 [0, target_h)
    int b = blockIdx.z;                            // Batch 索引 [0, batch_size)

    if (b >= batch_size || x >= target_w || y >= target_h) return;

    int src_w = src_widths[b];
    int src_h = src_heights[b];
    int src_stride = src_strides[b];
    PixelFormat fmt = formats[b];
    const uint8_t* src_buf = src_ptrs[b];

    // 計算保持長寬比的 Letterbox 比例
    float rx = (float)target_w / src_w;
    float ry = (float)target_h / src_h;
    float r = fminf(rx, ry);

    int new_unpad_w = __float2int_rn(src_w * r);
    int new_unpad_h = __float2int_rn(src_h * r);

    float pad_x = (target_w - new_unpad_w) / 2.0f;
    float pad_y = (target_h - new_unpad_h) / 2.0f;

    // 第一個線程記錄該 batch 的 Affine Meta
    if (x == 0 && y == 0) {
        affine_meta[b].scale = r;
        affine_meta[b].pad_x = pad_x;
        affine_meta[b].pad_y = pad_y;
        affine_meta[b].src_w = src_w;
        affine_meta[b].src_h = src_h;
    }

    // 將目標 (x, y) 映射回原圖 (src_x, src_y)
    float src_x_f = (x - pad_x) / r;
    float src_y_f = (y - pad_y) / r;

    float r_val = 0.4470588f; // 預設 Padding 灰色 (114.0 / 255.0)
    float g_val = 0.4470588f;
    float b_val = 0.4470588f;

    if (src_x_f >= 0.0f && src_x_f < src_w && src_y_f >= 0.0f && src_y_f < src_h) {
        // 雙線性內插 (Bilinear Interpolation)
        int x1 = (int)src_x_f;
        int y1 = (int)src_y_f;
        int x2 = min(x1 + 1, src_w - 1);
        int y2 = min(y1 + 1, src_h - 1);

        float dx = src_x_f - x1;
        float dy = src_y_f - y1;
        float w1 = (1.0f - dx) * (1.0f - dy);
        float w2 = dx * (1.0f - dy);
        float w3 = (1.0f - dx) * dy;
        float w4 = dx * dy;

        auto get_pixel_rgb = [&](int px, int py, float& out_r, float& out_g, float& out_b) {
            if (fmt == PixelFormat::BGR24) {
                const uint8_t* p = src_buf + py * src_stride + px * 3;
                out_b = p[0] / 255.0f;
                out_g = p[1] / 255.0f;
                out_r = p[2] / 255.0f;
            } else if (fmt == PixelFormat::RGB24) {
                const uint8_t* p = src_buf + py * src_stride + px * 3;
                out_r = p[0] / 255.0f;
                out_g = p[1] / 255.0f;
                out_b = p[2] / 255.0f;
            } else if (fmt == PixelFormat::RGBA32) {
                const uint8_t* p = src_buf + py * src_stride + px * 4;
                out_r = p[0] / 255.0f;
                out_g = p[1] / 255.0f;
                out_b = p[2] / 255.0f;
            } else if (fmt == PixelFormat::NV12) {
                // NV12 解碼邏輯 (Y 平面 + UV 交錯平面)
                const uint8_t* y_plane = src_buf;
                const uint8_t* uv_plane = src_buf + src_stride * src_h;
                float y_pixel = y_plane[py * src_stride + px];
                int uv_idx = (py / 2) * src_stride + (px / 2) * 2;
                float u_pixel = uv_plane[uv_idx] - 128.0f;
                float v_pixel = uv_plane[uv_idx + 1] - 128.0f;

                float r_raw = y_pixel + 1.402f * v_pixel;
                float g_raw = y_pixel - 0.344136f * u_pixel - 0.714136f * v_pixel;
                float b_raw = y_pixel + 1.772f * u_pixel;

                out_r = fminf(fmaxf(r_raw, 0.0f), 255.0f) / 255.0f;
                out_g = fminf(fmaxf(g_raw, 0.0f), 255.0f) / 255.0f;
                out_b = fminf(fmaxf(b_raw, 0.0f), 255.0f) / 255.0f;
            }
        };

        float r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4;
        get_pixel_rgb(x1, y1, r1, g1, b1);
        get_pixel_rgb(x2, y1, r2, g2, b2);
        get_pixel_rgb(x1, y2, r3, g3, b3);
        get_pixel_rgb(x2, y2, r4, g4, b4);

        r_val = r1 * w1 + r2 * w2 + r3 * w3 + r4 * w4;
        g_val = g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4;
        b_val = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4;
    }

    // 寫入 NCHW 佈局的廣義 Tensor [B, 3, target_h, target_w]
    int plane_size = target_h * target_w;
    int b_offset = b * 3 * plane_size;
    int pixel_idx = y * target_w + x;

    dst_tensor[b_offset + 0 * plane_size + pixel_idx] = r_val;
    dst_tensor[b_offset + 1 * plane_size + pixel_idx] = g_val;
    dst_tensor[b_offset + 2 * plane_size + pixel_idx] = b_val;
}

void launch_batched_preprocess(
    const std::vector<const void*>& src_ptrs,
    const std::vector<int>& src_widths,
    const std::vector<int>& src_heights,
    const std::vector<int>& src_strides,
    const std::vector<PixelFormat>& formats,
    float* dst_batch_tensor_gpu,
    AffineMatrix* affine_meta_gpu,
    int batch_size,
    int target_w,
    int target_h,
    cudaStream_t stream
) {
    if (batch_size <= 0) return;

    // 將參數暫時複製至 GPU (非同步拷貝)
    const void** d_src_ptrs;
    int* d_src_widths;
    int* d_src_heights;
    int* d_src_strides;
    PixelFormat* d_formats;

    CUDA_CHECK(cudaMallocAsync(&d_src_ptrs, batch_size * sizeof(void*), stream));
    CUDA_CHECK(cudaMallocAsync(&d_src_widths, batch_size * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_src_heights, batch_size * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_src_strides, batch_size * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_formats, batch_size * sizeof(PixelFormat), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_src_ptrs, src_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src_widths, src_widths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src_heights, src_heights.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src_strides, src_strides.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_formats, formats.data(), batch_size * sizeof(PixelFormat), cudaMemcpyHostToDevice, stream));

    dim3 block(16, 16, 1);
    dim3 grid(
        (target_w + block.x - 1) / block.x,
        (target_h + block.y - 1) / block.y,
        batch_size
    );

    preprocess_letterbox_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t* const*>(d_src_ptrs),
        d_src_widths,
        d_src_heights,
        d_src_strides,
        d_formats,
        dst_batch_tensor_gpu,
        affine_meta_gpu,
        batch_size,
        target_w,
        target_h
    );

    CUDA_CHECK(cudaFreeAsync(d_src_ptrs, stream));
    CUDA_CHECK(cudaFreeAsync(d_src_widths, stream));
    CUDA_CHECK(cudaFreeAsync(d_src_heights, stream));
    CUDA_CHECK(cudaFreeAsync(d_src_strides, stream));
    CUDA_CHECK(cudaFreeAsync(d_formats, stream));
}

} // namespace vonnx
