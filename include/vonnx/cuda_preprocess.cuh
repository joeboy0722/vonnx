#pragma once

#include "vonnx/common.hpp"

namespace vonnx {

// Letterbox 縮放比率與偏移詮釋資料
struct AffineMatrix {
    float scale;   // 縮放比例
    float pad_x;   // X 軸 Fill Padding (像素)
    float pad_y;   // Y 軸 Fill Padding (像素)
    int src_w;     // 原始寬度
    int src_h;     // 原始高度
};

// 手寫 Fused CUDA Preprocess 算子
// 支援一併執行：格式轉換 (BGR/RGB/NV12) -> Bilinear Letterbox 保比例縮放 -> 歸一化 (0-1) -> HWC 轉 CHW 記憶體佈局
void launch_batched_preprocess(
    const std::vector<const void*>& src_ptrs,
    const std::vector<int>& src_widths,
    const std::vector<int>& src_heights,
    const std::vector<int>& src_strides,
    const std::vector<PixelFormat>& formats,
    float* dst_batch_tensor_gpu,     // 輸出 [B, 3, target_h, target_w] 的 GPU 記憶體
    AffineMatrix* affine_meta_gpu,   // 輸出至 GPU 的 Affine 變換矩陣 (單批 B 個)
    int batch_size,
    int target_w,
    int target_h,
    cudaStream_t stream
);

} // namespace vonnx
