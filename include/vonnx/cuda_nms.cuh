#pragma once

#include "vonnx/common.hpp"
#include "vonnx/cuda_preprocess.cuh"

namespace vonnx {

// 最多允許單幀解碼與保留的邊界框上限 (防止 GPU 記憶體溢出)
constexpr int MAX_BOXES_PER_IMAGE = 1024;

// GPU 端內部用的解碼 BBox 記憶體結構
struct alignas(16) DeviceBBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
    int is_suppressed; // 0: 保留, 1: 被 NMS 抑制
};

// 批次 CUDA NMS 介面 (特別適配 YOLOv8 與 YOLOv11 的 Output Shape: [B, 4 + C, N])
void launch_batched_yolo_nms(
    const float* d_output_tensor,   // ONNX 原始輸出指標 [B, 4 + C, num_anchors]
    const AffineMatrix* d_affine_meta, // GPU 上的 Affine 反向變換矩陣
    int batch_size,
    int num_classes,                // 例如 80 類 COCO
    int num_anchors,                // 例如 8400 (640x640 下的 anchor 數)
    float conf_threshold,           // 信心分數門檻 (例如 0.25)
    float iou_threshold,            // NMS IoU 重疊門檻 (例如 0.45)
    DeviceBBox* d_out_boxes,        // GPU 輸出緩衝區 [B * MAX_BOXES_PER_IMAGE]
    int* d_out_counts,              // GPU 各張圖片實際保留的框數 [B]
    cudaStream_t stream
);

} // namespace vonnx
