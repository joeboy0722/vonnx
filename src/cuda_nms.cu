#include "vonnx/cuda_nms.cuh"

namespace vonnx {

// IoU 計算輔助 Device 函數
__device__ inline float compute_iou(const DeviceBBox& a, const DeviceBBox& b) {
    float inter_x1 = fmaxf(a.x1, b.x1);
    float inter_y1 = fmaxf(a.y1, b.y1);
    float inter_x2 = fminf(a.x2, b.x2);
    float inter_y2 = fminf(a.y2, b.y2);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// 階段一：YOLOv8/v11 解碼與門檻過濾 CUDA Kernel
__global__ void decode_yolo_boxes_kernel(
    const float* d_output_tensor,
    const AffineMatrix* d_affine_meta,
    int num_classes,
    int num_anchors,
    float conf_threshold,
    DeviceBBox* d_out_boxes,
    int* d_out_counts,
    int batch_size
) {
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x; // [0, num_anchors)
    int b = blockIdx.y;                                     // [0, batch_size)

    if (b >= batch_size || anchor_idx >= num_anchors) return;

    int channels = 4 + num_classes;
    int batch_stride = channels * num_anchors;
    const float* batch_tensor = d_output_tensor + b * batch_stride;

    // 在 num_classes 中尋找最高得分與對應類別
    float max_score = -1.0f;
    int best_class = -1;

    for (int c = 0; c < num_classes; ++c) {
        float score = batch_tensor[(4 + c) * num_anchors + anchor_idx];
        if (score > max_score) {
            max_score = score;
            best_class = c;
        }
    }

    // 若信心分數大於門檻，則進行解碼
    if (max_score >= conf_threshold) {
        float cx = batch_tensor[0 * num_anchors + anchor_idx];
        float cy = batch_tensor[1 * num_anchors + anchor_idx];
        float w  = batch_tensor[2 * num_anchors + anchor_idx];
        float h  = batch_tensor[3 * num_anchors + anchor_idx];

        // 轉換為 Letterbox 座標
        float box_x1 = cx - w * 0.5f;
        float box_y1 = cy - h * 0.5f;
        float box_x2 = cx + w * 0.5f;
        float box_y2 = cy + h * 0.5f;

        // 利用 Affine 逆矩陣映射回原始圖片座標
        AffineMatrix meta = d_affine_meta[b];
        float orig_x1 = (box_x1 - meta.pad_x) / meta.scale;
        float orig_y1 = (box_y1 - meta.pad_y) / meta.scale;
        float orig_x2 = (box_x2 - meta.pad_x) / meta.scale;
        float orig_y2 = (box_y2 - meta.pad_y) / meta.scale;

        // 裁剪邊界至原圖長寬範圍內
        orig_x1 = fminf(fmaxf(orig_x1, 0.0f), (float)meta.src_w);
        orig_y1 = fminf(fmaxf(orig_y1, 0.0f), (float)meta.src_h);
        orig_x2 = fminf(fmaxf(orig_x2, 0.0f), (float)meta.src_w);
        orig_y2 = fminf(fmaxf(orig_y2, 0.0f), (float)meta.src_h);

        // 使用原子加法取得目前寫入位置
        int count_idx = atomicAdd(&d_out_counts[b], 1);
        if (count_idx < MAX_BOXES_PER_IMAGE) {
            DeviceBBox& box = d_out_boxes[b * MAX_BOXES_PER_IMAGE + count_idx];
            box.x1 = orig_x1;
            box.y1 = orig_y1;
            box.x2 = orig_x2;
            box.y2 = orig_y2;
            box.score = max_score;
            box.class_id = best_class;
            box.is_suppressed = 0;
        }
    }
}

// 階段二：GPU Parallel NMS Kernel
__global__ void fast_gpu_nms_kernel(
    DeviceBBox* d_boxes,
    const int* d_counts,
    float iou_threshold,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 候選框 i
    int b = blockIdx.y;                             // Batch 索引

    if (b >= batch_size) return;

    int count = min(d_counts[b], MAX_BOXES_PER_IMAGE);
    if (i >= count) return;

    DeviceBBox* batch_boxes = d_boxes + b * MAX_BOXES_PER_IMAGE;
    DeviceBBox box_i = batch_boxes[i];

    // 如果已經被抑制則跳過
    if (box_i.is_suppressed) return;

    // 將 box_i 與該圖片中的所有其他 box_j 進行平行 IoU 比較
    for (int j = 0; j < count; ++j) {
        if (i == j) continue;
        DeviceBBox box_j = batch_boxes[j];

        // 僅比較相同類別的目標
        if (box_i.class_id == box_j.class_id && !box_j.is_suppressed) {
            // 如果分數較低，或分數相同但 index 較大，且 IoU 大於門檻，則抑制該框
            if (box_i.score < box_j.score || (box_i.score == box_j.score && i > j)) {
                if (compute_iou(box_i, box_j) > iou_threshold) {
                    batch_boxes[i].is_suppressed = 1;
                    break;
                }
            }
        }
    }
}

void launch_batched_yolo_nms(
    const float* d_output_tensor,
    const AffineMatrix* d_affine_meta,
    int batch_size,
    int num_classes,
    int num_anchors,
    float conf_threshold,
    float iou_threshold,
    DeviceBBox* d_out_boxes,
    int* d_out_counts,
    cudaStream_t stream
) {
    if (batch_size <= 0) return;

    // 清空輸出計數器
    CUDA_CHECK(cudaMemsetAsync(d_out_counts, 0, batch_size * sizeof(int), stream));

    // 啟動階段一：解碼與信心門檻篩選
    int block_size = 256;
    dim3 grid_decode((num_anchors + block_size - 1) / block_size, batch_size);

    decode_yolo_boxes_kernel<<<grid_decode, block_size, 0, stream>>>(
        d_output_tensor,
        d_affine_meta,
        num_classes,
        num_anchors,
        conf_threshold,
        d_out_boxes,
        d_out_counts,
        batch_size
    );

    // 啟動階段二：平行 NMS 抑制
    dim3 grid_nms((MAX_BOXES_PER_IMAGE + block_size - 1) / block_size, batch_size);
    fast_gpu_nms_kernel<<<grid_nms, block_size, 0, stream>>>(
        d_out_boxes,
        d_out_counts,
        iou_threshold,
        batch_size
    );
}

} // namespace vonnx
