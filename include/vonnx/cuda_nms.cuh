#pragma once
#include "vonnx/cuda_preprocess.cuh"
namespace vonnx {
constexpr int MAX_BOXES_PER_IMAGE = 300;
struct alignas(16) DeviceBBox {
    float x1, y1, x2, y2, score;
    int class_id, anchor_index, is_suppressed;
};
class NMSWorkspace {
  public:
    DeviceBuffer<float> scores, sorted_scores;
    DeviceBuffer<int> indices, sorted_indices;
    DeviceBuffer<uint8_t> scratch;
    size_t scratch_bytes{};
    NMSWorkspace(int batch, int anchors, cudaStream_t stream);
};
void launch_batched_yolo_nms(const float *, const AffineMatrix *, int, int, int, int, float, float, int,
                             NMSWorkspace &, DeviceBBox *, int *, cudaStream_t);
void launch_pose(const float *, const DeviceBBox *, const int *, const AffineMatrix *, int, int, int, int,
                 int, int, float *, cudaStream_t);
struct MaskPlan {
    int batch, box, x, y, width, height, left, top, right, bottom;
    size_t offset;
};
void launch_masks(const float *, const float *, const DeviceBBox *, const AffineMatrix *, const MaskPlan *,
                  int, int, int, int, int, int, int, uint8_t *, cudaStream_t);
} // namespace vonnx
