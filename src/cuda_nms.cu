#include "vonnx/cuda_nms.cuh"
#include <cub/cub.cuh>
namespace vonnx {
NMSWorkspace::NMSWorkspace(int batch, int anchors, cudaStream_t stream) {
    scores.allocate(size_t(batch) * anchors);
    sorted_scores.allocate(size_t(batch) * anchors);
    indices.allocate(size_t(batch) * anchors);
    sorted_indices.allocate(size_t(batch) * anchors);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, scratch_bytes, scores.get(),
                                                         sorted_scores.get(), indices.get(),
                                                         sorted_indices.get(), anchors, 0, 32, stream));
    scratch.allocate(scratch_bytes);
}
__global__ void score_kernel(const float *pred, int nc, int channels, int n, float *scores, int *indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (i >= n)
        return;
    const float *p = pred + b * channels * n;
    float score = 0;
    for (int c = 0; c < nc; ++c)
        score = fmaxf(score, p[(4 + c) * n + i]);
    scores[b * n + i] = score;
    indices[b * n + i] = i;
}
__device__ DeviceBBox decode(const float *p, int nc, int n, int i) {
    DeviceBBox box{};
    box.anchor_index = i;
    box.score = -1;
    for (int c = 0; c < nc; ++c)
        if (p[(4 + c) * n + i] > box.score) {
            box.score = p[(4 + c) * n + i];
            box.class_id = c;
        }
    box.x1 = p[i] - p[2 * n + i] * .5f;
    box.y1 = p[n + i] - p[3 * n + i] * .5f;
    box.x2 = p[i] + p[2 * n + i] * .5f;
    box.y2 = p[n + i] + p[3 * n + i] * .5f;
    return box;
}
__device__ float iou(const DeviceBBox &a, const DeviceBBox &b) {
    float inter =
        fmaxf(0.f, fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1)) * fmaxf(0.f, fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1));
    float area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return area > 0 ? inter / area : 0;
}
__global__ void greedy(const float *pred, const int *order, const float *scores, const AffineMatrix *affine,
                       int nc, int channels, int n, float conf, float threshold, int max_det, DeviceBBox *out,
                       int *counts) {
    int b = blockIdx.x;
    const float *p = pred + b * channels * n;
    DeviceBBox *boxes = out + b * MAX_BOXES_PER_IMAGE;
    int count = 0;
    for (int k = 0; k < n && count < max_det; ++k) {
        if (scores[b * n + k] <= conf)
            break;
        auto box = decode(p, nc, n, order[b * n + k]);
        bool keep = true;
        for (int j = 0; j < count; ++j)
            if (box.class_id == boxes[j].class_id && iou(box, boxes[j]) > threshold) {
                keep = false;
                break;
            }
        if (keep)
            boxes[count++] = box;
    }
    auto a = affine[b];
    for (int k = 0; k < count; ++k) {
        auto &box = boxes[k];
        box.x1 = fminf(fmaxf((box.x1 - a.pad_x) / a.scale, 0.f), float(a.src_w));
        box.x2 = fminf(fmaxf((box.x2 - a.pad_x) / a.scale, 0.f), float(a.src_w));
        box.y1 = fminf(fmaxf((box.y1 - a.pad_y) / a.scale, 0.f), float(a.src_h));
        box.y2 = fminf(fmaxf((box.y2 - a.pad_y) / a.scale, 0.f), float(a.src_h));
    }
    counts[b] = count;
}
void launch_batched_yolo_nms(const float *pred, const AffineMatrix *affine, int batch, int nc, int channels,
                             int n, float conf, float threshold, int max_det, NMSWorkspace &ws,
                             DeviceBBox *boxes, int *counts, cudaStream_t stream) {
    score_kernel<<<dim3((n + 255) / 256, batch), 256, 0, stream>>>(pred, nc, channels, n, ws.scores.get(),
                                                                   ws.indices.get());
    CUDA_CHECK(cudaGetLastError());
    for (int b = 0; b < batch; ++b)
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            ws.scratch.get(), ws.scratch_bytes, ws.scores.get() + b * n, ws.sorted_scores.get() + b * n,
            ws.indices.get() + b * n, ws.sorted_indices.get() + b * n, n, 0, 32, stream));
    greedy<<<batch, 1, 0, stream>>>(pred, ws.sorted_indices.get(), ws.sorted_scores.get(), affine, nc,
                                    channels, n, conf, threshold, max_det, boxes, counts);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void pose_kernel(const float *pred, const DeviceBBox *boxes, const int *counts,
                            const AffineMatrix *affine, int nc, int channels, int n, int kp, int kd,
                            float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (i >= counts[b] * kp)
        return;
    int box = i / kp, k = i % kp, idx = boxes[b * MAX_BOXES_PER_IMAGE + box].anchor_index;
    auto a = affine[b];
    const float *p = pred + b * channels * n;
    float *v = out + (b * MAX_BOXES_PER_IMAGE * kp + i) * 3;
    v[0] = fminf(fmaxf((p[(4 + nc + k * kd) * n + idx] - a.pad_x) / a.scale, 0.f), float(a.src_w));
    v[1] = fminf(fmaxf((p[(5 + nc + k * kd) * n + idx] - a.pad_y) / a.scale, 0.f), float(a.src_h));
    v[2] = kd == 3 ? p[(6 + nc + k * kd) * n + idx] : 1.f;
}
void launch_pose(const float *p, const DeviceBBox *boxes, const int *counts, const AffineMatrix *a, int batch,
                 int nc, int channels, int n, int kp, int kd, float *out, cudaStream_t s) {
    pose_kernel<<<dim3((MAX_BOXES_PER_IMAGE * kp + 255) / 256, batch), 256, 0, s>>>(p, boxes, counts, a, nc,
                                                                                    channels, n, kp, kd, out);
    CUDA_CHECK(cudaGetLastError());
}
__device__ float logit(const float *p, const float *proto, int idx, int nc, int n, int m, int mh, int mw,
                       int x, int y) {
    float v = 0;
    for (int c = 0; c < m; ++c)
        v += p[(4 + nc + c) * n + idx] * proto[(c * mh + y) * mw + x];
    return v;
}
__global__ void mask_kernel(const float *pred, const float *protos, const DeviceBBox *boxes,
                            const AffineMatrix *affine, const MaskPlan *plans, int nc, int channels, int n,
                            int m, int mh, int mw, uint8_t *out) {
    auto plan = plans[blockIdx.y];
    auto a = affine[plan.batch];
    int idx = boxes[plan.batch * MAX_BOXES_PER_IMAGE + plan.box].anchor_index;
    const float *p = pred + plan.batch * channels * n;
    const float *proto = protos + plan.batch * m * mh * mw;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < plan.width * plan.height;
         i += gridDim.x * blockDim.x) {
        int x = plan.x + i % plan.width, y = plan.y + i / plan.width;
        float sx = (x + .5f) * (plan.right - plan.left) / a.src_w - .5f,
              sy = (y + .5f) * (plan.bottom - plan.top) / a.src_h - .5f;
        sx = fmaxf(sx, 0.f);
        sy = fmaxf(sy, 0.f);
        int x0 = int(sx), y0 = int(sy);
        float dx = sx - x0, dy = sy - y0;
        x0 = min(x0, plan.right - plan.left - 1);
        y0 = min(y0, plan.bottom - plan.top - 1);
        int x1 = min(x0 + 1, plan.right - plan.left - 1), y1 = min(y0 + 1, plan.bottom - plan.top - 1);
        float v =
            (1 - dy) * ((1 - dx) * logit(p, proto, idx, nc, n, m, mh, mw, x0 + plan.left, y0 + plan.top) +
                        dx * logit(p, proto, idx, nc, n, m, mh, mw, x1 + plan.left, y0 + plan.top)) +
            dy * ((1 - dx) * logit(p, proto, idx, nc, n, m, mh, mw, x0 + plan.left, y1 + plan.top) +
                  dx * logit(p, proto, idx, nc, n, m, mh, mw, x1 + plan.left, y1 + plan.top));
        auto box = boxes[plan.batch * MAX_BOXES_PER_IMAGE + plan.box];
        bool inside = x >= box.x1 && x < box.x2 && y >= box.y1 && y < box.y2;
        out[plan.offset + i] = inside && v > 0 ? 1 : 0;
    }
}
void launch_masks(const float *p, const float *proto, const DeviceBBox *boxes, const AffineMatrix *a,
                  const MaskPlan *plans, int count, int nc, int channels, int n, int m, int mh, int mw,
                  uint8_t *out, cudaStream_t s) {
    if (count) {
        mask_kernel<<<dim3(128, count), 256, 0, s>>>(p, proto, boxes, a, plans, nc, channels, n, m, mh, mw,
                                                     out);
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace vonnx
