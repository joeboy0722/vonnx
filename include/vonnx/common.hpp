#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

// CUDA 錯誤檢查巨集
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

namespace vonnx {

// 影像色彩與資料格式列舉
enum class PixelFormat {
    BGR24,   // 3 channel OpenCV 標準格式 (HWC)
    RGB24,   // 3 channel RGB 格式 (HWC)
    RGBA32,  // 4 channel RGBA 格式 (HWC)
    NV12     // YUV420 雙平面 NV12 碼流
};

// 單一偵測邊界框結構體
struct BBox {
    float x1;       // 左上角 X 座標
    float y1;       // 左上角 Y 座標
    float x2;       // 右下角 X 座標
    float y2;       // 右下角 Y 座標
    float score;    // 信心分數
    int class_id;   // 類別 ID
};

// 單張影像的偵測結果
struct DetectionResult {
    uint64_t stream_id;       // 輸入串流識別碼
    uint64_t frame_id;        // 幀次序標籤
    std::vector<BBox> boxes;  // 篩選後保留的邊界框
};

// 傳入 vONNX 的非同步影像資料包
struct FrameTask {
    uint64_t stream_id;           // 串流識別碼
    uint64_t frame_id;            // 幀編號
    const void* data_ptr;         // 數據指標 (Host 記憶體或 GPU 記憶體)
    bool is_gpu_ptr;              // 是否已在 GPU 上
    int width;                    // 原始寬度
    int height;                   // 原始高度
    int step;                     // 每一列的 Pitch / Stride (bytes)
    PixelFormat format;           // 色彩格式
    cudaStream_t user_stream;     // 呼叫端傳入的 CUDA Stream (選用)
};

// Paged Pool 頁面配置單元
struct PageSlot {
    int page_id;                  // 頁面索引 ID
    void* gpu_ptr;                // 該頁面的 GPU 記憶體首地址
    size_t size_bytes;            // 頁面可用容量 (bytes)
    bool is_free;                 // 是否處於空閒狀態
};

} // namespace vonnx
