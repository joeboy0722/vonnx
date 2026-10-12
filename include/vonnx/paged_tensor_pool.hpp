#pragma once

#include "vonnx/common.hpp"
#include "vonnx/cuda_resources.hpp"
#include <mutex>
#include <queue>
#include <vector>

namespace vonnx {

// vONNX 分頁 GPU 記憶體池 (Paged Tensor Memory Pool)
// 類似 vLLM PagedAttention 思想，預先在 VRAM 配置頁塊，消除執行期的 cudaMalloc 延遲
class PagedTensorPool {
  public:
    // 建構子：指定頁面數量與單一頁面大小 (bytes)
    PagedTensorPool(size_t max_pages, size_t page_size_bytes);
    ~PagedTensorPool();

    // 禁止複製
    PagedTensorPool(const PagedTensorPool &) = delete;
    PagedTensorPool &operator=(const PagedTensorPool &) = delete;

    // 配置單個 GPU 頁塊
    int allocate_page();

    // 釋放指定 ID 的 GPU 頁塊
    void free_page(int page_id);

    // 取得指定 page_id 的 GPU 記憶體指標
    void *get_page_ptr(int page_id);

    // 取得當前剩餘可用頁數
    size_t free_page_count();

    // 取得單頁大小
    size_t get_page_size() const { return page_size_bytes_; }

  private:
    size_t max_pages_;
    size_t page_size_bytes_;
    DeviceBuffer<uint8_t> storage_;
    void *base_gpu_ptr_;               // 整塊預配置 GPU 記憶體起點
    std::vector<PageSlot> page_slots_; // 所有 Slot 的詮釋資料
    std::queue<int> free_queue_;       // 空閒頁面佇列
    std::mutex pool_mutex_;            // 多執行緒安全鎖
};

} // namespace vonnx
