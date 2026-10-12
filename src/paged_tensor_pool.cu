#include "vonnx/paged_tensor_pool.hpp"

namespace vonnx {

PagedTensorPool::PagedTensorPool(size_t max_pages, size_t page_size_bytes)
    : max_pages_(max_pages), page_size_bytes_(page_size_bytes), base_gpu_ptr_(nullptr) {

    size_t total_bytes = max_pages_ * page_size_bytes_;
    storage_.allocate(total_bytes);
    base_gpu_ptr_ = storage_.get();

    page_slots_.resize(max_pages_);
    uint8_t *byte_ptr = static_cast<uint8_t *>(base_gpu_ptr_);

    for (size_t i = 0; i < max_pages_; ++i) {
        page_slots_[i].page_id = static_cast<int>(i);
        page_slots_[i].gpu_ptr = byte_ptr + (i * page_size_bytes_);
        page_slots_[i].size_bytes = page_size_bytes_;
        page_slots_[i].is_free = true;
        free_queue_.push(static_cast<int>(i));
    }
}

PagedTensorPool::~PagedTensorPool() = default;

int PagedTensorPool::allocate_page() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_queue_.empty()) {
        throw std::runtime_error("PagedTensorPool is full, no available GPU pages!");
    }
    int page_id = free_queue_.front();
    free_queue_.pop();
    page_slots_[page_id].is_free = false;
    return page_id;
}

void PagedTensorPool::free_page(int page_id) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (page_id < 0 || static_cast<size_t>(page_id) >= max_pages_) {
        throw std::out_of_range("Invalid page_id for free_page!");
    }
    if (!page_slots_[page_id].is_free) {
        page_slots_[page_id].is_free = true;
        free_queue_.push(page_id);
    }
}

void *PagedTensorPool::get_page_ptr(int page_id) {
    if (page_id < 0 || static_cast<size_t>(page_id) >= max_pages_) {
        throw std::out_of_range("Invalid page_id for get_page_ptr!");
    }
    return page_slots_[page_id].gpu_ptr;
}

size_t PagedTensorPool::free_page_count() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return free_queue_.size();
}

} // namespace vonnx
