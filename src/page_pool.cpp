#include "page_pool.h"
#include <cstdio>

namespace diskann {

void PagePool::init(uint64_t num_pages, uint64_t page_size) {
  destroy();
  page_size_ = page_size;
  num_pages_allocated_ = num_pages;
  freelist_.reserve(num_pages);
  // allocate one big contiguous aligned block (page-aligned)
  big_block_ = nullptr;
  alloc_aligned((void**)&big_block_, num_pages * page_size_, page_size_);
  // slice into pages
  for (uint64_t i = 0; i < num_pages; ++i) {
    freelist_.push_back(big_block_ + i * page_size_);
  }
  // 清空零引用候选队列
  while (!zero_ref_queue_.empty()) {
    unsigned tmp;
    zero_ref_queue_.try_pop(tmp);
  }
}

void PagePool::destroy() {
  std::lock_guard<std::mutex> lk(freelist_mutex_);
  if (big_block_ != nullptr) {
    aligned_free(big_block_);
    big_block_ = nullptr;
  }
  freelist_.clear();
  page_size_ = 0;
  num_pages_allocated_ = 0;
  // clear entries
  page_entries_.clear();
  // 清空零引用候选队列
  while (!zero_ref_queue_.empty()) {
    unsigned tmp;
    zero_ref_queue_.try_pop(tmp);
  }
}

void PagePool::clear_cache() {
  std::vector<char*> to_return_all;
  for (auto it = page_entries_.begin(); it != page_entries_.end(); ++it) {
    if (it->second.buf != nullptr) to_return_all.push_back(it->second.buf);
  }
  page_entries_.clear();
  if (!to_return_all.empty()) {
    std::lock_guard<std::mutex> lk(freelist_mutex_);
    for (char* buf : to_return_all) {
      freelist_.push_back(buf);
    }
  }
  // 清空零引用候选队列
  while (!zero_ref_queue_.empty()) {
    unsigned tmp;
    zero_ref_queue_.try_pop(tmp);
  }
}

char* PagePool::acquire() {
  {
    std::lock_guard<std::mutex> lk(freelist_mutex_);
    if (!freelist_.empty()) {
      char* p = freelist_.back();
      freelist_.pop_back();
      return p;
    }
  }
  return nullptr;
}

void PagePool::release(char* buf) {
  if (buf == nullptr) return;
  std::lock_guard<std::mutex> lk(freelist_mutex_);
  freelist_.push_back(buf);
}

char* PagePool::enter_page(unsigned page_id) {
  PageMap::accessor acc;
  if (!page_entries_.find(acc, page_id)) return nullptr;
  acc->second.refcount.fetch_add(1, std::memory_order_relaxed);
  char* ret = acc->second.buf;
  acc.release();
  return ret;
}

bool PagePool::contains(unsigned page_id) {
  PageMap::const_accessor acc;
  bool found = page_entries_.find(acc, page_id);
  if (found) acc.release();
  return found;
}

void PagePool::leave_page(unsigned page_id) {
  PageMap::accessor acc;
  if (!page_entries_.find(acc, page_id)) return;
  uint32_t prev = acc->second.refcount.load(std::memory_order_relaxed);
  if (prev == 0) {
    fprintf(stderr, "[PagePool] leave_page 下溢: page %u\n", page_id);
    acc.release();
    return;
  }
  uint32_t rc = acc->second.refcount.fetch_sub(1, std::memory_order_relaxed) - 1;
  if (rc == 0) { 
    // 引用计数降为 0：加入候选队列
    zero_ref_queue_.push(page_id);
  }
  acc.release();
}

char* PagePool::add_page(unsigned page_id, char* buf) {
  PageMap::accessor acc;
  if (page_entries_.insert(acc, page_id)) {
    acc->second.buf = buf;
    // 直接发布为零引用，并加入候选队列，避免再次查表与原子操作
    acc->second.refcount.store(0, std::memory_order_relaxed);
    acc.release();
    zero_ref_queue_.push(page_id);
    return buf;
  } else {
    char* ret = acc->second.buf;
    acc.release();
    // 发布命中仅返回已有缓冲区，不改变引用计数与 LRU，以免意外固定住
    return ret;
  }
}

} // namespace diskann


