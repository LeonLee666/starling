#pragma once
#include <cstdint>
#include <vector>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <oneapi/tbb/concurrent_queue.h>
#include "utils.h"

namespace diskann {

// A simple threadsafe pool of fixed-size pages (SECTOR_LEN bytes, 4KB)
class PagePool {
 public:
  PagePool() = default;
  ~PagePool() { destroy(); }

  void init(uint64_t num_pages, uint64_t page_size);
  void destroy();
  // clear all cached pages from the buffer pool back to freelist
  void clear_cache();

  // acquire a page buffer (aligned to page_size). returns nullptr on failure
  char* acquire();
  // return a page buffer back to pool
  void release(char* buf);

  // shared page mapping APIs (thread-safe)
  // if page exists, increments refcount and returns its buffer; else returns nullptr
  char* enter_page(unsigned page_id);
  // check existence without changing refcount/LRU
  bool contains(unsigned page_id);
  // publish a freshly loaded page into cache; if already exists, increments refcount and returns existing buffer
  // may return a different canonical buffer than the input; caller should release its own buffer if different
  char* add_page(unsigned page_id, char* buf);
  // release a reference to a cached page; when refcount reaches zero, evict and return buffer to freelist
  void leave_page(unsigned page_id);

 private:
  uint64_t page_size_{0};
  std::vector<char*> freelist_;
  std::mutex freelist_mutex_;

  // If non-null, all pages are slices of this contiguous block
  char* big_block_{nullptr};
  uint64_t num_pages_allocated_{0};

  struct PageEntry {
    char* buf{nullptr};
    std::atomic<uint32_t> refcount{0};
  };
  using PageMap = oneapi::tbb::concurrent_hash_map<unsigned, PageEntry>;
  PageMap page_entries_;

  // Zero-ref candidates queue (multi-producer, multi-consumer)
  oneapi::tbb::concurrent_queue<unsigned> zero_ref_queue_;
};

} // namespace diskann


