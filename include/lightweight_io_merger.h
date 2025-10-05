// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <folly/AtomicHashMap.h>

namespace diskann {

class LightweightIOMerger {
public:
    // Simplified IO key encoding for AtomicHashMap compatibility
    // Since len is fixed (e.g., SECTOR_LEN = 4096), we only need to encode offset
    // Use offset directly as the key (supports full 64-bit address space)
    static inline uint64_t encode_io_key(uint64_t offset, uint64_t /*len*/) {
        return offset;  // Direct mapping for maximum simplicity and performance
    }
    
    static inline uint64_t decode_io_key(uint64_t key) {
        return key;  // Direct mapping
    }

private:

    // Minimal IO state management - lock-free design
    struct IOState {
        std::atomic<bool> completed{false};
        std::atomic<char*> cached_data{nullptr};  // Lock-free atomic pointer
        std::atomic<bool> deleted{false};         // Soft delete flag for cleanup
        uint64_t len;
        
        explicit IOState(uint64_t length) : len(length) {}
        
        ~IOState() {
            char* data = cached_data.load(std::memory_order_relaxed);
            if (data) {
                delete[] data;
            }
        }
        
        // 禁用拷贝和移动
        IOState(const IOState&) = delete;
        IOState& operator=(const IOState&) = delete;
        IOState(IOState&&) = delete;
        IOState& operator=(IOState&&) = delete;
    };
    
    // Global in-flight IO table - Folly's lock-free AtomicHashMap for maximum performance
    // Key: encoded uint64_t (offset + len), Value: IOState shared_ptr
    // Pre-allocated with 1M entries (~16MB memory)
    static folly::AtomicHashMap<uint64_t, std::shared_ptr<IOState>> in_flight_ios_;
    
    // FIFO tracking for cache eviction - circular buffer of keys
    static constexpr size_t MAX_CACHE_SIZE = 1000000;  
    static std::array<std::atomic<uint64_t>, MAX_CACHE_SIZE> fifo_keys_;
    static std::atomic<uint64_t> fifo_head_;  // Monotonically increasing counter
    
public:
    // Follower request information
    struct FollowerRequest {
        uint64_t key;  // Encoded IO key
        void* buf;
        uint64_t len;
    };
    
    // Batch context to pass between submit and wait (instead of thread_local)
    struct BatchContext {
        std::vector<std::shared_ptr<IOState>> leaders;
        std::vector<AlignedRead> ios;
        std::vector<FollowerRequest> followers;
    };
    
    // 批量执行可能合并的IO
    static int submit_merged_batch(LinuxAlignedFileReader* reader,
                                   std::vector<AlignedRead>& read_reqs,
                                   IOContext& ctx,
                                   BatchContext& batch_ctx);
    
    // 等待批量IO完成（需要reader参数）
    static void wait_merged_batch(LinuxAlignedFileReader* reader, 
                                   IOContext& ctx, 
                                   int n_ops,
                                   BatchContext& batch_ctx);
    
    // Remove specific cache entry
    static void remove_cache_entry(uint64_t key);
    
    // Get current cache statistics
    static size_t get_cache_size() {
        uint64_t head = fifo_head_.load(std::memory_order_acquire);
        return std::min<size_t>(head, MAX_CACHE_SIZE);
    }
    
    static constexpr size_t get_max_cache_size() {
        return MAX_CACHE_SIZE;
    }
    
    // Clear all cached IO entries - useful for starting fresh between test runs
    static void clear_all_cache();
    
    static uint64_t make_key(uint64_t offset, uint64_t len) {
        return encode_io_key(offset, len);
    }
    
private:
    // Track new key in FIFO and evict old ones if necessary
    static void track_key_in_fifo(uint64_t key);
    
};

} // namespace diskann
