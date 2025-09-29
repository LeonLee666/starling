// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "lightweight_io_merger.h"
#include "linux_aligned_file_reader.h"
#include "pq_flash_index.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>
#include <immintrin.h>  // For _mm_pause()

namespace diskann {

// Static member definition - AtomicHashMap with 1M pre-allocated slots
// This provides lock-free performance for high-concurrency IO merging
// Key is encoded uint64_t combining offset and length
folly::AtomicHashMap<uint64_t, std::shared_ptr<LightweightIOMerger::IOState>>
    LightweightIOMerger::in_flight_ios_(1000000);  // 1M entries, ~16MB memory

int LightweightIOMerger::submit_merged_batch(LinuxAlignedFileReader* reader,
                                            std::vector<AlignedRead>& read_reqs,
                                            IOContext& ctx,
                                            BatchContext& batch_ctx) {
    if (read_reqs.empty()) {
        return 0;
    }

    static thread_local uint64_t submit_count = 0;
    if (++submit_count % 1000 == 0) {
        cleanup_expired_cache();
    }
    
    std::vector<AlignedRead> actual_io_reqs;
    std::vector<std::shared_ptr<IOState>> batch_leaders;
    
    for (auto& req : read_reqs) {
        uint64_t offset = req.offset;
        uint64_t len = req.len;
        void* buf = req.buf;
        
        if (buf == nullptr || len == 0) {
            std::cerr << "ERROR: Invalid request in batch - buf=" << buf << ", len=" << len << std::endl;
            continue;
        }
        
        uint64_t key = make_key(offset, len);
        std::shared_ptr<IOState> io_state;
        
        // Try to create IOState first
        try {
            io_state = std::make_shared<IOState>(len);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to create IOState in batch: " << e.what() << std::endl;
            continue;
        }
        
        // Atomically find or create IO state (lock-free with AtomicHashMap)
        auto result = in_flight_ios_.insert(key, io_state);
        if (result.second) {
            // Successfully inserted, become leader
            // leader needs to perform actual IO
            req.buf = buf; // ensure buffer points to the correct address
            actual_io_reqs.push_back(req);
            batch_leaders.push_back(io_state);
        } else {
            // Already exists, become follower
            io_state = result.first->second;
            // Skip if entry is marked as deleted
            if (io_state && io_state->deleted.load(std::memory_order_acquire)) {
                // Entry is being deleted, retry as leader
                result = in_flight_ios_.insert(key, std::make_shared<IOState>(len));
                if (result.second) {
                    io_state = result.first->second;
                    req.buf = buf;
                    actual_io_reqs.push_back(req);
                    batch_leaders.push_back(io_state);
                    continue;
                }
            }
            if (io_state) {
                FollowerRequest follower_req;
                follower_req.key = key;
                follower_req.buf = buf;
                follower_req.len = len;
                batch_ctx.followers.push_back(follower_req);
            } else {
                std::cerr << "ERROR: IOState is null for offset " << offset << std::endl;
            }
            continue;
        }
    }
    // Submit all actual IO requests at once
    if (!actual_io_reqs.empty()) {
        // Store batch information in the context passed by caller
        batch_ctx.leaders = std::move(batch_leaders);
        batch_ctx.ios = std::move(actual_io_reqs);
        return reader->submit_reqs(batch_ctx.ios, ctx);
    }
    return 0;
}

void LightweightIOMerger::wait_merged_batch(LinuxAlignedFileReader* reader, 
                                           IOContext& ctx, 
                                           int n_ops,
                                           BatchContext& batch_ctx) {
    if (reader == nullptr) {
        return;
    }
    
    // Wait for all leader IOs to complete (only if we have actual IOs)
    if (n_ops > 0) {
        try {
            reader->get_events(ctx, n_ops);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to get IO events: " << e.what() << std::endl;
            return;
        }
    }
    
    // Get current batch leader and IO request information from the passed context
    auto& current_leaders = batch_ctx.leaders;
    auto& current_ios = batch_ctx.ios;
    
    // Critical fix: handle data caching for each leader IO
    if (current_leaders.size() != current_ios.size()) {
        std::cerr << "ERROR: Leader count mismatch with IO count!" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < current_leaders.size(); ++i) {
        auto& io_state = current_leaders[i];
        auto& io_req = current_ios[i];
        if (!io_state) {
            std::cerr << "ERROR: IOState is null for leader " << i << std::endl;
            continue;
        }
        
        if (io_state->completed.load()) {
            continue;
        }
        
        try {
            // Cache leader's data (already read into io_req.buf) - lock-free approach
            if (io_req.buf != nullptr) {
                // Allocate and copy data for caching
                char* cached_buffer = new char[io_state->len];
                std::memcpy(cached_buffer, io_req.buf, io_state->len);
                
                // Atomically update cached_data pointer with compare-exchange
                char* expected = nullptr;
                if (io_state->cached_data.compare_exchange_strong(expected, cached_buffer,
                                                                   std::memory_order_release,
                                                                   std::memory_order_relaxed)) {
                    // Successfully installed our buffer, mark as completed
                    io_state->completed.store(true, std::memory_order_release);
                } else {
                    // Race condition: another thread cached it first (extremely rare)
                    delete[] cached_buffer;
                    io_state->completed.store(true, std::memory_order_release);
                }
            } else {
                std::cerr << "ERROR: Leader buffer is null for offset " << io_req.offset << std::endl;
                // Mark as completed even with null buffer
                io_state->completed.store(true, std::memory_order_release);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to cache leader data for offset " << io_req.offset 
                      << ": " << e.what() << std::endl;
            // Mark as completed to unblock followers
            io_state->completed.store(true, std::memory_order_release);
        }
    }
    
    auto& current_followers = batch_ctx.followers;
    
    // Lambda function for lightweight spin-wait on IO completion
    auto spin_wait_for_completion = [](const std::shared_ptr<IOState>& io_state) -> bool {
        const int MAX_SPIN_ITERATIONS = 10000;
        const int YIELD_THRESHOLD = 1000;
        
        for (int spin = 0; spin < MAX_SPIN_ITERATIONS; ++spin) {
            if (io_state->completed.load(std::memory_order_acquire)) {
                return true;
            }
            // Use pause instruction to reduce CPU power and improve hyper-threading
            _mm_pause();
            // Yield to other threads after spinning for a while
            if (spin > YIELD_THRESHOLD && spin % 100 == 0) {
                std::this_thread::yield();
            }
        }
        
        return false;
    };
    
    for (size_t idx = 0; idx < current_followers.size(); ++idx) {
        auto& follower = current_followers[idx];
        if (!follower.buf) {
            std::cerr << "ERROR: Follower " << idx << " has null buffer" << std::endl;
            continue;
        }
        
        // Lookup IOState from global hash map (wait-free read with AtomicHashMap)
        auto it = in_flight_ios_.find(follower.key);
        if (it != in_flight_ios_.end()) {
            auto io_state = it->second;
            // Skip if marked as deleted
            if (io_state && io_state->deleted.load(std::memory_order_acquire)) {
                continue;
            }
            if (!io_state) {
                std::cerr << "ERROR: IOState is null for follower request" << std::endl;
                continue;
            }
            
            // Check if already completed first (fast path)
            bool completed = io_state->completed.load(std::memory_order_acquire);
            if (!completed) {
                // Slow path: spin-wait for leader to complete
                completed = spin_wait_for_completion(io_state);
            }
            
            if (completed) {
                char* cached_ptr = io_state->cached_data.load(std::memory_order_acquire);
                if (cached_ptr != nullptr) {
                    std::memcpy(follower.buf, cached_ptr, follower.len);
                } else {
                    uint64_t offset = LightweightIOMerger::decode_io_key(follower.key);
                    std::cerr << "WARNING: No cached data available for follower (offset=" 
                              << offset << ")" << std::endl;
                }
            } else {
                uint64_t offset = LightweightIOMerger::decode_io_key(follower.key);
                std::cerr << "ERROR: Timeout waiting for leader IO after spin-wait (offset=" 
                          << offset << ")" << std::endl;
            }
        } else {
            uint64_t offset = LightweightIOMerger::decode_io_key(follower.key);
            std::cerr << "ERROR: Cannot find IOState for follower (offset=" 
                      << offset << ")" << std::endl;
        }
    }
}

void LightweightIOMerger::cleanup_expired_cache() {
    return;
}

void LightweightIOMerger::remove_cache_entry(uint64_t key) {
    // AtomicHashMap doesn't support erase, use soft delete instead
    auto it = in_flight_ios_.find(key);
    if (it != in_flight_ios_.end() && it->second) {
        it->second->deleted.store(true, std::memory_order_release);
    }
}


} // namespace diskann
