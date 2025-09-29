// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "io_merge_wrapper.h"
#include "lightweight_io_merger.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

namespace diskann {

// Thread-local batch context shared between submit and wait calls
static thread_local LightweightIOMerger::BatchContext g_batch_ctx;

// Thread-local storage for current batch keys (shared between submit and wait)
// Keys are now encoded as uint64_t (offset only, since len is fixed)
static thread_local std::vector<uint64_t> g_batch_keys;

int IOMergeWrapper::submit_reqs_merged(LinuxAlignedFileReader* reader,
                                       std::vector<AlignedRead>& read_reqs, 
                                       IOContext& ctx) {
    if (reader == nullptr || read_reqs.empty()) {
        return 0;
    }
    
    // Critical: Clear previous batch data to avoid contamination
    g_batch_ctx.leaders.clear();
    g_batch_ctx.ios.clear();
    g_batch_ctx.followers.clear();
    
    // Store all keys from this batch for cleanup
    g_batch_keys.clear();
    for (const auto& req : read_reqs) {
        g_batch_keys.push_back(LightweightIOMerger::make_key(req.offset, req.len));
    }
    
    int n_ops = LightweightIOMerger::submit_merged_batch(reader, read_reqs, ctx, g_batch_ctx);
    
    return n_ops;
}

void IOMergeWrapper::get_events_merged(LinuxAlignedFileReader* reader,
                                       IOContext& ctx, 
                                       int n_ops) {
    if (reader == nullptr) {
        return;
    }
    
    // Use the same thread_local batch context as submit
    LightweightIOMerger::wait_merged_batch(reader, ctx, n_ops, g_batch_ctx);
    
    // **TESTING: Don't remove anything, enable full cross-batch reuse**
    // cleanup_expired_cache is also disabled to test if it's the problem
}


} // namespace diskann