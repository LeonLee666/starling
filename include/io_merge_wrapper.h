// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "linux_aligned_file_reader.h"
#include "aligned_file_reader.h"
#include "lightweight_io_merger.h"
#include <vector>
#include <memory>
#include <mutex>

namespace diskann {

// IO合并包装器 - 使用轻量级线程协作式IO合并
class IOMergeWrapper {
public:
    // IO合并接口 - 使用调用方提供的 reader (thread_local batch context managed internally)
    static int submit_reqs_merged(LinuxAlignedFileReader* reader, 
                                   std::vector<AlignedRead>& read_reqs, 
                                   IOContext& ctx);
    static void get_events_merged(LinuxAlignedFileReader* reader, 
                                   IOContext& ctx, 
                                   int n_ops);
};

} // namespace diskann