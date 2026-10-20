// Supplemental real-CUDA timing witness. Existing integrity tests stay unchanged.
#include "strata/core/branch_digest.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <iterator>
#include <iostream>
#include <vector>

int main() {
    using namespace strata::core;
    std::vector<uint8_t> input((std::istreambuf_iterator<char>(std::cin)), {});
    constexpr uint64_t page_bytes = 65536;
    if (input.empty() || input.size() % page_bytes) return 2;
    uint8_t* device = nullptr;
    if (cudaMalloc(&device,input.size()) != cudaSuccess ||
        cudaMemcpy(device,input.data(),input.size(),cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    const auto baseline = branch_digest_counters();
    {
        BranchDigestWorkspace workspace;
        const auto n = input.size()/page_bytes;
        if (!workspace.reserve_pages(n,page_bytes)) return 4;
        std::vector<const uint8_t*> pages;
        for (size_t i=n;i;--i) pages.push_back(device+(i-1)*page_bytes);
        std::vector<BranchDigest> result;
        // Warm-up outside the timed measurement.
        if (!workspace.digest_pages(pages,page_bytes,result)) return 5;
        const auto before = branch_digest_counters();
        const auto start = std::chrono::steady_clock::now();
        for (int i=0;i<3;++i) if (!workspace.digest_pages(pages,page_bytes,result)) return 6;
        const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
        const auto after = branch_digest_counters();
        if (after.workspace_allocations != before.workspace_allocations) return 7;
        std::printf("{\"digests\":[");
        for (size_t i=0;i<result.size();++i) {
            if (i) std::putchar(',');
            std::putchar('"');
            for (auto byte:result[i].bytes) std::printf("%02x",byte);
            std::putchar('"');
        }
        std::printf("],\"wall_ns\":%llu,\"plan_upload_ns\":%llu,\"page_hash_ns\":%llu,"
                    "\"page_digest_download_ns\":%llu,\"device_hashed_bytes\":%llu,\"kernel_calls\":%llu",
            (unsigned long long)wall,
            (unsigned long long)(after.plan_upload_ns-before.plan_upload_ns),
            (unsigned long long)(after.page_hash_ns-before.page_hash_ns),
            (unsigned long long)(after.page_digest_download_ns-before.page_digest_download_ns),
            (unsigned long long)(after.device_hashed_bytes-before.device_hashed_bytes),
            (unsigned long long)(after.kernel_calls-before.kernel_calls));
    }
    if (cudaFree(device) != cudaSuccess) return 8;
    std::printf(",\"workspace_live_bytes\":%llu}\n",
        (unsigned long long)(branch_digest_counters().workspace_live_bytes-baseline.workspace_live_bytes));
}
