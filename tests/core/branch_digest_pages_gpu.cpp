// Discontiguous pages in reversed address order; compared with hashlib by Python.
#include "strata/core/branch_digest.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
    using namespace strata::core;
    if (argc != 2 || cudaFree(nullptr) != cudaSuccess) return 2;
    const uint64_t page_bytes = std::strtoull(argv[1], nullptr, 10);
    std::vector<uint8_t> input((std::istreambuf_iterator<char>(std::cin)), {});
    if (!page_bytes || page_bytes % branch_digest_tile_bytes || input.empty() || input.size() % page_bytes) return 3;
    uint8_t* device = nullptr;
    if (cudaMalloc(&device, input.size()) != cudaSuccess ||
        cudaMemcpy(device, input.data(), input.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 4;
    const auto baseline = branch_digest_counters();
    {
        BranchDigestWorkspace workspace;
        const auto n = input.size() / page_bytes;
        if (workspace.reserve_pages(0, page_bytes) || workspace.reserve_pages(n, page_bytes-1) ||
            workspace.reserve_pages(std::numeric_limits<uint64_t>::max(), page_bytes)) return 5;
        if (!workspace.reserve_pages(n, page_bytes)) return 6;
        std::vector<const uint8_t*> pages;
        for (size_t i = n; i; --i) pages.push_back(device + (i-1)*page_bytes);
        std::vector<BranchDigest> result;
        if (workspace.digest_pages({}, page_bytes, result) ||
            workspace.digest_pages({nullptr}, page_bytes, result) ||
            workspace.digest_pages(pages, page_bytes-1, result)) return 7;
        const auto before = branch_digest_counters();
        if (!workspace.digest_pages(pages, page_bytes, result)) return 8;
        const auto after = branch_digest_counters();
        if (after.workspace_allocations != before.workspace_allocations) return 9;
        std::printf("{\"digests\":[");
        for (size_t i = 0; i < result.size(); ++i) {
            if (i) std::putchar(',');
            std::putchar('"');
            for (auto byte : result[i].bytes) std::printf("%02x", byte);
            std::putchar('"');
        }
        std::printf("],\"device_hashed_bytes\":%llu,\"kernel_calls\":%llu,\"plan_copy_bytes\":%llu",
                    (unsigned long long)(after.device_hashed_bytes-before.device_hashed_bytes),
                    (unsigned long long)(after.kernel_calls-before.kernel_calls),
                    (unsigned long long)(after.plan_copy_bytes-before.plan_copy_bytes));
    }
    if (cudaFree(device) != cudaSuccess) return 10;
    const auto after = branch_digest_counters();
    const auto live = after.workspace_live_bytes - baseline.workspace_live_bytes;
    std::printf(",\"workspace_live_bytes\":%llu}\n", (unsigned long long)live);
    return live ? 11 : 0;
}
