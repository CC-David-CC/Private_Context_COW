// Real-CUDA witness for persistent digest scratch, including fresh-byte checks.
#include "strata/core/branch_digest.hpp"
#include <cuda_runtime.h>
#include <algorithm>
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
        std::vector<BranchDigest> original;
        if (!workspace.digest_pages(pages,page_bytes,original)) return 5;
        const BranchDigest* result = nullptr;
        uint64_t count = 0;
        if (!workspace.digest_pages_view(pages,page_bytes,result,count) || count != original.size() ||
            !std::equal(original.begin(),original.end(),result)) return 6;
        const auto address = result;
        const auto before = branch_digest_counters();
        const auto start = std::chrono::steady_clock::now();
        for (int i=0;i<3;++i) {
            if (!workspace.digest_pages_view(pages,page_bytes,result,count) || result != address ||
                count != original.size()) return 7;
        }
        const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
        const auto after = branch_digest_counters();
        if (!std::equal(original.begin(),original.end(),result)) return 18;
        if (after.workspace_allocations != before.workspace_allocations ||
            after.host_workspace_allocations != before.host_workspace_allocations) return 8;
        std::printf("{\"digests\":[");
        for (size_t i=0;i<count;++i) {
            if (i) std::putchar(',');
            std::putchar('"');
            for (auto byte:result[i].bytes) std::printf("%02x",byte);
            std::putchar('"');
        }
        // A successful previous verification must not conceal a subsequent write.
        const uint8_t changed = input.back() ^ 1;
        if (cudaMemcpy(device+input.size()-1,&changed,1,cudaMemcpyHostToDevice) != cudaSuccess ||
            !workspace.digest_pages_view(pages,page_bytes,result,count) ||
            result[page_bytes/branch_digest_tile_bytes-1] == original[page_bytes/branch_digest_tile_bytes-1]) return 9;
        if (cudaMemcpy(device+input.size()-1,&input.back(),1,cudaMemcpyHostToDevice) != cudaSuccess) return 10;
        auto reversed = pages;
        std::reverse(reversed.begin(),reversed.end());
        if (!workspace.digest_pages_view(reversed,page_bytes,result,count)) return 11;
        const auto per_page = page_bytes/branch_digest_tile_bytes;
        for (size_t i=0;i<n;++i)
            if (!std::equal(original.begin()+i*per_page,original.begin()+(i+1)*per_page,
                            result+(n-i-1)*per_page)) return 12;
        if (!workspace.digest_pages_view({pages[0]},page_bytes,result,count) || count != per_page ||
            !std::equal(original.begin(),original.begin()+per_page,result)) return 13;
        if (workspace.digest_pages_view({nullptr},page_bytes,result,count) || result || count) return 14;
        if (!workspace.digest_pages_view(pages,page_bytes,result,count) ||
            !std::equal(original.begin(),original.end(),result)) return 15;
        std::printf("],\"wall_ns\":%llu,\"plan_upload_ns\":%llu,\"page_hash_ns\":%llu,"
                    "\"page_digest_download_ns\":%llu,\"device_hashed_bytes\":%llu,\"kernel_calls\":%llu,"
                    "\"host_allocations_during_measurement\":%llu,\"fresh_mutation_detected\":true",
            (unsigned long long)wall,
            (unsigned long long)(after.plan_upload_ns-before.plan_upload_ns),
            (unsigned long long)(after.page_hash_ns-before.page_hash_ns),
            (unsigned long long)(after.page_digest_download_ns-before.page_digest_download_ns),
            (unsigned long long)(after.device_hashed_bytes-before.device_hashed_bytes),
            (unsigned long long)(after.kernel_calls-before.kernel_calls),
            (unsigned long long)(after.host_workspace_allocations-before.host_workspace_allocations));
    }
    if (cudaFree(device) != cudaSuccess) return 16;
    const auto after = branch_digest_counters();
    const auto live = after.workspace_live_bytes-baseline.workspace_live_bytes;
    const auto host_live = after.host_workspace_live_bytes-baseline.host_workspace_live_bytes;
    std::printf(",\"workspace_live_bytes\":%llu,\"host_workspace_live_bytes\":%llu}\n",
        (unsigned long long)live,(unsigned long long)host_live);
    return live || host_live ? 17 : 0;
}
