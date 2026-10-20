// Standalone diagnostic driver for the exact production comparison kernel.
// This translation unit includes branch_digest.cu so it can launch its
// internal hash_compare_pages kernel without widening the production API.
#include "../../src/core/branch_digest.cu"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

static bool ok(cudaError_t error) { return error == cudaSuccess; }

int main(int argc, char** argv) {
    using namespace strata::core;
    if (argc != 2) return 2;
    char* end = nullptr;
    const auto data_bytes = std::strtoull(argv[1], &end, 10);
    constexpr uint64_t page_bytes = 65536;
    constexpr uint64_t tile_bytes = branch_digest_tile_bytes;
    if (!end || *end || !data_bytes || data_bytes % page_bytes) return 2;
    const uint64_t page_count = data_bytes / page_bytes;
    const uint64_t tiles_per_page = page_bytes / tile_bytes;
    const uint64_t tile_count = page_count * tiles_per_page;
    const uint64_t seal_bytes = tile_count * sizeof(BranchDigest);

    std::vector<uint8_t> input(data_bytes);
    std::vector<BranchDigest> host_seals(tile_count);
    std::cin.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(data_bytes));
    std::cin.read(reinterpret_cast<char*>(host_seals.data()), static_cast<std::streamsize>(seal_bytes));
    if (!std::cin || std::cin.peek() != std::char_traits<char>::eof()) return 3;

    uint8_t* d_input = nullptr;
    BranchDigest* d_seals = nullptr;
    const uint8_t** d_pages = nullptr;
    const BranchDigest** d_expected = nullptr;
    unsigned long long* d_mismatch = nullptr;
    if (!ok(cudaMalloc(&d_input, data_bytes)) || !ok(cudaMalloc(&d_seals, seal_bytes)) ||
        !ok(cudaMalloc(&d_pages, page_count * sizeof(*d_pages))) ||
        !ok(cudaMalloc(&d_expected, page_count * sizeof(*d_expected))) ||
        !ok(cudaMalloc(&d_mismatch, sizeof(*d_mismatch)))) return 4;
    if (!ok(cudaMemcpy(d_input, input.data(), data_bytes, cudaMemcpyHostToDevice)) ||
        !ok(cudaMemcpy(d_seals, host_seals.data(), seal_bytes, cudaMemcpyHostToDevice))) return 5;

    std::vector<const uint8_t*> pages(page_count);
    std::vector<const BranchDigest*> expected(page_count);
    for (uint64_t page = 0; page < page_count; ++page) {
        pages[page] = d_input + page * page_bytes;
        expected[page] = d_seals + page * tiles_per_page;
    }

    cudaEvent_t total_start{}, kernel_start{}, kernel_stop{}, total_stop{};
    if (!ok(cudaEventCreate(&total_start)) || !ok(cudaEventCreate(&kernel_start)) ||
        !ok(cudaEventCreate(&kernel_stop)) || !ok(cudaEventCreate(&total_stop))) return 6;
    // One untimed compile/cache warm-up, checked against the CPU-provided seals.
    unsigned long long warmup_mismatch = 0;
    if (!ok(cudaMemcpy(d_pages, pages.data(), page_count * sizeof(*d_pages), cudaMemcpyHostToDevice)) ||
        !ok(cudaMemcpy(d_expected, expected.data(), page_count * sizeof(*d_expected), cudaMemcpyHostToDevice)) ||
        !ok(cudaMemset(d_mismatch, 0xff, sizeof(*d_mismatch)))) return 7;
    hash_compare_pages<<<unsigned((tile_count + 127) / 128), 128>>>(
        d_pages, d_expected, tiles_per_page, tile_count, d_mismatch);
    if (!ok(cudaGetLastError()) || !ok(cudaDeviceSynchronize()) ||
        !ok(cudaMemcpy(&warmup_mismatch, d_mismatch, sizeof(warmup_mismatch), cudaMemcpyDeviceToHost)) ||
        warmup_mismatch != std::numeric_limits<unsigned long long>::max()) return 8;

    std::vector<float> total_ms, kernel_ms;
    std::vector<uint64_t> host_wall_ns;
    for (int repetition = 0; repetition < 3; ++repetition) {
        auto wall_start = std::chrono::steady_clock::now();
        if (!ok(cudaEventRecord(total_start)) ||
            !ok(cudaMemcpy(d_pages, pages.data(), page_count * sizeof(*d_pages), cudaMemcpyHostToDevice)) ||
            !ok(cudaMemcpy(d_expected, expected.data(), page_count * sizeof(*d_expected), cudaMemcpyHostToDevice)) ||
            !ok(cudaMemset(d_mismatch, 0xff, sizeof(*d_mismatch))) ||
            !ok(cudaEventRecord(kernel_start))) return 9;
        hash_compare_pages<<<unsigned((tile_count + 127) / 128), 128>>>(
            d_pages, d_expected, tiles_per_page, tile_count, d_mismatch);
        if (!ok(cudaGetLastError()) || !ok(cudaEventRecord(kernel_stop)) || !ok(cudaDeviceSynchronize())) return 10;
        unsigned long long mismatch = 0;
        if (!ok(cudaMemcpy(&mismatch, d_mismatch, sizeof(mismatch), cudaMemcpyDeviceToHost)) ||
            mismatch != std::numeric_limits<unsigned long long>::max() ||
            !ok(cudaEventRecord(total_stop)) || !ok(cudaEventSynchronize(total_stop))) return 11;
        host_wall_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_start).count());
        float total = 0, kernel = 0;
        if (!ok(cudaEventElapsedTime(&total, total_start, total_stop)) ||
            !ok(cudaEventElapsedTime(&kernel, kernel_start, kernel_stop))) return 12;
        total_ms.push_back(total);
        kernel_ms.push_back(kernel);
    }
    std::printf("{\"status\":\"PASS\",\"bytes_per_compare\":%llu,\"tiles_per_compare\":%llu,"
                "\"warmup_compares\":1,\"measured_compares\":3,\"mismatch\":null,"
                "\"fresh_hash_bytes\":%llu,\"kernel_launches\":3,"
                "\"event_total_ms\":[%.6f,%.6f,%.6f],"
                "\"event_kernel_ms\":[%.6f,%.6f,%.6f],"
                "\"host_wall_ns\":[%llu,%llu,%llu]}\n",
        (unsigned long long)data_bytes, (unsigned long long)tile_count,
        (unsigned long long)(3 * data_bytes),
        total_ms[0], total_ms[1], total_ms[2],
        kernel_ms[0], kernel_ms[1], kernel_ms[2],
        (unsigned long long)host_wall_ns[0], (unsigned long long)host_wall_ns[1],
        (unsigned long long)host_wall_ns[2]);
    cudaEventDestroy(total_start); cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop); cudaEventDestroy(total_stop);
    cudaFree(d_mismatch); cudaFree(d_expected); cudaFree(d_pages); cudaFree(d_seals); cudaFree(d_input);
    return 0;
}
