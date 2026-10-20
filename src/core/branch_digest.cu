#include "strata/core/branch_sha256.hpp"
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
#include "strata/core/branch_gpu_compat.hpp"
#else
#include <cuda_runtime.h>
#endif
#include <atomic>
#include <chrono>
#include <exception>
#include <limits>

namespace strata::core {
namespace {
struct Counters {
    std::atomic<uint64_t> device_hashed_bytes{0}, digest_copy_bytes{0}, kernel_calls{0};
    std::atomic<uint64_t> workspace_allocations{0}, workspace_releases{0}, workspace_live_bytes{0};
    std::atomic<uint64_t> plan_copy_bytes{0};
    std::atomic<uint64_t> plan_upload_ns{0}, page_hash_ns{0}, page_digest_download_ns{0};
    std::atomic<uint64_t> host_workspace_allocations{0}, host_workspace_live_bytes{0};
    std::atomic<uint64_t> expected_seal_plan_copy_bytes{0}, compare_flag_init_bytes{0}, compare_result_copy_bytes{0};
    std::atomic<uint64_t> seal_compare_ns{0}, seal_result_download_ns{0};
    std::atomic<uint64_t> compare_workspace_live_bytes{0};
};
Counters& counters() { static auto* value = new Counters; return *value; }
uint64_t tiles(uint64_t bytes) { return bytes ? (bytes - 1) / branch_digest_tile_bytes + 1 : 1; }

// Device-only SHA-256 candidate. Keep the CPU reference in branch_sha256.hpp
// independent; the rolling schedule retains all 64 rounds and every input
// byte while replacing the 64-word expanded schedule with 16 rotating words.
__device__ __forceinline__ uint32_t device_rotr(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}
__device__ __forceinline__ void branch_sha256_device(const uint8_t* input, uint64_t bytes,
                                                      BranchDigest& digest) {
    constexpr uint32_t constants[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                         0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const uint64_t blocks = bytes / 64 + 1 + (bytes % 64 > 55);
    const uint64_t padded_bytes = blocks * 64, bits = bytes * 8;
    for (uint64_t block = 0; block < blocks; ++block) {
        uint32_t words[16]{};
        for (unsigned i = 0; i < 64; ++i) {
            const auto position = block * 64 + i;
            uint8_t value = 0;
            if (position < bytes) value = input[position];
            else if (position == bytes) value = 0x80;
            else if (position >= padded_bytes - 8)
                value = uint8_t(bits >> (8 * (padded_bytes - 1 - position)));
            words[i / 4] |= uint32_t(value) << (24 - 8 * (i % 4));
        }
        uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
        uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
        #pragma unroll
        for (unsigned i = 0; i < 64; ++i) {
            if (i >= 16) {
                const auto x = words[(i - 15) & 15], y = words[(i - 2) & 15];
                const auto s0 = device_rotr(x, 7) ^ device_rotr(x, 18) ^ (x >> 3);
                const auto s1 = device_rotr(y, 17) ^ device_rotr(y, 19) ^ (y >> 10);
                words[i & 15] += s0 + words[(i - 7) & 15] + s1;
            }
            const auto s1 = device_rotr(e, 6) ^ device_rotr(e, 11) ^ device_rotr(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto t1 = h + s1 + choice + constants[i] + words[i & 15];
            const auto s0 = device_rotr(a, 2) ^ device_rotr(a, 13) ^ device_rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + majority;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    for (unsigned i = 0; i < 32; ++i) digest.bytes[i] = uint8_t(state[i / 4] >> (24 - 8 * (i % 4)));
}

__global__ void hash_tiles(const uint8_t* input, uint64_t bytes, uint64_t count, BranchDigest* result) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto offset = index * branch_digest_tile_bytes;
    const auto remaining = bytes - offset;
    const auto length = remaining < branch_digest_tile_bytes ? remaining : branch_digest_tile_bytes;
    branch_sha256_device(input + offset, length, result[index]);
}
__global__ void hash_pages(const uint8_t* const* pages, uint64_t per_page, uint64_t count,
                           BranchDigest* result) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    branch_sha256_device(pages[index / per_page] + (index % per_page) * branch_digest_tile_bytes,
                         branch_digest_tile_bytes, result[index]);
}
__global__ void hash_compare_pages(const uint8_t* const* pages,
                                   const BranchDigest* const* expected_seals,
                                   uint64_t per_page, uint64_t count,
                                   unsigned long long* first_mismatch) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    BranchDigest actual{};
    branch_sha256_device(pages[index / per_page] + (index % per_page) * branch_digest_tile_bytes,
                         branch_digest_tile_bytes, actual);
    const auto& expected = expected_seals[index / per_page][index % per_page];
    bool equal = true;
    #pragma unroll
    for (int byte = 0; byte < 32; ++byte) equal = equal && actual.bytes[byte] == expected.bytes[byte];
    if (!equal) atomicMin(first_mismatch, static_cast<unsigned long long>(index));
}
}
BranchDigestCounters branch_digest_counters() {
    const auto& c = counters();
    return {c.device_hashed_bytes.load(), c.digest_copy_bytes.load(), c.kernel_calls.load(),
            c.workspace_allocations.load(), c.workspace_releases.load(), c.workspace_live_bytes.load(),
            c.plan_copy_bytes.load(), c.plan_upload_ns.load(), c.page_hash_ns.load(),
            c.page_digest_download_ns.load(), c.host_workspace_allocations.load(),
            c.host_workspace_live_bytes.load(), c.expected_seal_plan_copy_bytes.load(),
            c.compare_flag_init_bytes.load(), c.compare_result_copy_bytes.load(),
            c.seal_compare_ns.load(), c.seal_result_download_ns.load(), c.compare_workspace_live_bytes.load()};
}
BranchDigestWorkspace::~BranchDigestWorkspace() {
    int previous=-1;
    if (owner_device_>=0 && (cudaGetDevice(&previous)!=cudaSuccess ||
        (previous!=owner_device_ && cudaSetDevice(owner_device_)!=cudaSuccess))) std::terminate();
    counters().host_workspace_live_bytes -= host_.capacity() * sizeof(BranchDigest);
    if (pages_) {
        if (cudaFree(pages_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= page_capacity_ * sizeof(*pages_);
    }
    if (expected_seals_) {
        if (cudaFree(expected_seals_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= expected_seal_capacity_ * sizeof(*expected_seals_);
        counters().compare_workspace_live_bytes -= expected_seal_capacity_ * sizeof(*expected_seals_);
    }
    if (mismatch_tile_) {
        if (cudaFree(mismatch_tile_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= sizeof(*mismatch_tile_);
        counters().compare_workspace_live_bytes -= sizeof(*mismatch_tile_);
    }
    if (device_) {
        if (cudaFree(device_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= capacity_ * sizeof(BranchDigest);
    }
    if (owner_device_>=0 && previous!=owner_device_ && cudaSetDevice(previous)!=cudaSuccess) std::terminate();
}
bool BranchDigestWorkspace::reserve(uint64_t bytes) {
    int current=-1;
    if (cudaGetDevice(&current)!=cudaSuccess || (owner_device_>=0 && current!=owner_device_)) return false;
    const auto count = tiles(bytes);
    if (count <= capacity_) return true;
    // Bound the launch dimension and byte multiplication before any allocation.
    if (count > uint64_t(std::numeric_limits<int>::max()) * 128 ||
        count > std::numeric_limits<size_t>::max() / sizeof(BranchDigest)) return false;
    BranchDigest* next = nullptr;
    if (cudaMalloc(&next, count * sizeof(BranchDigest)) != cudaSuccess) return false;
    if (device_) {
        if (cudaFree(device_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= capacity_ * sizeof(BranchDigest);
    }
    device_ = next; capacity_ = count;
    owner_device_=current;
    ++counters().workspace_allocations;
    counters().workspace_live_bytes += count * sizeof(BranchDigest);
    return true;
}
bool BranchDigestWorkspace::digest(const uint8_t* input, uint64_t bytes, std::vector<BranchDigest>& result) {
    int current=-1;
    if (cudaGetDevice(&current)!=cudaSuccess || current!=owner_device_) return false;
    const auto count = tiles(bytes);
    if (!input || !device_ || count > capacity_) return false;
    std::vector<BranchDigest> output;
    try { output.resize(count); } catch (...) { return false; }
    hash_tiles<<<unsigned((count + 127) / 128), 128>>>(input, bytes, count, device_);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
    ++counters().kernel_calls;
    counters().device_hashed_bytes += bytes;
    if (cudaMemcpy(output.data(), device_, count * sizeof(BranchDigest), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    counters().digest_copy_bytes += count * sizeof(BranchDigest);
    result.swap(output);
    return true;
}
bool BranchDigestWorkspace::reserve_pages(uint64_t pages, uint64_t bytes) {
    if (!pages || !bytes || bytes % branch_digest_tile_bytes ||
        pages > std::numeric_limits<uint64_t>::max() / bytes ||
        pages > std::numeric_limits<size_t>::max() / sizeof(*pages_) || !reserve(pages * bytes)) return false;
    const auto count = pages * (bytes / branch_digest_tile_bytes);
    const auto previous = host_.capacity();
    try { if (host_.size() < count) host_.resize(count); } catch (...) { return false; }
    if (host_.capacity() != previous) {
        ++counters().host_workspace_allocations;
        counters().host_workspace_live_bytes += (host_.capacity()-previous) * sizeof(BranchDigest);
    }
    if (pages <= page_capacity_) return reserve_compare(pages);
    const uint8_t** next = nullptr;
    if (cudaMalloc(&next, pages * sizeof(*pages_)) != cudaSuccess) return false;
    if (pages_) {
        if (cudaFree(pages_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= page_capacity_ * sizeof(*pages_);
    }
    pages_ = next; page_capacity_ = pages;
    ++counters().workspace_allocations;
    counters().workspace_live_bytes += pages * sizeof(*pages_);
    return reserve_compare(pages);
}
bool BranchDigestWorkspace::reserve_compare(uint64_t pages) {
    if (!pages || pages > std::numeric_limits<size_t>::max() / sizeof(*expected_seals_)) return false;
    if (pages > expected_seal_capacity_) {
        const BranchDigest** next = nullptr;
        if (cudaMalloc(&next, pages * sizeof(*expected_seals_)) != cudaSuccess) return false;
        if (expected_seals_) {
            if (cudaFree(expected_seals_) != cudaSuccess) std::terminate();
            ++counters().workspace_releases;
            counters().workspace_live_bytes -= expected_seal_capacity_ * sizeof(*expected_seals_);
            counters().compare_workspace_live_bytes -= expected_seal_capacity_ * sizeof(*expected_seals_);
        }
        expected_seals_ = next;
        expected_seal_capacity_ = pages;
        ++counters().workspace_allocations;
        counters().workspace_live_bytes += pages * sizeof(*expected_seals_);
        counters().compare_workspace_live_bytes += pages * sizeof(*expected_seals_);
    }
    if (!mismatch_tile_) {
        if (cudaMalloc(&mismatch_tile_, sizeof(*mismatch_tile_)) != cudaSuccess) return false;
        ++counters().workspace_allocations;
        counters().workspace_live_bytes += sizeof(*mismatch_tile_);
        counters().compare_workspace_live_bytes += sizeof(*mismatch_tile_);
    }
    return true;
}
bool BranchDigestWorkspace::digest_pages(const std::vector<const uint8_t*>& pages, uint64_t bytes,
                                         std::vector<BranchDigest>& result) {
    if (pages.empty() || !bytes || bytes % branch_digest_tile_bytes ||
        pages.size() > std::numeric_limits<uint64_t>::max() / bytes) return false;
    const auto count = pages.size() * (bytes / branch_digest_tile_bytes);
    if (count > capacity_) return false;
    std::vector<BranchDigest> output;
    try { output.resize(count); } catch (...) { return false; }
    if (!digest_pages_into(pages,bytes,output.data(),output.size())) return false;
    result.swap(output);
    return true;
}
bool BranchDigestWorkspace::digest_pages_view(const std::vector<const uint8_t*>& pages, uint64_t bytes,
                                              const BranchDigest*& result, uint64_t& count) {
    result = nullptr; count = 0;
    if (!digest_pages_into(pages,bytes,host_.data(),host_.size())) return false;
    result = host_.data(); count = pages.size() * (bytes / branch_digest_tile_bytes);
    return true;
}
bool BranchDigestWorkspace::compare_pages_view(
        const std::vector<const uint8_t*>& pages,
        const std::vector<const BranchDigest*>& expected_seals,
        uint64_t bytes, uint64_t& mismatch_tile) {
    mismatch_tile = std::numeric_limits<uint64_t>::max();
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != owner_device_ || pages.empty() ||
        pages.size() != expected_seals.size() || pages.size() > page_capacity_ || !pages_ ||
        !bytes || bytes % branch_digest_tile_bytes ||
        pages.size() > std::numeric_limits<uint64_t>::max() / bytes ||
        !reserve_compare(pages.size())) return false;
    const uint64_t per_page = bytes / branch_digest_tile_bytes;
    const uint64_t count = pages.size() * per_page;
    if (count > capacity_ || count > uint64_t(std::numeric_limits<int>::max()) * 128) return false;
    for (size_t i = 0; i < pages.size(); ++i)
        if (!pages[i] || !expected_seals[i]) return false;
    const auto page_plan_bytes = pages.size() * sizeof(*pages_);
    auto start = std::chrono::steady_clock::now();
    const auto page_upload = cudaMemcpy(pages_, pages.data(), page_plan_bytes, cudaMemcpyHostToDevice);
    counters().plan_upload_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (page_upload != cudaSuccess) return false;
    counters().plan_copy_bytes += page_plan_bytes;
    start = std::chrono::steady_clock::now();
    const auto seal_upload = cudaMemcpy(expected_seals_, expected_seals.data(),
        pages.size() * sizeof(*expected_seals_), cudaMemcpyHostToDevice);
    counters().plan_upload_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (seal_upload != cudaSuccess) return false;
    counters().expected_seal_plan_copy_bytes += pages.size() * sizeof(*expected_seals_);
    if (cudaMemset(mismatch_tile_, 0xff, sizeof(*mismatch_tile_)) != cudaSuccess) return false;
    counters().compare_flag_init_bytes += sizeof(*mismatch_tile_);

    start = std::chrono::steady_clock::now();
    hash_compare_pages<<<unsigned((count + 127) / 128), 128>>>(
        pages_, expected_seals_, per_page, count,
        reinterpret_cast<unsigned long long*>(mismatch_tile_));
    const auto launch = cudaGetLastError();
    const auto complete = launch == cudaSuccess ? cudaDeviceSynchronize() : launch;
    counters().seal_compare_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (complete != cudaSuccess) return false;
    ++counters().kernel_calls;
    counters().device_hashed_bytes += pages.size() * bytes;
    start = std::chrono::steady_clock::now();
    const auto download = cudaMemcpy(&mismatch_tile, mismatch_tile_, sizeof(mismatch_tile), cudaMemcpyDeviceToHost);
    counters().seal_result_download_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (download != cudaSuccess) return false;
    counters().compare_result_copy_bytes += sizeof(mismatch_tile);
    return true;
}
bool BranchDigestWorkspace::digest_pages_into(const std::vector<const uint8_t*>& pages, uint64_t bytes,
                                             BranchDigest* result, uint64_t result_capacity) {
    int current=-1;
    if (cudaGetDevice(&current)!=cudaSuccess || current!=owner_device_) return false;
    if (pages.empty() || pages.size() > page_capacity_ || !pages_ || !device_ ||
        !bytes || bytes % branch_digest_tile_bytes ||
        pages.size() > std::numeric_limits<uint64_t>::max() / bytes) return false;
    const auto count = pages.size() * (bytes / branch_digest_tile_bytes);
    if (count > capacity_ || count > result_capacity || !result) return false;
    for (const auto* page : pages) if (!page) return false;
    const auto plan_bytes = pages.size() * sizeof(*pages_);
    auto start = std::chrono::steady_clock::now();
    const auto upload = cudaMemcpy(pages_, pages.data(), plan_bytes, cudaMemcpyHostToDevice);
    counters().plan_upload_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count();
    if (upload != cudaSuccess) return false;
    counters().plan_copy_bytes += plan_bytes;
    start = std::chrono::steady_clock::now();
    hash_pages<<<unsigned((count + 127) / 128), 128>>>(pages_, bytes / branch_digest_tile_bytes, count, device_);
    const auto launch = cudaGetLastError();
    const auto complete = launch == cudaSuccess ? cudaDeviceSynchronize() : launch;
    counters().page_hash_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count();
    if (complete != cudaSuccess) return false;
    ++counters().kernel_calls;
    counters().device_hashed_bytes += pages.size() * bytes;
    start = std::chrono::steady_clock::now();
    const auto download = cudaMemcpy(result, device_, count * sizeof(BranchDigest), cudaMemcpyDeviceToHost);
    counters().page_digest_download_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count();
    if (download != cudaSuccess) return false;
    counters().digest_copy_bytes += count * sizeof(BranchDigest);
    return true;
}
} // namespace strata::core
