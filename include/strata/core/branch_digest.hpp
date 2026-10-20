#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::core {
class BranchPages;
struct BranchDigest { uint8_t bytes[32]{}; };
inline bool operator==(const BranchDigest& a, const BranchDigest& b) {
    return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}
inline constexpr uint64_t branch_digest_tile_bytes = 4096;
struct BranchDigestCounters {
    uint64_t device_hashed_bytes = 0, digest_copy_bytes = 0, kernel_calls = 0;
    uint64_t workspace_allocations = 0, workspace_releases = 0, workspace_live_bytes = 0;
    uint64_t plan_copy_bytes = 0; // actual host-to-device page-address lists
    // Blocking wall time for digest_pages only, including existing waits.
    // Not GPU-event time; digest()/snapshot auditing has separate scope.
    uint64_t plan_upload_ns = 0, page_hash_ns = 0, page_digest_download_ns = 0;
    uint64_t host_workspace_allocations = 0, host_workspace_live_bytes = 0;
    // The COW preparation fast path compares fresh device hashes against the
    // live PhysicalPage seal mirror and returns only one mismatch index.
    uint64_t expected_seal_plan_copy_bytes = 0, compare_flag_init_bytes = 0, compare_result_copy_bytes = 0;
    uint64_t seal_compare_ns = 0, seal_result_download_ns = 0;
    uint64_t compare_workspace_live_bytes = 0;
};
BranchDigestCounters branch_digest_counters();

// Private scratch allocated before entering a memory-pressure operation. Each
// 4096-byte tile is independently SHA-256 hashed on device; only the ordered
// digests cross to the host. This is integrity metadata, never model state.
class BranchDigestWorkspace {
public:
    BranchDigestWorkspace() = default;
    ~BranchDigestWorkspace();
    BranchDigestWorkspace(const BranchDigestWorkspace&) = delete;
    BranchDigestWorkspace& operator=(const BranchDigestWorkspace&) = delete;
    bool reserve(uint64_t maximum_input_bytes);
    bool digest(const uint8_t* device, uint64_t bytes, std::vector<BranchDigest>& result);
    // Reserve before the execution lease starts. Each input page has the same
    // tile-aligned extent; output order is page order, then tile order.
    bool reserve_pages(uint64_t maximum_pages, uint64_t page_bytes);
    bool digest_pages(const std::vector<const uint8_t*>& pages, uint64_t page_bytes,
                      std::vector<BranchDigest>& result);
    // Borrow fresh results from preallocated private host scratch. Valid only
    // until another view call, reserve_pages(), or destruction. No cached hash
    // success: every call rehashes all input bytes. Failure clears both outputs.
    bool digest_pages_view(const std::vector<const uint8_t*>& pages, uint64_t page_bytes,
                           const BranchDigest*& result, uint64_t& count);
private:
    friend class BranchPages;
    // Freshly SHA-256 every tile, compare against each page's device-resident
    // expected seal, and return UINT64_MAX on match or the first tile index on
    // mismatch. The host reads only this scalar; this is not a cached result.
    bool compare_pages_view(const std::vector<const uint8_t*>& pages,
                            const std::vector<const BranchDigest*>& expected_seals,
                            uint64_t page_bytes, uint64_t& mismatch_tile);
    int owner_device_ = -1;
    bool digest_pages_into(const std::vector<const uint8_t*>& pages, uint64_t page_bytes,
                           BranchDigest* result, uint64_t result_capacity);
    bool reserve_compare(uint64_t pages);
    BranchDigest* device_ = nullptr;
    uint64_t capacity_ = 0;
    const uint8_t** pages_ = nullptr;
    uint64_t page_capacity_ = 0;
    const BranchDigest** expected_seals_ = nullptr;
    uint64_t expected_seal_capacity_ = 0;
    uint64_t* mismatch_tile_ = nullptr;
    std::vector<BranchDigest> host_;
};
} // namespace strata::core
