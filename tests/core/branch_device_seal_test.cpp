// Real BranchSession all-pool COW integrity gate; no model forward is run.
#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "branch_test_extent.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
static int checks = 0, failures = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
static void require(bool ok, const char* message) {
    check(ok, message);
    if (!ok) std::exit(2);
}
static void gpu_check(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "FAIL: CUDA line %d `%s`: %s\n", line, expression,
                     cudaGetErrorString(result));
        ++checks;
        ++failures;
        std::exit(2);
    }
    ++checks;
}
#define gpu(expr) gpu_check((expr), #expr, __LINE__)

static void write_prepared_range(const BranchSession& session, int64_t position,
                                 int64_t count, unsigned char value) {
    const auto shape = strata::kernels::qsa_real_shapes();
    for (const auto& pool : session.pools()) {
        uint64_t first, end;
        if (pool.pooled) {
            first = position / shape.idx_block;
            end = (position + count) / shape.idx_block + 1;
        } else {
            first = position / shape.page_size * shape.page_size;
            end = (position + count + shape.page_size - 1) / shape.page_size * shape.page_size;
        }
        const auto offset = first * pool.bytes_per_cell;
        const auto bytes = (end - first) * pool.bytes_per_cell;
        gpu(cudaMemset(static_cast<unsigned char*>(pool.address) + offset, value, bytes));
    }
}

static void seed_session_pools(const BranchSession& session, uint64_t seed) {
    const auto pools = session.pools();
    const auto mapped = branch_test_mapped_bytes_by_pool(session);
    require(mapped.size() == pools.size(), "mapped extent inventory matches pools");
    uint64_t pool_index = 0;
    for (const auto& pool : pools) {
        require(mapped[pool_index] > 0 && mapped[pool_index] <= branch_test_reserved_capacity(pool),
                "seed pool mapping is a positive prefix within rounded reserved capacity");
        const auto extent = mapped[pool_index];
        auto* base = static_cast<unsigned char*>(pool.address);
        gpu(cudaMemset(base, static_cast<int>((seed + pool_index * 37) & 0xff), extent));
        uint64_t tile_index = 0;
        for (uint64_t offset = 0; offset < extent; offset += branch_digest_tile_bytes, ++tile_index) {
            const uint64_t marker[2] = {
                0x4d4630434f575f31ULL ^ (seed + pool_index * 0x100000001b3ULL),
                (pool_index << 48) ^ tile_index
            };
            const auto marker_bytes = std::min<uint64_t>(sizeof(marker), extent - offset);
            gpu(cudaMemcpy(base + offset, marker, marker_bytes, cudaMemcpyHostToDevice));
        }
        ++pool_index;
    }
}

int main() {
    gpu(cudaSetDevice(0));
    ModelGeometry geometry;
    geometry.n_layers = 4;
    geometry.ssm_state_size = 2;
    geometry.ssm_v_heads = 2;
    geometry.ssm_conv_channels = 8;
    qsa_set_kv_int8(false); qsa_set_kv_q4(false); qsa_set_kv_hybrid(false);
    qsa_set_kv_elastic(false, 0); qsa_set_kv_resident(0);
    SessionState state;
    QsaState qsa;
    state.max_cells = 8192; state.layer_hi = 4;
    state.gdn_alloc = 3; state.qsa_alloc = 1; state.qsa_states = &qsa;
    void* arena = nullptr;
    gpu(cudaMalloc(&arena, qsa_state_bytes(geometry, state.max_cells)));
    require(qsa_state_init(geometry, state.max_cells, arena, qsa) != 0, "initialize QSA");
    qsa_state_zero(qsa, geometry, nullptr);
    ConversationStateSizes sizes;
    std::string error;
    require(conversation_state_sizes(geometry, sizes, error), "state sizes");
    gpu(cudaMalloc(&state.gdn_state, sizes.gdn));
    gpu(cudaMalloc(&state.ple_hist, sizes.ple));
    gpu(cudaMemset(state.gdn_state, 0, sizes.gdn));
    gpu(cudaMemset(state.ple_hist, 0, sizes.ple));

    const auto page_baseline = BranchPages::counters();
    const auto digest_baseline = branch_digest_counters();
    {
        BranchSession session(state, geometry, "device-seal-compare-fixture");
        require(session.bind(error), "bind");
        require(session.begin(0, 4097, error), "begin initial prefix");
        seed_session_pools(session, 0xC0FFEE);
        require(session.finish(std::vector<int32_t>(4097, 42), {1.0f, 2.0f}, error), "finish prefix");
        auto root = session.freeze(error);
        require(bool(root), "freeze root");
        const auto root_pages = session.ownership();
        uint64_t sealed_bytes = 0;
        for (const auto& page : root_pages) sealed_bytes += page.bytes;
        const auto seals_frozen = BranchPages::counters();
        check(seals_frozen.seal_live_bytes > page_baseline.seal_live_bytes,
              "freeze allocates expected mirrors for sealed physical pages");
        check(seals_frozen.seal_upload_bytes > page_baseline.seal_upload_bytes,
              "mirrors are populated from CPU-validated seals");

        int device_count = 0;
        gpu(cudaGetDeviceCount(&device_count));
        if (device_count > 1) {
            gpu(cudaSetDevice(1));
            check(!session.begin(4097, 1, error), "wrong-device prepare fails closed");
            gpu(cudaSetDevice(0));
            require(session.restore(root, error), "restore recovers after wrong-device refusal");
            check(session.ids().size() == 4097, "wrong-device refusal consumes no token");
        }

        // Regression: all sealed bytes are freshly SHA-256 hashed, while only
        // a scalar mismatch index returns to the host (not one digest per tile).
        const auto digest_before = branch_digest_counters();
        require(session.begin(4097, 1, error), "prepare next token");
        const auto digest_after = branch_digest_counters();
        check(digest_after.device_hashed_bytes - digest_before.device_hashed_bytes == sealed_bytes,
              "every sealed tile is freshly hashed");
        check(digest_after.digest_copy_bytes - digest_before.digest_copy_bytes == 0,
              "prepare does not download the full digest vector");
        check(digest_after.compare_result_copy_bytes - digest_before.compare_result_copy_bytes == sizeof(uint64_t),
              "prepare reads only the mismatch index");
        check(digest_after.expected_seal_plan_copy_bytes > digest_before.expected_seal_plan_copy_bytes,
              "expected mirror pointers are uploaded for this comparison");
        check(digest_after.compare_flag_init_bytes - digest_before.compare_flag_init_bytes == sizeof(uint64_t),
              "comparison mismatch flag initialization is counted separately");
        write_prepared_range(session, 4097, 1, 0x33);
        require(session.finish({43}, {3.0f, 4.0f}, error), "finish child token");

        // A corruption at the first byte of the first page is detected before
        // any pool detaches, even though that page is outside the write tail.
        require(session.restore(root, error), "restore root before first-byte corruption");
        const auto before_first_reject = session.ownership();
        require(!before_first_reject.empty(), "root pages before first-byte corruption");
        VmmMapping first_alias;
        require(first_alias.reserve(vmm_granularity()), "reserve first-page alias");
        require(first_alias.map(0, before_first_reject.front().driver_handle, true), "map first page writable");
        uint8_t first_original = 0;
        gpu(cudaMemcpy(&first_original, first_alias.base(), 1, cudaMemcpyDeviceToHost));
        const uint8_t first_corrupted = first_original ^ 1;
        check(first_corrupted != first_original, "first-byte fixture changes the byte");
        gpu(cudaMemcpy(first_alias.base(), &first_corrupted, 1, cudaMemcpyHostToDevice));
        gpu(cudaDeviceSynchronize());
        require(first_alias.release(), "release first-page alias");
        const auto pages_before_first = BranchPages::counters();
        check(!session.begin(4097, 1, error), "first-page corruption rejects prepare");
        const auto pages_after_first = BranchPages::counters();
        check(pages_after_first.cow_pages == pages_before_first.cow_pages,
              "first-byte rejection performs no COW copies");
        check(pages_after_first.copied_bytes == pages_before_first.copied_bytes,
              "first-byte rejection performs no copied-byte mutation");
        check(pages_after_first.mappings == pages_before_first.mappings &&
              pages_after_first.unmappings == pages_before_first.unmappings,
              "first-byte rejection changes no VMM mappings");
        check(pages_after_first.last_integrity_mismatch_tile == 0,
              "first-page corruption reports the first tile index");
        check(session.ids().size() == 4097, "first-byte rejection consumes no token");
        VmmMapping restore_alias;
        require(restore_alias.reserve(vmm_granularity()), "reserve restore alias");
        require(restore_alias.map(0, before_first_reject.front().driver_handle, true), "map first page to restore");
        gpu(cudaMemcpy(restore_alias.base(), &first_original, 1, cudaMemcpyHostToDevice));
        gpu(cudaDeviceSynchronize());
        require(restore_alias.release(), "release first-page restore alias");
        require(session.restore(root, error), "restore root after first-byte corruption fixture");

        // Corrupt the final pool after restoring the root. It must fail before
        // any pool is detached, copied, or advanced to another token.
        const auto before_restore = branch_digest_counters();
        require(session.restore(root, error), "restore root before corruption");
        const auto after_restore = branch_digest_counters();
        check(after_restore.digest_copy_bytes > before_restore.digest_copy_bytes,
              "snapshot restore retains full CPU-visible digest verification");
        const auto restored = session.ownership();
        require(!restored.empty(), "root has physical pages");
        VmmMapping alias;
        require(alias.reserve(vmm_granularity()), "reserve alias");
        require(alias.map(0, restored.back().driver_handle, true), "map late pool writable");
        uint8_t previous_byte = 0;
        gpu(cudaMemcpy(&previous_byte, alias.base() + vmm_granularity() - 1, 1, cudaMemcpyDeviceToHost));
        const uint8_t corrupted_byte = previous_byte ^ 1;
        check(corrupted_byte != previous_byte, "corruption fixture changes the sealed final byte");
        gpu(cudaMemcpy(alias.base() + vmm_granularity() - 1, &corrupted_byte, 1, cudaMemcpyHostToDevice));
        gpu(cudaDeviceSynchronize());
        require(alias.release(), "release alias");
        const auto pages_before_reject = BranchPages::counters();
        const auto digest_before_reject = branch_digest_counters();
        check(!session.begin(4097, 1, error), "late-pool corruption rejects prepare");
        const auto pages_after_reject = BranchPages::counters();
        const auto digest_after_reject = branch_digest_counters();
        check(pages_after_reject.cow_pages == pages_before_reject.cow_pages,
              "rejection performs no COW copies");
        check(pages_after_reject.copied_bytes == pages_before_reject.copied_bytes,
              "rejection performs no copied-byte mutation");
        check(pages_after_reject.mappings == pages_before_reject.mappings &&
              pages_after_reject.unmappings == pages_before_reject.unmappings,
              "rejection changes no VMM mappings");
        check(digest_after_reject.device_hashed_bytes - digest_before_reject.device_hashed_bytes == sealed_bytes,
              "corrupt attempt also hashes every pool before rejecting");
        check(digest_after_reject.compare_result_copy_bytes - digest_before_reject.compare_result_copy_bytes ==
              sizeof(uint64_t), "corrupt attempt returns only the scalar mismatch index");
        const auto after_reject = session.ownership();
        check(after_reject.size() == restored.size(), "ownership count remains stable after rejection");
        for (size_t i = 0; i < restored.size() && i < after_reject.size(); ++i)
            check(after_reject[i].allocation_id == restored[i].allocation_id,
                  "rejection preserves every physical page identity");
        check(pages_after_reject.last_integrity_mismatch_tile == sealed_bytes / branch_digest_tile_bytes - 1,
              "late final-page corruption reports its final tile index");
        check(session.ids().size() == 4097, "rejection consumes no token");
    }
    const auto page_after = BranchPages::counters();
    const auto digest_after = branch_digest_counters();
    check(page_after.live_bytes == page_baseline.live_bytes, "physical pages released");
    check(page_after.allocations - page_baseline.allocations ==
          page_after.releases - page_baseline.releases, "physical page lifetime balanced");
    check(page_after.seal_live_bytes == page_baseline.seal_live_bytes,
          "GPU seal mirrors released with their physical pages");
    check(page_after.seal_allocations - page_baseline.seal_allocations ==
          page_after.seal_releases - page_baseline.seal_releases,
          "GPU seal mirror allocations and releases balance");
    check(digest_after.compare_workspace_live_bytes == digest_baseline.compare_workspace_live_bytes,
          "comparison pointer and flag workspace released");
    gpu(cudaFreeHost(qsa.host_step)); gpu(cudaFreeHost(qsa.host_pos));
    gpu(cudaFree(state.gdn_state)); gpu(cudaFree(state.ple_hist)); gpu(cudaFree(arena));
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"model_forward\":false}\n",
                failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
