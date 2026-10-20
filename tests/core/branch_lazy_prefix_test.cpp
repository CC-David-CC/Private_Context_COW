// CUDA fixture for sparse physical-prefix materialization. This is a native
// unit test only; it does not run a model forward or qualify a backend.
#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace strata::core;

static int checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}
static void gpu(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "FAIL: CUDA line %d `%s`: %s\n", line, expression,
                     cudaGetErrorString(result));
        std::exit(2);
    }
}
#define GPU(expr) gpu((expr), #expr, __LINE__)

static uint64_t mapped_bytes(const std::vector<BranchPageInfo>& pages) {
    uint64_t bytes = 0;
    for (const auto& page : pages) bytes += page.bytes;
    return bytes;
}

static uint64_t required_prefix_bytes(const BranchSession& session, int64_t consumed) {
    const auto shape = strata::kernels::qsa_real_shapes();
    const auto granularity = vmm_granularity();
    const auto kv_cells = ((consumed + shape.page_size - 1) / shape.page_size) * shape.page_size;
    uint64_t total = 0;
    for (const auto& pool : session.pools()) {
        const uint64_t cells = pool.pooled
            ? static_cast<uint64_t>(consumed / shape.idx_block + 1)
            : static_cast<uint64_t>(kv_cells);
        const uint64_t required = cells * pool.bytes_per_cell;
        total += ((required + granularity - 1) / granularity) * granularity;
    }
    return total;
}

static void write_prepared_range(const BranchSession& session, int64_t position,
                                 int64_t count, unsigned char value) {
    const auto shape = strata::kernels::qsa_real_shapes();
    for (const auto& pool : session.pools()) {
        const uint64_t offset = pool.pooled
            ? static_cast<uint64_t>(position / shape.idx_block) * pool.bytes_per_cell
            : static_cast<uint64_t>(position) * pool.bytes_per_cell;
        const uint64_t bytes = pool.pooled
            ? static_cast<uint64_t>((position + count) / shape.idx_block + 1 - position / shape.idx_block) *
                  pool.bytes_per_cell
            : static_cast<uint64_t>(count) * pool.bytes_per_cell;
        GPU(cudaMemset(static_cast<unsigned char*>(pool.address) + offset, value,
                       static_cast<size_t>(bytes)));
    }
}

int main() {
    GPU(cudaSetDevice(0));
    ModelGeometry geometry;
    geometry.n_layers = 4;
    geometry.ssm_state_size = 2;
    geometry.ssm_v_heads = 2;
    geometry.ssm_conv_channels = 8;
    qsa_set_kv_int8(false);
    qsa_set_kv_q4(false);
    qsa_set_kv_hybrid(false);
    qsa_set_kv_elastic(false, 0);
    qsa_set_kv_resident(0);

    // 32K cells spans many VMM granules, while the committed prefix below only
    // needs a small fraction of them. Keep the original arena intact so the
    // test exercises the real session pointer rebinding path.
    constexpr int64_t capacity = 32768;
    constexpr int64_t consumed = 4097;
    constexpr int64_t larger_consumed = 12000;
    SessionState state;
    QsaState qsa;
    state.max_cells = capacity;
    state.layer_hi = 4;
    state.gdn_alloc = 3;
    state.qsa_alloc = 1;
    state.qsa_states = &qsa;
    void* arena = nullptr;
    GPU(cudaMalloc(&arena, qsa_state_bytes(geometry, capacity)));
    check(qsa_state_init(geometry, capacity, arena, qsa) != 0, "initialize full-capacity QSA arena");
    qsa_state_zero(qsa, geometry, nullptr);
    ConversationStateSizes sizes;
    std::string error;
    check(conversation_state_sizes(geometry, sizes, error), "recurrent-state sizes");
    GPU(cudaMalloc(&state.gdn_state, sizes.gdn));
    GPU(cudaMalloc(&state.ple_hist, sizes.ple));
    GPU(cudaMemset(state.gdn_state, 0, sizes.gdn));
    GPU(cudaMemset(state.ple_hist, 0, sizes.ple));
    GPU(cudaDeviceSynchronize());
    void* const original_kv = qsa.k_pool ? static_cast<void*>(qsa.k_pool)
        : qsa.k_q ? static_cast<void*>(qsa.k_q)
        : static_cast<void*>(qsa.k_q4);

    const auto page_baseline = BranchPages::counters();
    const auto digest_baseline = branch_digest_counters();
    {
        BranchSession session(state, geometry, "lazy-prefix-fixture");
        check(session.bind(error), "bind full virtual capacity");
        const auto bound_pools = session.pools();
        const auto after_bind = BranchPages::counters();
        check(after_bind.live_bytes == page_baseline.live_bytes,
              "binding an empty session reserves virtual capacity without physical KV pages");
        check(after_bind.allocations == page_baseline.allocations && after_bind.releases == page_baseline.releases,
              "binding allocates no transient physical KV pages");
        check(session.counters().binding_copy_bytes == 0,
              "empty-session binding copies no unused full-capacity KV bytes");
        check(!bound_pools.empty(), "QSA pools are bound");
        check(bound_pools.front().bytes > 0, "full virtual pool capacity is retained");
        void* const stable_address = bound_pools.front().address;
        check(stable_address != original_kv, "session pointer is rebound into stable virtual mapping");

        check(session.begin(0, consumed, error), "begin first prefix");
        write_prepared_range(session, 0, consumed, 0x35);
        std::vector<int32_t> ids(static_cast<size_t>(consumed), 42);
        check(session.finish(ids, {1.0f, -1.0f}, error), "finish first prefix");
        auto root = session.freeze(error);
        check(bool(root), "freeze first-prefix checkpoint");

        const auto ownership = session.ownership();
        const uint64_t actual_bytes = mapped_bytes(ownership);
        const uint64_t expected_bytes = required_prefix_bytes(session, consumed);
        uint64_t full_capacity_bytes = 0;
        for (const auto& pool : session.pools()) {
            const auto g = vmm_granularity();
            full_capacity_bytes += ((pool.bytes + g - 1) / g) * g;
        }
        std::printf("{\"event\":\"first-prefix-extent\",\"capacity_cells\":%lld,\"consumed\":%lld,\"mapped_bytes\":%llu,\"expected_prefix_bytes\":%llu,\"full_capacity_bytes\":%llu}\n",
                    (long long)capacity, (long long)consumed,
                    (unsigned long long)actual_bytes,
                    (unsigned long long)expected_bytes,
                    (unsigned long long)full_capacity_bytes);
        check(expected_bytes * 2 < full_capacity_bytes,
              "fixture distinguishes prefix extent from full capacity");
        check(actual_bytes == expected_bytes,
              "checkpoint owns only the page-rounded consumed prefix, not max capacity");
        check(session.pools().front().address == stable_address,
              "growing the physical prefix preserves the full virtual address");
        check(session.pools().front().bytes == bound_pools.front().bytes,
              "growing the physical prefix preserves virtual capacity");

        const std::vector<float> first_frontier{1.0f, -1.0f};
        SavedConversation host_image;
        check(session.capture_host(host_image, error), "capture a sparse prefix to host");
        check(session.reset(error), "clear sparse prefix before host restore");
        check(mapped_bytes(session.ownership()) == 0, "reset unmaps sparse physical prefix");
        check(session.restore_host(host_image, ids, first_frontier, error), "restore host prefix into sparse mapping");
        check(mapped_bytes(session.ownership()) == expected_bytes, "host restore maps only the validated prefix");
        check(session.ids() == ids && session.logits() == first_frontier,
              "host restore preserves exact history and frontier");

        const auto disk_path = std::filesystem::temp_directory_path() /
            ("strata-lazy-prefix-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        size_t disk_bytes = 0;
        check(session.write(disk_path.string(), disk_bytes, error), "stream-write sparse prefix");
        check(disk_bytes > 0 && std::filesystem::file_size(disk_path) == disk_bytes,
              "stream-write reports exact saved bytes");
        check(session.reset(error), "clear sparse prefix before streamed restore");
        check(session.load(disk_path.string(), ids, first_frontier, error), "streamed-load sparse prefix");
        std::filesystem::remove(disk_path);
        check(mapped_bytes(session.ownership()) == expected_bytes, "streamed load maps only validated extents");
        check(session.ids() == ids && session.logits() == first_frontier,
              "streamed load preserves exact history and supplied frontier");
        check(session.restore(root, error), "restore sealed root after host/disk round trips");

        // Corrupt the final byte of the last allocated pool page, including
        // page padding beyond the logical consumed cells. A request that would
        // grow several pages must reject before any mapping or COW mutation.
        const auto root_pages = session.ownership();
        check(!root_pages.empty(), "root has allocated prefix pages");
        VmmMapping tail_alias;
        check(tail_alias.reserve(vmm_granularity()), "reserve partial-tail corruption alias");
        check(tail_alias.map(0, root_pages.back().driver_handle, true), "map last pool page for corruption");
        uint8_t tail_original = 0;
        GPU(cudaMemcpy(&tail_original, tail_alias.base() + vmm_granularity() - 1, 1, cudaMemcpyDeviceToHost));
        const uint8_t tail_corrupted = tail_original ^ 1;
        GPU(cudaMemcpy(tail_alias.base() + vmm_granularity() - 1, &tail_corrupted, 1, cudaMemcpyHostToDevice));
        GPU(cudaDeviceSynchronize());
        check(tail_alias.release(), "release partial-tail corruption alias");
        const auto before_growth_reject = BranchPages::counters();
        const auto before_growth_hash = branch_digest_counters();
        check(!session.begin(consumed, larger_consumed - consumed, error),
              "corrupt last-pool partial-tail page rejects before prefix growth");
        const auto after_growth_reject = BranchPages::counters();
        const auto after_growth_hash = branch_digest_counters();
        const uint64_t root_extent_bytes = mapped_bytes(root_pages);
        check(after_growth_hash.device_hashed_bytes - before_growth_hash.device_hashed_bytes == root_extent_bytes,
              "rejected growth freshly hashes every allocated byte across all pools");
        check(after_growth_reject.integrity_mismatches == before_growth_reject.integrity_mismatches + 1,
              "last-pool partial-tail corruption is reported as an integrity mismatch");
        check(after_growth_reject.last_integrity_mismatch_tile ==
                  root_extent_bytes / branch_digest_tile_bytes - 1,
              "last-pool partial-tail corruption reports final sealed tile index");
        check(after_growth_reject.allocations == before_growth_reject.allocations &&
              after_growth_reject.mappings == before_growth_reject.mappings &&
              after_growth_reject.unmappings == before_growth_reject.unmappings &&
              after_growth_reject.cow_pages == before_growth_reject.cow_pages &&
              after_growth_reject.copied_bytes == before_growth_reject.copied_bytes,
              "corruption rejection changes no mappings, allocations, or tail ownership");
        check(session.ids().size() == static_cast<size_t>(consumed),
              "corruption rejection consumes no additional token");
        VmmMapping restore_alias;
        check(restore_alias.reserve(vmm_granularity()), "reserve alias to restore test byte");
        check(restore_alias.map(0, root_pages.back().driver_handle, true), "map last page to restore test byte");
        GPU(cudaMemcpy(restore_alias.base() + vmm_granularity() - 1, &tail_original, 1, cudaMemcpyHostToDevice));
        GPU(cudaDeviceSynchronize());
        check(restore_alias.release(), "release alias after restoring test byte");
        check(session.restore(root, error), "recover the committed root after rejected growth");

        // The optimization may leave future pages unmapped, but every sealed
        // page that exists must still receive a fresh full-byte hash.
        const auto digest_before = branch_digest_counters();
        check(session.begin(consumed, 1, error), "begin append after captured prefix");
        const auto digest_after = branch_digest_counters();
        check(digest_after.device_hashed_bytes - digest_before.device_hashed_bytes == actual_bytes,
              "append freshly hashes every byte of every extant sealed page");
        check(session.pools().front().address == stable_address,
              "append preserves stable virtual address");
        check(session.finish({43}, {2.0f, -2.0f}, error), "finish appended token");
        auto small_child = session.freeze(error);
        check(bool(small_child), "freeze small child");
        check(session.restore(root, error), "restore small root before larger growth");

        const int64_t extension = larger_consumed - consumed;
        check(session.begin(consumed, extension, error), "grow past several physical-page boundaries");
        write_prepared_range(session, consumed, extension, 0x67);
        check(session.finish(std::vector<int32_t>(static_cast<size_t>(extension), 44),
                             {3.0f, -3.0f}, error), "finish larger child");
        auto larger = session.freeze(error);
        check(bool(larger), "freeze larger child");
        const uint64_t larger_bytes = mapped_bytes(session.ownership());
        const uint64_t expected_larger_bytes = required_prefix_bytes(session, larger_consumed);
        check(larger_bytes == expected_larger_bytes,
              "larger checkpoint owns the page-rounded larger prefix only");
        check(larger_bytes > actual_bytes, "larger history grows physical extent");
        check(session.pools().front().address == stable_address,
              "growing past page boundaries preserves virtual address");

        SavedConversation large_host_image;
        const auto large_ids = session.ids();
        const std::vector<float> large_frontier{3.0f, -3.0f};
        check(session.capture_host(large_host_image, error), "capture larger sparse prefix to host");
        check(session.reset(error), "clear larger sparse prefix before host restore");
        check(session.restore_host(large_host_image, large_ids, large_frontier, error),
              "restore larger host prefix into sparse mapping");
        check(mapped_bytes(session.ownership()) == expected_larger_bytes,
              "larger host restore maps the larger validated extent");
        check(session.ids() == large_ids && session.logits() == large_frontier,
              "larger host restore preserves exact history and frontier");
        check(session.restore(root, error), "restore small sealed root after larger host restore");

        check(session.restore(root, error), "restore first-prefix checkpoint");
        check(mapped_bytes(session.ownership()) == expected_bytes,
              "restoring smaller image shrinks to its saved physical extent");
        check(session.restore(larger, error), "restore larger image after smaller image");
        check(mapped_bytes(session.ownership()) == expected_larger_bytes,
              "restoring larger image regrows its saved physical extent");
        check(session.pools().front().address == stable_address,
              "restore preserves stable virtual address");
        check(session.reset(error), "reset session while retaining saved prefix");
        check(mapped_bytes(session.ownership()) == 0,
              "reset releases physical prefix pages but keeps virtual reservation");
        check(session.pools().front().address == stable_address,
              "reset preserves stable virtual address");
        check(session.pools().front().bytes == bound_pools.front().bytes,
              "reset retains full future virtual capacity");
    }
    check((qsa.k_pool ? static_cast<void*>(qsa.k_pool)
                      : qsa.k_q ? static_cast<void*>(qsa.k_q)
                                : static_cast<void*>(qsa.k_q4)) == original_kv,
          "session teardown restores borrowed model pointers");
    const auto page_end = BranchPages::counters();
    const auto digest_end = branch_digest_counters();
    check(page_end.live_bytes == page_baseline.live_bytes, "all derived physical pages released");
    check(digest_end.device_hashed_bytes > digest_baseline.device_hashed_bytes,
          "fixture exercised real fresh device hashing");
    GPU(cudaFreeHost(qsa.host_step));
    GPU(cudaFreeHost(qsa.host_pos));
    GPU(cudaFree(state.gdn_state));
    GPU(cudaFree(state.ple_hist));
    GPU(cudaFree(arena));
    std::printf("{\"status\":\"PASS\",\"checks\":%d,\"model_forward\":false,\"qualification\":\"native-unit-only\"}\n", checks);
}
