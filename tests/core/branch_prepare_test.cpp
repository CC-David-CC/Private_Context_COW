// Real CUDA preparation across separate pools. No model or numerical gate claim.
#include "strata/core/branch_session.hpp"
#include "branch_test_extent.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
static int checks = 0, failures = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
static void require(bool ok, const char* message) {
    check(ok, message);
    if (!ok) std::exit(2);
}
static void gpu(cudaError_t result) { require(result == cudaSuccess, cudaGetErrorString(result)); }

static void exercise(int format) {
    ModelGeometry geometry;
    geometry.n_layers = 4;
    geometry.ssm_state_size = 2; geometry.ssm_v_heads = 2; geometry.ssm_conv_channels = 8;
    qsa_set_kv_int8(format == 1); qsa_set_kv_q4(format == 2); qsa_set_kv_hybrid(format == 3);
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
    gpu(cudaMalloc(&state.gdn_state, sizes.gdn)); gpu(cudaMalloc(&state.ple_hist, sizes.ple));
    gpu(cudaMemset(state.gdn_state, 0, sizes.gdn)); gpu(cudaMemset(state.ple_hist, 0, sizes.ple));
    const auto before = BranchPages::counters();
    const auto scratch_before = branch_digest_counters();
    {
        BranchSession session(state, geometry, "batched-preparation-fixture");
        require(session.bind(error), "bind");
        require(session.begin(0, 4097, error), "begin prefix");
        const auto pools = session.pools();
        const auto mapped = branch_test_mapped_bytes_by_pool(session);
        check(mapped.size() == pools.size(), "mapped extent inventory matches pools");
        for (size_t i = 0; i < pools.size(); ++i) {
            check(mapped[i] > 0 && mapped[i] <= branch_test_reserved_capacity(pools[i]),
                  "prepared pool mapping is a positive prefix within rounded reserved capacity");
            gpu(cudaMemset(pools[i].address, 21, mapped[i]));
        }
        require(session.finish(std::vector<int32_t>(4097, 42), {1.0f, 2.0f}, error), "finish prefix");
        auto root = session.freeze(error);
        require(bool(root), "retain root");
        const auto root_pages = session.ownership();
        uint64_t hashed_bytes = 0, launches = 0;
        double elapsed = 0;
        for (int step = 0; step < 9; ++step) {
            const auto pages = session.ownership();
            uint64_t sealed_bytes = 0, sealed_pages = 0;
            for (size_t i = 0; i < pages.size(); ++i) {
                if (pages[i].allocation_id == root_pages[i].allocation_id) {
                    sealed_bytes += pages[i].bytes; ++sealed_pages;
                }
            }
            const auto digest_before = branch_digest_counters();
            const auto page_before = BranchPages::counters();
            const auto start = std::chrono::steady_clock::now();
            require(session.begin(4097 + step, 1, error), "prepare next token");
            elapsed += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto digest_after = branch_digest_counters();
            const auto page_after = BranchPages::counters();
            check(digest_after.kernel_calls - digest_before.kernel_calls == (sealed_pages ? 1 : 0),
                  "one integrity launch across every sealed pool");
            check(digest_after.device_hashed_bytes - digest_before.device_hashed_bytes == sealed_bytes,
                  "every sealed byte checked, including pools outside the tail");
            check(page_after.integrity_checked_pages - page_before.integrity_checked_pages == sealed_pages,
                  "every sealed page compared");
            check(digest_after.workspace_allocations == digest_before.workspace_allocations,
                  "verification scratch prepared before decoding");
            if (step) check(page_after.cow_pages == page_before.cow_pages, "private tail reused without copying");
            hashed_bytes += digest_after.device_hashed_bytes - digest_before.device_hashed_bytes;
            launches += digest_after.kernel_calls - digest_before.kernel_calls;
            for (const auto& pool : session.pools()) {
                const auto cell = pool.pooled ? (4097 + step) / 4 : 4097 + step;
                gpu(cudaMemset(static_cast<uint8_t*>(pool.address) + cell * pool.bytes_per_cell,
                               77, pool.bytes_per_cell));
            }
            require(session.finish({43}, {3.0f, 4.0f}, error), "finish child token");
        }
        auto child = session.freeze(error);
        require(bool(child), "retain child");
        require(session.restore(root, error), "restore root before corruption");
        const auto restored = session.ownership();
        VmmMapping alias;
        require(alias.reserve(vmm_granularity()), "reserve corrupting alias");
        require(alias.map(0, restored.back().driver_handle, true), "map last pool writable");
        gpu(cudaMemset(alias.base() + vmm_granularity() - 1, 22, 1));
        gpu(cudaDeviceSynchronize());
        require(alias.release(), "release alias");
        const auto corruption_before = BranchPages::counters();
        check(!session.begin(4097, 1, error), "late-pool corruption refuses preparation");
        const auto rejected = session.ownership();
        check(BranchPages::counters().cow_pages == corruption_before.cow_pages,
              "all pool checks precede any tail copy");
        for (size_t i = 0; i < rejected.size(); ++i)
            check(rejected[i].allocation_id == restored[i].allocation_id,
                  "rejected preparation leaves all page ownership unchanged");
        check(session.ids().size() == 4097, "failed preparation consumes no token");
        require(session.restore(child, error), "uncorrupted child remains restorable");
        std::printf("{\"format\":%d,\"prepare_seconds\":%.9f,\"hashed_bytes\":%llu,\"launches\":%llu}\n",
                    format, elapsed, (unsigned long long)hashed_bytes, (unsigned long long)launches);
    }
    const auto after = BranchPages::counters();
    const auto scratch_after = branch_digest_counters();
    check(after.live_bytes == before.live_bytes, "all pages released");
    check(scratch_after.workspace_live_bytes == scratch_before.workspace_live_bytes, "all scratch released");
    check(scratch_after.workspace_allocations - scratch_before.workspace_allocations ==
          scratch_after.workspace_releases - scratch_before.workspace_releases, "scratch releases balanced");
    gpu(cudaFreeHost(qsa.host_step)); gpu(cudaFreeHost(qsa.host_pos));
    gpu(cudaFree(state.gdn_state)); gpu(cudaFree(state.ple_hist)); gpu(cudaFree(arena));
}
int main() {
    gpu(cudaSetDevice(0));
    for (int format = 0; format < 4; ++format) exercise(format);
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"model_forward\":false}\n",
                failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
