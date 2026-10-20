// GPU byte-audit regression. Pattern data exercises real CUDA storage; no model forward.
#include "strata/core/branch_session.hpp"
#include "branch_test_extent.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

using namespace strata::core;
static int checks = 0;
static void check(bool ok, const char* text) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", text); std::exit(1); }
}
static void gpu(cudaError_t result) { check(result == cudaSuccess, cudaGetErrorString(result)); }
static std::vector<char> read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    check(bool(file), "read actual audit file");
    return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}
static void exercise(int format) {
    ModelGeometry g;
    g.n_layers = 4;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    qsa_set_kv_int8(format == 1); qsa_set_kv_q4(format == 2); qsa_set_kv_hybrid(format == 3);
    qsa_set_kv_elastic(false, 0); qsa_set_kv_resident(0);
    SessionState state;
    QsaState q;
    state.max_cells = 8192; state.layer_hi = 4;
    state.gdn_alloc = 3; state.qsa_alloc = 1; state.qsa_states = &q;
    void* arena = nullptr;
    gpu(cudaMalloc(&arena, qsa_state_bytes(g, state.max_cells)));
    check(qsa_state_init(g, state.max_cells, arena, q) != 0, "actual QSA layout");
    qsa_state_zero(q, g, nullptr);
    std::string error;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g, sizes, error), "running sizes");
    gpu(cudaMalloc(&state.gdn_state, sizes.gdn));
    gpu(cudaMalloc(&state.ple_hist, sizes.ple));
    gpu(cudaMemset(state.gdn_state, 19, sizes.gdn));
    gpu(cudaMemset(state.ple_hist, 23, sizes.ple));
    const auto baseline = BranchPages::counters();
    const auto path = std::filesystem::temp_directory_path() /
        ("strata-audit-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        BranchSession session(state, g, "audit-fixture");
        check(session.bind(error), "bind actual CUDA pages");
        BranchSessionAudit invalid;
        check(!session.audit({}, path.string(), invalid, error), "missing image refused before file creation");
        check(!std::filesystem::exists(path), "invalid audit creates no file");
        check(session.begin(0, 2049, error), "begin partial physical page fixture");
        const auto pools = session.pools();
        const auto mapped = branch_test_mapped_bytes_by_pool(session);
        check(mapped.size() == pools.size(), "mapped extent inventory matches pools");
        uint64_t total_capacity = 0, total_mapped = 0;
        for (size_t i = 0; i < pools.size(); ++i) {
            const auto rounded_capacity = branch_test_reserved_capacity(pools[i]);
            check(mapped[i] > 0 && mapped[i] <= rounded_capacity,
                  "audit fixture mapping is a positive prefix within rounded reserved capacity");
            check(rounded_capacity <= std::numeric_limits<uint64_t>::max() - total_capacity,
                  "audit fixture rounded capacity sum does not overflow");
            total_capacity += rounded_capacity;
            total_mapped += mapped[i];
            gpu(cudaMemset(pools[i].address, 37, mapped[i]));
        }
        check(total_mapped < total_capacity,
              "short audit fixture keeps virtual capacity larger than mapped physical extent");
        gpu(cudaMemset(q.idx_block_pos, 0, sizeof(int32_t)));
        check(session.finish(std::vector<int32_t>(2049, 42), {1.0f, 2.0f}, error), "consume prefix");
        auto root = session.freeze(error);
        check(bool(root), "retain immutable root");
        const auto address = session.pools().front().address;
        const auto before = BranchPages::counters();
        const auto session_before = session.counters();
        BranchSessionAudit first;
        check(session.audit(root, path.string(), first, error), "audit saved pages without activation");
        auto root_bytes = read(path);
        check(first.bytes == root_bytes.size(), "reported bytes equal actual file extent");
        check(first.consumed == 2049 && first.qsa_page_cells == 4, "real consumed and QSA geometry");
        uint64_t offset = 0, gpu_bytes = 0;
        bool full_prefix_page = false, unused_tail_in_mapped_page = false;
        bool gdn = false, frontier = false, ids = false;
        for (const auto& region : first.regions) {
            check(region.offset == offset && region.bytes > 0, "complete contiguous audit coverage");
            offset += region.bytes;
            if (region.page.bytes) {
                check(region.bytes == vmm_granularity(), "whole actual allocation page audited");
                check(region.filled_prefix_bytes <= region.bytes, "prefix coverage never exceeds page");
                full_prefix_page |= region.filled_prefix_bytes == region.bytes;
                unused_tail_in_mapped_page |= region.filled_prefix_bytes < region.bytes;
                gpu_bytes += region.bytes;
                std::vector<char> actual(region.bytes);
                const auto pool = session.pools().at((size_t)region.pool);
                gpu(cudaMemcpy(actual.data(), (char*)pool.address + region.pool_offset,
                               actual.size(), cudaMemcpyDeviceToHost));
                check(std::equal(actual.begin(), actual.end(), root_bytes.begin() + region.offset),
                      "audit is the actual complete GPU bytes, including padding");
            } else if (region.kind == "gdn") {
                gdn = true;
                check(region.bytes == sizes.gdn, "all recurrent and convolution bytes included");
                check(std::vector<char>(root_bytes.begin()+region.offset, root_bytes.begin()+offset) ==
                      std::vector<char>(sizes.gdn, 19), "saved recurrence payload verified independently");
            } else if (region.kind == "logits") frontier = true;
            else if (region.kind == "ids") ids = true;
        }
        check(offset == first.bytes && gdn && frontier && ids, "all exported sections accounted");
        check(unused_tail_in_mapped_page,
              "mapped partial physical page distinguishes consumed prefix from tail padding");
        if (format == 0) check(full_prefix_page, "FP16 fixture spans at least one populated CUDA page");
        check(BranchPages::counters().audit_read_bytes - before.audit_read_bytes == gpu_bytes,
              "audit counter measures successful GPU transfers");
        check(BranchPages::counters().allocations == before.allocations, "audit allocates no GPU pages");
        check(BranchPages::counters().copied_bytes == before.copied_bytes, "audit does not inflate COW bytes");
        check(session.begin(2049, 1, error), "prepare child partial tail");
        for (const auto& pool : session.pools()) {
            const size_t cell = pool.pooled ? 2049 / 4 : 2049;
            gpu(cudaMemset((char*)pool.address + cell * pool.bytes_per_cell, 71, pool.bytes_per_cell));
        }
        gpu(cudaMemset(state.gdn_state, 91, sizes.gdn));
        check(session.finish({43}, {3.0f, 4.0f}, error), "consume child");
        auto child = session.freeze(error);
        check(bool(child), "retain child");
        BranchSessionAudit second;
        check(session.audit(root, path.string(), second, error), "audit parent while child remains active");
        check(read(path) == root_bytes, "all parent page, recurrence, history and frontier bytes unchanged");
        check(session.ids().size() == 2050 && session.ids().back() == 43, "audit never activates parent");
        check(session.logits() == std::vector<float>({3.0f, 4.0f}), "audit never replaces frontier");
        check(session.pools().front().address == address, "audit preserves graph-visible addresses");
        check(session.counters().restores == session_before.restores, "audit performs no hidden restore");
        check(session.audit(child, path.string(), second, error), "audit child as adversarial counterexample");
        check(read(path) != root_bytes, "audit distinguishes an actually modified child");
    }
    std::filesystem::remove(path);
    const auto end = BranchPages::counters();
    check(end.live_bytes == baseline.live_bytes, "audit retains no allocation after teardown");
    check(end.allocations-baseline.allocations == end.releases-baseline.releases, "balanced physical releases");
    check(end.mappings-baseline.mappings == end.unmappings-baseline.unmappings, "balanced audit mappings");
    check(end.address_reservations-baseline.address_reservations == end.address_releases-baseline.address_releases,
          "balanced audit virtual addresses");
    gpu(cudaFreeHost(q.host_step)); gpu(cudaFreeHost(q.host_pos));
    gpu(cudaFree(state.gdn_state)); gpu(cudaFree(state.ple_hist)); gpu(cudaFree(arena));
}
int main() {
    gpu(cudaSetDevice(0));
    for (int format = 0; format < 4; ++format) exercise(format);
    std::printf("{\"status\":\"PASS\",\"checks\":%d,\"formats\":4,\"model_forward\":false}\n", checks);
}
