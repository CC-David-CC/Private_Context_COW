#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/qsa.hpp"
#include "branch_test_extent.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;

namespace {
struct DeviceRange { uintptr_t base; uint64_t bytes; };
std::vector<DeviceRange> g_targets;
std::atomic<bool> g_armed{false};
std::atomic<unsigned> g_matching_calls{0};
std::atomic<unsigned> g_completed_calls{0};
std::atomic<unsigned> g_injected_failures{0};
unsigned g_checks = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
void gpu(cudaError_t error, const char* what) {
    ++g_checks;
    if (error != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(error));
        std::exit(1);
    }
}

std::vector<unsigned char> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(bool(input), "open audit bytes");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct TempDirectory {
    std::filesystem::path path;
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("strata-branch-apply-failure-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};
} // namespace

extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);

// Only fail copies whose destination belongs to this test's BranchSession K/V pools.
// The first matching copy is performed and synchronized for real; the next one fails.
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source, size_t bytes,
                                                cudaMemcpyKind kind, cudaStream_t stream) {
    if (g_armed.load(std::memory_order_relaxed) && bytes && kind == cudaMemcpyDefault) {
        const uintptr_t dst = reinterpret_cast<uintptr_t>(destination);
        bool is_target = false;
        for (const auto& range : g_targets) {
            if (dst >= range.base && dst - range.base < range.bytes &&
                bytes <= range.bytes - (dst - range.base)) {
                is_target = true;
                break;
            }
        }
        if (is_target) {
            const unsigned call = g_matching_calls.fetch_add(1, std::memory_order_relaxed);
            if (call == 0) {
                const cudaError_t copied = __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
                if (copied != cudaSuccess) return copied;
                const cudaError_t ordered = cudaStreamSynchronize(stream);
                if (ordered == cudaSuccess) g_completed_calls.fetch_add(1, std::memory_order_relaxed);
                return ordered;
            }
            g_injected_failures.fetch_add(1, std::memory_order_relaxed);
            return cudaErrorUnknown;
        }
    }
    return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

int main() {
    gpu(cudaSetDevice(0), "select test device");
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

    SessionState state;
    QsaState qsa;
    state.max_cells = 8192;
    state.layer_hi = 4;
    state.gdn_alloc = 3;
    state.qsa_alloc = 1;
    state.qsa_states = &qsa;
    void* arena = nullptr;
    gpu(cudaMalloc(&arena, qsa_state_bytes(geometry, state.max_cells)), "allocate QSA arena");
    check(qsa_state_init(geometry, state.max_cells, arena, qsa) != 0, "initialize QSA state");
    qsa_state_zero(qsa, geometry, nullptr);

    ConversationStateSizes sizes;
    std::string error;
    check(conversation_state_sizes(geometry, sizes, error), "compute running state sizes");
    gpu(cudaMalloc(&state.gdn_state, sizes.gdn), "allocate GDN state");
    gpu(cudaMalloc(&state.ple_hist, sizes.ple), "allocate PLE history");
    gpu(cudaMemset(state.gdn_state, 0, sizes.gdn), "initialize GDN state");
    gpu(cudaMemset(state.ple_hist, 0, sizes.ple), "initialize PLE history");

    TempDirectory temporary;
    const auto checkpoint = temporary.path / "sibling.session";
    const auto parent_before = temporary.path / "parent.before";
    const auto parent_after = temporary.path / "parent.after";
    const auto sibling_before = temporary.path / "sibling.before";
    const auto sibling_after = temporary.path / "sibling.after";
    const auto baseline = BranchPages::counters();
    {
        BranchSession session(state, geometry, "post-apply-failure-test");
        check(session.bind(error), "bind session");

        check(session.begin(0, 4097, error), "begin parent prefix");
        const auto pools = session.pools();
        check(!pools.empty(), "bound session has K/V pools");
        const auto parent_mapped = branch_test_mapped_bytes_by_pool(session);
        check(parent_mapped.size() == pools.size(), "parent mapped extent inventory matches pools");
        for (size_t i = 0; i < pools.size(); ++i) {
            check(parent_mapped[i] > 0 && parent_mapped[i] <= branch_test_reserved_capacity(pools[i]),
                  "parent pool mapping is a positive prefix within rounded reserved capacity");
            gpu(cudaMemset(pools[i].address, 0x21, parent_mapped[i]), "fill parent pools");
        }
        gpu(cudaMemset(qsa.idx_dead, 0x31, sizes.dead), "fill parent index spare row");
        gpu(cudaMemset(qsa.idx_tail, 0x32, sizes.tail), "fill parent index tail");
        gpu(cudaMemset(qsa.idx_block_pos, 0, sizeof(int32_t)), "initialize parent block position");
        std::vector<int32_t> parent_ids(4097, 41);
        check(session.finish(parent_ids, {1.0f, 2.0f}, error), "finish parent prefix");
        const auto parent = session.freeze(error);
        check(bool(parent), "publish parent snapshot");

        check(session.restore(parent, error), "restore parent for sibling");
        check(session.begin(4097, 1, error), "begin sibling suffix");
        for (const auto& pool : session.pools()) {
            const size_t at = pool.pooled ? (4097 / strata::kernels::qsa_real_shapes().idx_block) * pool.bytes_per_cell
                                           : 4097 * pool.bytes_per_cell;
            gpu(cudaMemset(static_cast<unsigned char*>(pool.address) + at, 0x72,
                           pool.bytes_per_cell), "write sibling K/V suffix");
        }
        gpu(cudaMemset(state.gdn_state, 0x73, sizes.gdn), "write sibling GDN state");
        gpu(cudaMemset(state.ple_hist, 0x74, sizes.ple), "write sibling PLE history");
        std::vector<int32_t> sibling_ids = parent_ids;
        sibling_ids.push_back(42);
        check(session.finish({42}, {3.0f, 4.0f}, error), "finish sibling suffix");
        const auto sibling = session.freeze(error);
        check(bool(sibling), "publish sibling snapshot");

        size_t checkpoint_bytes = 0;
        check(session.write(checkpoint.string(), checkpoint_bytes, error), "write sibling cold checkpoint");
        check(checkpoint_bytes == std::filesystem::file_size(checkpoint), "checkpoint size receipt matches file");
        check(session.load(checkpoint.string(), sibling_ids, {3.0f, 4.0f}, error),
              "unarmed positive control loads the same checkpoint");
        check(session.ids() == sibling_ids && session.logits() == std::vector<float>({3.0f, 4.0f}),
              "positive control restores expected history and frontier");
        BranchSessionAudit parent_audit, sibling_audit;
        check(session.audit(parent, parent_before.string(), parent_audit, error), "audit parent before failure");
        check(session.audit(sibling, sibling_before.string(), sibling_audit, error), "audit sibling before failure");
        check(parent_audit.bytes > 0 && sibling_audit.bytes > 0, "audits cover saved physical and controller bytes");

        g_targets.clear();
        const auto load_pools = session.pools();
        const auto load_mapped = branch_test_mapped_bytes_by_pool(session);
        check(load_mapped.size() == load_pools.size(), "load mapped extent inventory matches pools");
        for (size_t i = 0; i < load_pools.size(); ++i) {
            check(load_mapped[i] > 0 && load_mapped[i] <= branch_test_reserved_capacity(load_pools[i]),
                  "load pool mapping is a positive prefix within rounded reserved capacity");
            g_targets.push_back({reinterpret_cast<uintptr_t>(load_pools[i].address), load_mapped[i]});
        }
        g_matching_calls.store(0, std::memory_order_relaxed);
        g_completed_calls.store(0, std::memory_order_relaxed);
        g_injected_failures.store(0, std::memory_order_relaxed);
        g_armed.store(true, std::memory_order_release);
        const bool loaded = session.load(checkpoint.string(), sibling_ids, {3.0f, 4.0f}, error);
        g_armed.store(false, std::memory_order_release);

        check(!loaded, "injected post-apply copy failure rejects cold load");
        check(g_matching_calls.load() == 2, "failure injected on second K/V pool copy");
        check(g_completed_calls.load() == 1, "first actual K/V copy completed and synchronized");
        check(g_injected_failures.load() == 1, "exactly one targeted copy was failed");
        check(error.find("conversation snapshot copy: ") == 0,
              "load reports the targeted device-copy error, not an unrelated validation refusal");
        check(!session.freeze(error), "failed load cannot publish the partially applied session");
        check(!session.begin(4098, 1, error), "failed load cannot begin another write transaction");

        BranchSessionAudit parent_after_audit, sibling_after_audit;
        check(session.audit(parent, parent_after.string(), parent_after_audit, error), "audit parent after failure");
        check(session.audit(sibling, sibling_after.string(), sibling_after_audit, error), "audit sibling after failure");
        check(parent_audit.bytes == parent_after_audit.bytes && read_file(parent_before) == read_file(parent_after),
              "parent full page/controller bytes survive failed apply");
        check(sibling_audit.bytes == sibling_after_audit.bytes && read_file(sibling_before) == read_file(sibling_after),
              "sibling full page/controller bytes survive failed apply");
    }
    g_targets.clear();
    check(BranchPages::counters().live_bytes == baseline.live_bytes, "release all test-owned GPU pages");
    gpu(cudaFreeHost(qsa.host_step), "free host step");
    gpu(cudaFreeHost(qsa.host_pos), "free host positions");
    gpu(cudaFree(state.gdn_state), "free GDN state");
    gpu(cudaFree(state.ple_hist), "free PLE history");
    gpu(cudaFree(arena), "free QSA arena");
    std::printf("{\"status\":\"PASS\",\"checks\":%u,\"apply_copy_successes\":1,\"injected_apply_failures\":1,\"model_forward\":false}\n",
                g_checks);
    return 0;
}
