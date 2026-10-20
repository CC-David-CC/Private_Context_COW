// Same per-tile integrity contract, with fewer launches for adjacent sealed pages.
#include "strata/core/branch_pages.hpp"
#include "strata/core/branch_digest.hpp"
#include <cuda_runtime.h>
#include <cstdio>

using namespace strata::core;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !vmm_available()) return 2;
    const auto g = vmm_granularity();
    BranchPages pages;
    CHECK(pages.reserve(6*g));
    CHECK(pages.grow(6*g));
    CHECK(cudaMemset(pages.data(), 17, 6*g) == cudaSuccess);
    auto root = pages.freeze();
    CHECK(root != nullptr);
    auto before = branch_digest_counters();
    CHECK(pages.prepare_write(0, 0));
    auto after = branch_digest_counters();
    CHECK(after.device_hashed_bytes-before.device_hashed_bytes == 6*g);
    CHECK(after.kernel_calls-before.kernel_calls == 1);
    CHECK(after.workspace_allocations == before.workspace_allocations);
    CHECK(pages.prepare_write(2*g+3, 1));
    CHECK(cudaMemset(pages.data()+2*g+3, 19, 1) == cudaSuccess);
    before = branch_digest_counters();
    CHECK(pages.prepare_write(0, 0));
    after = branch_digest_counters();
    CHECK(after.device_hashed_bytes-before.device_hashed_bytes == 5*g);
    CHECK(after.kernel_calls-before.kernel_calls == 2);
    CHECK(after.workspace_allocations == before.workspace_allocations);
    // Mutate a shared page through a separate writable mapping. Every sealed
    // page must still be checked, including the far end of the second run.
    VmmMapping alias;
    const auto owned = pages.ownership();
    CHECK(alias.reserve(g));
    CHECK(alias.map(0, owned[5].driver_handle, true));
    CHECK(cudaMemset(alias.base()+g-1, 20, 1) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(!pages.prepare_write(0, 0));
    CHECK(!pages.freeze());
    CHECK(alias.release());
    {
        BranchPages long_prefix;
        CHECK(long_prefix.reserve(21*g));
        CHECK(long_prefix.grow(21*g));
        auto saved = long_prefix.freeze();
        CHECK(saved != nullptr);
        before = branch_digest_counters();
        CHECK(long_prefix.prepare_write(0, 0));
        after = branch_digest_counters();
        CHECK(after.device_hashed_bytes-before.device_hashed_bytes == 21*g);
        CHECK(after.kernel_calls-before.kernel_calls == 2);
        CHECK(after.workspace_allocations == before.workspace_allocations);
    }
    std::printf("{\"status\":\"%s\",\"failures\":%d}\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
