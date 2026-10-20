#pragma once

#include "strata/core/branch_session.hpp"
#include "strata/core/vmm.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Test-only view of the currently mapped physical extent for each reserved
// BranchSession pool. Pool bytes are virtual capacity; ownership is the mapped
// prefix that a fixture may safely read or write.
inline std::vector<uint64_t> branch_test_mapped_bytes_by_pool(
    const strata::core::BranchSession& session) {
    const auto pools = session.pools();
    std::vector<uint64_t> result(pools.size(), 0);
    for (const auto& page : session.ownership()) {
        if (page.pool < 0 || static_cast<size_t>(page.pool) >= result.size())
            throw std::logic_error("BranchSession ownership has an invalid pool index");
        auto& bytes = result[static_cast<size_t>(page.pool)];
        if (page.bytes > std::numeric_limits<uint64_t>::max() - bytes)
            throw std::overflow_error("BranchSession mapped extent byte count overflow");
        bytes += page.bytes;
    }
    return result;
}

inline uint64_t branch_test_reserved_capacity(const strata::core::BranchSessionPool& pool) {
    const uint64_t granularity = strata::core::vmm_granularity();
    if (!granularity) throw std::logic_error("VMM granularity is unavailable");
    const uint64_t pages = pool.bytes / granularity + (pool.bytes % granularity != 0);
    if (pages > std::numeric_limits<uint64_t>::max() / granularity)
        throw std::overflow_error("rounded BranchSession capacity overflow");
    return pages * granularity;
}
