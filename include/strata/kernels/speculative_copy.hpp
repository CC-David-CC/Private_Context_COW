#pragma once
#include <cstddef>
namespace strata::kernels {
// Independent, non-overlapping device-to-device copies.
struct SpeculativeCopy { void* dst; const void* src; size_t bytes; };
bool speculative_copy(const SpeculativeCopy* device_copies, size_t count, void* stream);
}
