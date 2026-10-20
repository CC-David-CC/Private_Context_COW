#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>
#include <cstdint>

namespace strata::core::conversation_detail {

// Thread-local execution witness: completed CUDA copies, not planned tensor sizes.
// Callers bracket a restore to measure its transfers independently of capture.
struct CopyCounters { uint64_t bytes = 0, calls = 0; };
inline thread_local CopyCounters completed_copies;

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    cudaStream_t s = streams[(size_t) dev];
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) { completed_copies.bytes += bytes; ++completed_copies.calls; }
    return e;
}

} // namespace strata::core::conversation_detail
