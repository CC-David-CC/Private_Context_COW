#pragma once

// Minimal CUDA-runtime spelling bridge for the existing, HIP-compiled branch
// page and digest sources. Include this before any CUDA runtime include and
// only for the bounded model-free HIP path.
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaFree hipFree
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaStream_t hipStream_t
#define cudaStreamNonBlocking hipStreamNonBlocking

namespace strata::core::detail {
template <class T> inline hipError_t branch_hip_malloc(T** ptr, size_t bytes) {
    return hipMalloc(reinterpret_cast<void**>(ptr), bytes);
}
}
#define cudaMalloc(ptr, bytes) ::strata::core::detail::branch_hip_malloc((ptr), (bytes))
#endif
