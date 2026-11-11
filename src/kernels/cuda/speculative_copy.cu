#include "strata/kernels/speculative_copy.hpp"
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::kernels {
namespace {
__global__ void copy_kernel(const SpeculativeCopy* copies, size_t count) {
    const size_t index=static_cast<size_t>(blockIdx.x)*4+threadIdx.x/32;
    if (index>=count) return;
    const auto item=copies[index];
    auto* dst=static_cast<unsigned char*>(item.dst);
    const auto* src=static_cast<const unsigned char*>(item.src);
    for (size_t i=threadIdx.x%32;i<item.bytes;i+=32) dst[i]=src[i];
}
}
bool speculative_copy(const SpeculativeCopy* copies, size_t count, void* stream) {
    if (!count) return true;
    copy_kernel<<<static_cast<unsigned>((count+3)/4),128,0,static_cast<cudaStream_t>(stream)>>>(copies,count);
    return cudaGetLastError()==cudaSuccess;
}
}
