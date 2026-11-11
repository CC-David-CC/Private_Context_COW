// DFlash2 tap must match the completed residual stream used by PCC.
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/gr.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc((void**)&p, n * sizeof(float))); }
    ~Buffer() { cudaFree(p); }
};
int main() try {
    constexpr int rows = 3, n = 2560, hc = 4;
    std::vector<float> residual(rows*hc*n), output(rows*n), inject(rows*hc);
    for (size_t i=0; i<residual.size(); ++i) residual[i] = 0.2f*std::sin(float(i)*0.019f);
    for (size_t i=0; i<output.size(); ++i) output[i] = 0.3f*std::cos(float(i)*0.013f);
    for (size_t i=0; i<inject.size(); ++i) inject[i] = float(int(i%7)-3)*0.4f;
    Buffer r(residual.size()), o(output.size()), g(inject.size()), scratch(hc*n), fused(rows*n), ref(rows*n);
    ck(cudaMemcpy(r.p,residual.data(),residual.size()*sizeof(float),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(o.p,output.data(),output.size()*sizeof(float),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(g.p,inject.data(),inject.size()*sizeof(float),cudaMemcpyHostToDevice));
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    strata::kernels::df2_tap_gr(r.p,hc*n,o.p,n,g.p,hc,fused.p,n,rows,n,hc,stream);
    const strata::kernels::GrShapes shape{n,hc,1};
    for (int i=0; i<rows; ++i) {
        strata::kernels::gr_write(r.p+i*hc*n,o.p+i*n,g.p+i*hc,shape,scratch.p,stream);
        strata::kernels::df2_tap_rows(scratch.p,ref.p+i*n,n,n,hc,1,stream);
    }
    ck(cudaStreamSynchronize(stream));
    std::vector<float> a(rows*n),b(rows*n);
    ck(cudaMemcpy(a.data(),fused.p,a.size()*sizeof(float),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(),ref.p,b.size()*sizeof(float),cudaMemcpyDeviceToHost));
    ck(cudaStreamDestroy(stream));
    float max_err=0;
    for(size_t i=0; i<a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite tap");
        max_err=std::max(max_err,std::abs(a[i]-b[i]));
        if (std::abs(a[i]-b[i]) > 1e-5f + 1e-5f*std::abs(b[i])) throw std::runtime_error("tap differs from completed residual");
    }
    std::printf("DFlash2 tap matches completed residual, max error %.9g\n",max_err);
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"DFlash2 tap test: %s\n",e.what()); return 1; }
