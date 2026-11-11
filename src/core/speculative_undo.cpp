#include "strata/core/speculative_undo.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <new>

namespace strata::core {
namespace {
bool fail(std::string& error, const char* message) { error=message; return false; }
}
SpeculativeUndo::~SpeculativeUndo() { if (candidates_) cudaFree(candidates_); if (saved_) cudaFree(saved_); }

bool SpeculativeUndo::capture(SessionState& state, const ModelGeometry& g, int64_t first, int count, std::string& error) {
    if (saved_ || !slots_.empty() || first < 1 || count < 1 || count > 8 || first > state.max_cells ||
        count > state.max_cells-first || !state.qsa_states || state.qsa_alloc <= 0 ||
        state.qsa_ord0 < 0 || state.qsa_ord0 + state.qsa_alloc > g.n_qsa_layers() ||
        g.n_head_kv <= 0 || g.head_dim <= 0 || g.idx_key_dim <= 0)
        return fail(error,"invalid speculative window or QSA geometry");
    const auto shape=strata::kernels::qsa_real_shapes();
    if (shape.page_size <= 0 || shape.idx_block <= 0) return fail(error,"invalid QSA page geometry");
    first_=first; count_=count;
    try {
        auto add=[&](void* target, size_t len, int64_t at, bool pooled) {
            if (!target || len > std::numeric_limits<size_t>::max()-bytes_) throw std::bad_alloc();
            slots_.push_back({target,bytes_,len,at,pooled}); bytes_+=len;
        };
        for (int64_t z=0; z<state.qsa_alloc; ++z) {
            auto& q=state.qsa_states[state.qsa_ord0+z];
            if (q.kv_int8 || q.kv_q4 || q.kv_hybrid || q.kv_elastic >= 0 ||
                q.n_pages != q.n_slots || !q.k_pool || !q.v_pool || !q.idx_pooled || !q.idx_block_pos ||
                (first+count)/shape.idx_block >= q.idx_pooled_rows)
                return fail(error,"speculative undo needs fully resident FP16 QSA pools");
            for (int64_t t=first; t<first+count; ++t) for (int64_t h=0; h<g.n_head_kv; ++h) {
                const int64_t row=((t/shape.page_size)*g.n_head_kv+h)*shape.page_size+t%shape.page_size;
                add(q.k_pool+row*g.head_dim,(size_t)g.head_dim*sizeof(uint16_t),t,false);
                add(q.v_pool+row*g.head_dim,(size_t)g.head_dim*sizeof(uint16_t),t,false);
            }
            for (int64_t row=first/shape.idx_block; row<=(first+count)/shape.idx_block; ++row)
                add(q.idx_pooled+row*g.idx_key_dim,(size_t)g.idx_key_dim*sizeof(float),row,true);
            add(q.idx_block_pos,sizeof(int32_t),-1,true);
        }
    } catch (...) { return fail(error,"cannot allocate speculative undo metadata"); }
    if (cudaDeviceSynchronize()!=cudaSuccess || cudaMalloc((void**)&saved_,bytes_)!=cudaSuccess)
        return fail(error,"cannot allocate speculative undo on the GPU");
    for (const auto& slot:slots_)
        if (cudaMemcpy(saved_+slot.offset,slot.target,slot.bytes,cudaMemcpyDeviceToDevice)!=cudaSuccess)
            return fail(error,"cannot save speculative preimage");
    if (cudaDeviceSynchronize()!=cudaSuccess) return fail(error,"cannot synchronize speculative preimage");
    return true;
}

bool SpeculativeUndo::capture_candidates(std::string& error) {
    if (!saved_ || candidates_) return fail(error,"invalid speculative candidate capture");
    if (cudaDeviceSynchronize()!=cudaSuccess || cudaMalloc((void**)&candidates_,bytes_)!=cudaSuccess)
        return fail(error,"cannot allocate speculative candidates on the GPU");
    for (const auto& slot:slots_)
        if (cudaMemcpy(candidates_+slot.offset,slot.target,slot.bytes,cudaMemcpyDeviceToDevice)!=cudaSuccess)
            return fail(error,"cannot save speculative candidates");
    if (cudaDeviceSynchronize()!=cudaSuccess) return fail(error,"cannot synchronize speculative candidates");
    return true;
}

bool SpeculativeUndo::reapply_candidates(std::string& error) {
    if (!candidates_) return fail(error,"missing speculative candidates");
    if (cudaDeviceSynchronize()!=cudaSuccess) return fail(error,"cannot synchronize speculative reapply");
    for (const auto& slot:slots_)
        if (cudaMemcpy(slot.target,candidates_+slot.offset,slot.bytes,cudaMemcpyDeviceToDevice)!=cudaSuccess)
            return fail(error,"cannot reapply speculative candidates");
    if (cudaDeviceSynchronize()!=cudaSuccess) return fail(error,"cannot synchronize speculative candidates");
    return true;
}

bool SpeculativeUndo::restore_suffix(int accepted, std::string& error) {
    if (!saved_ || accepted < 0 || accepted > count_) return fail(error,"invalid speculative accepted prefix");
    if (cudaDeviceSynchronize()!=cudaSuccess) return fail(error,"cannot synchronize target commit");
    const auto block=strata::kernels::qsa_real_shapes().idx_block;
    const int64_t consumed=first_+accepted;
    const bool no_complete=first_/block == consumed/block;
    for (const auto& slot:slots_) {
        const bool restore=slot.pooled ? (slot.position == -1 ? no_complete :
            (no_complete ? slot.position >= first_/block : slot.position > consumed/block)) : slot.position >= consumed;
        if (restore && cudaMemcpy(slot.target,saved_+slot.offset,slot.bytes,cudaMemcpyDeviceToDevice)!=cudaSuccess)
            return fail(error,"cannot restore rejected speculative state");
    }
    if (cudaDeviceSynchronize()!=cudaSuccess) return fail(error,"cannot synchronize rejected-state restore");
    return true;
}
}
