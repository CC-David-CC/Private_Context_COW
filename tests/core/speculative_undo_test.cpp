// Real CUDA storage: rejected speculative rows must restore their preimage.
#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/speculative_undo.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace strata::core;
static void require(bool ok, const char* message) { if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); } }
static void gpu(cudaError_t e) { require(e==cudaSuccess,cudaGetErrorString(e)); }
static unsigned char byte_at(void* p) { unsigned char x=0; gpu(cudaMemcpy(&x,p,1,cudaMemcpyDeviceToHost)); return x; }
static void exercise(int first, int keep) {
    ModelGeometry g; g.n_layers=4; g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    qsa_set_kv_int8(false); qsa_set_kv_q4(false); qsa_set_kv_hybrid(false);
    qsa_set_kv_elastic(false,0); qsa_set_kv_resident(0);
    SessionState state; QsaState q; state.max_cells=64; state.layer_hi=4; state.gdn_alloc=3; state.qsa_alloc=1; state.qsa_states=&q;
    void* arena=nullptr; gpu(cudaMalloc(&arena,qsa_state_bytes(g,state.max_cells)));
    require(qsa_state_init(g,state.max_cells,arena,q)!=0,"initialize QSA"); qsa_state_zero(q,g,nullptr);
    ConversationStateSizes sizes; std::string error;
    require(conversation_state_sizes(g,sizes,error),error.c_str());
    gpu(cudaMalloc(&state.gdn_state,sizes.gdn)); gpu(cudaMalloc(&state.ple_hist,sizes.ple));
    gpu(cudaMemset(state.gdn_state,0,sizes.gdn)); gpu(cudaMemset(state.ple_hist,0,sizes.ple));
    {
        BranchSession session(state,g,"speculative-undo-fixture");
        require(session.bind(error),error.c_str());
        require(session.begin(0,first,error),error.c_str());
        require(session.finish(std::vector<int32_t>(first,1),{1.0f},error),error.c_str());
        require(session.begin(first,7,error),error.c_str());
        const auto shape=strata::kernels::qsa_real_shapes();
        auto rowptr=[&](uint16_t* pool,int t,int h) {
            const int64_t row=((t/shape.page_size)*g.n_head_kv+h)*shape.page_size+t%shape.page_size;
            return pool+row*g.head_dim;
        };
        for (int t=first;t<first+7;++t) for(int h=0;h<g.n_head_kv;++h) {
            gpu(cudaMemset(rowptr(q.k_pool,t,h),17,g.head_dim*sizeof(uint16_t)));
            gpu(cudaMemset(rowptr(q.v_pool,t,h),17,g.head_dim*sizeof(uint16_t)));
        }
        const int lo=first/shape.idx_block, hi=(first+7)/shape.idx_block;
        for(int i=lo;i<=hi;++i) gpu(cudaMemset(q.idx_pooled+i*g.idx_key_dim,17,g.idx_key_dim*sizeof(float)));
        gpu(cudaMemset(q.idx_block_pos,17,sizeof(int32_t)));
        if (first == 1 && keep == 0) {
            SpeculativeUndo invalid;
            require(!invalid.capture(state,g,0,7,error),"reject first position zero");
            require(!invalid.capture(state,g,first,0,error),"reject empty window");
            require(!invalid.capture(state,g,first,9,error),"reject oversized window");
            require(!invalid.capture(state,g,state.max_cells,1,error),"reject window beyond capacity");
            require(!invalid.capture_candidates(error),"reject candidates before capture");
            require(!invalid.reapply_candidates(error),"reject reapply before capture");
            require(!invalid.restore_suffix(0,error),"reject restore before capture");
        }
        SpeculativeUndo undo;
        require(undo.capture(state,g,first,7,error),error.c_str());
        require(!undo.capture(state,g,first,7,error),"reject duplicate capture");
        require(!undo.reapply_candidates(error),"reject reapply before candidate capture");
        for (int t=first;t<first+7;++t) for(int h=0;h<g.n_head_kv;++h) {
            gpu(cudaMemset(rowptr(q.k_pool,t,h),33,g.head_dim*sizeof(uint16_t)));
            gpu(cudaMemset(rowptr(q.v_pool,t,h),33,g.head_dim*sizeof(uint16_t)));
        }
        for(int i=lo;i<=hi;++i) gpu(cudaMemset(q.idx_pooled+i*g.idx_key_dim,33,g.idx_key_dim*sizeof(float)));
        gpu(cudaMemset(q.idx_block_pos,33,sizeof(int32_t)));
        require(undo.capture_candidates(error),error.c_str());
        require(!undo.capture_candidates(error),"reject duplicate candidate capture");
        const int final=(first+keep)/shape.idx_block;
        const bool no_complete=lo==final;
        if (!no_complete) {
            for(int i=lo;i<=final;++i) gpu(cudaMemset(q.idx_pooled+i*g.idx_key_dim,44,g.idx_key_dim*sizeof(float)));
            gpu(cudaMemset(q.idx_block_pos,44,sizeof(int32_t)));
        }
        require(undo.restore_suffix(keep,error),error.c_str());
        for(int t=first;t<first+7;++t) for(int h=0;h<g.n_head_kv;++h) {
            const unsigned char expected=t<first+keep?33:17;
            require(byte_at(rowptr(q.k_pool,t,h))==expected,"key row restore");
            require(byte_at(rowptr(q.v_pool,t,h))==expected,"value row restore");
        }
        for(int i=lo;i<=hi;++i) require(byte_at(q.idx_pooled+i*g.idx_key_dim)==(!no_complete && i<=final?44:17),"index row restore");
        require(byte_at(q.idx_block_pos)==(no_complete?17:44),"index position restore");
        require(undo.reapply_candidates(error),error.c_str());
        for (int t=first;t<first+7;++t) for(int h=0;h<g.n_head_kv;++h) {
            require(byte_at(rowptr(q.k_pool,t,h))==33,"reapply key row");
            require(byte_at(rowptr(q.v_pool,t,h))==33,"reapply value row");
        }
        for(int i=lo;i<=hi;++i) require(byte_at(q.idx_pooled+i*g.idx_key_dim)==33,"reapply index row");
        require(byte_at(q.idx_block_pos)==33,"reapply index position");
        require(undo.restore_suffix(0,error),error.c_str());
        for (int t=first;t<first+7;++t) for(int h=0;h<g.n_head_kv;++h)
            require(byte_at(rowptr(q.k_pool,t,h))==17 && byte_at(rowptr(q.v_pool,t,h))==17,"cancel whole window");
        require(!undo.restore_suffix(8,error),"reject invalid accepted count");
        if (keep > 0) {
            require(undo.reapply_candidates(error),error.c_str());
            if (!no_complete) {
                for(int i=lo;i<=final;++i) gpu(cudaMemset(q.idx_pooled+i*g.idx_key_dim,44,g.idx_key_dim*sizeof(float)));
                gpu(cudaMemset(q.idx_block_pos,44,sizeof(int32_t)));
            }
            require(undo.restore_suffix(keep,error),error.c_str());
            require(session.finish_speculative_prefix(std::vector<int32_t>(keep,4),{2.0f},error),error.c_str());
            auto image=session.freeze(error); require(bool(image),error.c_str());
            if (keep < 7) {
                require(session.begin_speculative_replay(first+keep,first,7,error),error.c_str());
                require(undo.reapply_candidates(error),error.c_str());
                require(undo.restore_suffix(keep+1,error),error.c_str());
                require(session.finish({4},{3.0f},error),error.c_str());
                require(session.restore(image,error),error.c_str());
                for(int t=first;t<first+7;++t) for(int h=0;h<g.n_head_kv;++h) {
                    const unsigned char expected=t<first+keep?33:17;
                    require(byte_at(rowptr(q.k_pool,t,h))==expected,"replay kept snapshot keys");
                    require(byte_at(rowptr(q.v_pool,t,h))==expected,"replay kept snapshot values");
                }
            }
        } else {
            require(!session.finish_speculative_prefix({}, {2.0f}, error),"reject empty speculative prefix");
        }

    }
    cudaFree(state.ple_hist); cudaFree(state.gdn_state); cudaFree(arena);
}
int main() { for(int pos=1;pos<=8;++pos) for(int keep=0;keep<=7;++keep) exercise(pos,keep); std::puts("speculative undo: 64 cases passed"); }
