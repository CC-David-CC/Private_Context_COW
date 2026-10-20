// Explicitly test the caller-managed promise, including its loss of detection.
// This fixture is NOT a registered corruption-detection acceptance test.
#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
static int checks=0,failures=0;
static void check(bool ok,const char* message) {
    ++checks; if (!ok) { ++failures; std::fprintf(stderr,"FAIL: %s\n",message); }
}
static void require(bool ok,const char* message) { check(ok,message); if (!ok) std::exit(2); }
static void gpu(cudaError_t result) { require(result==cudaSuccess,cudaGetErrorString(result)); }
static void exercise(int format) {
    ModelGeometry g; g.n_layers=4; g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    qsa_set_kv_int8(format==1); qsa_set_kv_q4(format==2); qsa_set_kv_hybrid(format==3);
    qsa_set_kv_elastic(false,0); qsa_set_kv_resident(0);
    SessionState ss; QsaState q;
    ss.max_cells=8192; ss.layer_hi=4; ss.gdn_alloc=3; ss.qsa_alloc=1; ss.qsa_states=&q;
    void* arena=nullptr;
    gpu(cudaMalloc(&arena,qsa_state_bytes(g,ss.max_cells)));
    require(qsa_state_init(g,ss.max_cells,arena,q)!=0,"initialize QSA"); qsa_state_zero(q,g,nullptr);
    ConversationStateSizes sizes; std::string error;
    require(conversation_state_sizes(g,sizes,error),"running-state sizes");
    gpu(cudaMalloc(&ss.gdn_state,sizes.gdn)); gpu(cudaMalloc(&ss.ple_hist,sizes.ple));
    gpu(cudaMemset(ss.gdn_state,0,sizes.gdn)); gpu(cudaMemset(ss.ple_hist,0,sizes.ple));
    const auto baseline=BranchPages::counters();
    const auto scratch_baseline=branch_digest_counters();
    {
        BranchSession session(ss,g,"external-integrity-fixture"); require(session.bind(error),"bind");
        require(session.begin(0,4097,error),"prefix begin");
        for (const auto& p:session.pools()) gpu(cudaMemset(p.address,21,p.bytes));
        require(session.finish(std::vector<int32_t>(4097,42),{1.f,2.f},error),"prefix finish");
        auto root=session.freeze(error); require(bool(root),"freeze still seals bytes");
        const auto owned=session.ownership();
        auto before=branch_digest_counters();
        require(session.begin(4097,1,error),"prepare private tails");
        auto after=branch_digest_counters();
        check(after.kernel_calls==before.kernel_calls,"caller-managed write performs no hash launch");
        check(after.device_hashed_bytes==before.device_hashed_bytes,"caller-managed write reads no sealed bytes for hashing");
        check(BranchPages::counters().cow_pages>baseline.cow_pages,"real shared tails are copied");
        for (const auto& p:session.pools()) {
            const auto cell=p.pooled?4097/4:4097;
            gpu(cudaMemset(static_cast<uint8_t*>(p.address)+cell*p.bytes_per_cell,77,p.bytes_per_cell));
        }
        require(session.finish({43},{3.f,4.f},error),"child finish");
        auto child=session.freeze(error); require(bool(child),"child freeze keeps full integrity checks");
        require(session.restore(root,error),"root restore remains byte-checked");
        const auto restored=session.ownership();
        for (size_t i=0;i<owned.size();++i) check(restored[i].allocation_id==owned[i].allocation_id,"ordered root handles restored");
        std::vector<uint8_t> bytes(vmm_granularity());
        gpu(cudaMemcpy(bytes.data(),session.pools().front().address,bytes.size(),cudaMemcpyDeviceToHost));
        check(bytes==std::vector<uint8_t>(bytes.size(),21),"parent's first page unchanged by child writes");
        const auto copies=BranchPages::counters().cow_pages;
        check(!session.begin(4096,1,error),"wrong position refused before preparation");
        check(!session.begin(4097,ss.max_cells,error),"capacity overflow refused before preparation");
        check(BranchPages::counters().cow_pages==copies,"invalid requests do not change ownership");
        VmmMapping alias; require(alias.reserve(vmm_granularity()),"alias reserve");
        require(alias.map(0,restored.front().driver_handle,true),"intentional external promise violation");
        gpu(cudaMemset(alias.base(),22,1)); gpu(cudaDeviceSynchronize()); require(alias.release(),"alias release");
        before=branch_digest_counters();
        const bool accepted=session.begin(4097,1,error);
        check(accepted,"external policy does not detect a changed sealed byte before write");
        after=branch_digest_counters();
        check(after.kernel_calls==before.kernel_calls,"promise-violation case also has no hash launch");
        // An independent observer sees the violation. No claim of guard equivalence.
        uint8_t changed=0;
        gpu(cudaMemcpy(&changed,session.pools().front().address,1,cudaMemcpyDeviceToHost));
        check(changed==22,"external byte audit observes deliberate corruption");
        std::printf("{\"format\":%d,\"external_promise_violation_detected_before_write\":%s}\n",format,accepted?"false":"true");
    }
    check(BranchPages::counters().live_bytes==baseline.live_bytes,"pages released");
    check(branch_digest_counters().workspace_live_bytes==scratch_baseline.workspace_live_bytes,"scratch released");
    gpu(cudaFreeHost(q.host_step)); gpu(cudaFreeHost(q.host_pos));
    gpu(cudaFree(ss.gdn_state)); gpu(cudaFree(ss.ple_hist)); gpu(cudaFree(arena));
}
int main() {
    gpu(cudaSetDevice(0)); for (int format=0;format<4;++format) exercise(format);
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"contract\":\"caller-managed-write-integrity\",\"model_forward\":false}\n",
                failures?"FAIL":"PASS",checks,failures);
    return failures?1:0;
}
