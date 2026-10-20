#include "fixture_data.hpp"
#include "strata/core/branch_pages.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using strata::core::BranchPages;
namespace {
int failures = 0;
uint64_t check_count = 0, passed_checks = 0;
#define CHECK(x, label) do { const bool check_ok=bool(x); ++check_count; if(check_ok) ++passed_checks; else ++failures; std::printf("{\"type\":\"check\",\"id\":\"%s\",\"ok\":%s,\"line\":%d}\n", label, check_ok?"true":"false", __LINE__); } while (0)
#define REQUIRE(x, label, code) do { if (!(x)) { CHECK(false,label); return code; } } while (0)
#define REQUIRE_STAGE(x, label) do { if (!(x)) { CHECK(false,label); return 1; } } while (0)
__global__ void fill_fixture(uint8_t* p, uint64_t n, uint64_t seed, uint32_t region) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = cow_fixture::byte_at(seed, region, 0, 0, 0, i);
}
__global__ void overwrite(uint8_t* p, uint64_t begin, uint64_t n, uint8_t value) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[begin+i] = value;
}
__global__ void make_stage(float* p, uint64_t n, uint64_t seed, int stage) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<n) p[i]=float((i*17+seed%997+stage*31)%1009)*0.03125f;
}
__global__ void stage_transform(const float* in, float* out, uint64_t n, float scale, float bias) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<n) { volatile const float product=in[i]*scale; out[i]=product+bias; }
}
__global__ void reset_stage(float* p,uint64_t n) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) p[i]=0.0f;
}
__global__ void make_canary(uint8_t* p,uint64_t n,uint64_t seed,int device) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) p[i]=cow_fixture::canary_byte(seed,device,0,0,0,i);
}
struct StageSlot {
    int device;
    BranchPages* pool;
    strata::core::BranchPageSnapshot snapshot;
    uint64_t page_bytes;
};
bool restore_stage_group(const std::vector<StageSlot>& slots, size_t expected_count) {
    if (expected_count<2 || expected_count>3 || slots.size()!=expected_count) return false;
    // Validate the whole group against physical-page ownership metadata before
    // changing either active mapping. This catches missing or cross-device state.
    BranchPages probes[3];
    for (size_t i=0;i<slots.size();++i) {
        const auto& slot=slots[i];
        if (slot.device!=int(i) || !slot.pool || !slot.snapshot || !slot.page_bytes) return false;
        const auto pages=BranchPages::ownership(slot.snapshot);
        if (pages.size()!=1 || pages.front().device!=slot.device || pages.front().bytes!=slot.page_bytes) return false;
        if (cudaSetDevice(slot.device)!=cudaSuccess || strata::core::vmm_granularity()!=slot.page_bytes || !slot.pool->data()) return false;
        // Restore each saved image into a temporary lease first. This invokes
        // the production integrity verifier for every stage while leaving the
        // active group untouched if any image is missing or damaged.
        if (!probes[i].restore(slot.snapshot)) return false;
    }
    for (size_t i=0;i<slots.size();++i)
        if (cudaSetDevice(int(i))!=cudaSuccess || !probes[i].release()) return false;
    for (const auto& slot:slots) {
        if (cudaSetDevice(slot.device)!=cudaSuccess || !slot.pool->restore(slot.snapshot)) return false;
    }
    return true;
}
bool pull(const uint8_t* device, std::vector<uint8_t>& host) {
    return cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
}

int run_stage_case(uint64_t seed, int stage_count) {
    const int failures_before=failures;
    const uint64_t checks_before=check_count, passed_before=passed_checks;
    const auto stage_baseline=BranchPages::counters();
    constexpr uint64_t n=65536; constexpr float scale=0.75f, bias=-1.25f;
    uint64_t stage_cow_pages=0,stage_copied_bytes=0;
    bool stage_group_restored=false;
    bool group_preflight_before_restore=false,missing_stage_rejected=false;
    std::array<BranchPages,3> stages;
    std::array<uint8_t*,3> canaries{{nullptr,nullptr,nullptr}};
    std::array<uint8_t*,3> stage_ptrs{{nullptr,nullptr,nullptr}};
    std::array<uint64_t,3> grans{{0,0,0}};
    std::array<strata::core::BranchPageSnapshot,3> snapshots;
    const size_t bytes=n*sizeof(float);
    for (int device=0;device<stage_count;++device) {
        REQUIRE_STAGE(cudaSetDevice(device)==cudaSuccess,"stage_select_device");
        grans[device]=strata::core::vmm_granularity();
        REQUIRE_STAGE(grans[device]>=bytes && stages[device].reserve(grans[device]) && stages[device].grow(grans[device]),"stage_branchpages_alloc");
        stage_ptrs[device]=stages[device].data();
        REQUIRE_STAGE(stage_ptrs[device]!=nullptr,"stage_owned_device_pointer");
        REQUIRE_STAGE(cudaMalloc(&canaries[device],4096)==cudaSuccess,"stage_canary_alloc");
        make_canary<<<16,256>>>(canaries[device],4096,seed,device);
        REQUIRE_STAGE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"stage_canary_gpu_init");
    }
    int peer=0; REQUIRE_STAGE(cudaDeviceCanAccessPeer(&peer,1,0)==cudaSuccess && peer,"required_peer_access_0_1");
    REQUIRE_STAGE(cudaSetDevice(1)==cudaSuccess,"peer_enable_device");
    const cudaError_t enabled=cudaDeviceEnablePeerAccess(0,0);
    REQUIRE_STAGE(enabled==cudaSuccess || enabled==cudaErrorPeerAccessAlreadyEnabled,"peer_enable");
    if (enabled==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
    REQUIRE_STAGE(cudaSetDevice(0)==cudaSuccess,"stage_seed_device");
    make_stage<<<(n+255)/256,256>>>(reinterpret_cast<float*>(stage_ptrs[0]),n,seed,0);
    REQUIRE_STAGE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"stage0_gpu_init_sync_before_handoff");
    REQUIRE_STAGE(cudaSetDevice(1)==cudaSuccess,"stage1_transfer_device");
    REQUIRE_STAGE(cudaMemcpyPeerAsync(stage_ptrs[1],1,stage_ptrs[0],0,bytes)==cudaSuccess,"stage_0_to_1_peer_transfer");
    stage_transform<<<(n+255)/256,256>>>(reinterpret_cast<float*>(stage_ptrs[1]),reinterpret_cast<float*>(stage_ptrs[1]),n,scale,bias);
    REQUIRE_STAGE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"stage1_gpu_math");
    std::vector<float> expected(n), expected_prev(n), got(n);
    for(uint64_t i=0;i<n;++i) expected_prev[i]=float((i*17+seed%997)%1009)*0.03125f;
    std::vector<std::string> routes{"GENERATED", "CUDA_PEER_0_TO_1"};
    if (stage_count==3) {
        // Force the mixed topology's third hop through explicitly pinned host staging.
        // No CUDA peer-copy call is made for 1 -> 2, even if that edge is available.
        void* pinned=nullptr;
        REQUIRE_STAGE(cudaHostAlloc(&pinned,bytes,cudaHostAllocPortable)==cudaSuccess,"stage2_pinned_staging_alloc");
        REQUIRE_STAGE(cudaSetDevice(1)==cudaSuccess && cudaMemcpyAsync(pinned,stage_ptrs[1],bytes,cudaMemcpyDeviceToHost)==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"stage1_to_pinned_sync");
        REQUIRE_STAGE(cudaSetDevice(2)==cudaSuccess && cudaMemcpyAsync(stage_ptrs[2],pinned,bytes,cudaMemcpyHostToDevice)==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"pinned_to_stage2_sync");
        REQUIRE_STAGE(cudaFreeHost(pinned)==cudaSuccess,"stage2_pinned_staging_free");
        stage_transform<<<(n+255)/256,256>>>(reinterpret_cast<float*>(stage_ptrs[2]),reinterpret_cast<float*>(stage_ptrs[2]),n,scale,bias);
        REQUIRE_STAGE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"stage2_gpu_math");
        routes.push_back("HOST_PINNED_STAGING_1_TO_2");
    }
    std::vector<std::vector<float>> expected_stages(stage_count,std::vector<float>(n));
    expected_stages[0]=expected_prev;
    for(int stage=1;stage<stage_count;++stage) {
        for(uint64_t i=0;i<n;++i) { volatile const float product=expected_stages[stage-1][i]*scale; expected_stages[stage][i]=product+bias; }
        REQUIRE_STAGE(cudaSetDevice(stage)==cudaSuccess && cudaMemcpy(got.data(),stage_ptrs[stage],bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"stage_readback");
        CHECK(got==expected_stages[stage],"stage_independent_cpu_arithmetic_match");
    }
    REQUIRE_STAGE(cudaSetDevice(0)==cudaSuccess && cudaMemcpy(got.data(),stage_ptrs[0],bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"stage0_readback");
    CHECK(got==expected_stages[0],"stage0_independent_cpu_arithmetic_match");
    for(int stage=0;stage<stage_count;++stage)
        std::printf("{\"type\":\"stage_frame\",\"id\":\"stage_%d\",\"device\":%d,\"elements\":%llu,\"bytes\":%llu,\"route\":\"%s\"}\n",
            stage,stage,(unsigned long long)n,(unsigned long long)bytes,stage==0?"GPU_GENERATOR":stage==1?"CUDA_PEER_0_TO_1":"HOST_PINNED_STAGING_1_TO_2");
    std::vector<StageSlot> slots;
    for(int stage=0;stage<stage_count;++stage) {
        REQUIRE_STAGE(cudaSetDevice(stage)==cudaSuccess,"capture_stage_device");
        snapshots[stage]=stages[stage].freeze();
        REQUIRE_STAGE(bool(snapshots[stage]),"capture_stage_snapshot");
        slots.push_back({stage,&stages[stage],snapshots[stage],grans[stage]});
    }
    const auto cow_before=BranchPages::counters();
    for(int stage=0;stage<stage_count;++stage) {
        REQUIRE_STAGE(cudaSetDevice(stage)==cudaSuccess && stages[stage].prepare_write(0,bytes),"stage_prepare_write");
        reset_stage<<<(n+255)/256,256>>>(reinterpret_cast<float*>(stage_ptrs[stage]),n);
        REQUIRE_STAGE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"stage_mutation");
    }
    const auto cow_after_mutation=BranchPages::counters();
    stage_cow_pages=cow_after_mutation.cow_pages-cow_before.cow_pages;
    stage_copied_bytes=cow_after_mutation.copied_bytes-cow_before.copied_bytes;
    CHECK(stage_cow_pages==uint64_t(stage_count),"stage_group_cow_pages");
    uint64_t expected_copied=0; for(int i=0;i<stage_count;++i) expected_copied+=grans[i];
    CHECK(stage_copied_bytes==expected_copied,"stage_group_copied_bytes");
    std::vector<StageSlot> missing(slots.begin(),slots.end()-1);
    missing_stage_rejected=!restore_stage_group(missing,size_t(stage_count));
    CHECK(missing_stage_rejected,"missing_stage_rejected");
    CHECK(BranchPages::counters().cow_pages==cow_after_mutation.cow_pages,"missing_stage_no_side_effect");
    const std::vector<float> zero(n,0.0f);
    for(int stage=0;stage<stage_count;++stage) {
        REQUIRE_STAGE(cudaSetDevice(stage)==cudaSuccess && cudaMemcpy(got.data(),stage_ptrs[stage],bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"missing_stage_readback");
        CHECK(got==zero,"missing_stage_keeps_active_mutation");
    }
    stage_group_restored=restore_stage_group(slots,size_t(stage_count));
    REQUIRE_STAGE(stage_group_restored,"restore_complete_stage_group");
    group_preflight_before_restore=true;
    for(int stage=0;stage<stage_count;++stage) {
        REQUIRE_STAGE(cudaSetDevice(stage)==cudaSuccess && cudaMemcpy(got.data(),stage_ptrs[stage],bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"restored_stage_readback");
        CHECK(got==expected_stages[stage],"restored_stage_cpu_match");
        std::vector<uint8_t> canary_bytes(4096);
        REQUIRE_STAGE(cudaMemcpy(canary_bytes.data(),canaries[stage],4096,cudaMemcpyDeviceToHost)==cudaSuccess,"stage_canary_readback");
        for(uint64_t i=0;i<4096;++i) if(canary_bytes[i]!=cow_fixture::canary_byte(seed,stage,0,0,0,i)) { CHECK(false,"stage_canary_unchanged"); break; }
    }
    for(int stage=stage_count-1;stage>=0;--stage) {
        REQUIRE_STAGE(cudaSetDevice(stage)==cudaSuccess,"stage_cleanup_device");
        if(canaries[stage]) CHECK(cudaFree(canaries[stage])==cudaSuccess,"stage_canary_free");
        CHECK(stages[stage].release(),"stage_branchpages_release");
    }
    slots.clear();
    missing.clear();
    for(int stage=0;stage<stage_count;++stage) snapshots[stage].reset();
    const auto stage_after=BranchPages::counters();
    CHECK(stage_after.live_bytes==stage_baseline.live_bytes,"stage_group_live_bytes_cleanup");
    CHECK(stage_after.allocations-stage_baseline.allocations==stage_after.releases-stage_baseline.releases,"stage_group_alloc_cleanup");
    CHECK(stage_after.mappings-stage_baseline.mappings==stage_after.unmappings-stage_baseline.unmappings,"stage_group_mapping_cleanup");
    CHECK(stage_after.integrity_mismatches==stage_baseline.integrity_mismatches,"stage_group_integrity");
    std::printf("{\"type\":\"result\",\"id\":\"%s\",\"status\":\"%s\",\"stage_count\":%d,\"checks\":%llu,\"passed_checks\":%llu,\"peer_0_to_1\":%s,\"peer_0_to_1_bytes\":%llu,\"route_1_to_2\":\"%s\",\"host_staged_bytes\":%llu,\"group_preflight_before_restore\":%s,\"group_restored\":%s,\"missing_stage_rejected_without_partial_restore\":%s,\"stage_group_cow_pages\":%llu,\"stage_group_copied_bytes\":%llu,\"allocations\":%llu,\"releases\":%llu,\"mappings\":%llu,\"unmappings\":%llu,\"integrity_checked_pages\":%llu,\"integrity_mismatches\":%llu,\"live_bytes_after\":%llu}\n",
        stage_count==2?"stage_branchpages_group_2card":"stage_branchpages_group_3card_mixed",failures==failures_before?"PASS":"FAIL",stage_count,
        (unsigned long long)(check_count-checks_before),(unsigned long long)(passed_checks-passed_before),peer?"true":"false",(unsigned long long)bytes,
        stage_count==3?"HOST_PINNED_STAGING":"NONE",(unsigned long long)(stage_count==3?bytes:0),group_preflight_before_restore?"true":"false",stage_group_restored?"true":"false",missing_stage_rejected?"true":"false",
        (unsigned long long)stage_cow_pages,(unsigned long long)stage_copied_bytes,
        (unsigned long long)(stage_after.allocations-stage_baseline.allocations),(unsigned long long)(stage_after.releases-stage_baseline.releases),
        (unsigned long long)(stage_after.mappings-stage_baseline.mappings),(unsigned long long)(stage_after.unmappings-stage_baseline.unmappings),
        (unsigned long long)(stage_after.integrity_checked_pages-stage_baseline.integrity_checked_pages),(unsigned long long)(stage_after.integrity_mismatches-stage_baseline.integrity_mismatches),(unsigned long long)stage_after.live_bytes);
    return failures==failures_before?0:1;
}
int run_split_stages(uint64_t seed) {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess) {
        std::printf("{\"type\":\"result\",\"id\":\"stage_branchpages_group_2card\",\"status\":\"UNRUN\",\"reason\":\"cuda_device_count_failed\"}\n");
        std::printf("{\"type\":\"result\",\"id\":\"stage_branchpages_group_3card_mixed\",\"status\":\"UNRUN\",\"reason\":\"cuda_device_count_failed\"}\n");
        return 0;
    }
    if(count<2) {
        std::printf("{\"type\":\"result\",\"id\":\"stage_branchpages_group_2card\",\"status\":\"UNRUN\",\"reason\":\"fewer_than_two_CUDA_devices\"}\n");
        std::printf("{\"type\":\"result\",\"id\":\"stage_branchpages_group_3card_mixed\",\"status\":\"UNRUN\",\"reason\":\"fewer_than_three_CUDA_devices\"}\n");
        return 0;
    }
    const int two_card=run_stage_case(seed,2);
    if(count<3) {
        std::printf("{\"type\":\"result\",\"id\":\"stage_branchpages_group_3card_mixed\",\"status\":\"UNRUN\",\"reason\":\"fewer_than_three_CUDA_devices\"}\n");
        return two_card;
    }
    const int mixed_three_card=run_stage_case(seed,3);
    return two_card||mixed_three_card;
}
bool check_fixture(const std::vector<uint8_t>& bytes, const std::vector<uint8_t>& reference,
                  uint64_t changed_begin=UINT64_MAX, uint64_t changed_size=0, uint8_t changed_value=0) {
    if (bytes.size()!=reference.size()) return false;
    for (uint64_t i=0; i<bytes.size(); ++i) {
        const uint8_t want = i>=changed_begin && i-changed_begin<changed_size ? changed_value : reference[i];
        if (bytes[i] != want) { std::fprintf(stderr,"{\"type\":\"mismatch\",\"offset\":%llu,\"expected\":%u,\"actual\":%u}\n",
            (unsigned long long)i, (unsigned)want, (unsigned)bytes[i]); return false; }
    }
    return true;
}
}

int main(int argc, char** argv) {
    uint64_t seed=0x4d465f434f575f31ULL, logical_pages=8;
    if (argc>1) seed=std::strtoull(argv[1],nullptr,0);
    if (argc>2) logical_pages=std::strtoull(argv[2],nullptr,0);
    if (argc!=4) { std::fprintf(stderr,"usage: seeded_cow_test SEED LOGICAL_PAGES INDEPENDENT_CPU_EXPECTED_FILE\n"); return 2; }
    if (!logical_pages || logical_pages>8192) { std::fprintf(stderr,"invalid logical_pages (expected 1..8192)\n"); return 2; }
    if (cudaFree(nullptr)!=cudaSuccess || !strata::core::vmm_available()) {
        std::fprintf(stderr,"{\"type\":\"result\",\"id\":\"seeded_cow\",\"status\":\"BLOCKED\",\"reason\":\"CUDA_VMM_unavailable\"}\n"); return 2;
    }
    const uint64_t logical_bytes=logical_pages*cow_fixture::logical_page_bytes;
    const uint64_t gran=strata::core::vmm_granularity();
    const uint64_t physical_bytes=std::max<uint64_t>(2*gran,((logical_bytes+gran-1)/gran)*gran);
    std::vector<uint8_t> expected(physical_bytes);
    std::ifstream oracle(argv[3],std::ios::binary);
    if(!oracle || !oracle.read(reinterpret_cast<char*>(expected.data()),static_cast<std::streamsize>(expected.size())) || oracle.peek()!=EOF) {
        std::fprintf(stderr,"independent CPU oracle size/read failure\n"); return 2;
    }
    const auto baseline=BranchPages::counters();
    const uint64_t cow_checks_before=check_count,cow_passed_before=passed_checks;
    {
        BranchPages root, sibling;
        REQUIRE(root.reserve(physical_bytes),"reserve_root",3);
        REQUIRE(root.grow(physical_bytes),"grow_root",3);
        fill_fixture<<<unsigned((physical_bytes+255)/256),256>>>(root.data(),physical_bytes,seed,cow_fixture::region_canonical);
        REQUIRE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"gpu_fixture_fill",4);
        std::vector<uint8_t> root_out(physical_bytes), sibling_out(physical_bytes);
        CHECK(pull(root.data(),root_out) && root_out==expected,"gpu_against_cpu_fixture");
        auto frozen=root.freeze();
        REQUIRE(bool(frozen),"freeze_root",5);
        REQUIRE(sibling.restore(frozen),"restore_sibling",5);
        const auto before=BranchPages::counters();
        const auto root_before=root.ownership();
        // One write spans two adjacent physical chunks, touching only those chunks.
        const uint64_t begin=gran-7, bytes=23;
        REQUIRE(root.prepare_write(begin,bytes),"prepare_boundary_cow",6);
        overwrite<<<1,64>>>(root.data(),begin,bytes,0xD7);
        REQUIRE(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"gpu_boundary_write",7);
        CHECK(pull(root.data(),root_out) && check_fixture(root_out,expected,begin,bytes,0xD7),"root_after_cow");
        CHECK(pull(sibling.data(),sibling_out) && sibling_out==expected,"sibling_still_original");
        const auto after_cow=BranchPages::counters();
        const auto root_after=root.ownership();
        const auto sibling_after=sibling.ownership();
        CHECK(after_cow.cow_pages-before.cow_pages==2,"cow_physical_pages");
        CHECK(after_cow.copied_bytes-before.copied_bytes==2*gran,"cow_copy_bytes");
        CHECK(root_before.size()>=2 && root_after.size()==root_before.size(),"two_page_root_shape");
        if(root_before.size()>=2 && root_after.size()>=2) {
            CHECK(root_before[0].allocation_id!=root_after[0].allocation_id && root_before[1].allocation_id!=root_after[1].allocation_id,"touched_pages_detached");
            CHECK(root_before[0].allocation_id==sibling_after[0].allocation_id && root_before[1].allocation_id==sibling_after[1].allocation_id,"sibling_retains_original_pages");
        }
        CHECK(root_out!=sibling_out,"branches_diverge");
        // Snapshot verification hashes the live physical pages through BranchPages' SHA-256 path.
        auto changed=root.freeze();
        REQUIRE(bool(changed),"freeze_changed",8);
        REQUIRE(root.restore(frozen),"restore_root_snapshot",8);
        CHECK(pull(root.data(),root_out) && root_out==expected,"root_snapshot_restored");
        REQUIRE(sibling.prepare_write(gran+1,1),"prepare_sibling_cow",9);
        overwrite<<<1,1>>>(sibling.data(),gran+1,1,0xA9);
        REQUIRE(cudaDeviceSynchronize()==cudaSuccess,"sibling_write",10);
        CHECK(pull(sibling.data(),sibling_out) && check_fixture(sibling_out,expected,gran+1,1,0xA9),"sibling_write_validated");
        REQUIRE(root.restore(changed),"restore_changed_snapshot",11);
        CHECK(pull(root.data(),root_out) && check_fixture(root_out,expected,begin,bytes,0xD7),"changed_snapshot_restored");
    }
    const auto after=BranchPages::counters();
    CHECK(after.live_bytes==baseline.live_bytes,"live_bytes_cleanup");
    CHECK(after.allocations-baseline.allocations==after.releases-baseline.releases,"allocation_cleanup");
    CHECK(after.mappings-baseline.mappings==after.unmappings-baseline.unmappings,"mapping_cleanup");
    CHECK(after.integrity_mismatches==baseline.integrity_mismatches,"hash_integrity_no_mismatch");
    CHECK(after.integrity_checked_pages>baseline.integrity_checked_pages,"snapshot_integrity_hashing_exercised");
    const uint64_t cow_checks=check_count-cow_checks_before,cow_passed=passed_checks-cow_passed_before;
    run_split_stages(seed);
    std::printf("{\"type\":\"result\",\"id\":\"seeded_cow\",\"status\":\"%s\",\"seed\":%llu,\"logical_pages\":%llu,\"logical_bytes\":%llu,\"materialized_bytes\":%llu,\"granularity\":%llu,\"checks\":%llu,\"passed_checks\":%llu,\"all_checks\":%llu,\"all_passed_checks\":%llu,\"allocations\":%llu,\"releases\":%llu,\"cow_pages\":%llu,\"copied_bytes\":%llu,\"integrity_checked_pages\":%llu,\"integrity_mismatches\":%llu,\"live_bytes_after\":%llu,\"failures\":%d}\n",
      failures?"FAIL":"PASS",(unsigned long long)seed,(unsigned long long)logical_pages,(unsigned long long)logical_bytes,(unsigned long long)physical_bytes,(unsigned long long)gran,
      (unsigned long long)cow_checks,(unsigned long long)cow_passed,(unsigned long long)check_count,(unsigned long long)passed_checks,
      (unsigned long long)(after.allocations-baseline.allocations),(unsigned long long)(after.releases-baseline.releases),
      (unsigned long long)(after.cow_pages-baseline.cow_pages),(unsigned long long)(after.copied_bytes-baseline.copied_bytes),
      (unsigned long long)(after.integrity_checked_pages-baseline.integrity_checked_pages),
      (unsigned long long)(after.integrity_mismatches-baseline.integrity_mismatches),(unsigned long long)after.live_bytes,failures);
    return failures?1:0;
}
