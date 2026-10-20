#include "strata/core/vmm.hpp"
#include "strata/core/branch_gpu_compat.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#if !defined(STRATA_VMM_BASELINE)
#include "strata/core/branch_pages.hpp"
#endif

namespace {
unsigned check_sequence = 0;
struct LateChunkCleanup {
    strata::core::VmmChunk chunk = 0;
    ~LateChunkCleanup() {
        if (chunk && !strata::core::vmm_chunk_release(chunk)) {
            std::fputs("HIP_LATE_CLEANUP:FAIL\n", stderr);
            std::_Exit(96);
        }
        if (chunk) std::puts("HIP_LATE_CLEANUP:PASS");
    }
};
// Construct before default-priority adapter registries so destruction runs
// after them. The handle itself is assigned in main after HIP initialization.
LateChunkCleanup late_chunk_cleanup __attribute__((init_priority(65534)));
}

#define CHECK(condition) do { \
    const unsigned check_id = ++check_sequence; \
    if (!(condition)) { \
        std::fprintf(stderr,"HIP_CHECK_FAIL:%03u:%s (%s:%d)\n",check_id,#condition,__FILE__,__LINE__); \
        std::abort(); \
    } \
    std::printf("HIP_CHECK_PASS:%03u:%s\n",check_id,#condition); \
} while (0)

namespace {
int readonly_write_witness() {
    int device=-1;
    if (hipGetDevice(&device)!=hipSuccess || !strata::core::vmm_available()) return 31;
    const auto granularity=strata::core::vmm_granularity();
    const auto chunk=strata::core::vmm_chunk_new();
    if (!granularity || !chunk) return 32;
    strata::core::VmmMapping mapping;
    if (!mapping.reserve(granularity) || !mapping.map(0,chunk,false)) return 33;
    const char ready[]="READY\n";
    (void)write(STDOUT_FILENO,ready,sizeof(ready)-1);
    alarm(15);
    const hipError_t write_result=hipMemset(mapping.base(),0xa5,4);
    const hipError_t result=write_result==hipSuccess?hipDeviceSynchronize():write_result;
    if (result==hipSuccess) {
        const char unexpected[]="WRITE_UNEXPECTED\n";
        (void)write(STDOUT_FILENO,unexpected,sizeof(unexpected)-1);
        return 0;
    }
    char rejected[64];
    const int count=std::snprintf(rejected,sizeof(rejected),"WRITE_REJECTED:%d\n",static_cast<int>(result));
    if (count>0) (void)write(STDOUT_FILENO,rejected,static_cast<size_t>(count));
    if (result!=hipErrorIllegalAddress && result!=hipErrorInvalidValue) return 42;
    // The runtime can leave this child's HIP context in an error state after
    // the protected write. Let process teardown release its VMM handle.
    _exit(41);
}

bool isolated_readonly_write_rejected() {
    int fds[2];
    if (pipe(fds)!=0) return false;
    const pid_t child=fork();
    if (child<0) return false;
    if (child==0) {
        close(fds[0]);
        if (dup2(fds[1],STDOUT_FILENO)<0) _exit(90);
        close(fds[1]);
        char executable[4096];
        const auto length=readlink("/proc/self/exe",executable,sizeof(executable)-1);
        if (length<=0 || static_cast<size_t>(length)>=sizeof(executable)) _exit(91);
        executable[length]='\0';
        execl(executable,executable,"--readonly-write-witness",static_cast<char*>(nullptr));
        _exit(92);
    }
    close(fds[1]);
    const int flags=fcntl(fds[0],F_GETFL,0);
    if (flags<0 || fcntl(fds[0],F_SETFL,flags|O_NONBLOCK)<0) {
        kill(child,SIGKILL);
        (void)waitpid(child,nullptr,0);
        close(fds[0]);
        return false;
    }
    char output[256]{};
    size_t used=0;
    bool eof=false,exited=false;
    int status=0;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    while (!(eof && exited) && std::chrono::steady_clock::now()<deadline) {
        if (!eof) {
            char overflow[128];
            char* destination=used<sizeof(output)-1?output+used:overflow;
            const size_t capacity=used<sizeof(output)-1?sizeof(output)-1-used:sizeof(overflow);
            const auto n=read(fds[0],destination,capacity);
            if (n>0 && destination==output+used) used+=static_cast<size_t>(n);
            else if (n==0) eof=true;
            else if (n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) break;
        }
        if (!exited) {
            const pid_t result=waitpid(child,&status,WNOHANG);
            if (result==child) exited=true;
            else if (result<0 && errno!=EINTR) break;
        }
        if (!(eof && exited)) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    close(fds[0]);
    if (!exited) {
        kill(child,SIGKILL);
        (void)waitpid(child,&status,0);
        return false;
    }
    const bool ready=std::strstr(output,"READY\n")!=nullptr;
    const char* rejected=std::strstr(output,"WRITE_REJECTED:");
    const int error_code=rejected?std::atoi(rejected+15):-1;
    std::fprintf(stderr,"read-only child output: %sstatus=%d signal=%d\n",output,
        WIFEXITED(status)?WEXITSTATUS(status):-1,WIFSIGNALED(status)?WTERMSIG(status):0);
    return ready && WIFEXITED(status) && WEXITSTATUS(status)==41 &&
        (error_code==hipErrorIllegalAddress || error_code==hipErrorInvalidValue);
}
}

int main(int argc, char** argv) {
    if (argc==2 && std::strcmp(argv[1],"--readonly-write-witness")==0) return readonly_write_witness();
    using namespace strata::core;
    int device = -1;
    CHECK(hipGetDevice(&device) == hipSuccess);
    CHECK(vmm_available());
    const uint64_t granularity = vmm_granularity();
    CHECK(granularity != 0);

    const auto before = vmm_mapping_counters();
    const VmmChunk chunk = vmm_chunk_new();
    CHECK(chunk != 0);

    VmmMapping owner;
    CHECK(owner.reserve(granularity));
#if !defined(STRATA_VMM_BASELINE)
    CHECK(owner.device() == device);
    CHECK(owner.granularity() == granularity);
#endif
    CHECK(owner.capacity() == granularity);
    CHECK(owner.map(0, chunk, true));

    std::vector<uint8_t> expected(static_cast<size_t>(granularity));
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    CHECK(hipMemcpy(owner.base(), expected.data(), expected.size(), hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipDeviceSynchronize() == hipSuccess);

    unsigned long long access = 0;
    hipMemLocation location{};
    location.type = hipMemLocationTypeDevice;
    location.id = device;
    CHECK(hipMemGetAccess(&access, &location, owner.base()) == hipSuccess);
    CHECK(access == hipMemAccessFlagsProtReadWrite);

    // Change access in place, then verify the driver reports read-only access.
    CHECK(owner.protect(0, false));
    CHECK(hipMemGetAccess(&access, &location, owner.base()) == hipSuccess);
    CHECK(access == hipMemAccessFlagsProtRead);
    CHECK(owner.protect(0, true));

    // A second VA borrows the same physical chunk read-only. Copies through
    // that alias must observe the owner bytes without acquiring write access.
    VmmMapping reader;
    CHECK(reader.reserve(granularity));
#if !defined(STRATA_VMM_BASELINE)
    CHECK(reader.device() == device);
#endif
    CHECK(reader.map(0, chunk, false));
    CHECK(hipMemGetAccess(&access, &location, reader.base()) == hipSuccess);
    CHECK(access == hipMemAccessFlagsProtRead);
    std::vector<uint8_t> actual(expected.size());
    CHECK(hipMemcpy(actual.data(), reader.base(), actual.size(), hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(actual == expected);
#if !defined(STRATA_VMM_BASELINE)
    CHECK(isolated_readonly_write_rejected());
#endif

    CHECK(reader.unmap(0));
    CHECK(reader.release());
    CHECK(owner.unmap(0));
    CHECK(owner.release());
    CHECK(vmm_chunk_release(chunk));

    // Exercise the same allocation/map/access/unmap lifecycle through VmmRange.
#if !defined(STRATA_VMM_BASELINE)
    VmmRange range;
    CHECK(range.reserve(granularity));
    CHECK(range.device() == device);
    VmmChunk ranged = vmm_chunk_new();
    CHECK(ranged != 0);
    CHECK(range.map_range(0, 1, [ranged]() mutable { const auto h = ranged; ranged = 0; return h; }));
    CHECK(range.mapped(0));
    CHECK(range.mapped_count() == 1);
    auto handed_back = range.unmap(0);
    CHECK(handed_back != 0);
    CHECK(vmm_chunk_release(handed_back));
    range.release();
#endif

    const auto after = vmm_mapping_counters();
    const uint64_t mapping_count =
#if defined(STRATA_VMM_BASELINE)
        2;
#else
        3;
#endif
    CHECK(after.mappings - before.mappings == mapping_count);
    CHECK(after.unmappings - before.unmappings == mapping_count);
    CHECK(after.reservations - before.reservations == mapping_count);
    CHECK(after.address_releases - before.address_releases == mapping_count);
    std::printf("HIP_VMM_METRICS:{\"mappings_delta\":%llu,\"unmappings_delta\":%llu,\"reservations_delta\":%llu,\"address_releases_delta\":%llu}\n",
        static_cast<unsigned long long>(after.mappings-before.mappings),
        static_cast<unsigned long long>(after.unmappings-before.unmappings),
        static_cast<unsigned long long>(after.reservations-before.reservations),
        static_cast<unsigned long long>(after.address_releases-before.address_releases));
#if !defined(STRATA_VMM_BASELINE)
    int device_count=0;
    CHECK(hipGetDeviceCount(&device_count)==hipSuccess);
    if (device_count>1) {
        VmmMapping owned;
        const VmmChunk owned_chunk=vmm_chunk_new();
        CHECK(owned_chunk && owned.reserve(granularity) && owned.map(0,owned_chunk,true));
        const int other_device=(device+1)%device_count;
        CHECK(hipSetDevice(other_device)==hipSuccess);
        const auto before_reject=vmm_mapping_counters();
        CHECK(!owned.protect(0,false));
        CHECK(!owned.clear());
        CHECK(!owned.release());
        CHECK(!vmm_chunk_release(owned_chunk));
        const auto after_reject=vmm_mapping_counters();
        CHECK(before_reject.mappings==after_reject.mappings);
        CHECK(before_reject.unmappings==after_reject.unmappings);
        CHECK(before_reject.address_releases==after_reject.address_releases);
        CHECK(hipSetDevice(device)==hipSuccess);
        CHECK(owned.release());
        CHECK(vmm_chunk_release(owned_chunk));
        std::puts("HIP VMM wrong-device operations reject without mutation: PASS");
    } else {
        std::puts("HIP VMM wrong-device rejection: UNRUN (one device available)");
    }

    // Native BranchPages COW mechanics: a frozen page is shared, the child's
    // first write allocates and copies one replacement, and parent bytes stay
    // identical. This tests GPU page behavior, not model inference.
    BranchPages parent;
    CHECK(parent.reserve(granularity));
    CHECK(parent.grow(granularity));
    std::vector<uint8_t> page(static_cast<size_t>(granularity));
    for (size_t i=0;i<page.size();++i) page[i]=static_cast<uint8_t>((i*29u+3u)&0xffu);
    CHECK(hipMemcpy(parent.data(),page.data(),page.size(),hipMemcpyHostToDevice)==hipSuccess);
    CHECK(hipDeviceSynchronize()==hipSuccess);
    const auto snapshot=parent.freeze();
    CHECK(snapshot);
    BranchPages child;
    CHECK(child.reserve(granularity));
    CHECK(child.restore(snapshot));
    const auto before_cow=BranchPages::counters();
    CHECK(child.prepare_write(0,1));
    const uint8_t changed=static_cast<uint8_t>(page[0]^0xffu);
    CHECK(hipMemcpy(child.data(),&changed,1,hipMemcpyHostToDevice)==hipSuccess);
    CHECK(hipDeviceSynchronize()==hipSuccess);
    std::vector<uint8_t> saved;
    CHECK(BranchPages::read_page(snapshot,0,saved));
    CHECK(saved==page);
    uint8_t parent_first=0,child_first=0;
    CHECK(hipMemcpy(&parent_first,parent.data(),1,hipMemcpyDeviceToHost)==hipSuccess);
    CHECK(hipMemcpy(&child_first,child.data(),1,hipMemcpyDeviceToHost)==hipSuccess);
    CHECK(parent_first==page[0]);
    CHECK(child_first==changed);
    const auto after_cow=BranchPages::counters();
    CHECK(after_cow.cow_pages-before_cow.cow_pages==1);
    CHECK(after_cow.copied_bytes-before_cow.copied_bytes==granularity);
    const auto parent_ownership=parent.ownership();
    const auto child_ownership=child.ownership();
    CHECK(parent_ownership.size()==1 && child_ownership.size()==1);
    CHECK(parent_ownership[0].allocation_id!=child_ownership[0].allocation_id);
    std::printf("HIP_COW_METRICS:{\"cow_pages_delta\":%llu,\"copied_bytes_delta\":%llu,\"granularity\":%llu,\"parent_allocations\":%zu,\"child_allocations\":%zu}\n",
        static_cast<unsigned long long>(after_cow.cow_pages-before_cow.cow_pages),
        static_cast<unsigned long long>(after_cow.copied_bytes-before_cow.copied_bytes),
        static_cast<unsigned long long>(granularity),
        parent_ownership.size(),child_ownership.size());
#endif
    late_chunk_cleanup.chunk=vmm_chunk_new();
    CHECK(late_chunk_cleanup.chunk!=0);
    std::puts("HIP VMM mapping, protection, copy, range, cleanup, and native BranchPages COW: PASS");
}
