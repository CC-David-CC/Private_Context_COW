#include "strata/core/vmm.hpp"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <atomic>
#include <exception>
#include <limits>
#include <map>
#include <unordered_map>

namespace strata::core {
namespace {

struct FailureLog {
    std::mutex mutex;
    VmmFailureCounters counters;
};
FailureLog& failure_log() {
    // VMM also runs from the pools' static destructors.
    static auto* log = new FailureLog;
    return *log;
}
bool driver_result(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return true;
    auto& log = failure_log();
    std::lock_guard<std::mutex> lock(log.mutex);
    ++log.counters.driver_failures;
    if (result == CUDA_ERROR_OUT_OF_MEMORY) ++log.counters.out_of_memory_failures;
    log.counters.last_error = static_cast<int>(result);
    log.counters.last_operation = operation;
    return false;
}

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api(int device) {
    static auto* values = new std::map<int, Api>;
    static auto* mutex = new std::mutex;
    std::lock_guard<std::mutex> lock(*mutex);
    auto found = values->find(device);
    if (found != values->end()) return found->second;
    auto& a = (*values)[device];
    a.dev = device;
    [&] {
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        if (device < 0) return;
        if (!resolve("cuDeviceGetAttribute", attr) || !resolve("cuMemGetAllocationGranularity", granularity) ||
            !resolve("cuMemAddressReserve", a.reserve) || !resolve("cuMemAddressFree", a.free_va) ||
            !resolve("cuMemCreate", a.create) || !resolve("cuMemRelease", a.release) || !resolve("cuMemMap", a.map) ||
            !resolve("cuMemUnmap", a.unmap) || !resolve("cuMemSetAccess", a.access))
            return;
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) a.dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.dev;
        size_t g = 0;
        if (granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        a.gran = g;
        a.ok = true;
    }();
    return a;
}
const Api& api() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) device = -1;
    return api(device);
}
struct ChunkRegistry {
    std::mutex mutex;
    std::unordered_map<VmmChunk,int> devices;
};
ChunkRegistry& chunks() { static auto* value=new ChunkRegistry; return *value; }
struct DeviceGuard {
    int previous=-1;
    bool changed=false;
    explicit DeviceGuard(int target) {
        if (target < 0) return;
        if (cudaGetDevice(&previous)!=cudaSuccess) std::terminate();
        if (target!=previous) {
            if (cudaSetDevice(target)!=cudaSuccess) std::terminate();
            changed=true;
        }
    }
    ~DeviceGuard() { if (changed && cudaSetDevice(previous)!=cudaSuccess) std::terminate(); }
};

}  // namespace

bool vmm_available() { return api().ok; }
int vmm_current_device() { int d=-1; return cudaGetDevice(&d)==cudaSuccess?d:-1; }
int vmm_chunk_device(VmmChunk handle) {
    auto& c=chunks(); std::lock_guard<std::mutex> lock(c.mutex);
    auto found=c.devices.find(handle); return found==c.devices.end()?-1:found->second;
}
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }
VmmFailureCounters vmm_failure_counters() {
    auto& log = failure_log();
    std::lock_guard<std::mutex> lock(log.mutex);
    return log.counters;
}

namespace {
std::atomic<uint64_t> mapping_events{0}, unmapping_events{0};
std::atomic<uint64_t> reservation_events{0}, address_release_events{0};
}
VmmMappingCounters vmm_mapping_counters() {
    return {mapping_events.load(),unmapping_events.load(),reservation_events.load(),address_release_events.load()};
}
VmmMapping::~VmmMapping() { DeviceGuard guard(device_); if (!release()) std::terminate(); }
bool VmmMapping::reserve(uint64_t bytes) {
    const auto& a = api();
    if (!a.ok || !bytes || bytes>std::numeric_limits<uint64_t>::max()-a.gran+1 || !release()) return false;
    const auto n = (bytes+a.gran-1)/a.gran;
    std::vector<VmmChunk> handles;
    try { handles.resize(n,0); } catch (...) { return false; }
    CUdeviceptr p = 0;
    if (!driver_result(a.reserve(&p,n*a.gran,0,0,0),"cuMemAddressReserve")) return false;
    ++reservation_events;
    handles_.swap(handles);
    base_ = p;
    device_=a.dev;
    granularity_=a.gran;
    return true;
}
bool VmmMapping::map(int64_t index, VmmChunk handle, bool writable) {
    if (!base_ || vmm_current_device()!=device_ || index<0 || (size_t)index>=handles_.size() ||
        handles_[index] || !handle || vmm_chunk_device(handle)!=device_) return false;
    const auto& a = api();
    if (!driver_result(a.map(base_+index*a.gran,a.gran,0,handle,0),"cuMemMap")) return false;
    handles_[index]=handle;
    ++mapping_events;
    if (protect(index,writable)) return true;
    if (!unmap(index)) std::terminate(); // never let a failed borrow release a still-mapped handle
    return false;
}
bool VmmMapping::protect(int64_t index, bool writable) {
    if (vmm_current_device()!=device_ || index<0 || (size_t)index>=handles_.size() || !handles_[index]) return false;
    const auto& a = api();
    CUmemAccessDesc access{};
    access.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id=a.dev;
    access.flags=writable?CU_MEM_ACCESS_FLAGS_PROT_READWRITE:CU_MEM_ACCESS_FLAGS_PROT_READ;
    return driver_result(a.access(base_+index*a.gran,a.gran,&access,1),"cuMemSetAccess");
}
bool VmmMapping::unmap(int64_t index) {
    if (vmm_current_device()!=device_ || index<0 || (size_t)index>=handles_.size()) return false;
    if (!handles_[index]) return true;
    const auto& a=api();
    if (!driver_result(a.unmap(base_+index*a.gran,a.gran),"cuMemUnmap")) return false;
    handles_[index]=0;
    ++unmapping_events;
    return true;
}
bool VmmMapping::release() {
    if (!base_) return true;
    if (vmm_current_device()!=device_) return false;
    if (!clear()) return false;
    if (!driver_result(api().free_va(base_,handles_.size()*api().gran),"cuMemAddressFree")) return false;
    ++address_release_events;
    base_=0;
    handles_.clear();
    device_=-1; granularity_=0;
    return true;
}
bool VmmMapping::clear() {
    if (base_ && vmm_current_device()!=device_) return false;
    for (size_t i=0;i<handles_.size();++i) if (!unmap(i)) return false;
    return true;
}

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemGenericAllocationHandle h = 0;
    if (!driver_result(a.create(&h, (size_t) a.gran, &prop, 0), "cuMemCreate")) return 0;
    try {
        auto& c=chunks(); std::lock_guard<std::mutex> lock(c.mutex);
        c.devices.emplace((VmmChunk)h,a.dev);
    } catch (...) { a.release(h); return 0; }
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    (void)vmm_chunk_release(h);
}
bool vmm_chunk_release(VmmChunk h) {
    if (!h) return true;
    auto& c=chunks(); std::lock_guard<std::mutex> lock(c.mutex);
    auto found=c.devices.find(h);
    if (found==c.devices.end() || found->second!=vmm_current_device()) return false;
    const auto& a=api(found->second);
    if (!a.ok || !driver_result(a.release((CUmemGenericAllocationHandle)h),"cuMemRelease")) return false;
    c.devices.erase(found); return true;
}

VmmRange::~VmmRange() { DeviceGuard guard(device_); release(); }
bool VmmRange::reserve(uint64_t bytes) {
    if (base_ && vmm_current_device()!=device_) return false;
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0 || bytes>std::numeric_limits<uint64_t>::max()-a.gran+1) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    CUdeviceptr p = 0;
    if (!driver_result(a.reserve(&p, (size_t) (n * a.gran), 0, 0, 0), "cuMemAddressReserve")) return false;
    base_ = (unsigned long long) p;
    h_.assign((size_t) n, 0);
    device_=a.dev; granularity_=a.gran;
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    if (vmm_current_device()!=device_) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    driver_result(a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * a.gran)), "cuMemAddressFree");
    base_ = 0;
    h_.clear();
    device_=-1; granularity_=0;
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    if (vmm_current_device()!=device_ || vmm_chunk_device(h)!=device_) return false;
    const Api& a = api();
    if (!driver_result(a.map((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0,
                            (CUmemGenericAllocationHandle) h, 0), "cuMemMap"))
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (vmm_current_device()!=device_) return false;
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = a.dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return driver_result(a.access((CUdeviceptr) (base_ + (uint64_t) lo * a.gran),
                                 (size_t) ((uint64_t) (hi - lo) * a.gran), &d, 1), "cuMemSetAccess");
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (vmm_current_device()!=device_ || !mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (!driver_result(a.unmap((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran), "cuMemUnmap")) return 0;
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace strata::core

#else  // HIP: no virtual memory here; the elastic K/V stays off

namespace strata::core {
bool vmm_available() { return false; }
int vmm_current_device() { return -1; }
int vmm_chunk_device(VmmChunk) { return -1; }
VmmMappingCounters vmm_mapping_counters() { return {}; }
VmmFailureCounters vmm_failure_counters() { return {}; }
VmmMapping::~VmmMapping() = default;
bool VmmMapping::reserve(uint64_t) { return false; }
bool VmmMapping::release() { return true; }
bool VmmMapping::clear() { return true; }
bool VmmMapping::map(int64_t,VmmChunk,bool) { return false; }
bool VmmMapping::unmap(int64_t) { return false; }
bool VmmMapping::protect(int64_t,bool) { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool vmm_chunk_release(VmmChunk h) { return h==0; }
bool VmmRange::reserve(uint64_t) { return false; }
VmmRange::~VmmRange() = default;
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace strata::core

#endif
