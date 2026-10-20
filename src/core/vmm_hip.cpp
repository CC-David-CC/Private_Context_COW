#include "strata/core/vmm.hpp"
#include <hip/hip_runtime.h>

#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace strata::core {
namespace {
struct FailureLog { std::mutex mutex; VmmFailureCounters counters; };
FailureLog& failure_log() { static auto* value = new FailureLog; return *value; }
bool driver_result(hipError_t result, const char* operation) {
    if (result == hipSuccess) return true;
    auto& log = failure_log();
    std::lock_guard<std::mutex> lock(log.mutex);
    ++log.counters.driver_failures;
    if (result == hipErrorOutOfMemory) ++log.counters.out_of_memory_failures;
    log.counters.last_error = static_cast<int>(result);
    log.counters.last_operation = operation;
    return false;
}

struct DeviceApi { int device=-1; uint64_t granularity=0; bool ok=false; };
struct DeviceApiRegistry { std::mutex mutex; std::unordered_map<int, DeviceApi> devices; };
DeviceApiRegistry& device_api_registry() { static auto* value = new DeviceApiRegistry; return *value; }
DeviceApi api_for(int device) {
    auto& registry = device_api_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto [it, inserted] = registry.devices.try_emplace(device);
    if (inserted) {
        auto& a = it->second;
        a.device = device;
        int supported = 0;
        if (hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, device) != hipSuccess || !supported)
            return a;
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        size_t granularity = 0;
        if (hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || !granularity)
            return a;
        a.granularity = granularity;
        a.ok = true;
    }
    return it->second;
}
int current_device() { int device=-1; return hipGetDevice(&device)==hipSuccess ? device : -1; }

struct Allocation { hipMemGenericAllocationHandle_t handle{}; int device=-1; uint64_t granularity=0; };
struct AllocationRegistry { std::mutex mutex; std::unordered_map<VmmChunk, Allocation> chunks; };
AllocationRegistry& allocation_registry() { static auto* value = new AllocationRegistry; return *value; }
VmmChunk chunk_key(hipMemGenericAllocationHandle_t handle) {
    return static_cast<VmmChunk>(reinterpret_cast<uintptr_t>(handle));
}
bool allocation_for(VmmChunk key, Allocation& result) {
    auto& registry = allocation_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it=registry.chunks.find(key);
    if (it==registry.chunks.end()) return false;
    result=it->second;
    return true;
}
bool record_allocation(hipMemGenericAllocationHandle_t handle, int device, uint64_t granularity) {
    auto& registry = allocation_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    try { return registry.chunks.emplace(chunk_key(handle), Allocation{handle,device,granularity}).second; }
    catch (...) { return false; }
}
void forget_allocation(VmmChunk key) {
    auto& registry = allocation_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.chunks.erase(key);
}

struct MappingCounters {
    std::atomic<uint64_t> mappings{0},unmappings{0},reservations{0},address_releases{0};
};
MappingCounters& mapping_counts() { static auto* value=new MappingCounters; return *value; }
bool compatible_chunk(VmmChunk key, int device, uint64_t granularity) {
    Allocation allocation;
    return key && allocation_for(key,allocation) && allocation.device==device && allocation.granularity==granularity;
}
hipMemGenericAllocationHandle_t allocation_handle(VmmChunk key) {
    Allocation allocation;
    return allocation_for(key, allocation) ? allocation.handle : hipMemGenericAllocationHandle_t{};
}
bool allocation_prop(int device, hipMemAllocationProp& prop) {
    prop={};
    prop.type=hipMemAllocationTypePinned;
    prop.location.type=hipMemLocationTypeDevice;
    prop.location.id=device;
    return true;
}

class DeviceGuard {
public:
    explicit DeviceGuard(int target) : previous_(current_device()), changed_(previous_>=0 && previous_!=target) {
        ok_=previous_>=0 && (!changed_ || hipSetDevice(target)==hipSuccess);
    }
    ~DeviceGuard() { if (ok_ && changed_ && hipSetDevice(previous_)!=hipSuccess) std::terminate(); }
    bool ok() const { return ok_; }
private: int previous_=-1; bool changed_=false,ok_=false;
};
}

bool vmm_available() { const int d=current_device(); return d>=0 && api_for(d).ok; }
int vmm_current_device() { return current_device(); }
uint64_t vmm_granularity() { const int d=current_device(); if (d<0) return 0; const auto a=api_for(d); return a.ok?a.granularity:0; }
VmmFailureCounters vmm_failure_counters() {
    auto& log=failure_log(); std::lock_guard<std::mutex> lock(log.mutex); return log.counters;
}
VmmMappingCounters vmm_mapping_counters() {
    const auto& c=mapping_counts();
    return {c.mappings.load(),c.unmappings.load(),c.reservations.load(),c.address_releases.load()};
}
int vmm_chunk_device(VmmChunk h) { Allocation a; return allocation_for(h,a)?a.device:-1; }

VmmMapping::~VmmMapping() {
    if (!base_) return;
    DeviceGuard guard(device_);
    if (!guard.ok() || !release()) std::terminate();
}
bool VmmMapping::reserve(uint64_t bytes) {
    const int device=current_device();
    if (device<0 || !bytes) return false;
    const auto a=api_for(device);
    const uint64_t granularity=a.granularity;
    if (!a.ok || bytes>std::numeric_limits<uint64_t>::max()-granularity+1 || !release()) return false;
    const uint64_t n=(bytes+granularity-1)/granularity;
    if (!n || n>std::numeric_limits<size_t>::max() || n>std::numeric_limits<size_t>::max()/granularity) return false;
    std::vector<VmmChunk> handles;
    try { handles.resize(static_cast<size_t>(n),0); } catch (...) { return false; }
    void* address=nullptr;
    if (!driver_result(hipMemAddressReserve(&address,static_cast<size_t>(n*granularity),0,nullptr,0),"hipMemAddressReserve")) return false;
    ++mapping_counts().reservations;
    handles_.swap(handles);
    base_=reinterpret_cast<uintptr_t>(address);
    device_=device;
    granularity_=granularity;
    return true;
}
bool VmmMapping::map(int64_t index, VmmChunk handle, bool writable) {
    if (!base_ || current_device()!=device_ || index<0 || static_cast<size_t>(index)>=handles_.size() || handles_[index] ||
        !compatible_chunk(handle,device_,granularity_)) return false;
    void* address=reinterpret_cast<void*>(base_+static_cast<uint64_t>(index)*granularity_);
    if (!driver_result(hipMemMap(address,granularity_,0,allocation_handle(handle),0),"hipMemMap")) return false;
    handles_[index]=handle;
    ++mapping_counts().mappings;
    if (protect(index,writable)) return true;
    if (!unmap(index)) std::terminate();
    return false;
}
bool VmmMapping::protect(int64_t index, bool writable) {
    if (!base_ || current_device()!=device_ || index<0 || static_cast<size_t>(index)>=handles_.size() || !handles_[index]) return false;
    hipMemAccessDesc access{};
    access.location.type=hipMemLocationTypeDevice;
    access.location.id=device_;
    access.flags=writable?hipMemAccessFlagsProtReadWrite:hipMemAccessFlagsProtRead;
    void* address=reinterpret_cast<void*>(base_+static_cast<uint64_t>(index)*granularity_);
    return driver_result(hipMemSetAccess(address,granularity_,&access,1),"hipMemSetAccess");
}
bool VmmMapping::unmap(int64_t index) {
    if (!base_ || current_device()!=device_ || index<0 || static_cast<size_t>(index)>=handles_.size()) return false;
    if (!handles_[index]) return true;
    void* address=reinterpret_cast<void*>(base_+static_cast<uint64_t>(index)*granularity_);
    if (!driver_result(hipMemUnmap(address,granularity_),"hipMemUnmap")) return false;
    handles_[index]=0;
    ++mapping_counts().unmappings;
    return true;
}
bool VmmMapping::clear() {
    if (base_ && current_device()!=device_) return false;
    for (size_t i=0;i<handles_.size();++i) if (!unmap(static_cast<int64_t>(i))) return false;
    return true;
}
bool VmmMapping::release() {
    if (!base_) return true;
    if (current_device()!=device_ || !clear()) return false;
    const size_t bytes=handles_.size()*static_cast<size_t>(granularity_);
    if (!driver_result(hipMemAddressFree(reinterpret_cast<void*>(base_),bytes),"hipMemAddressFree")) return false;
    ++mapping_counts().address_releases;
    base_=0;
    handles_.clear();
    device_=-1;
    granularity_=0;
    return true;
}

VmmChunk vmm_chunk_new() {
    const int device=current_device();
    if (device<0) return 0;
    const auto a=api_for(device);
    if (!a.ok) return 0;
    hipMemAllocationProp prop{};
    allocation_prop(device,prop);
    hipMemGenericAllocationHandle_t handle{};
    if (!driver_result(hipMemCreate(&handle,static_cast<size_t>(a.granularity),&prop,0),"hipMemCreate")) return 0;
    const VmmChunk key=chunk_key(handle);
    if (!key) {
        (void)driver_result(hipMemRelease(handle),"hipMemRelease");
        return 0;
    }
    if (!record_allocation(handle,device,a.granularity)) {
        (void)driver_result(hipMemRelease(handle),"hipMemRelease");
        return 0;
    }
    return key;
}
void vmm_chunk_free(VmmChunk handle) { (void)vmm_chunk_release(handle); }
bool vmm_chunk_release(VmmChunk key) {
    if (!key) return true;
    Allocation allocation;
    if (!allocation_for(key,allocation) || current_device()!=allocation.device) return false;
    if (!driver_result(hipMemRelease(allocation.handle),"hipMemRelease")) return false;
    forget_allocation(key);
    return true;
}

VmmRange::~VmmRange() {
    if (!base_) return;
    DeviceGuard guard(device_);
    if (!guard.ok()) std::terminate();
    release();
}
bool VmmRange::reserve(uint64_t bytes) {
    if (base_) {
        release();
        if (base_) return false;
    }
    const int device=current_device();
    if (device<0 || !bytes) return false;
    const auto a=api_for(device);
    const uint64_t granularity=a.granularity;
    if (!a.ok || bytes>std::numeric_limits<uint64_t>::max()-granularity+1) return false;
    const uint64_t n=(bytes+granularity-1)/granularity;
    if (!n || n>std::numeric_limits<size_t>::max() || n>std::numeric_limits<size_t>::max()/granularity) return false;
    void* address=nullptr;
    if (!driver_result(hipMemAddressReserve(&address,static_cast<size_t>(n*granularity),0,nullptr,0),"hipMemAddressReserve")) return false;
    try { h_.assign(static_cast<size_t>(n),0); }
    catch (...) {
        if (!driver_result(hipMemAddressFree(address,static_cast<size_t>(n*granularity)),"hipMemAddressFree")) std::terminate();
        return false;
    }
    base_=reinterpret_cast<uintptr_t>(address);
    device_=device;
    granularity_=granularity;
    ++mapping_counts().reservations;
    return true;
}
void VmmRange::release() {
    if (!base_) return;
    if (current_device()!=device_) return;
    for (int64_t i=0;i<chunks();++i) {
        const VmmChunk handle=unmap(i);
        if (handle && !vmm_chunk_release(handle)) std::terminate();
    }
    const size_t bytes=h_.size()*static_cast<size_t>(granularity_);
    if (!driver_result(hipMemAddressFree(reinterpret_cast<void*>(base_),bytes),"hipMemAddressFree")) std::terminate();
    ++mapping_counts().address_releases;
    base_=0;
    h_.clear();
    device_=-1;
    granularity_=0;
}
int64_t VmmRange::mapped_count() const {
    int64_t count=0;
    for (const VmmChunk handle:h_) count+=handle!=0;
    return count;
}
bool VmmRange::map_one(int64_t index, VmmChunk key) {
    if (!base_ || current_device()!=device_ || index<0 || index>=chunks() || mapped(index) ||
        !compatible_chunk(key,device_,granularity_)) return false;
    void* address=reinterpret_cast<void*>(base_+static_cast<uint64_t>(index)*granularity_);
    if (!driver_result(hipMemMap(address,granularity_,0,allocation_handle(key),0),"hipMemMap")) return false;
    h_[static_cast<size_t>(index)]=key;
    ++mapping_counts().mappings;
    return true;
}
bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi<=lo) return true;
    if (!base_ || current_device()!=device_ || lo<0 || hi>chunks()) return false;
    hipMemAccessDesc access{};
    access.location.type=hipMemLocationTypeDevice;
    access.location.id=device_;
    access.flags=hipMemAccessFlagsProtReadWrite;
    void* address=reinterpret_cast<void*>(base_+static_cast<uint64_t>(lo)*granularity_);
    return driver_result(hipMemSetAccess(address,static_cast<size_t>(hi-lo)*granularity_,&access,1),"hipMemSetAccess");
}
bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo,hi)) return true;
    bool ok=true;
    for (int64_t i=lo;i<hi;++i) {
        const VmmChunk handle=unmap(i);
        if (handle && !vmm_chunk_release(handle)) ok=false;
    }
    if (!ok) std::terminate();
    return false;
}
VmmChunk VmmRange::unmap(int64_t index) {
    if (!mapped(index) || current_device()!=device_) return 0;
    const auto key=h_[static_cast<size_t>(index)];
    void* address=reinterpret_cast<void*>(base_+static_cast<uint64_t>(index)*granularity_);
    if (!driver_result(hipMemUnmap(address,granularity_),"hipMemUnmap")) return 0;
    h_[static_cast<size_t>(index)]=0;
    ++mapping_counts().unmappings;
    return key;
}
} // namespace strata::core
