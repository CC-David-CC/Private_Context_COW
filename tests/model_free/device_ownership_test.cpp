// Real driver allocation ownership witness. No model and no mocked GPU.
#include "strata/core/vmm.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int count = 0, failures = 0, checks = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("{\"status\":\"BLOCKED\",\"reason\":\"no CUDA device\"}"); return 2;
    }
    std::vector<int> devices;
    for (int i=1; i<argc; ++i) devices.push_back(std::atoi(argv[i]));
    if (devices.empty()) devices.push_back(0);
    for (int dev : devices) {
        if (dev < 0 || dev >= count || cudaSetDevice(dev) != cudaSuccess ||
            cudaFree(nullptr) != cudaSuccess || !strata::core::vmm_available()) {
            std::printf("{\"id\":\"MF-02-VMM\",\"device\":%d,\"status\":\"BLOCKED\"}\n",dev);
            return 2;
        }
        const auto handle = strata::core::vmm_chunk_new();
        CUmemAllocationProp prop{};
        const auto query = handle ? cuMemGetAllocationPropertiesFromHandle(&prop,handle) : CUDA_ERROR_INVALID_HANDLE;
        ++checks;
        const bool ok = query == CUDA_SUCCESS && prop.location.id == dev;
        if (!ok) ++failures;
        std::printf("{\"id\":\"MF-02-ALLOCATION-DEVICE\",\"requested_device\":%d,"
                    "\"actual_device\":%d,\"query_error\":%d,\"granularity\":%llu,\"status\":\"%s\"}\n",
                    dev,prop.location.id,int(query),(unsigned long long)strata::core::vmm_granularity(),ok?"PASS":"FAIL");
        if (handle && !strata::core::vmm_chunk_release(handle)) { ++checks; ++failures; }
    }
    if (devices.size()>1) {
        cudaSetDevice(devices[0]);
        const auto handle=strata::core::vmm_chunk_new();
        strata::core::VmmMapping mapping;
        const auto g=strata::core::vmm_granularity();
        const bool initialized=handle && mapping.reserve(g) && mapping.map(0,handle,true);
        ++checks; if (!initialized) ++failures;
        cudaSetDevice(devices[1]);
        const auto before=strata::core::vmm_mapping_counters();
        const bool rejected=initialized && !mapping.protect(0,false) && !mapping.clear() &&
            !mapping.release() && !strata::core::vmm_chunk_release(handle);
        const auto after=strata::core::vmm_mapping_counters();
        const bool unchanged=before.unmappings==after.unmappings && before.address_releases==after.address_releases;
        ++checks; if (!rejected || !unchanged) ++failures;
        std::printf("{\"id\":\"MF-02-WRONG-DEVICE-REJECTION\",\"status\":\"%s\",\"unchanged\":%s}\n",
                    rejected&&unchanged?"PASS":"FAIL",unchanged?"true":"false");
        cudaSetDevice(devices[0]);
        const bool closed=mapping.release() && strata::core::vmm_chunk_release(handle);
        ++checks; if (!closed) ++failures;
        std::printf("{\"id\":\"MF-02-OWNER-CONTINUES\",\"status\":\"%s\"}\n",closed?"PASS":"FAIL");
    }
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"skips\":0}\n",
                failures?"FAIL":"PASS",checks,failures);
    return failures ? 1 : 0;
}
