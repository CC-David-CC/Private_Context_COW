#include "fixture_data.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <sys/statvfs.h>
#include <unistd.h>

// Generate canonical bytes on the GPU. The CPU reference is a separate
// fixture_cpu process, so this runner never validates a generator against itself.
__global__ void generate_fixture(uint8_t* out,uint64_t n,uint64_t seed,uint64_t offset) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=cow_fixture::byte_at(seed,cow_fixture::region_canonical,0,0,0,offset+i);
}
static bool number(const char* s,uint64_t& v) {
    char* end=nullptr; const auto x=std::strtoull(s,&end,0); if(!s[0]||!end||*end) return false; v=x; return true;
}
static std::string quote_shell(const std::string& value) {
    std::string out="'";
    for(char c:value) { if(c=='\'') out += "'\\''"; else out += c; }
    return out+"'";
}
static bool read_exact(FILE* stream,uint8_t* dst,size_t bytes) {
    size_t done=0;
    while(done<bytes) { const size_t n=std::fread(dst+done,1,bytes-done,stream); if(!n) return false; done+=n; }
    return true;
}
static bool sha256_file(const std::string& path,std::string& out) {
    const std::string command="sha256sum -- "+quote_shell(path);
    FILE* hash=popen(command.c_str(),"r");
    if(!hash) return false;
    char line[512]{};
    const bool read=std::fgets(line,sizeof(line),hash)!=nullptr;
    const int status=pclose(hash);
    if(!read||status!=0||std::strlen(line)<64) return false;
    out.assign(line,64);
    return true;
}
static uint64_t micros(std::chrono::steady_clock::time_point a,std::chrono::steady_clock::time_point b) {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(b-a).count());
}
static int error(const char* id,int code,uint64_t offset=UINT64_MAX) {
    if(offset==UINT64_MAX) std::printf("{\"type\":\"error\",\"id\":\"%s\",\"code\":%d}\n",id,code);
    else std::printf("{\"type\":\"error\",\"id\":\"%s\",\"code\":%d,\"offset\":%llu}\n",id,code,(unsigned long long)offset);
    return code;
}

int main(int argc,char** argv) {
    uint64_t total=0,chunk=4ULL*1024*1024,seed=0x5052455353555245ULL;
    uint64_t resident_bytes=0,resident_percent=0,headroom=1024ULL*1024*1024;
    uint64_t disk_window=0,disk_reserve=8ULL*1024*1024*1024;
    std::string path,expected_path,oracle_bin="./build/fixture_cpu";
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"--bytes")&&i+1<argc) { if(!number(argv[++i],total)) return 2; }
        else if(!std::strcmp(argv[i],"--chunk-bytes")&&i+1<argc) { if(!number(argv[++i],chunk)) return 2; }
        else if(!std::strcmp(argv[i],"--seed")&&i+1<argc) { if(!number(argv[++i],seed)) return 2; }
        else if(!std::strcmp(argv[i],"--output")&&i+1<argc) path=argv[++i];
        else if(!std::strcmp(argv[i],"--expected")&&i+1<argc) expected_path=argv[++i];
        else if(!std::strcmp(argv[i],"--oracle-bin")&&i+1<argc) oracle_bin=argv[++i];
        else if(!std::strcmp(argv[i],"--resident-bytes")&&i+1<argc) { if(!number(argv[++i],resident_bytes)) return 2; }
        else if(!std::strcmp(argv[i],"--resident-percent")&&i+1<argc) { if(!number(argv[++i],resident_percent)) return 2; }
        else if(!std::strcmp(argv[i],"--headroom-bytes")&&i+1<argc) { if(!number(argv[++i],headroom)) return 2; }
        else if(!std::strcmp(argv[i],"--disk-window-bytes")&&i+1<argc) { if(!number(argv[++i],disk_window)) return 2; }
        else if(!std::strcmp(argv[i],"--disk-reserve-bytes")&&i+1<argc) { if(!number(argv[++i],disk_reserve)) return 2; }
        else { std::fprintf(stderr,"usage: fixture_pressure --bytes N --output FILE (--expected CPU_FILE | --oracle-bin CPU_EXE) [--resident-bytes N | --resident-percent 25|60|90] [--chunk-bytes N] [--disk-window-bytes N] [--headroom-bytes N] [--disk-reserve-bytes N]\n"); return 2; }
    }
    if(!total||!chunk||path.empty()||chunk>256ULL*1024*1024||
       (resident_percent && resident_bytes) || (resident_percent && resident_percent!=25 && resident_percent!=60 && resident_percent!=90)) return 2;
    if(!disk_window) disk_window=total;
    if(!disk_window) return 2;
    if(cudaFree(nullptr)!=cudaSuccess) { std::fprintf(stderr,"CUDA unavailable\n"); return error("cuda_init",2); }

    const auto parent=std::filesystem::path(path).parent_path();
    const auto space=std::filesystem::space(parent.empty()?std::filesystem::path("."):parent);
    if(disk_window>(UINT64_MAX-disk_reserve)/2 || space.available<2*disk_window || space.available-2*disk_window<disk_reserve) {
        std::printf("{\"type\":\"result\",\"id\":\"fixture_pressure\",\"status\":\"BLOCKED\",\"reason\":\"two_disk_windows_plus_reserve_exceeds_free_space\",\"available_bytes\":%llu,\"disk_window_bytes\":%llu,\"required_bytes\":%llu}\n",
            (unsigned long long)space.available,(unsigned long long)disk_window,(unsigned long long)(disk_window>(UINT64_MAX-disk_reserve)/2?UINT64_MAX:2*disk_window+disk_reserve));
        return 10;
    }

    uint8_t* device_chunk=nullptr;
    if(cudaMalloc(&device_chunk,chunk)!=cudaSuccess) return error("staging_allocation",3);
    size_t free_bytes=0,total_device_bytes=0;
    if(cudaMemGetInfo(&free_bytes,&total_device_bytes)!=cudaSuccess) return error("device_memory_query",3);
    const uint64_t admitted_free=free_bytes>headroom?free_bytes-headroom:0;
    if(resident_percent) resident_bytes=(admitted_free/100)*resident_percent;
    const uint64_t resident_requested=resident_bytes;
    const uint64_t generation_bytes=std::max<uint64_t>(total,resident_bytes);
    uint8_t* device_resident=nullptr;
    std::string resident_status="UNREQUESTED";
    if(resident_bytes) {
        if(resident_bytes<=total && resident_bytes<=admitted_free && cudaMalloc(&device_resident,resident_bytes)==cudaSuccess) {
            resident_status="ALLOCATED";
        }
        else { resident_bytes=0; resident_status="UNAVAILABLE_OR_EXCEEDS_TOTAL_BYTES"; (void)cudaGetLastError(); }
    }

    std::vector<uint8_t> expected(chunk),output(chunk);
    FILE* oracle=nullptr;
    std::ifstream expected_file;
    if(!expected_path.empty()) {
        expected_file.open(expected_path,std::ios::binary);
        if(!expected_file) return error("open_expected_file",4);
    } else {
        const std::string command=quote_shell(oracle_bin)+" "+std::to_string(seed)+" 1 1 1 "+std::to_string(generation_bytes);
        oracle=popen(command.c_str(),"r");
        if(!oracle) return error("start_cpu_oracle",4);
    }

    uint64_t written=0,chunk_id=0,window_index=0,window_start=0,window_written=0;
    uint64_t window_actual_bytes=0,total_reread_bytes=0;
    uint64_t write_us=0,fsync_us=0,readback_us=0,total_write_us=0,total_fsync_us=0,total_readback_us=0;
    uint64_t gpu_generate_us=0,gpu_chunk_readback_us=0,resident_readback_us=0;
    FILE* output_file=nullptr,*oracle_window_file=nullptr;
    const std::string oracle_window_path=path+".oracle-window";
    auto finish_window=[&]() -> bool {
        if(!output_file) return true;
        auto t0=std::chrono::steady_clock::now();
        const bool flush_ok=std::fflush(output_file)==0;
        const bool sync_ok=flush_ok && ::fsync(::fileno(output_file))==0;
        const bool oracle_flush=std::fflush(oracle_window_file)==0;
        const bool oracle_sync=oracle_flush && ::fsync(::fileno(oracle_window_file))==0;
        auto t1=std::chrono::steady_clock::now(); fsync_us=micros(t0,t1);
        const bool close_ok=std::fclose(output_file)==0; output_file=nullptr;
        const bool oracle_close=std::fclose(oracle_window_file)==0; oracle_window_file=nullptr;
        if(!sync_ok||!close_ok||!oracle_sync||!oracle_close) return false;
        std::ifstream disk(path,std::ios::binary);
        std::ifstream saved_oracle(oracle_window_path,std::ios::binary);
        if(!disk||!saved_oracle) return false;
        t0=std::chrono::steady_clock::now();
        uint64_t remaining=window_written;
        while(remaining) {
            const size_t n=size_t(std::min<uint64_t>(remaining,output.size()));
            if(!disk.read(reinterpret_cast<char*>(output.data()),std::streamsize(n))) return false;
            if(!saved_oracle.read(reinterpret_cast<char*>(expected.data()),std::streamsize(n))) return false;
            const bool bytes_match=std::equal(output.begin(),output.begin()+n,expected.begin());
            if(!bytes_match) return false;
            remaining-=n; window_actual_bytes+=n;
        }
        char extra=0; if(disk.get(extra)||saved_oracle.get(extra)) return false;
        t1=std::chrono::steady_clock::now(); readback_us=micros(t0,t1);
        std::string expected_hex,read_hex;
        if(!sha256_file(oracle_window_path,expected_hex)||!sha256_file(path,read_hex)) return false;
        const bool match=expected_hex==read_hex && window_actual_bytes==window_written;
        std::printf("{\"type\":\"window\",\"id\":\"window_%llu\",\"offset\":%llu,\"bytes\":%llu,\"reread_bytes\":%llu,\"oracle_sha256\":\"%s\",\"readback_sha256\":\"%s\",\"byte_exact_oracle_match\":%s,\"hash_match\":%s,\"write_us\":%llu,\"fsync_us\":%llu,\"readback_us\":%llu}\n",
            (unsigned long long)window_index,(unsigned long long)window_start,(unsigned long long)window_written,(unsigned long long)window_actual_bytes,
            expected_hex.c_str(),read_hex.c_str(),match?"true":"false",match?"true":"false",(unsigned long long)write_us,(unsigned long long)fsync_us,(unsigned long long)readback_us);
        total_reread_bytes+=window_actual_bytes;
        total_write_us+=write_us; total_fsync_us+=fsync_us; total_readback_us+=readback_us;
        ++window_index; window_start+=window_written; window_written=window_actual_bytes=write_us=fsync_us=readback_us=0;
        return match;
    };

    while(written<generation_bytes) {
        uint64_t n=std::min<uint64_t>(chunk,generation_bytes-written);
        const bool on_disk=written<total;
        if(on_disk) {
            n=std::min<uint64_t>(n,total-written);
            const uint64_t in_window=written-window_start;
            if(!in_window) {
                output_file=std::fopen(path.c_str(),"wb");
                oracle_window_file=std::fopen(oracle_window_path.c_str(),"wb");
                if(!output_file||!oracle_window_file) return error("open_disk_window",5,written);
            }
            n=std::min<uint64_t>(n,disk_window-in_window);
        }
        if(expected_file.is_open()) {
        if(!expected_file.read(reinterpret_cast<char*>(expected.data()),std::streamsize(n))) return error("read_cpu_expected",6,written);
        } else if(!read_exact(oracle,expected.data(),size_t(n))) return error("read_cpu_oracle",6,written);
        auto gpu_t0=std::chrono::steady_clock::now();
        generate_fixture<<<unsigned((n+255)/256),256>>>(device_chunk,n,seed,written);
        if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess) return error("gpu_fixture_generate",7,written);
        auto gpu_t1=std::chrono::steady_clock::now(); gpu_generate_us+=micros(gpu_t0,gpu_t1);
        gpu_t0=std::chrono::steady_clock::now();
        if(cudaMemcpy(output.data(),device_chunk,size_t(n),cudaMemcpyDeviceToHost)!=cudaSuccess) return error("gpu_fixture_readback",7,written);
        gpu_t1=std::chrono::steady_clock::now(); gpu_chunk_readback_us+=micros(gpu_t0,gpu_t1);
        if(!std::equal(output.begin(),output.begin()+n,expected.begin())) {
            uint64_t j=0; while(j<n&&output[j]==expected[j]) ++j;
            return error("gpu_cpu_byte_mismatch",8,written+j);
        }
        if(on_disk) {
            const auto t0=std::chrono::steady_clock::now();
            if(std::fwrite(output.data(),1,size_t(n),output_file)!=n) return error("write_disk_window",9,written);
            const auto t1=std::chrono::steady_clock::now(); write_us+=micros(t0,t1);
            if(std::fwrite(expected.data(),1,size_t(n),oracle_window_file)!=n) return error("write_oracle_window",9,written);
            window_written+=n;
        }
        std::printf("{\"type\":\"chunk\",\"id\":\"fixture_%llu\",\"offset\":%llu,\"bytes\":%llu,\"resident_bytes\":%llu,\"staging_device_bytes\":%llu,\"status\":\"PASS\"}\n",
            (unsigned long long)chunk_id,(unsigned long long)written,(unsigned long long)n,(unsigned long long)resident_bytes,(unsigned long long)chunk);
        if(device_resident && written<resident_bytes) {
            const uint64_t resident_n=std::min<uint64_t>(n,resident_bytes-written);
            const auto resident_t0=std::chrono::steady_clock::now();
            if(cudaMemcpy(device_resident+written,device_chunk,size_t(resident_n),cudaMemcpyDeviceToDevice)!=cudaSuccess ||
               cudaMemcpy(output.data(),device_resident+written,size_t(resident_n),cudaMemcpyDeviceToHost)!=cudaSuccess ||
               !std::equal(output.begin(),output.begin()+resident_n,expected.begin())) return error("resident_readback",11,written);
            const auto resident_t1=std::chrono::steady_clock::now(); resident_readback_us+=micros(resident_t0,resident_t1);
        }
        written+=n; ++chunk_id;
        if(on_disk && (window_written==disk_window || written==total)) if(!finish_window()) return error("window_reread_hash",12,window_start);
    }
    if(expected_file.is_open()) { char extra=0; if(expected_file.get(extra)) return 6; }
    bool oracle_ok=true;
    if(oracle) { oracle_ok=std::fgetc(oracle)==EOF && pclose(oracle)==0; oracle=nullptr; }
    if(device_resident) resident_status="PASS_BYTE_EXACT_PER_CHUNK";
    const bool freed_resident=!device_resident || cudaFree(device_resident)==cudaSuccess;
    const bool freed_chunk=cudaFree(device_chunk)==cudaSuccess;
    const bool stream_ok=oracle_ok&&freed_chunk&&total_reread_bytes==total;
    const bool resident_ok=freed_resident;
    std::printf("{\"type\":\"result\",\"id\":\"fixture_pressure\",\"status\":\"%s\",\"stream_status\":\"%s\",\"resident_status\":\"%s\",\"total_bytes\":%llu,\"resident_bytes_requested\":%llu,\"resident_bytes_allocated\":%llu,\"resident_bytes_verified\":%llu,\"chunks\":%llu,\"chunk_bytes\":%llu,\"disk_window_bytes\":%llu,\"windows\":%llu,\"reread_bytes\":%llu,\"max_resident_host_bytes\":%llu,\"staging_device_bytes\":%llu,\"gpu_free_bytes_before_resident\":%llu,\"gpu_headroom_bytes\":%llu,\"disk_window_plus_reserve_bytes\":%llu,\"gpu_generate_us\":%llu,\"gpu_chunk_readback_us\":%llu,\"resident_readback_us\":%llu,\"dense_write_us\":%llu,\"fsync_us\":%llu,\"disk_readback_us\":%llu,\"dense_window_file\":\"%s\",\"oracle_window_file\":\"%s\"}\n",
        stream_ok&&resident_ok?(resident_requested&&!device_resident?"PASS_STAGING_ONLY":"PASS"):"FAIL",stream_ok?"PASS":"FAIL",resident_status.c_str(),(unsigned long long)total,(unsigned long long)resident_requested,(unsigned long long)resident_bytes,(unsigned long long)(device_resident?resident_bytes:0),
        (unsigned long long)chunk_id,(unsigned long long)chunk,(unsigned long long)disk_window,(unsigned long long)window_index,(unsigned long long)total_reread_bytes,(unsigned long long)(2*chunk),(unsigned long long)chunk,
        (unsigned long long)free_bytes,(unsigned long long)headroom,(unsigned long long)(2*disk_window+disk_reserve),
        (unsigned long long)gpu_generate_us,(unsigned long long)gpu_chunk_readback_us,(unsigned long long)resident_readback_us,
        (unsigned long long)total_write_us,(unsigned long long)total_fsync_us,(unsigned long long)total_readback_us,path.c_str(),oracle_window_path.c_str());
    return stream_ok&&resident_ok?0:10;
}
