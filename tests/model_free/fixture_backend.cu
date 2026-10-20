// Test-only native memory backend. Actual BranchPages; no model inference.
#include "fixture_data.hpp"
#include "strata/core/branch_pages.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/core/branch_gpu_compat.hpp"
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using strata::core::BranchPages;
using strata::core::BranchPageSnapshot;
static void gpu(cudaError_t code) { if (code!=cudaSuccess) throw std::runtime_error("GPU operation failed"); }
__global__ void fill_rows(uint8_t* p,const int32_t* ids,uint64_t count,uint64_t start,uint64_t seed,unsigned layer) {
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count*32) return;
    const auto at=start*32+i;
    p[at]=cow_fixture::byte_at(seed,0,layer,uint64_t(ids[i/32]),i%32,at);
}
__global__ void advance(uint32_t* r,const int32_t* ids,uint64_t n,uint64_t* executed) {
    if (blockIdx.x || threadIdx.x) return;
    auto v=*r; for (uint64_t i=0;i<n;++i) v=(v*257+uint32_t(ids[i])+1)&65535;
    *r=v;
    *executed+=n;
}
struct Part {
    int device;
    std::unique_ptr<BranchPages> pages;
    uint32_t* running=nullptr;
    uint64_t* executed=nullptr;
    explicit Part(int d):device(d) {
        gpu(cudaSetDevice(device));pages=std::make_unique<BranchPages>();
        gpu(cudaMalloc(&running,sizeof(uint32_t)));
        gpu(cudaMalloc(&executed,sizeof(uint64_t)));gpu(cudaMemset(executed,0,8));
    }
    ~Part() { gpu(cudaSetDevice(device)); pages.reset(); if(running)gpu(cudaFree(running));if(executed)gpu(cudaFree(executed)); }
};
struct Image {
    std::vector<BranchPageSnapshot> parts;
    std::vector<uint32_t> running;
    std::vector<int32_t> ids;
};
template<class T> static void write_value(std::ostream& f,T v) { f.write(reinterpret_cast<const char*>(&v),sizeof(v)); }
template<class T> static T read_value(std::istream& f) { T v{}; f.read(reinterpret_cast<char*>(&v),sizeof(v));if(!f)throw std::runtime_error("truncated fixture image");return v; }
static void frontier(uint32_t r) {
    std::cout<<"\"logits\":[";
    for(unsigned k=0;k<64;++k) { if(k)std::cout<<',';std::cout<<((r^(k*197))&65535); }
    std::cout<<']';
}
static void history(const std::vector<int32_t>& ids) {
    std::cout<<"\"history\":[";for(size_t i=0;i<ids.size();++i){if(i)std::cout<<',';std::cout<<ids[i];}std::cout<<']';
}
int main(int argc,char**argv) {
    uint64_t seed=0x4d465f434f575f31ULL,capacity=12288;
    std::vector<int> devices{0};
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--capacity"&&i+1<argc)capacity=std::stoull(argv[++i]);
        else if(arg=="--seed"&&i+1<argc)seed=std::stoull(argv[++i],nullptr,0);
        else if(arg=="--gpu"&&i+1<argc){devices.clear();std::string s=argv[++i];std::replace(s.begin(),s.end(),',',' ');std::istringstream f(s);int d;while(f>>d)devices.push_back(d);}
        else return 2;
    }
    if(devices.empty()||!capacity||capacity>1048576) return 2;
    std::vector<std::unique_ptr<Part>> parts;
    for(int d:devices)parts.push_back(std::make_unique<Part>(d));
    std::map<uint64_t,Image> images;uint64_t next_image=0,work=0,captures=0,restores=0,written=0,loaded=0;
    std::vector<int32_t> ids;bool valid=false;
    auto reset=[&] {
        for(auto& p:parts){gpu(cudaSetDevice(p->device));if(!p->pages->reserve(capacity*32))throw std::runtime_error("fixture reserve failed");uint32_t r=seed&65535;gpu(cudaMemcpy(p->running,&r,4,cudaMemcpyHostToDevice));}
        ids.clear();valid=true;
    };
    auto running=[&] {uint32_t r=0;gpu(cudaSetDevice(parts.front()->device));gpu(cudaMemcpy(&r,parts.front()->running,4,cudaMemcpyDeviceToHost));return r;};
    auto reply_state=[&] {std::cout<<"{\"ok\":true,\"consumed\":"<<ids.size()<<',';history(ids);std::cout<<',';frontier(running());std::cout<<"}\n";};
    auto capture=[&] {
        if(!valid||ids.empty())throw std::runtime_error("no complete consumed fixture state");
        Image image;image.ids=ids;
        for(auto& p:parts){gpu(cudaSetDevice(p->device));gpu(cudaDeviceSynchronize());auto snap=p->pages->freeze();if(!snap)throw std::runtime_error("freeze failed");image.parts.push_back(snap);uint32_t r;gpu(cudaMemcpy(&r,p->running,4,cudaMemcpyDeviceToHost));image.running.push_back(r);}
        return image;
    };
    std::string line;
    while(std::getline(std::cin,line)) {
        try {
            std::istringstream command(line);std::string op;command>>op;
            if(op=="ready") {std::cout<<"{\"ok\":true,\"vocab\":64,\"capacity\":"<<capacity<<",\"GPU\":[";for(size_t i=0;i<devices.size();++i){if(i)std::cout<<',';std::cout<<devices[i];}std::cout<<"]}\n";}
            else if(op=="reset") {reset();std::cout<<"{\"ok\":true,\"consumed\":0}\n";}
            else if(op=="consume") {
                uint64_t n=0;command>>n;
                if(!valid||!n||n>capacity-ids.size())throw std::runtime_error("invalid consume boundary");
                std::vector<int32_t> tokens(n);
                for(auto& t:tokens)if(!(command>>t)||t<0||t>=64)throw std::runtime_error("invalid fixture token");
                valid=false;
                for(size_t layer=0;layer<parts.size();++layer){auto& p=parts[layer];gpu(cudaSetDevice(p->device));const auto end=(ids.size()+n)*32;if(!p->pages->grow(end)||!p->pages->prepare_write(ids.size()*32,n*32))throw std::runtime_error("fixture COW failed");int32_t* input=nullptr;gpu(cudaMalloc(&input,n*4));gpu(cudaMemcpy(input,tokens.data(),n*4,cudaMemcpyHostToDevice));fill_rows<<<unsigned((n*32+255)/256),256>>>(p->pages->data(),input,n,ids.size(),seed,unsigned(layer));advance<<<1,1>>>(p->running,input,n,p->executed);auto status=cudaGetLastError();auto complete=cudaDeviceSynchronize();gpu(cudaFree(input));gpu(status);gpu(complete);}
                work+=n;ids.insert(ids.end(),tokens.begin(),tokens.end());valid=true;
                std::cout<<"{\"ok\":true,\"consumed\":"<<ids.size()<<"}\n";
            }
            else if(op=="frontier") {if(!valid)throw std::runtime_error("invalidated fixture lease");std::cout<<"{\"ok\":true,\"consumed\":"<<ids.size()<<',';frontier(running());std::cout<<"}\n";}
            else if(op=="capture") {auto image=capture();uint64_t size=image.ids.size()*4;for(auto& s:image.parts)for(auto& p:BranchPages::ownership(s))size+=p.bytes;auto id=++next_image;images.emplace(id,std::move(image));++captures;std::cout<<"{\"ok\":true,\"image\":"<<id<<",\"consumed\":"<<ids.size()<<",\"identity\":\""<<getpid()<<':'<<id<<':'<<seed<<"\",\"bytes\":"<<size<<"}\n";}
            else if(op=="restore") {uint64_t id;command>>id;auto& image=images.at(id);if(image.parts.size()!=parts.size())throw std::runtime_error("incomplete group image");valid=false;for(size_t i=0;i<parts.size();++i){gpu(cudaSetDevice(parts[i]->device));if(!parts[i]->pages->restore(image.parts[i]))throw std::runtime_error("fixture restore failed");gpu(cudaMemcpy(parts[i]->running,&image.running[i],4,cudaMemcpyHostToDevice));}ids=image.ids;valid=true;++restores;reply_state();}
            else if(op=="serialize") {
                uint64_t id;std::string path;command>>id>>path;auto& image=images.at(id);std::ofstream f(path,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot write fixture image");
                f.write("CFB1",4);write_value(f,seed);write_value(f,capacity);write_value(f,uint32_t(parts.size()));write_value(f,uint64_t(image.ids.size()));for(int d:devices)write_value(f,int32_t(d));for(int32_t t:image.ids)write_value(f,t);
                for(size_t i=0;i<parts.size();++i){
                    gpu(cudaSetDevice(parts[i]->device));auto pages=BranchPages::ownership(image.parts[i]);
                    uint64_t bytes=0;for(auto& page:pages)bytes+=page.bytes;
                    write_value(f,bytes);write_value(f,image.running[i]);
                    for(size_t j=0;j<pages.size();++j){std::vector<uint8_t> bytes;if(!BranchPages::read_page(image.parts[i],j,bytes))throw std::runtime_error("fixture image audit failed");f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());}
                }
                f.flush();if(!f)throw std::runtime_error("fixture write failed");auto bytes=uint64_t(f.tellp());f.close();written+=bytes;std::cout<<"{\"ok\":true,\"bytes\":"<<bytes<<"}\n";
            }
            else if(op=="import") {
                std::string path;command>>path;std::ifstream f(path,std::ios::binary);char magic[4];f.read(magic,4);
                if(!f||std::string(magic,4)!="CFB1"||read_value<uint64_t>(f)!=seed||read_value<uint64_t>(f)!=capacity||read_value<uint32_t>(f)!=parts.size())throw std::runtime_error("incompatible fixture image");
                uint64_t n=read_value<uint64_t>(f);if(!n||n>capacity)throw std::runtime_error("invalid saved consumed count");
                for(int device:devices)if(read_value<int32_t>(f)!=device)throw std::runtime_error("fixture topology mismatch");
                std::vector<int32_t> saved_ids(n);for(auto& t:saved_ids){t=read_value<int32_t>(f);if(t<0||t>=64)throw std::runtime_error("invalid saved token");}
                std::vector<std::vector<uint8_t>> payloads;std::vector<uint32_t> rs;
                for(auto& part:parts){gpu(cudaSetDevice(part->device));const auto gran=strata::core::vmm_granularity();uint64_t bytes=read_value<uint64_t>(f);if(bytes!=((n*32+gran-1)/gran)*gran)throw std::runtime_error("fixture extent mismatch");uint32_t r=read_value<uint32_t>(f);if(r>65535)throw std::runtime_error("invalid running state");rs.push_back(r);payloads.emplace_back(bytes);f.read(reinterpret_cast<char*>(payloads.back().data()),bytes);if(!f)throw std::runtime_error("truncated fixture payload");}
                auto bytes=uint64_t(f.tellg());if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("extra fixture bytes");
                valid=false;
                for(size_t i=0;i<parts.size();++i){auto& p=parts[i];gpu(cudaSetDevice(p->device));if(!p->pages->reserve(capacity*32)||!p->pages->grow(n*32))throw std::runtime_error("fixture load allocation failed");gpu(cudaMemcpy(p->pages->data(),payloads[i].data(),payloads[i].size(),cudaMemcpyHostToDevice));gpu(cudaMemcpy(p->running,&rs[i],4,cudaMemcpyHostToDevice));}
                ids=std::move(saved_ids);valid=true;++restores;loaded+=bytes;reply_state();
            }
            else if(op=="counters") {uint64_t maximum=0,total=0;for(auto& p:parts){gpu(cudaSetDevice(p->device));uint64_t n=0;gpu(cudaMemcpy(&n,p->executed,8,cudaMemcpyDeviceToHost));maximum=std::max(maximum,n);total+=n;}auto c=BranchPages::counters();std::cout<<"{\"ok\":true,\"forward_tokens\":"<<maximum<<",\"gpu_step_executions\":"<<total<<",\"captures\":"<<captures<<",\"restores\":"<<restores<<",\"serialized_bytes\":"<<written<<",\"loaded_bytes\":"<<loaded<<",\"allocations\":"<<c.allocations<<",\"releases\":"<<c.releases<<",\"live_bytes\":"<<c.live_bytes<<",\"cow_pages\":"<<c.cow_pages<<",\"copied_bytes\":"<<c.copied_bytes<<",\"mappings\":"<<c.mappings<<",\"unmappings\":"<<c.unmappings<<"}\n";}
            else if(op=="drop") {uint64_t id;command>>id;images.erase(id);std::cout<<"{\"ok\":true}\n";}
            else if(op=="quit") {std::cout<<"{\"ok\":true}\n";std::cout.flush();break;}
            else throw std::runtime_error("unknown fixture operation");
        } catch(const std::exception& e) {std::cout<<"{\"ok\":false,\"error\":\""<<e.what()<<"\"}\n";}
        std::cout.flush();
    }
    return 0;
}
