// Fresh mixed-input oracle: no adapter, checkpoint, cache or state-import code.
#include "llama.h"
#include "qwen35_device.h"
#include "qwen35_math.h"
#include "qwen35_lora.h"
#include "nlohmann/json.hpp"
extern "C" {
#include "sha256/sha256.h"
}
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
static void check(bool v,const char *s){if(!v)throw std::runtime_error(s);}
template<class T>static void send(T v){std::cout.write((char*)&v,sizeof(v));}
static std::string hexhash(const std::vector<uint8_t>&b){unsigned char h[32];sha256_hash(h,b.data(),b.size());std::string s;const char *digits="0123456789abcdef";for(auto c:h){s+=digits[c>>4];s+=digits[c&15];}return s;}
struct Image{std::vector<float> values;};
struct Cell{int token,image,row;std::array<int32_t,3> pos;};
int main(int argc,char**argv){try{
 bool has_lora=argc==8||argc==9;int base_argc=has_lora?argc-3:argc;
 check(base_argc==5||base_argc==6,"Expected model, capacity, batch, threads and optional CUDA device/LoRA");
 if(has_lora)check(std::string(argv[base_argc])=="--lora","Unknown oracle adapter option");
 ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=99;
 ggml_backend_dev_t devices[2]={};qwen35_select_device(mp,base_argc==6?argv[5]:nullptr,devices);
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);check(bool(model),"Cannot load model");
 char arch[64]={};llama_model_meta_val_str(model.get(),"general.architecture",arch,sizeof(arch));check((std::string(arch)=="qwen35"||std::string(arch)=="wen1")&&llama_model_is_hybrid(model.get())&&llama_model_n_embd_inp(model.get())==4096,"Unsupported decoder");
 std::unique_ptr<llama_adapter_lora,decltype(&llama_adapter_lora_free)>adapter(has_lora?qwen35_load_adapter(model.get(),argv[base_argc+1]):nullptr,llama_adapter_lora_free);
 float adapter_scale=has_lora?std::stof(argv[base_argc+2]):1.f;
 auto vocab=llama_model_get_vocab(model.get());int nv=llama_vocab_n_tokens(vocab),capacity=std::stoi(argv[2]),limit=std::stoi(argv[3]),threads=std::stoi(argv[4]);check(capacity>=128&&capacity<=262144&&limit>0&&limit<=capacity&&threads>0,"Invalid context");
 std::string line;
 while(std::getline(std::cin,line)){
  if(line.rfind("T ",0)==0){auto t=line.substr(2);int n=-llama_tokenize(vocab,t.data(),t.size(),nullptr,0,true,false);check(n>0,"Tokenizer size");std::vector<llama_token>ids(n);check(llama_tokenize(vocab,t.data(),t.size(),ids.data(),n,true,false)==n,"Tokenizer output");send<int32_t>(n);std::cout.write((char*)ids.data(),n*4);std::cout.flush();continue;}
  check(line.rfind("M ",0)==0,"Unknown oracle operation");auto q=json::parse(line.substr(2));check(q.at("schema")==1,"Unknown oracle manifest");
  std::vector<Cell> cells;std::vector<Image> images;int cursor=0;
  for(auto &span:q.at("spans")){
   if(span.at("kind")=="text"){
    for(auto &v:span.at("tokens")){int t=v.get<int>();check(t>=0&&t<nv,"Token out of range");cells.push_back({t,-1,0,{cursor,cursor,cursor}});cursor++;}
   }else{
    check(span.at("kind")=="image","Unknown input kind");int rows=span.at("rows"),nx=span.at("grid_width"),ny=span.at("grid_height"),width=span.at("width"),pad=span.at("pad_token_id");
    check(rows>0&&rows<=1024&&nx>0&&ny>0&&int64_t(nx)*ny==rows&&width==4096&&pad>=0&&pad<nv,"Image geometry");
    std::string digest=span.at("embedding_sha256"),path=q.at("embedding_paths").at(digest);std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f),"Missing image payload");auto bytes=f.tellg();check(bytes==20+int64_t(rows)*width*4,"SVE1 byte count");std::vector<uint8_t>b(size_t(bytes),0);f.seekg(0);f.read((char*)b.data(),b.size());check(bool(f)&&hexhash(b)==digest,"Image payload checksum");int32_t h[5];memcpy(h,b.data(),20);check(h[0]==0x31455653&&h[1]==rows&&h[2]==nx&&h[3]==ny&&h[4]==width,"Image header mismatch");
    Image im;im.values.resize(size_t(rows)*width);memcpy(im.values.data(),b.data()+20,rows*width*4);for(float x:im.values)check(std::isfinite(x),"Nonfinite image");int id=images.size();images.push_back(std::move(im));
    for(int row=0;row<rows;++row)cells.push_back({pad,id,row,{cursor,cursor+row/nx,cursor+row%nx}});cursor+=std::max(nx,ny);
   }
   check(cells.size()<=size_t(capacity),"History capacity");
  }
  check(!cells.empty()&&cells.back().image<0,"Public frontier must end with text");
  auto cp=llama_context_default_params();cp.n_ctx=capacity;cp.n_batch=cp.n_ubatch=limit;cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=threads;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.no_perf=false;
  qwen35_context_math(cp);
  std::unique_ptr<llama_context,decltype(&llama_free)>ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx),"Fresh context creation");check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"Oracle context was not empty");
  if(adapter)qwen35_attach_adapter(ctx.get(),adapter.get(),adapter_scale);
  int offset=0;uint64_t work=0;
  for(auto &chunk:q.at("chunks")){
   check(chunk.size()==2,"Chunk shape");int start=chunk[0],n=chunk[1];check(start==offset&&n>0&&n<=limit&&start+n<=int(cells.size()),"Invalid native boundary");int image=cells[start].image;
   std::vector<llama_token>tokens;std::vector<float>embedding;std::vector<llama_pos>pos(n*4);std::vector<int32_t>seqcount(n,1);llama_seq_id seq=0;std::vector<llama_seq_id*>seqs(n,&seq);std::vector<int8_t>outputs(n,0);
   if(image<0)tokens.resize(n);else embedding.resize(size_t(n)*4096);
   for(int j=0;j<n;++j){auto &c=cells[start+j];check(c.image==image,"Batch crosses image/text boundary");
    if(image<0)tokens[j]=c.token;else memcpy(embedding.data()+size_t(j)*4096,images[image].values.data()+size_t(c.row)*4096,4096*4);
    for(int axis=0;axis<3;axis++)pos[axis*n+j]=c.pos[axis];pos[3*n+j]=image<0?c.pos[0]:0;
   }
   outputs[n-1]=image<0;llama_batch batch{};batch.n_tokens=n;batch.token=image<0?tokens.data():nullptr;batch.embd=image<0?nullptr:embedding.data();batch.pos=pos.data();batch.n_seq_id=seqcount.data();batch.seq_id=seqs.data();batch.logits=outputs.data();
   check(llama_decode(ctx.get(),batch)==0,"Fresh mixed forward failed");work+=n;offset+=n;
  }
  check(offset==int(cells.size()),"Missing native inputs");auto actual=llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0);check(actual==cursor-1,"Fresh temporal position mismatch");const float*frontier=llama_get_logits_ith(ctx.get(),-1);check(frontier,"Missing fresh frontier");for(int i=0;i<nv;++i)check(std::isfinite(frontier[i]),"Nonfinite fresh frontier");
  send<int32_t>(nv);send<uint64_t>(work);send<int32_t>(actual);std::cout.write((const char*)frontier,nv*4);std::cout.flush();
 }
 return 0;
}catch(const std::exception&e){std::cerr<<"MIXED ORACLE: "<<e.what()<<std::endl;return 2;}}
