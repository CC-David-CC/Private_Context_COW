// Qwen3.5 text state adapter. Whole-state copies, not physical page sharing.
// llama_state_* in the pinned engine stores hybrid memory but NOT next logits.
#include "llama.h"
#include "qwen35_device.h"
#include "qwen35_math.h"
#include "qwen35_lora.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
static thread_local std::string error_text;
struct Lease { llama_context *ctx; int vocab; uint64_t work=0, calls=0; bool valid=true; std::vector<float> frontier; };
static void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
#ifdef PCC_CTT_LLAMA_DENSE
static constexpr uint64_t CTT_STATE_MAGIC=0x3154534c4c414d41ULL;
#else
static constexpr uint64_t CTT_STATE_MAGIC=0x3154533533574f43ULL;
#endif
extern "C" {
const char *ctt_architecture_contract(){
#ifdef PCC_CTT_LLAMA_DENSE
 return "llama-dense-v1";
#else
 return "qwen35-hybrid-v1";
#endif
}
void *ctt_lora_load(void*m,const char*path){try{return qwen35_load_adapter((llama_model*)m,path);}catch(const std::exception&e){error_text=e.what();return nullptr;}}
void ctt_lora_free(void*a){llama_adapter_lora_free((llama_adapter_lora*)a);}
int ctt_lora_attach(void*x,void*a,float scale){try{qwen35_attach_adapter(((Lease*)x)->ctx,(llama_adapter_lora*)a,scale);return 0;}catch(const std::exception&e){error_text=e.what();return -1;}}
const char *ctt_error(){return error_text.c_str();}
static void *ctt_load_impl(const char *path,const char *device){
 try {
  ggml_backend_load_all();auto p=llama_model_default_params();p.n_gpu_layers=99;
  ggml_backend_dev_t devices[2]={};qwen35_select_device(p,device,devices);
  auto *m=llama_model_load_from_file(path,p);require(m,"Model load failed");
  char arch[64]={};llama_model_meta_val_str(m,"general.architecture",arch,sizeof(arch));
#ifdef PCC_CTT_LLAMA_DENSE
  if(std::string(arch)!="llama" || llama_model_is_hybrid(m) || llama_model_has_encoder(m)){
   llama_model_free(m);throw std::runtime_error("Requires qualified dense Llama text decoder");}
#else
  if((std::string(arch)!="qwen35" && std::string(arch)!="wen1") || !llama_model_is_hybrid(m) || llama_model_has_encoder(m)){
   llama_model_free(m);throw std::runtime_error("Requires qwen35 hybrid text decoder");}
#endif
  return m;
 }catch(const std::exception&e){error_text=e.what();return nullptr;}
}
void *ctt_load(const char *path){return ctt_load_impl(path,nullptr);}
void *ctt_load_device(const char *path,const char *device){
 if(!device||!*device){error_text="Missing CUDA device";return nullptr;}
 return ctt_load_impl(path,device);
}
void ctt_unload(void *m){llama_model_free((llama_model*)m);}
int ctt_vocab(void *m){return llama_vocab_n_tokens(llama_model_get_vocab((llama_model*)m));}
int ctt_meta(void *m,const char *key,char *dst,size_t n){return llama_model_meta_val_str((llama_model*)m,key,dst,n);}
int ctt_piece(void *m,int t,char *dst,int n){return llama_token_to_piece(llama_model_get_vocab((llama_model*)m),t,dst,n,0,true);}
bool ctt_eog(void *m,int t){return llama_vocab_is_eog(llama_model_get_vocab((llama_model*)m),t);}
int ctt_tokenize(void *m,const char *text,int len,int32_t *out,int n,bool add,bool special){return llama_tokenize(llama_model_get_vocab((llama_model*)m),text,len,out,n,add,special);}
void *ctt_new_config(void *m,int capacity,int batch,int threads){
 try{
  require(capacity>=128 && capacity<=262144 && batch>0 && batch<=capacity && threads>0,"Invalid context geometry");
  auto p=llama_context_default_params();p.n_ctx=capacity;p.n_batch=batch;p.n_ubatch=batch;p.n_seq_max=1;
  p.n_threads=p.n_threads_batch=threads;p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
  p.type_k=p.type_v=GGML_TYPE_F16;p.no_perf=false;
  qwen35_context_math(p);
  auto *ctx=llama_init_from_model((llama_model*)m,p);require(ctx,"Context creation failed");
  return new Lease{ctx,ctt_vocab(m)};
 }catch(const std::exception&e){error_text=e.what();return nullptr;}
}
void *ctt_new(void*m){return ctt_new_config(m,4096,512,8);}
void ctt_free(void*x){auto*l=(Lease*)x;llama_free(l->ctx);delete l;}
int ctt_position(void*x){return llama_memory_seq_pos_max(llama_get_memory(((Lease*)x)->ctx),0);}
int ctt_consume(void*x,const int32_t*tokens,int count){
 auto*l=(Lease*)x;
 try{
  require(l->valid,"Poisoned lease");require(count>0 && (uint32_t)count<=llama_n_batch(l->ctx),"Invalid native batch");
  int start=ctt_position(x)+1;require(start+count<=(int)llama_n_ctx(l->ctx),"Context exceeded");
  for(int i=0;i<count;i++)require(tokens[i]>=0 && tokens[i]<l->vocab,"Invalid token");
  auto b=llama_batch_init(count,0,1);b.n_tokens=count;
  for(int i=0;i<count;i++){b.token[i]=tokens[i];b.pos[i]=start+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=(i==count-1);}
  const int rc=llama_decode(l->ctx,b);llama_batch_free(b);
  require(rc==0,"Native forward failed; discard lease");l->work+=count;l->calls++;
  const float *logits=llama_get_logits_ith(l->ctx,-1);require(logits,"Missing frontier");
  l->frontier.assign(logits,logits+l->vocab);
  for(float v:l->frontier)require(std::isfinite(v),"Nonfinite frontier");
  require(ctt_position(x)==start+count-1,"Consumed boundary mismatch");return 0;
 }catch(const std::exception&e){l->valid=false;error_text=e.what();return -1;}
}
const float *ctt_logits(void*x){auto*l=(Lease*)x;return l->valid && !l->frontier.empty()?l->frontier.data():nullptr;}
uint64_t ctt_work(void*x){return ((Lease*)x)->work;}
uint64_t ctt_calls(void*x){return ((Lease*)x)->calls;}
// Header: magic, native-memory bytes, frontier count, followed by both payloads.
size_t ctt_state_size(void*x){auto*l=(Lease*)x;return l->valid && !l->frontier.empty()?24+llama_state_get_size(l->ctx)+4*l->frontier.size():0;}
size_t ctt_export(void*x,void*out,size_t size){
 auto*l=(Lease*)x;
 try{
  size_t need=ctt_state_size(x);require(need>24 && size==need,"Wrong state export size");
  uint64_t header[3]={CTT_STATE_MAGIC,llama_state_get_size(l->ctx),(uint64_t)l->vocab};
  auto*p=(uint8_t*)out;memcpy(p,header,24);
  require(llama_state_get_data(l->ctx,p+24,header[1])==header[1],"Incomplete native export");
  memcpy(p+24+header[1],l->frontier.data(),4*l->vocab);return need;
 }catch(const std::exception&e){error_text=e.what();return 0;}
}
size_t ctt_import(void*x,const void*in,size_t size){
 auto*l=(Lease*)x;
 try{
  require(l->valid && size>=24,"Invalid state import");uint64_t h[3];memcpy(h,in,24);
  require(h[0]==CTT_STATE_MAGIC && h[2]==(uint64_t)l->vocab,"State format/vocab mismatch");
  require(h[1]<=size-24 && size-24-h[1]==4*h[2],"Truncated or trailing state bytes");
  auto*p=(const uint8_t*)in;std::vector<float>f(l->vocab);memcpy(f.data(),p+24+h[1],4*h[2]);
  for(float v:f)require(std::isfinite(v),"Invalid saved frontier");
  require(llama_state_set_data(l->ctx,p+24,h[1])==h[1],"Incomplete hybrid memory import");
  l->frontier=std::move(f);return size;
 }catch(const std::exception&e){l->valid=false;error_text=e.what();return 0;}
}
}
