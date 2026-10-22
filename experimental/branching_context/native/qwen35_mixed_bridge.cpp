// Version 2 mixed state: logical cells, rotary cursor and incremental native input proofs.
// Whole-state copies, not physical page sharing. Version 1 text images stay distinct.
// llama_state_* in the pinned engine stores hybrid memory but NOT next logits.
#include "llama.h"
#include "qwen35_device.h"
#include "qwen35_math.h"
#include "qwen35_lora.h"
extern "C" {
#include "sha256/sha256.h"
}
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
static thread_local std::string error_text;
struct Lease { llama_context *ctx; int vocab; uint64_t work=0, calls=0; bool valid=true; std::vector<float> frontier; uint64_t cells=0,rotary=0; sha256_t cells_hash{},positions_hash{}; };
static void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
static constexpr char CELL_TAG[]="COW-QWEN35-CELLS-v2";
static constexpr char POS_TAG[]="COW-QWEN35-POS-v2";
static void update(sha256_t &s,const void*p,size_t n){sha256_update(&s,(const unsigned char*)p,n);}
static void position_proof(Lease*l,int t,int h,int w){int32_t v[3]={t,h,w};update(l->positions_hash,v,sizeof(v));}
struct StateHeader{uint64_t magic,native_bytes,vocab,cells,rotary;sha256_t cells_hash,positions_hash;};
static constexpr uint64_t STATE_MAGIC=0x3254533533574f43ULL;
extern "C" {
int ctt_state_version(){return 2;}
void *ctt_lora_load(void*m,const char*path){try{return qwen35_load_adapter((llama_model*)m,path);}catch(const std::exception&e){error_text=e.what();return nullptr;}}
void ctt_lora_free(void*a){llama_adapter_lora_free((llama_adapter_lora*)a);}
int ctt_lora_attach(void*x,void*a,float scale){try{qwen35_attach_adapter(((Lease*)x)->ctx,(llama_adapter_lora*)a,scale);return 0;}catch(const std::exception&e){error_text=e.what();return -1;}}
void ctt_poison(void*x){auto*l=(Lease*)x;l->valid=false;l->frontier.clear();}
int ctt_temporal_position(void*x){return llama_memory_seq_pos_max(llama_get_memory(((Lease*)x)->ctx),0);}
int ctt_proof(void*x,unsigned char*out,int*consumed,int*rotary){
 auto*l=(Lease*)x;if(!l->valid){error_text="Poisoned lease";return -1;}
 auto a=l->cells_hash,b=l->positions_hash;sha256_final(&a,out);sha256_final(&b,out+32);*consumed=l->cells;*rotary=l->rotary;return 0;
}
const char *ctt_error(){return error_text.c_str();}
static void *ctt_load_impl(const char *path,const char *device){
 try {
  ggml_backend_load_all();auto p=llama_model_default_params();p.n_gpu_layers=99;
  ggml_backend_dev_t devices[2]={};qwen35_select_device(p,device,devices);
  auto *m=llama_model_load_from_file(path,p);require(m,"Model load failed");
  char arch[64]={};llama_model_meta_val_str(m,"general.architecture",arch,sizeof(arch));
  if((std::string(arch)!="qwen35" && std::string(arch)!="wen1") || !llama_model_is_hybrid(m) || llama_model_has_encoder(m)){
   llama_model_free(m);throw std::runtime_error("Requires qwen35 hybrid text decoder");}
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
  auto*l=new Lease{ctx,ctt_vocab(m)};sha256_init(&l->cells_hash);sha256_init(&l->positions_hash);update(l->cells_hash,CELL_TAG,sizeof(CELL_TAG));update(l->positions_hash,POS_TAG,sizeof(POS_TAG));return l;
 }catch(const std::exception&e){error_text=e.what();return nullptr;}
}
void *ctt_new(void*m){return ctt_new_config(m,4096,512,8);}
void ctt_free(void*x){auto*l=(Lease*)x;llama_free(l->ctx);delete l;}
int ctt_position(void*x){return int(((Lease*)x)->cells)-1;}
int ctt_consume(void*x,const int32_t*tokens,int count){
 auto*l=(Lease*)x;
 try{
  require(l->valid,"Poisoned lease");require(count>0 && (uint32_t)count<=llama_n_batch(l->ctx),"Invalid native batch");
  int start=l->rotary;require(l->cells+count<=llama_n_ctx(l->ctx),"Context exceeded");
  for(int i=0;i<count;i++)require(tokens[i]>=0 && tokens[i]<l->vocab,"Invalid token");
  auto b=llama_batch_init(count,0,1);b.n_tokens=count;
  for(int i=0;i<count;i++){b.token[i]=tokens[i];b.pos[i]=start+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=(i==count-1);}
  const int rc=llama_decode(l->ctx,b);llama_batch_free(b);
  require(rc==0,"Native forward failed; discard lease");l->work+=count;l->calls++;
  for(int i=0;i<count;i++){const char tag='T';update(l->cells_hash,&tag,1);update(l->cells_hash,tokens+i,4);position_proof(l,start+i,start+i,start+i);}
  l->cells+=count;l->rotary+=count;
  const float *logits=llama_get_logits_ith(l->ctx,-1);require(logits,"Missing frontier");
  l->frontier.assign(logits,logits+l->vocab);
  for(float v:l->frontier)require(std::isfinite(v),"Nonfinite frontier");
  require(ctt_temporal_position(x)==start+count-1,"Consumed temporal boundary mismatch");return 0;
 }catch(const std::exception&e){l->valid=false;error_text=e.what();return -1;}
}
const float *ctt_logits(void*x){auto*l=(Lease*)x;return l->valid && !l->frontier.empty()?l->frontier.data():nullptr;}
uint64_t ctt_work(void*x){return ((Lease*)x)->work;}
uint64_t ctt_calls(void*x){return ((Lease*)x)->calls;}
// Image bytes are already admitted by COW; independently hash and validate the
// actual SVE1 payload before any forward. Each batch uses its original grid.
int ctt_consume_image(void*x,const void*data,size_t bytes,int pad,const char*expected){
 auto*l=(Lease*)x;try{
  require(l->valid&&data&&bytes>=20&&pad>=0&&pad<l->vocab,"Invalid image input");int32_t h[5];memcpy(h,data,20);
  int rows=h[1],nx=h[2],ny=h[3],width=h[4];require(h[0]==0x31455653&&rows>0&&rows<=1024&&nx>0&&ny>0&&int64_t(nx)*ny==rows&&width==4096,"Unsupported image geometry");
  require(bytes==20+size_t(rows)*width*4&&l->cells+rows<=llama_n_ctx(l->ctx),"Image size/capacity");
  unsigned char digest[32];sha256_hash(digest,(const unsigned char*)data,bytes);std::string actual;const char*hex="0123456789abcdef";for(auto c:digest){actual+=hex[c>>4];actual+=hex[c&15];}require(expected&&actual==expected,"Image byte checksum mismatch");
  auto values=(const unsigned char*)data+20;for(size_t i=0;i<size_t(rows)*width;i++){float v;memcpy(&v,values+i*4,4);require(std::isfinite(v),"Nonfinite image input");}
  int base=l->rotary;const char tag='I';update(l->cells_hash,&tag,1);int32_t meta[]={rows,nx,ny,width,pad};update(l->cells_hash,meta,sizeof(meta));update(l->cells_hash,actual.data(),actual.size());l->frontier.clear();
  for(int start=0;start<rows;start+=llama_n_batch(l->ctx)){
   int n=std::min(rows-start,int(llama_n_batch(l->ctx)));std::vector<float>embedding(size_t(n)*width);memcpy(embedding.data(),values+size_t(start)*width*4,size_t(n)*width*4);
   std::vector<llama_pos>pos(4*n);std::vector<int32_t>counts(n,1);llama_seq_id seq=0;std::vector<llama_seq_id*>seqs(n,&seq);std::vector<int8_t>outputs(n,0);
   for(int j=0;j<n;j++){int row=start+j;pos[j]=base;pos[n+j]=base+row/nx;pos[2*n+j]=base+row%nx;pos[3*n+j]=0;}
   llama_batch batch{};batch.n_tokens=n;batch.embd=embedding.data();batch.pos=pos.data();batch.n_seq_id=counts.data();batch.seq_id=seqs.data();batch.logits=outputs.data();
   require(llama_decode(l->ctx,batch)==0,"Native image forward failed");l->work+=n;l->calls++;l->cells+=n;
   for(int j=0;j<n;j++)position_proof(l,pos[j],pos[n+j],pos[2*n+j]);
  }
  l->rotary+=std::max(nx,ny);return 0;
 }catch(const std::exception&e){l->valid=false;error_text=e.what();return -1;}
}
size_t ctt_state_size(void*x){auto*l=(Lease*)x;return l->valid&&!l->frontier.empty()?sizeof(StateHeader)+llama_state_get_size(l->ctx)+4*l->frontier.size():0;}
size_t ctt_export(void*x,void*out,size_t size){auto*l=(Lease*)x;try{
 size_t need=ctt_state_size(x);require(need>sizeof(StateHeader)&&size==need,"Wrong mixed export size");require(l->cells>0&&l->rotary>0&&ctt_temporal_position(x)==int(l->rotary)-1,"Incomplete image frontier");
 StateHeader h{STATE_MAGIC,llama_state_get_size(l->ctx),(uint64_t)l->vocab,l->cells,l->rotary,l->cells_hash,l->positions_hash};auto*p=(uint8_t*)out;memcpy(p,&h,sizeof(h));
 require(llama_state_get_data(l->ctx,p+sizeof(h),h.native_bytes)==h.native_bytes,"Incomplete mixed native export");memcpy(p+sizeof(h)+h.native_bytes,l->frontier.data(),4*l->vocab);return need;
 }catch(const std::exception&e){error_text=e.what();return 0;}}
size_t ctt_import(void*x,const void*in,size_t size){auto*l=(Lease*)x;try{
 require(l->valid&&size>=sizeof(StateHeader),"Invalid mixed import");StateHeader h;memcpy(&h,in,sizeof(h));require(h.magic==STATE_MAGIC&&h.vocab==(uint64_t)l->vocab,"Mixed state format/vocab mismatch");
 require(h.cells>0&&h.cells<=llama_n_ctx(l->ctx)&&h.rotary>0&&h.rotary<=h.cells,"Invalid saved cursors");require(h.positions_hash.count==sizeof(POS_TAG)+12*h.cells&&h.cells_hash.count>=sizeof(CELL_TAG)&&h.cells_hash.count<=sizeof(CELL_TAG)+85*h.cells,"Invalid saved proof lengths");
 require(h.native_bytes<=size-sizeof(h)&&size-sizeof(h)-h.native_bytes==4*h.vocab,"Truncated or trailing mixed state");auto*p=(const uint8_t*)in;std::vector<float>f(l->vocab);memcpy(f.data(),p+sizeof(h)+h.native_bytes,4*h.vocab);for(float v:f)require(std::isfinite(v),"Invalid mixed frontier");
 require(llama_state_set_data(l->ctx,p+sizeof(h),h.native_bytes)==h.native_bytes,"Incomplete hybrid state import");require(ctt_temporal_position(x)==int(h.rotary)-1,"Saved temporal cursor differs from native memory");
 l->cells=h.cells;l->rotary=h.rotary;l->cells_hash=h.cells_hash;l->positions_hash=h.positions_hash;l->frontier=std::move(f);return size;
 }catch(const std::exception&e){l->valid=false;error_text=e.what();return 0;}}
}
