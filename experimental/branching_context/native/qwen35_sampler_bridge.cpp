// Reuse the existing native selectors; no Python reference on the hot path.
#include "qwen35_bridge.cpp"
#include "min_p_select.h"
#include "grammar_select.h"
extern "C" {
int ctt_select_logits_v2(const float *scores,size_t vocab,const uint8_t *mask,size_t bytes,
 double temperature,double min_p,uint64_t draw,size_t top_k,double top_p){
 try{return (int)min_p_select::select(scores,vocab,mask,bytes,temperature,min_p,draw,top_k,top_p).token;}
 catch(const std::exception&e){error_text=e.what();return -1;}
}
int ctt_sample_mask_v2(void *lease,const uint8_t *mask,size_t bytes,double temperature,
 double min_p,uint64_t draw,size_t top_k,double top_p){
 auto*l=(Lease*)lease;
 if(!l->valid || l->frontier.empty()){error_text="Missing consumed frontier";return -1;}
 return ctt_select_logits_v2(l->frontier.data(),l->vocab,mask,bytes,temperature,min_p,draw,top_k,top_p);
}
int ctt_choose_mask(void *lease,const uint8_t *mask,size_t bytes){
 auto*l=(Lease*)lease;
 try{require(l->valid && !l->frontier.empty(),"Missing consumed frontier");
 return (int)grammar_select::select(l->frontier.data(),l->vocab,mask,bytes,ctt_position(lease)+1,ctt_position(lease)+1).token;}
 catch(const std::exception&e){error_text=e.what();return -1;}
}
}
