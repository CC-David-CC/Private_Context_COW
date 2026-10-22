#pragma once
#include "llama.h"
#include <cmath>
#include <stdexcept>

// Model owns the base weights; caller releases all contexts before the adapter,
// and releases the adapter before its model. Attach only to an empty context.
static inline llama_adapter_lora * qwen35_load_adapter(llama_model *model,const char *path) {
    if(!model||!path||!*path)throw std::runtime_error("Missing LoRA model/path");
    auto *adapter=llama_adapter_lora_init(model,path);
    if(!adapter)throw std::runtime_error("Native LoRA load failed");
    return adapter;
}
static inline void qwen35_attach_adapter(llama_context *ctx,llama_adapter_lora *adapter,float scale) {
    if(!ctx||!adapter||!std::isfinite(scale)||scale<=0||scale>16)
        throw std::runtime_error("Invalid LoRA binding");
    if(llama_memory_seq_pos_max(llama_get_memory(ctx),0)!=-1)
        throw std::runtime_error("LoRA must be bound before any context input");
    if(llama_set_adapters_lora(ctx,&adapter,1,&scale)!=0)
        throw std::runtime_error("Native LoRA attachment failed");
}
