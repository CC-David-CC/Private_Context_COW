#pragma once
#include "llama.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

// Explicit reference precision. The launcher fingerprints these settings and
// rejects any attempt to restore an FP16 checkpoint into an FP32 context.
static void qwen35_context_math(llama_context_params & params) {
    const char * flag = std::getenv("COW_QWEN35_F32_ATTENTION");
    if (!flag || std::string(flag) == "0") return;
    if (std::string(flag) != "1") throw std::runtime_error("Invalid FP32 attention flag");
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = params.type_v = GGML_TYPE_F32;
}
