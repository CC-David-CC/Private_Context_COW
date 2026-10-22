#pragma once
#include "ggml-backend.h"
#include <stdexcept>
#include <string>

// Restrict each independent worker to its own CUDA device. No environment mutation
// and no implicit layer split over the other GPUs visible in this process.
static void qwen35_select_device(llama_model_params & params, const char * name,
                                ggml_backend_dev_t (&devices)[2]) {
    if (!name) return; // preserve the legacy load path
    if (std::string(name).rfind("CUDA", 0) != 0)
        throw std::runtime_error("Dense worker requires an explicit CUDA device");
    auto device = ggml_backend_dev_by_name(name);
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU)
        throw std::runtime_error("Selected CUDA device is unavailable");
    devices[0] = device;
    devices[1] = nullptr;
    params.devices = devices;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu = 0; // index within this model's one-element device list
}
