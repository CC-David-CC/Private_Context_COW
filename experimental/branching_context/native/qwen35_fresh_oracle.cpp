// Independent oracle: each forward allocates a new context and consumes all inputs.
#include "llama.h"
#include "qwen35_device.h"
#include "qwen35_math.h"
#include "qwen35_lora.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void send(T value) {
    std::cout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

int main(int argc, char ** argv) {
    try {
        bool has_lora=argc==8||argc==9;int base_argc=has_lora?argc-3:argc;
        if (base_argc != 5 && base_argc != 6) throw std::runtime_error("Expected model, capacity, batch, thread count and optional CUDA device/LoRA");
        if(has_lora&&std::string(argv[base_argc])!="--lora")throw std::runtime_error("Unknown oracle adapter option");
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        ggml_backend_dev_t devices[2]={};qwen35_select_device(mp,base_argc==6?argv[5]:nullptr,devices);
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Model load failed");
        char arch[64]={}; llama_model_meta_val_str(model,"general.architecture",arch,sizeof(arch));
        if ((std::string(arch)!="qwen35" && std::string(arch)!="wen1") || !llama_model_is_hybrid(model) || llama_model_has_encoder(model)) {
            throw std::runtime_error("Oracle requires qwen35 hybrid decoder");
        }
        std::unique_ptr<llama_adapter_lora,decltype(&llama_adapter_lora_free)>adapter(has_lora?qwen35_load_adapter(model,argv[base_argc+1]):nullptr,llama_adapter_lora_free);
        float adapter_scale=has_lora?std::stof(argv[base_argc+2]):1.f;
        auto * vocab = llama_model_get_vocab(model);
        const int32_t nv = llama_vocab_n_tokens(vocab);
        const int capacity=std::stoi(argv[2]), max_batch=std::stoi(argv[3]), threads=std::stoi(argv[4]);
        if(capacity<128 || max_batch<1 || max_batch>capacity || threads<1) throw std::runtime_error("Invalid oracle configuration");
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.rfind("T ", 0) == 0) {
                auto text = line.substr(2);
                int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
                std::vector<llama_token> ids(n);
                n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), true, false);
                if (n <= 0) throw std::runtime_error("Tokenization failed");
                send<int32_t>(n);
                std::cout.write(reinterpret_cast<const char *>(ids.data()), n*sizeof(llama_token));
                std::cout.flush();
                continue;
            }
            std::istringstream request(line);
            char command; int count=0; std::vector<int> boundaries;
            request >> command;
            if(command=='S') {
                int nb; request >> nb;
                if(nb<1 || nb>capacity) throw std::runtime_error("Invalid schedule length");
                for(int i=0;i<nb;i++) { int n=0;request>>n; if(n<1 || n>max_batch) throw std::runtime_error("Invalid batch boundary");boundaries.push_back(n);count+=n; }
            } else if(command=='F') {
                int chunk=0;request>>chunk>>count;
                if(chunk<1 || chunk>max_batch)throw std::runtime_error("Invalid chunk");
                for(int n=0;n<count;n+=chunk)boundaries.push_back(std::min(chunk,count-n));
            } else throw std::runtime_error("Unknown forward command");
            if(count<1 || count>capacity)throw std::runtime_error("Invalid history length");
            std::vector<llama_token> tokens(count);
            for (auto & token : tokens) {
                if (!(request >> token) || token < 0 || token >= nv) throw std::runtime_error("Invalid token");
            }
            auto cp = llama_context_default_params();
            cp.n_ctx = capacity;
            cp.n_batch = max_batch;
            cp.n_ubatch = max_batch;
            cp.n_seq_max = 1;
            cp.n_threads = cp.n_threads_batch = threads;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
  qwen35_context_math(cp);
            cp.no_perf = false;
            auto * ctx = llama_init_from_model(model, cp);
            if (!ctx) throw std::runtime_error("Fresh context allocation failed");
            if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != -1) throw std::runtime_error("Oracle context was not empty");
            if(adapter)qwen35_attach_adapter(ctx,adapter.get(),adapter_scale);
            uint64_t processed = 0;
            int start=0;
            for (int n : boundaries) {
                auto batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int j=0; j<n; ++j) {
                    batch.token[j] = tokens[start+j];
                    batch.pos[j] = start+j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = (j == n-1);
                }
                int rc = llama_decode(ctx, batch);
                llama_batch_free(batch);
                if (rc != 0) throw std::runtime_error("Oracle decode failed");
                processed += n; start += n;
            }
            const int32_t position = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            auto * logits = llama_get_logits_ith(ctx, -1);
            if (!logits || position != count-1) throw std::runtime_error("Oracle frontier mismatch");
            send<int32_t>(nv);
            send<uint64_t>(processed);
            send<int32_t>(position);
            std::cout.write(reinterpret_cast<const char *>(logits), nv*sizeof(float));
            std::cout.flush();
            llama_free(ctx);
        }
        adapter.reset();llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ORACLE ERROR: " << error.what() << '\n';
        return 2;
    }
}
