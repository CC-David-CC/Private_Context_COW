// Bounded native Wen forward for training validation, not an HTTP server.
#include "llama.h"
#include "ggml-backend.h"
#include "qwen35_device.h"
#include "qwen35_math.h"
#include "qwen35_lora.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
struct Capture { std::vector<float> hidden; };
static bool capture(ggml_tensor * t, bool ask, void * data) {
    const bool selected = std::string(t->name) == "result_norm";
    if (ask) return selected;
    if (!selected) return true;
    require(t->type == GGML_TYPE_F32 && t->ne[0] == 4096 && ggml_is_contiguous(t), "Unexpected head input");
    auto & h = static_cast<Capture *>(data)->hidden;
    h.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, h.data(), 0, h.size()*sizeof(float));
    return true;
}
static void write_vector(const std::filesystem::path & path, const float * values, size_t count) {
    for (size_t i = 0; i < count; ++i) require(std::isfinite(values[i]), "Nonfinite native output");
    std::ofstream f(path, std::ios::binary); require(bool(f), "Cannot open native result");
    f.write(reinterpret_cast<const char *>(values), count*sizeof(float)); require(bool(f), "Cannot write native result");
}
int main(int argc, char ** argv) { try {
    require(argc == 4, "Expected verified model path, private output directory and fixed profile");
    std::filesystem::path output(argv[2]); require(std::filesystem::is_directory(output), "Missing output directory");
    std::ifstream profile_file(argv[3]); json profile; profile_file >> profile;
    require(profile.at("schema") == 1 && profile.at("rope_factor").get<float>() >= 1.f,
            "Fixed experimental positional profile required");
    ggml_backend_load_all();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    ggml_backend_dev_t devices[2] = {};
    qwen35_select_device(mp, "CUDA0", devices);
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "Cannot load Wen");
    char architecture[64] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture));
    require(std::string(architecture) == "wen1" && llama_model_n_embd_inp(model.get()) == 4096, "Expected Wen geometry");
    auto * vocab = llama_model_get_vocab(model.get()); const int nv = llama_vocab_n_tokens(vocab);
    std::cout << json({{"ready", true}, {"vocab", nv}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        auto request = json::parse(line); const std::string op = request.at("op");
        if (op == "tokenize") {
            const std::string text = request.at("text");
            int count = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
            require(count > 0, "Empty tokenization"); std::vector<llama_token> ids(count);
            require(llama_tokenize(vocab, text.data(), text.size(), ids.data(), count, false, true) == count, "Tokenizer changed");
            std::cout << json({{"tokens", ids}}).dump() << std::endl; continue;
        }
        require(op == "forward", "Unsupported native operation");
        const std::string name = request.at("name");
        require(!name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c){ return std::isalnum(c) || c == '_'; }), "Invalid result name");
        auto ids = request.at("tokens").get<std::vector<llama_token>>();
        require(!ids.empty() && ids.size() <= 8192, "Bounded nonempty prefix required");
        for (auto token : ids) require(token >= 0 && token < nv, "Invalid token");
        std::vector<int32_t> logical(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) logical[i] = i;
        if (request.contains("positions")) logical = request.at("positions").get<std::vector<int32_t>>();
        require(logical.size() == ids.size(), "One logical position per physical token required");
        for (size_t i = 0; i < logical.size(); ++i)
            require(logical[i] >= 0 && logical[i] < (1<<24) && (i == 0 || logical[i] > logical[i-1]), "Invalid sparse positions");
        const int chunk = request.value("chunk", 64); require(chunk > 0 && chunk <= 64, "Invalid schedule");
        std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)> adapter(nullptr, llama_adapter_lora_free);
        if (request.contains("adapter")) adapter.reset(qwen35_load_adapter(model.get(), request.at("adapter").get<std::string>().c_str()));
        auto cp = llama_context_default_params();
        cp.n_ctx = 8192; cp.n_batch = cp.n_ubatch = 64; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        qwen35_context_math(cp);
        cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        cp.rope_freq_scale = 1.f/profile.at("rope_factor").get<float>();
        cp.yarn_orig_ctx = profile.at("rope_original_context").get<uint32_t>();
        cp.yarn_ext_factor = 1.f; cp.yarn_attn_factor = 1.f;
        cp.yarn_beta_fast = 32.f; cp.yarn_beta_slow = 1.f;
        Capture tape; const bool traced = request.value("capture", true);
        if (traced) { cp.cb_eval = capture; cp.cb_eval_user_data = &tape; }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "Cannot create fresh native context");
        if (adapter) qwen35_attach_adapter(ctx.get(), adapter.get(), 1.f);
        for (size_t offset = 0; offset < ids.size(); offset += chunk) {
            int count = std::min(size_t(chunk), ids.size()-offset);
            std::vector<llama_pos> positions(count*4);
            std::vector<int32_t> nseq(count, 1); llama_seq_id sequence = 0;
            std::vector<llama_seq_id *> sequences(count, &sequence);
            std::vector<int8_t> logits(count, 0); logits.back() = 1;
            for (int axis = 0; axis < 4; ++axis) for (int j = 0; j < count; ++j) positions[axis*count+j] = logical[offset+j];
            llama_batch batch{}; batch.n_tokens = count; batch.token = ids.data()+offset;
            batch.pos = positions.data(); batch.n_seq_id = nseq.data(); batch.seq_id = sequences.data(); batch.logits = logits.data();
            require(llama_decode(ctx.get(), batch) == 0, "Native forward failed");
        }
        require(!traced || tape.hidden.size() == 4096, "Expected one normalized final-position vector");
        const float * scores = llama_get_logits_ith(ctx.get(), -1); require(scores, "Missing native logits");
        if (traced) write_vector(output/(name+"_hidden.f32"), tape.hidden.data(), tape.hidden.size());
        write_vector(output/(name+"_logits.f32"), scores, nv);
        std::cout << json({{"name", name}, {"tokens", ids.size()}, {"chunk", chunk}, {"vocab", nv}, {"logical_end", logical.back()}}).dump() << std::endl;
    }
    return 0;
} catch (const std::exception & e) { std::cerr << "NATIVE TRAINING PROBE: " << e.what() << std::endl; return 2; } }
