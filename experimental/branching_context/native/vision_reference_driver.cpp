// Independent fresh session: no BranchSession or ConversationCache object,
// snapshot import, prefix lookup or restore command. CPU fixture supplies the
// rotary table independently of the production layout's coordinates.
#include "strata/core/branch_driver.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/prefill/prefill.hpp"
#include "vision_ingest.h"
#include <fstream>
#include <iostream>

namespace strata::core {
int branch_driver(const WeightTable& weights, const ModelGeometry& geometry, SessionState& state,
                  const VerifyHits& hits, const NativeHead* head, PoolMultiFn pool, void* user,
                  ExpertDispatch& dispatch, int pcie_mode, ExpertSource* source, const ExpertCache* cache,
                  const int32_t* residency, int64_t chunk, void* stream,
                  const std::function<bool(const float*, float*, std::string&)>& head_rows,
                  const std::vector<int64_t>&, const std::string&, int64_t) {
    std::string error;
    Verifier forward;
    if (!forward.init(weights, geometry, state, hits, head, 2, error)) return 1;
    forward.set_pcie_mode(pcie_mode);
    forward.set_head_sampling(false);
    dispatch.plan = forward.plan_sink();
    struct ClearPlan { ExpertDispatch& dispatch; ~ClearPlan() { dispatch.plan = nullptr; } } guard{dispatch};
    std::vector<float> logits(static_cast<size_t>(forward.vocab()));
    std::vector<int32_t> ids;
    std::vector<VisionPlacement> images;
    std::vector<std::pair<int64_t, int64_t>> chunks;
    strata::prefill::Prefill batched;
    if (!batched.init(weights, geometry, state, source, cache, residency, chunk, stream, error)) return 1;
    batched.on_chunk = [&](const float* rows, int64_t count, int64_t position, std::string& why) {
        const bool ok = head_rows(rows+(count-1)*geometry.hc*geometry.n_embd, logits.data(), why);
        if (ok) chunks.emplace_back(position, count);
        return ok;
    };
    std::cout << "{\"status\":\"ready\",\"reference\":\"fresh-no-checkpoints\",\"vocab\":"
              << forward.vocab() << ",\"capacity\":" << state.max_cells << "}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command, extra;
        input >> command;
        if (command == "quit") { std::cout << "{\"status\":\"closed\"}" << std::endl; return 0; }
        if (command == "logits") {
            std::string path;
            if (!(input >> std::quoted(path)) || (input >> extra) || ids.empty()) return 2;
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()), logits.size()*sizeof(float));
            out.close(); if (!out) return 1;
            std::cout << "{\"status\":\"ok\",\"bytes\":" << logits.size()*sizeof(float) << "}" << std::endl;
            continue;
        }
        if (command == "stats") {
            std::cout << "{\"status\":\"ok\",\"forward_tokens\":"
                      << batched.stats().tokens+forward.completed_forward_tokens
                      << ",\"prefill_tokens\":" << batched.stats().tokens
                      << ",\"prefill_chunks\":" << batched.stats().chunks << "}" << std::endl;
            continue;
        }
        std::vector<int32_t> tokens;
        cow_vision::Ingest prepared;
        chunks.clear();
        const int64_t start = ids.size();
        if (command == "feed_image") {
            std::string manifest, position_file;
            if (!(input >> std::quoted(manifest) >> std::quoted(position_file)) || (input >> extra)) return 2;
            prepared = cow_vision::prepare(manifest, ids, images, state.max_cells, forward.vocab(), geometry.n_embd);
            // Do not use prepared.layout.positions: independent fixture coordinates.
            const auto raw = read_vision_bytes(position_file, uint64_t(state.max_cells)*12);
            if (raw.size() != uint64_t(state.max_cells)*12 || !strata::kernels::mrope_table()) return 2;
            std::vector<int32_t> positions(raw.size()/4);
            for (size_t i = 0; i < positions.size(); ++i) {
                uint32_t value = 0;
                for (unsigned b = 0; b < 4; ++b) value |= uint32_t(raw[i*4+b]) << (8*b);
                if (value >= static_cast<uint64_t>(state.max_cells)) return 2;
                positions[i] = static_cast<int32_t>(value);
            }
            if (cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(const_cast<int32_t*>(strata::kernels::mrope_table()), positions.data(),
                           raw.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
            tokens = prepared.tokens;
            images = prepared.placements;
            batched.embd_rows = prepared.rows.data();
        } else if (command == "feed" || command == "decode") {
            int64_t token;
            while (input >> token) {
                if (token < 0 || token >= forward.vocab()) return 2;
                tokens.push_back(static_cast<int32_t>(token));
            }
            if (!input.eof()) return 2;
        } else return 2;
        if (tokens.empty() || tokens.size() > static_cast<size_t>(state.max_cells)-ids.size()) return 2;
        if (command == "decode") {
            if (tokens.size() != 1) return 2;
            int32_t unused = 0;
            dispatch.failed = false;
            if (!forward.run(1, tokens.data(), start, pool, user, &unused, error) || dispatch.failed ||
                !forward.commit(1, error) || !forward.copy_logits(0, logits.data())) return 1;
            chunks.emplace_back(start, 1);
        } else {
            const std::vector<int64_t> wide(tokens.begin(), tokens.end());
            if (!batched.run(wide.data(), wide.size(), start, error)) {
                std::cerr << error << std::endl; return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return 1;
        batched.embd_rows = nullptr;
        if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); })) return 1;
        ids.insert(ids.end(), tokens.begin(), tokens.end());
        std::cout << "{\"status\":\"ok\",\"consumed\":" << ids.size()
                  << ",\"position\":" << start << ",\"count\":" << tokens.size()
                  << ",\"chunks\":[";
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '[' << chunks[i].first << ',' << chunks[i].second << ']';
        }
        std::cout << "]}" << std::endl;
    }
    return 0;
}
} // namespace strata::core
