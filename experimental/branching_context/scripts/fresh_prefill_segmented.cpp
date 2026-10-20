#include <sstream>
#include <cstdlib>

#include "strata/core/branch_driver.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/prefill/prefill.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
namespace strata::core {
int branch_driver(const WeightTable& weights, const ModelGeometry& geometry, SessionState& state,
                  const VerifyHits& hits, const NativeHead* head, PoolMultiFn pool, void* user, ExpertDispatch& dispatch, int pcie_mode,
                  ExpertSource* source, const ExpertCache* cache, const int32_t* residency,
                  int64_t chunk, void* stream,
                  const std::function<bool(const float*, float*, std::string&)>& head_rows,
                  const std::vector<int64_t>& initial_tokens, const std::string& output, int64_t decode_steps) {
    // Fresh ordinary session: no BranchSession, store, checkpoint import or cache lookup.
    if (initial_tokens.empty() || output.empty()) return 2;
    std::string error;
    strata::prefill::Prefill batched;
    std::vector<float> logits(248320);
    if (!batched.init(weights, geometry, state, source, cache, residency, chunk, stream, error)) {
        std::cerr << error << std::endl; return 1;
    }
    std::vector<std::pair<int64_t,int64_t>> completed_chunks;
    batched.on_chunk = [&](const float* rows, int64_t count, int64_t position, std::string& e) {
        const bool ok = head_rows(rows + (count - 1) * geometry.hc * geometry.n_embd, logits.data(), e);
        std::cerr << "fresh prefill chunk position=" << position << " count=" << count << " ok=" << ok << std::endl;
        if (ok) completed_chunks.emplace_back(position, count);
        return ok;
    };
    const auto begin_prefill = std::chrono::steady_clock::now();
    // Requests are lengths, not a second token history. All tokens come from
    // initial_tokens; every request is independently passed to native Prefill.
    const char* supplied_requests = std::getenv("STRATA_PREFILL_REQUESTS");
    std::string request_text = supplied_requests ? supplied_requests : std::to_string(initial_tokens.size());
    std::replace(request_text.begin(), request_text.end(), ',', ' ');
    std::istringstream requests(request_text);
    std::vector<int64_t> request_counts;
    int64_t request_count = 0;
    while (requests >> request_count) {
        if (request_count <= 0) { std::cerr << "invalid request length"; return 2; }
        request_counts.push_back(request_count);
    }
    if (!requests.eof() || request_counts.empty()) return 2;
    size_t request_total = 0;
    for (auto count : request_counts) {
        if ((uint64_t)count > initial_tokens.size()-request_total) return 2;
        request_total += count;
    }
    if (request_total != initial_tokens.size()) return 2;
    std::ofstream request_frontiers(output + ".requests.f32", std::ios::binary);
    size_t request_offset = 0;
    for (auto request_count : request_counts) {
        if (!batched.run(initial_tokens.data()+request_offset, request_count, request_offset, error)) {
            std::cerr << error << std::endl; return 1;
        }
        request_offset += request_count;
        request_frontiers.write((const char*)logits.data(), logits.size()*sizeof(float));
        request_frontiers.flush();
        if (!request_frontiers) return 1;
    }
    if (request_offset != initial_tokens.size()) return 2;
    request_frontiers.close();
    std::ofstream file(output, std::ios::binary);
    file.write((const char*)logits.data(), logits.size()*sizeof(float));
    file.close();
    if (!file) return 1;
    const auto end_prefill = std::chrono::steady_clock::now();
    Verifier decode;
    if (!decode.init(weights, geometry, state, hits, head, 2, error)) { std::cerr << error; return 1; }
    decode.set_pcie_mode(pcie_mode); decode.set_head_sampling(false);
    // Optional audit-only request segmentation.  This remains one raw
    // Prefill+Verifier process: no BranchSession, Store, checkpoint image, or
    // COW state is created or imported.  Segment boundaries only rebind the
    // verifier plan as the serving backend does between generation requests.
    std::vector<int64_t> decode_segments;
    const char* supplied_segments = std::getenv("STRATA_DECODE_SEGMENTS");
    if (supplied_segments && *supplied_segments) {
        std::string segment_text(supplied_segments);
        std::replace(segment_text.begin(), segment_text.end(), ',', ' ');
        std::istringstream segment_stream(segment_text);
        int64_t count = 0;
        while (segment_stream >> count) {
            if (count <= 0) { std::cerr << "invalid decode segment length" << std::endl; return 2; }
            decode_segments.push_back(count);
        }
        if (!segment_stream.eof() || decode_segments.empty()) {
            std::cerr << "invalid decode segment schedule" << std::endl; return 2;
        }
    } else {
        if (decode_steps <= 0) { std::cerr << "missing decode segment schedule" << std::endl; return 2; }
        decode_segments.push_back(decode_steps);
    }
    int64_t requested_decode_steps = 0;
    for (const auto count : decode_segments) {
        if (count > decode_steps - requested_decode_steps) {
            std::cerr << "decode segments exceed max-new" << std::endl; return 2;
        }
        requested_decode_steps += count;
    }
    if (requested_decode_steps != decode_steps) {
        std::cerr << "decode segments must sum to max-new" << std::endl; return 2;
    }
    std::vector<int32_t> generated;
    std::ofstream trace(output + ".decode.f32", std::ios::binary);
    std::ofstream segment_frontiers(output + ".segments.f32", std::ios::binary);
    if (!trace || !segment_frontiers) { std::cerr << "cannot open decode frontier outputs" << std::endl; return 1; }
    std::vector<std::pair<int64_t,int64_t>> completed_segments;
    const auto begin_decode = std::chrono::steady_clock::now();
    bool stopped = false;
    for (const auto segment_limit : decode_segments) {
        // Rebind at each request boundary while retaining the same ordinary
        // verifier and recurrent model state; no prompt tokens are replayed.
        dispatch.plan = decode.plan_sink();
        const auto segment_start = (int64_t)generated.size();
        int64_t segment_completed = 0;
        for (int64_t j = 0; j < segment_limit; ++j) {
        const int64_t i = (int64_t)generated.size();
        // Normal greedy selection; consume EOS before stopping, as for every output.
        if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); })) {
            std::cerr << "nonfinite frontier" << std::endl; return 1;
        }
        int32_t token = 0;
        for (int32_t t = 1; t < (int32_t)logits.size(); ++t)
            if (logits[t] > logits[token]) token = t;
        int32_t unused = 0; dispatch.failed = false;
        if (!decode.run(1, &token, initial_tokens.size()+i, pool, user, &unused, error) || dispatch.failed ||
            !decode.commit(1, error) || !decode.copy_logits(0, logits.data())) { std::cerr << error; return 1; }
        generated.push_back(token);
        trace.write((const char*)logits.data(), logits.size()*sizeof(float));
        trace.flush();
        if (!trace) { std::cerr << "decode frontier write failed" << std::endl; return 1; }
        ++segment_completed;
        if ((i+1)%16 == 0) std::cerr << "fresh consumed outputs=" << i+1 << std::endl;
        if (token == 248044 || token == 248046) { stopped = true; break; }
        }
        dispatch.plan = nullptr;
        if (segment_completed > 0) {
            segment_frontiers.write((const char*)logits.data(), logits.size()*sizeof(float));
            segment_frontiers.flush();
            if (!segment_frontiers) { std::cerr << "segment frontier write failed" << std::endl; return 1; }
        }
        completed_segments.emplace_back(segment_start, segment_completed);
        if (stopped) break;
    }
    trace.close(); segment_frontiers.close(); dispatch.plan = nullptr;
    if (!trace || !segment_frontiers) return 1;
    std::ofstream schedule(output + ".schedule.json");
    schedule << "{\"prefill_operator\":\"batched-prefill\",\"prefill_requests\":[";
    for (size_t i=0; i<request_counts.size(); ++i) { if(i) schedule << ','; schedule << request_counts[i]; }
    schedule << "],\"prefill_chunks\":[";
    for (size_t i=0; i<completed_chunks.size(); ++i) {
        if(i) schedule << ',';
        schedule << '[' << completed_chunks[i].first << ',' << completed_chunks[i].second << ']';
    }
    schedule << "],\"decode_operator\":\"normal-verifier-decode\",\"requested_segments\":[";
    for (size_t i=0; i<decode_segments.size(); ++i) { if(i) schedule << ','; schedule << decode_segments[i]; }
    schedule << "],\"completed_segments\":[";
    for (size_t i=0; i<completed_segments.size(); ++i) {
        if(i) schedule << ',';
        schedule << "{\"generated_start\":" << completed_segments[i].first
                 << ",\"native_start_position\":" << initial_tokens.size()+completed_segments[i].first
                 << ",\"count\":" << completed_segments[i].second << '}';
    }
    schedule << "]}"; schedule.close(); if (!schedule) return 1;
    std::cout << "{\"consumed\":" << initial_tokens.size()+generated.size()
              << ",\"prompt_consumed\":" << initial_tokens.size()
              << ",\"decode_tokens\":" << generated.size()
              << ",\"normal_forward_tokens\":" << decode.completed_forward_tokens
              << ",\"normal_decode_windows\":" << decode.windows
              << ",\"stopping_reason\":\"" << (!generated.empty() && (generated.back()==248044 || generated.back()==248046) ? "eos" : "output_bound") << "\""
              << ",\"prefill_tokens\":" << batched.stats().tokens
              << ",\"prefill_chunks\":" << batched.stats().chunks
              << ",\"prefill_seconds\":" << std::chrono::duration<double>(end_prefill-begin_prefill).count()
              << ",\"decode_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin_decode).count()
              << ",\"generated\":[";
    for (size_t i=0;i<generated.size();++i) { if(i)std::cout << ','; std::cout << generated[i]; }
    std::cout << "]}" << std::endl;
    return 0;
}
}
