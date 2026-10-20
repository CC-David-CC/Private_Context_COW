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
#include <memory>
#include <tuple>
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
    std::unique_ptr<Verifier> decode = std::make_unique<Verifier>();
    if (!decode->init(weights, geometry, state, hits, head, 2, error)) { std::cerr << error; return 1; }
    decode->set_pcie_mode(pcie_mode); decode->set_head_sampling(false);
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
    int64_t interleave_after_segment = 0;
    std::vector<int64_t> appended_tokens;
    const char* interleave_env = std::getenv("STRATA_PREFILL_AFTER_SEGMENT");
    const char* append_env = std::getenv("STRATA_PREFILL_APPEND_TOKENS");
    if (interleave_env && *interleave_env) {
        try { interleave_after_segment = std::stoll(interleave_env); }
        catch (...) { return 2; }
        if (interleave_after_segment <= 0 ||
            interleave_after_segment >= static_cast<int64_t>(decode_segments.size()) ||
            !append_env || !*append_env) {
            std::cerr << "invalid interleaved prefill boundary" << std::endl; return 2;
        }
        std::string append_text(append_env);
        std::replace(append_text.begin(), append_text.end(), ',', ' ');
        std::istringstream append_stream(append_text);
        int64_t append_token = 0;
        while (append_stream >> append_token) {
            if (append_token < 0 || append_token >= static_cast<int64_t>(logits.size())) return 2;
            appended_tokens.push_back(append_token);
        }
        if (!append_stream.eof() || appended_tokens.empty()) return 2;
    } else if (append_env && *append_env) {
        std::cerr << "append tokens require an interleaved boundary" << std::endl; return 2;
    }
    const char* reset_segments_env = std::getenv("STRATA_RESET_VERIFIER_SEGMENTS");
    const bool reset_verifier_segments = reset_segments_env && std::string(reset_segments_env) == "1";
    std::vector<int32_t> forced_tokens;
    const char* supplied_forced = std::getenv("STRATA_FORCE_GENERATED_TOKENS");
    if (!supplied_forced || !*supplied_forced) {
        std::cerr << "missing forced token list" << std::endl; return 2;
    }
    std::string forced_text(supplied_forced);
    std::replace(forced_text.begin(), forced_text.end(), ',', ' ');
    std::istringstream forced_stream(forced_text);
    int64_t forced_token = 0;
    while (forced_stream >> forced_token) {
        if (forced_token < 0 || forced_token >= static_cast<int64_t>(logits.size())) {
            std::cerr << "forced token outside vocabulary" << std::endl; return 2;
        }
        forced_tokens.push_back(static_cast<int32_t>(forced_token));
    }
    if (!forced_stream.eof() || forced_tokens.size() != static_cast<size_t>(decode_steps)) {
        std::cerr << "invalid forced token schedule" << std::endl; return 2;
    }
    std::vector<int32_t> generated;
    std::ofstream trace(output + ".decode.f32", std::ios::binary);
    std::ofstream segment_frontiers(output + ".segments.f32", std::ios::binary);
    if (!trace || !segment_frontiers) { std::cerr << "cannot open decode frontier outputs" << std::endl; return 1; }
    std::vector<std::pair<int64_t,int64_t>> completed_segments;
    std::vector<std::tuple<int64_t,int64_t,int64_t>> positioned_segments;
    std::vector<std::pair<int64_t,int64_t>> interleaved_prefills;
    int64_t accumulated_forward_tokens = 0;
    int64_t accumulated_decode_windows = 0;
    int64_t session_position = static_cast<int64_t>(initial_tokens.size());
    const auto begin_decode = std::chrono::steady_clock::now();
    bool stopped = false;
    for (const auto segment_limit : decode_segments) {
        // Rebind at each request boundary while retaining the same ordinary
        // verifier and recurrent model state; no prompt tokens are replayed.
        dispatch.plan = decode->plan_sink();
        const auto segment_start = (int64_t)generated.size();
        const auto segment_position = session_position;
        int64_t segment_completed = 0;
        for (int64_t j = 0; j < segment_limit; ++j) {
        const int64_t i = (int64_t)generated.size();
        // Normal greedy selection; consume EOS before stopping, as for every output.
        if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); })) {
            std::cerr << "nonfinite frontier" << std::endl; return 1;
        }
        if (forced_tokens.size() != static_cast<size_t>(decode_steps)) {
            std::cerr << "forced token count differs from decode schedule" << std::endl; return 2;
        }
        const int32_t token = forced_tokens[static_cast<size_t>(i)];
        int32_t unused = 0; dispatch.failed = false;
        if (!decode->run(1, &token, session_position, pool, user, &unused, error) || dispatch.failed ||
            !decode->commit(1, error) || !decode->copy_logits(0, logits.data())) { std::cerr << error; return 1; }
        generated.push_back(token);
        ++session_position;
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
        positioned_segments.emplace_back(segment_start, segment_position, segment_completed);
        if (!stopped && interleave_after_segment == static_cast<int64_t>(completed_segments.size())) {
            // Match separate serving requests: release the old verifier, append
            // user input through native Prefill at the current absolute position,
            // then construct a fresh normal Verifier over that same raw state.
            accumulated_forward_tokens += decode->completed_forward_tokens;
            accumulated_decode_windows += decode->windows;
            decode.reset();
            dispatch.plan = nullptr;
            const auto append_position = session_position;
            const auto append_count = static_cast<int64_t>(appended_tokens.size());
            if (!batched.run(appended_tokens.data(), append_count, append_position, error)) {
                std::cerr << error << std::endl; return 1;
            }
            std::ofstream append_frontier(output + ".interleaved-prefill.f32", std::ios::binary);
            append_frontier.write((const char*)logits.data(), logits.size()*sizeof(float));
            append_frontier.flush(); append_frontier.close();
            if (!append_frontier) { std::cerr << "interleaved prefill frontier write failed" << std::endl; return 1; }
            interleaved_prefills.emplace_back(append_position, append_count);
            session_position += append_count;
            decode = std::make_unique<Verifier>();
            if (!decode->init(weights, geometry, state, hits, head, 2, error)) {
                std::cerr << error << std::endl; return 1;
            }
            decode->set_pcie_mode(pcie_mode); decode->set_head_sampling(false);
        } else if (!stopped && reset_verifier_segments &&
                   completed_segments.size() < decode_segments.size()) {
            accumulated_forward_tokens += decode->completed_forward_tokens;
            accumulated_decode_windows += decode->windows;
            decode.reset();
            dispatch.plan = nullptr;
            decode = std::make_unique<Verifier>();
            if (!decode->init(weights, geometry, state, hits, head, 2, error)) {
                std::cerr << error << std::endl; return 1;
            }
            decode->set_pcie_mode(pcie_mode); decode->set_head_sampling(false);
        }
        if (stopped) break;
    }
    if (decode) {
        accumulated_forward_tokens += decode->completed_forward_tokens;
        accumulated_decode_windows += decode->windows;
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
        schedule << "{\"generated_start\":" << std::get<0>(positioned_segments[i])
                 << ",\"native_start_position\":" << std::get<1>(positioned_segments[i])
                 << ",\"count\":" << completed_segments[i].second << '}';
    }
    schedule << "],\"operator_schedule\":[{\"operator\":\"batched-prefill\",\"purpose\":\"root\",\"position\":0,\"count\":" << initial_tokens.size() << '}';
    size_t schedule_prefill_index = 0;
    for (size_t i=0; i<positioned_segments.size(); ++i) {
        schedule << ",{\"operator\":\"normal-verifier-decode\",\"purpose\":\"segment-" << (i+1)
                 << "\",\"position\":" << std::get<1>(positioned_segments[i])
                 << ",\"count\":" << std::get<2>(positioned_segments[i]) << '}';
        if (interleave_after_segment == static_cast<int64_t>(i+1) &&
            schedule_prefill_index < interleaved_prefills.size()) {
            schedule << ",{\"operator\":\"batched-prefill\",\"purpose\":\"appended-user\",\"position\":"
                     << interleaved_prefills[schedule_prefill_index].first << ",\"count\":"
                     << interleaved_prefills[schedule_prefill_index].second << '}';
            ++schedule_prefill_index;
        }
    }
    schedule << "]}"; schedule.close(); if (!schedule) return 1;
    std::cout << "{\"consumed\":" << session_position
              << ",\"prompt_consumed\":" << initial_tokens.size()
              << ",\"decode_tokens\":" << generated.size()
              << ",\"normal_forward_tokens\":" << accumulated_forward_tokens
              << ",\"normal_decode_windows\":" << accumulated_decode_windows
              << ",\"stopping_reason\":\"" << (!generated.empty() && (generated.back()==248044 || generated.back()==248046) ? "eos" : "output_bound") << "\""
              << ",\"prefill_tokens\":" << batched.stats().tokens
              << ",\"initial_prefill_tokens\":" << initial_tokens.size()
              << ",\"appended_prefill_tokens\":" << (session_position-initial_tokens.size()-static_cast<int64_t>(generated.size()))
              << ",\"prefill_chunks\":" << batched.stats().chunks
              << ",\"prefill_seconds\":" << std::chrono::duration<double>(end_prefill-begin_prefill).count()
              << ",\"decode_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin_decode).count()
              << ",\"generated\":[";
    for (size_t i=0;i<generated.size();++i) { if(i)std::cout << ','; std::cout << generated[i]; }
    std::cout << "],\"operator_schedule\":[{\"operator\":\"batched-prefill\",\"purpose\":\"root\",\"position\":0,\"count\":" << initial_tokens.size() << '}';
    size_t stdout_prefill_index = 0;
    for (size_t i=0; i<positioned_segments.size(); ++i) {
        std::cout << ",{\"operator\":\"normal-verifier-decode\",\"purpose\":\"segment-" << (i+1)
                  << "\",\"position\":" << std::get<1>(positioned_segments[i])
                  << ",\"count\":" << std::get<2>(positioned_segments[i]) << '}';
        if (interleave_after_segment == static_cast<int64_t>(i+1) &&
            stdout_prefill_index < interleaved_prefills.size()) {
            std::cout << ",{\"operator\":\"batched-prefill\",\"purpose\":\"appended-user\",\"position\":"
                      << interleaved_prefills[stdout_prefill_index].first << ",\"count\":"
                      << interleaved_prefills[stdout_prefill_index].second << '}';
            ++stdout_prefill_index;
        }
    }
    std::cout << ']';
    std::cout << '}' << std::endl;
    return 0;
}
}
