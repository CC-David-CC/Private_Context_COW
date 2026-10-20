#include "strata/prefill/prefill.hpp"
#include "vision_ingest.h"
#include "vision_dry.h"
#include <chrono>
#include "strata/core/branch_driver.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/branch_session.hpp"
#include <algorithm>
#include <cmath>
#include "strata/core/branch_handles.hpp"
#include "strata/core/conversation_cache.hpp"
#include <exception>
#include <fstream>
#include "grammar_select.h"
#include "min_p_select.h"
#include "mirostat2_select.h"
#include "adaptive_p_select.h"
#include <fstream>
#include <iterator>
#include <charconv>
#include <limits>
#include <iomanip>
#include <iostream>
#include <sstream>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace strata::core {
namespace {
void error_reply(const std::string& message) {
    std::cout << "{\"status\":\"error\",\"message\":\"";
    for (unsigned char c : message) {
        if (c == '"' || c == '\\') std::cout << '\\';
        std::cout << (c < 32 ? ' ' : (char)c);
    }
    std::cout << "\"}" << std::endl;
}
}
int branch_driver(const WeightTable& weights, const ModelGeometry& geometry, SessionState& state,
                  const VerifyHits& hits, const NativeHead* head, PoolMultiFn pool, void* user,
                  ExpertDispatch& dispatch, int pcie_mode, ExpertSource* source, const ExpertCache* cache, const int32_t* residency,
                  int64_t chunk, void* stream,
                  const std::function<bool(const float*, float*, std::string&)>& head_rows,
                  const std::vector<int64_t>& initial_tokens, const std::string& output,
                  int64_t decode_steps) {
    std::string error;
    // Images never leave this process; model identity is fixed by its loaded
    // weights. The API validates the external model/build contract before launch.
    BranchSession session(state, geometry, "loaded-model-single-process-v1");
    if (!session.bind(error)) { error_reply(error); return 1; }
    Verifier forward;
    // Strata requires capacity for at least two rows even when run() consumes
    // just one. The actual forced forward below always passes T=1.
    if (!forward.init(weights, geometry, state, hits, head, 2, error)) { error_reply(error); return 1; }
    // The verifier consumes the GPU plan produced by this exact dispatch.
    // Without this binding, resident experts are omitted from both CPU and GPU work.
    forward.set_pcie_mode(pcie_mode);
    dispatch.plan = forward.plan_sink();
    struct PlanReset {
        ExpertDispatch& dispatch;
        ~PlanReset() { dispatch.plan = nullptr; }
    } plan_reset{dispatch};
    bool& pool_failed = dispatch.failed;
    forward.set_head_sampling(false); // forced inputs; the API owns RNG/grammar
    BranchHandleCounters handle_work;
    BranchHandles<BranchSessionImage> images(handle_work);
    ConversationCache ram_cache(size_t(1024) * 1024 * 1024, 8);
    uint64_t ram_hits = 0, ram_misses = 0;
    std::vector<float> logits((size_t)forward.vocab());
    uint64_t grammar_selection_calls = 0, grammar_selection_logit_bytes = 0, grammar_selection_mask_bytes = 0, sampling_calls = 0, sampling_logit_bytes = 0, sampling_mask_bytes = 0, sampling_ns = 0, sampling_v2_calls = 0, sampling_v2_logit_bytes = 0, sampling_v2_mask_bytes = 0, sampling_v2_ns = 0, sampling_v3_calls = 0, sampling_v3_logit_bytes = 0, sampling_v3_mask_bytes = 0, sampling_v3_ns = 0, dry_penalized_candidates = 0; uint64_t sampling_v4_calls = 0, sampling_v4_logit_bytes = 0, sampling_v4_mask_bytes = 0, sampling_v4_ns = 0, typical_filter_calls = 0, dynatemp_filter_calls = 0, dry_penalized_candidates_v4 = 0; uint64_t sampling_v5_calls = 0, sampling_v5_logit_bytes = 0, sampling_v5_mask_bytes = 0, sampling_v5_ns = 0, sampling_v5_activation_draws = 0, sampling_v5_activated_calls = 0, sampling_v5_removed_candidates = 0, typical_filter_calls_v5 = 0, dynatemp_filter_calls_v5 = 0, dry_penalized_candidates_v5 = 0; uint64_t sampling_adaptive_p_calls = 0, sampling_adaptive_p_logit_bytes = 0, sampling_adaptive_p_mask_bytes = 0, sampling_adaptive_p_ns = 0; uint64_t sampling_mirostat2_calls = 0, sampling_mirostat2_logit_bytes = 0, sampling_mirostat2_mask_bytes = 0, sampling_mirostat2_ns = 0;
    std::cout << "{\"status\":\"ready\",\"vocab\":" << forward.vocab()
              << ",\"capacity\":" << state.max_cells
              << ",\"native_request_protocol\":\"consumed-greedy-ack-v1\",\"selected_token_protocol\":\"consumed-selected-receipt-v1\",\"grammar_selection_protocol\":\"finite-mask-argmax-v1\",\"sampling_protocol\":\"full-vocab-adaptive-p-u53-v7\",\"adaptive_p_protocol\":\"full-vocab-adaptive-p-u53-v7\",\"mirostat2_protocol\":\"full-vocab-mirostat2-u53-v6\",\"sampling_v6_protocol\":\"full-vocab-mirostat2-u53-v6\",\"sampling_v5_protocol\":\"full-vocab-xtc-min-p-u53-v5\",\"sampling_v4_protocol\":\"full-vocab-typical-dyna-min-p-u53-v4\",\"sampling_v3_protocol\":\"full-vocab-dry-min-p-u53-v3\",\"sampling_v2_protocol\":\"full-vocab-min-p-u53-v2\",\"sampling_v1_protocol\":\"full-vocab-min-p-u53-v1\",\"mirostat2_feedback_protocol\":\"mirostat2-state-v1\",\"adaptive_p_feedback_protocol\":\"adaptive-p-state-v1\",\"adaptive_p_feedback_version\":1,\"mirostat2_feedback_version\":1}" << std::endl;

    strata::prefill::Prefill batched;
    if (chunk <= 0 || !batched.init(weights, geometry, state, source, cache, residency, chunk, stream, error)) {
        error_reply(error.empty() ? "missing real batched prefill" : error); return 1;
    }
    std::vector<std::pair<int64_t,int64_t>> request_chunks;
    batched.on_chunk = [&](const float* rows, int64_t count, int64_t position, std::string& e) {
        const bool ok = head_rows(rows + (count - 1) * geometry.hc * geometry.n_embd, logits.data(), e);
        if (ok) request_chunks.emplace_back(position, count);
        return ok;
    };
    uint64_t prefill_tokens = 0, prefill_chunks = 0;
    uint64_t decode_prepare_ns = 0, decode_compute_ns = 0, decode_finish_ns = 0, decode_timed_tokens = 0;

    uint64_t vision_validated_bytes = 0;
    uint64_t sampling_mixed_dry_calls=0,sampling_mixed_dry_ns=0,
        sampling_mixed_dry_logit_bytes=0,sampling_mixed_dry_mask_bytes=0;
    uint64_t sampling_image_dry_combination_calls=0;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string command;
        in >> command;
        error.clear();
        if (!session.input_images().empty() && (command == "park" || command == "activate")) {
            error_reply("legacy native host cache has no canonical image lookup"); continue;
        }
        if (command == "quit") {
            std::cout << "{\"status\":\"closed\"}" << std::endl;
            return 0;
        }
        if (command == "reset") {
            if (!session.reset(error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"consumed\":0}" << std::endl;
        } else if (command == "vision_info") {
            const auto plan = vision_layout(session.ids(), session.input_images(), state.max_cells,
                                            forward.vocab(), geometry.n_embd, 1024);
            std::cout << "{\"status\":\"ok\",\"vision_protocol\":\"explicit-image-rows-mrope-v3\","
                      << "\"enabled\":" << (strata::kernels::mrope_table() ? "true" : "false")
                      << ",\"consumed\":" << plan.consumed << ",\"rotary_cursor\":" << plan.rotary_cursor
                      << ",\"position_sha256\":\"" << vision_position_digest(plan) << "\""
                      << ",\"native_input_sha256\":\"" << vision_input_digest(session.ids(), session.input_images()) << "\""
                      << ",\"hot_image_snapshots\":true"
                      << ",\"cold_image_snapshots\":true"
                      << ",\"images\":" << session.input_images().size() << ",\"image_rows\":" << plan.image_rows
                      << ",\"vision_dry_protocol\":\"mixed-text-runs-dry-u53-v1\"" << ",\"vision_dry_sampling_protocol\":\"mixed-text-runs-sampling-u53-v2\"" << ",\"vision_sampling_protocols\":[\"full-vocab-typical-dyna-min-p-u53-v4\"]" << ",\"validated_embedding_bytes\":" << vision_validated_bytes
                      << ",\"position_uploaded_bytes\":" << session.counters().position_uploaded_bytes << "}" << std::endl;
        } else if (command == "load_image") {
            std::string path, metadata, logits_path, extra;
            if (!(in >> std::quoted(path) >> std::quoted(metadata) >> std::quoted(logits_path)) ||
                (in >> extra) || !strata::kernels::mrope_table()) {
                error_reply("load_image requires --vision, state, input metadata and frontier paths"); continue;
            }
            cow_vision::RestoreInput expected;
            try { expected = cow_vision::restore_metadata(metadata, state.max_cells, forward.vocab(), geometry.n_embd); }
            catch (const std::exception& e) { error_reply(e.what()); continue; }
            std::ifstream frontier(logits_path, std::ios::binary | std::ios::ate);
            const auto bytes = static_cast<std::streamsize>(forward.vocab())*sizeof(float);
            if (!frontier || frontier.tellg() != bytes) { error_reply("invalid image restore frontier extent"); continue; }
            std::vector<float> next_logits(static_cast<size_t>(forward.vocab()));
            frontier.seekg(0); frontier.read(reinterpret_cast<char*>(next_logits.data()), bytes);
            if (!frontier || !session.load_inputs(path, expected.tokens, expected.placements, next_logits, error)) {
                error_reply(error.empty() ? "incomplete image restore frontier" : error); continue;
            }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "feed_image") {
            std::string manifest, extra;
            if (!(in >> std::quoted(manifest)) || (in >> extra) || !strata::kernels::mrope_table()) {
                error_reply("feed_image requires --vision and one quoted manifest path"); continue;
            }
            cow_vision::Ingest prepared;
            try {
                prepared = cow_vision::prepare(manifest, session.ids(), session.input_images(),
                    state.max_cells, forward.vocab(), geometry.n_embd);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const auto position = static_cast<int64_t>(session.ids().size());
            request_chunks.clear();
            std::vector<int64_t> wide(prepared.tokens.begin(), prepared.tokens.end());
            batched.embd_rows = prepared.rows.data();
            const bool ok = session.begin_inputs(prepared.tokens, prepared.placements, forward.vocab(), error) &&
                batched.run(wide.data(), wide.size(), position, error) &&
                session.finish(prepared.tokens, logits, error);
            batched.embd_rows = nullptr; // finish synchronized all input-row reads before buffer release.
            if (!ok) { error_reply(error.empty() ? "image prefill failed" : error); return 1; }
            vision_validated_bytes += prepared.read_bytes;
            prefill_tokens = batched.stats().tokens;
            prefill_chunks = batched.stats().chunks;
            std::cout << "{\"status\":\"ok\",\"operator\":\"batched-prefill\",\"position\":" << position
                      << ",\"consumed\":" << session.ids().size() << ",\"count\":" << wide.size()
                      << ",\"position_sha256\":\"" << vision_position_digest(prepared.layout) << "\",\"chunks\":[";
            for (size_t i = 0; i < request_chunks.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << '[' << request_chunks[i].first << ',' << request_chunks[i].second << ']';
            }
            std::cout << "]}" << std::endl;
        } else if (command == "feed") {
            std::vector<int32_t> tokens;
            int64_t token;
            bool valid = true;
            while (in >> token) {
                if (token < 0 || token >= forward.vocab()) { valid = false; break; }
                tokens.push_back((int32_t)token);
            }
            if (!valid || !in.eof() || tokens.empty() || tokens.size() > (size_t)state.max_cells - session.ids().size()) {
                error_reply("invalid token list or context capacity"); continue;
            }
            request_chunks.clear();
            std::vector<int64_t> wide(tokens.begin(), tokens.end());
            const auto position = (int64_t)session.ids().size();
            if (!session.begin(position, (int64_t)tokens.size(), error) ||
                !batched.run(wide.data(), (int64_t)wide.size(), position, error) ||
                !session.finish(tokens, logits, error)) {
                error_reply(error.empty() ? "batched prefill failed" : error); continue;
            }
            prefill_tokens = batched.stats().tokens;
            prefill_chunks = batched.stats().chunks;

            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"operator\":\"batched-prefill\",\"position\":" << position
                      << ",\"count\":" << tokens.size() << ",\"chunks\":[";
            for (size_t i=0;i<request_chunks.size();++i) {
                if(i) std::cout << ',';
                std::cout << '[' << request_chunks[i].first << ',' << request_chunks[i].second << ']';
            }
            std::cout << "]}" << std::endl;
            continue;
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
         } else if (command == "select_mask") {
            int64_t expected_consumed = 0;
            std::string mask_path, extra;
            if (!(in >> expected_consumed >> std::quoted(mask_path)) || (in >> extra)) {
                error_reply("select_mask requires expected consumed count and mask path"); continue;
            }
            const auto actual_consumed = session.ids().size();
            if (actual_consumed == 0 || expected_consumed <= 0 ||
                static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("select_mask consumed count mismatch"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("select_mask frontier extent mismatch"); continue;
            }
            const size_t mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("select_mask could not open mask file"); continue; }
            std::vector<uint8_t> mask(mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(mask_bytes)) {
                error_reply("select_mask truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("select_mask extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("select_mask mask read failed"); continue; }
            grammar_select::Result selection{};
            try {
                selection = grammar_select::select(frontier.data(), vocab, mask.data(), mask.size(),
                    actual_consumed, static_cast<size_t>(expected_consumed));
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            ++grammar_selection_calls;
            grammar_selection_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            grammar_selection_mask_bytes += selection.mask_bytes;
            std::cout << "{\"status\":\"ok\",\"selection_protocol\":\"finite-mask-argmax-v1\",\"consumed\":"
                      << actual_consumed << ",\"token\":" << selection.token
                      << ",\"logits_count\":" << selection.logits_count
                      << ",\"mask_bytes\":" << selection.mask_bytes << "}" << std::endl;
                } else if (command == "sample_mask") {
            int64_t expected_consumed = 0;
            double temperature = 0.0, min_p = 0.0;
            uint64_t draw53 = 0;
            std::string mask_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> draw53 >> std::quoted(mask_path)) || (in >> extra)) {
                error_reply("sample_mask requires consumed, temperature, min_p, draw53, and quoted mask path"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                draw53 >= (uint64_t(1) << 53)) {
                error_reply("sample_mask parameters are outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask mask read failed"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select(frontier.data(), vocab, mask.data(), mask.size(),
                                                 temperature, min_p, draw53);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            sampling_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            ++sampling_calls;
            sampling_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_mask_bytes += selection.mask_bytes;
            std::cout << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                      << "\"protocol\":\"full-vocab-min-p-u53-v1\",\"token\":" << selection.token
                      << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                      << ",\"mask_bytes\":" << selection.mask_bytes
                      << ",\"legal_count\":" << selection.legal_count
                      << ",\"kept_count\":" << selection.kept_count
                      << ",\"draw53\":" << draw53 << "}" << std::endl;
                } else if (command == "sample_mask_v2") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0;
            uint64_t draw53 = 0;
            std::string top_k_text, mask_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53 >> std::quoted(mask_path)) || (in >> extra)) {
                error_reply("sample_mask_v2 requires consumed temperature min_p top_k top_p draw53 and quoted mask path"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_v2 top_k must be an exact nonnegative integer"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v2 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature < 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 ||
                (temperature == 0.0 ? draw53 != 0 : draw53 >= (uint64_t(1) << 53))) {
                error_reply("sample_mask_v2 parameters are outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v2 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v2 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v2 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v2 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v2 mask read failed"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t selection_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += selection_elapsed_ns;
            sampling_v2_ns += selection_elapsed_ns;
            ++sampling_calls;
            ++sampling_v2_calls;
            sampling_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_v2_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_mask_bytes += selection.mask_bytes;
            sampling_v2_mask_bytes += selection.mask_bytes;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                      << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                      << "\"protocol\":\"full-vocab-min-p-u53-v2\",\"token\":" << selection.token
                      << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                      << ",\"mask_bytes\":" << selection.mask_bytes
                      << ",\"legal_count\":" << selection.legal_count
                      << ",\"kept_count\":" << selection.kept_count
                      << ",\"top_k\":" << top_k_arg << ",\"top_p\":" << top_p
                      << ",\"draw53\":" << draw53 << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
                } else if (command == "sample_mask_image_dry_v1") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)) || (in >> extra)) {
                error_reply("sample_mask_image_dry_v1 requires consumed temperature min_p top_k top_p draw53 mask_path dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_image_dry_v1 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_image_dry_v1 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature < 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                (temperature == 0.0 ? draw53 != 0 : draw53 >= (uint64_t(1) << 53))) {
                error_reply("sample_mask_image_dry_v1 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_image_dry_v1 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_image_dry_v1 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_image_dry_v1 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_image_dry_v1 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_image_dry_v1 mask read failed"); continue; }

            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            const auto window = cow_vision_dry::project(session.ids(), session.input_images(),
                static_cast<size_t>(dry_last_n_arg));
            const size_t expected_history = window.tokens.size();
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_image_dry_v1 history/breaker file is missing or malformed"); continue;
            }
            if ((dry_last_n_arg == 0 && !breakers.empty()) || breakers.size() > history.size()) {
                error_reply("sample_mask_image_dry_v1 history/breaker count mismatch"); continue;
            }
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != window.tokens[i]) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_image_dry_v1 history is not the native session tail"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                size_t affected = 0;
                const auto adjusted = min_p_select::apply_dry_penalties(frontier.data(), vocab,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    &affected, &window.boundaries);
                selection = min_p_select::select(adjusted.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p);
                selection.dry_penalized_candidates = affected;
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_mixed_dry_ns += elapsed_ns;
            ++sampling_calls; ++sampling_mixed_dry_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes;
            sampling_mixed_dry_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.mask_bytes;
            sampling_mixed_dry_mask_bytes += selection.mask_bytes;
            dry_penalized_candidates += selection.dry_penalized_candidates;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"mixed-text-runs-dry-u53-v1\",\"token\":" << selection.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.mask_bytes << ",\"legal_count\":" << selection.legal_count
                << ",\"kept_count\":" << selection.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"temperature\":" << temperature << ",\"min_p\":" << min_p << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"text_tokens\":" << window.text_count << ",\"dry_boundary_count\":" << window.boundaries.size() << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.dry_penalized_candidates << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
                } else if (command == "sample_mask_v3") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)) || (in >> extra)) {
                error_reply("sample_mask_v3 requires consumed temperature min_p top_k top_p draw53 mask_path dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_v3 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v3 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature < 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                (temperature == 0.0 ? draw53 != 0 : draw53 >= (uint64_t(1) << 53))) {
                error_reply("sample_mask_v3 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v3 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v3 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v3 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v3 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v3 mask read failed"); continue; }

            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            const size_t expected_history = dry_last_n_arg == 0 ? 0 :
                std::min<uint64_t>(actual_consumed, dry_last_n_arg);
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_v3 history/breaker file is missing or malformed"); continue;
            }
            if ((dry_last_n_arg == 0 && !breakers.empty()) || breakers.size() > history.size()) {
                error_reply("sample_mask_v3 history/breaker count mismatch"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_v3 history is not the native session tail"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select_with_dry(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)});
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_v3_ns += elapsed_ns;
            ++sampling_calls; ++sampling_v3_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes;
            sampling_v3_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.mask_bytes;
            sampling_v3_mask_bytes += selection.mask_bytes;
            dry_penalized_candidates += selection.dry_penalized_candidates;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"full-vocab-dry-min-p-u53-v3\",\"token\":" << selection.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.mask_bytes << ",\"legal_count\":" << selection.legal_count
                << ",\"kept_count\":" << selection.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.dry_penalized_candidates << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
                } else if (command == "sample_mask_image_v4") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            double typical_p = 1.0, dyna_range = 0.0, dyna_exponent = 1.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)
                    >> typical_p >> dyna_range >> dyna_exponent) || (in >> extra)) {
                error_reply("sample_mask_image_v4 requires consumed T min_p top_k top_p draw53 mask_path dryMult dryBase dryAllowed dryN history breakers typicalP dynaRange dynaExponent"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_image_v4 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_image_v4 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                !std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0 ||
                !std::isfinite(dyna_range) || dyna_range < 0.0 ||
                !std::isfinite(dyna_exponent) || dyna_exponent <= 0.0 ||
                !std::isfinite(temperature + dyna_range)) {
                error_reply("sample_mask_image_v4 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_image_v4 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_image_v4 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_image_v4 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_image_v4 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_image_v4 mask read failed"); continue; }

            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const auto window = cow_vision_dry::project(session.ids(), session.input_images(),
                dry_active ? static_cast<size_t>(dry_last_n_arg) : 0);
            const size_t expected_history = window.tokens.size();
            std::string boundary_json = "[";
            for (size_t i = 0; i < window.boundaries.size(); ++i) {
                if (i != 0) boundary_json += ",";
                boundary_json += std::to_string(window.boundaries[i]);
            }
            boundary_json += "]";
            std::vector<uint32_t> history, breakers;
            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max()) || breakers.size() > history.size()) {
                error_reply("sample_mask_image_v4 history/breaker file is missing or malformed"); continue;
            }
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != window.tokens[i]) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_image_v4 history is not the native session tail"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                size_t segmented_affected = 0;
                const auto adjusted = min_p_select::apply_dry_penalties(frontier.data(), vocab,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    &segmented_affected, &window.boundaries);
                selection = min_p_select::select_v4(adjusted.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53,
                    {}, {}, min_p_select::DryConfig{0.0, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    typical_p, dyna_range, dyna_exponent);
                selection.dry_penalized_candidates = segmented_affected;
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_v4_ns += elapsed_ns;
            ++sampling_calls; ++sampling_v4_calls; ++sampling_image_dry_combination_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes; sampling_v4_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.mask_bytes; sampling_v4_mask_bytes += selection.mask_bytes;
            if (selection.typical_computed) ++typical_filter_calls;
            if (selection.dynatemp_computed) ++dynatemp_filter_calls;
            dry_penalized_candidates += selection.dry_penalized_candidates;
            dry_penalized_candidates_v4 += selection.dry_penalized_candidates;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"full-vocab-typical-dyna-min-p-u53-v4\",\"token\":" << selection.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.mask_bytes << ",\"legal_count\":" << selection.legal_count
                << ",\"min_p_kept_count\":" << selection.min_p_kept_count
                << ",\"top_k_kept_count\":" << selection.top_k_kept_count
                << ",\"typical_kept_count\":" << selection.typical_kept_count
                << ",\"kept_count\":" << selection.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"vision_dry_sampling_protocol\":\"mixed-text-runs-sampling-u53-v2\"" << ",\"text_tokens\":" << window.text_count << ",\"dry_boundary_count\":" << window.boundaries.size() << ",\"dry_boundaries\":" << boundary_json << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.dry_penalized_candidates
                << ",\"typical_p\":" << typical_p << ",\"dynatemp_range\":" << dyna_range
                << ",\"dynatemp_exponent\":" << dyna_exponent
                << ",\"effective_temperature\":" << selection.effective_temperature << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
        } else if (command == "sample_mask_v4") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            double typical_p = 1.0, dyna_range = 0.0, dyna_exponent = 1.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)
                    >> typical_p >> dyna_range >> dyna_exponent) || (in >> extra)) {
                error_reply("sample_mask_v4 requires consumed T min_p top_k top_p draw53 mask_path dryMult dryBase dryAllowed dryN history breakers typicalP dynaRange dynaExponent"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_v4 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v4 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                !std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0 ||
                !std::isfinite(dyna_range) || dyna_range < 0.0 ||
                !std::isfinite(dyna_exponent) || dyna_exponent <= 0.0 ||
                !std::isfinite(temperature + dyna_range)) {
                error_reply("sample_mask_v4 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v4 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v4 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v4 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v4 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v4 mask read failed"); continue; }

            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = dry_active ?
                std::min<uint64_t>(actual_consumed, dry_last_n_arg) : 0;
            std::vector<uint32_t> history, breakers;
            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max()) || breakers.size() > history.size()) {
                error_reply("sample_mask_v4 history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_v4 history is not the native session tail"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select_v4(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    typical_p, dyna_range, dyna_exponent);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_v4_ns += elapsed_ns;
            ++sampling_calls; ++sampling_v4_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes; sampling_v4_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.mask_bytes; sampling_v4_mask_bytes += selection.mask_bytes;
            if (selection.typical_computed) ++typical_filter_calls;
            if (selection.dynatemp_computed) ++dynatemp_filter_calls;
            dry_penalized_candidates += selection.dry_penalized_candidates;
            dry_penalized_candidates_v4 += selection.dry_penalized_candidates;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"full-vocab-typical-dyna-min-p-u53-v4\",\"token\":" << selection.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.mask_bytes << ",\"legal_count\":" << selection.legal_count
                << ",\"min_p_kept_count\":" << selection.min_p_kept_count
                << ",\"top_k_kept_count\":" << selection.top_k_kept_count
                << ",\"typical_kept_count\":" << selection.typical_kept_count
                << ",\"kept_count\":" << selection.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.dry_penalized_candidates
                << ",\"typical_p\":" << typical_p << ",\"dynatemp_range\":" << dyna_range
                << ",\"dynatemp_exponent\":" << dyna_exponent
                << ",\"effective_temperature\":" << selection.effective_temperature << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
        } else if (command == "sample_mask_v5") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0, draw53_second = min_p_select::kXtcSecondDrawAbsent;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            double typical_p = 1.0, dyna_range = 0.0, dyna_exponent = 1.0; double xtc_probability = 0.0, xtc_threshold = 0.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)
                    >> typical_p >> dyna_range >> dyna_exponent >> xtc_probability >> xtc_threshold >> draw53_second) || (in >> extra)) {
                error_reply("sample_mask_v5 requires v4 arguments followed by xtcProbability xtcThreshold secondDraw"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_v5 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v5 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 || draw53 >= (uint64_t(1) << 53) || (draw53_second >= (uint64_t(1) << 53) && draw53_second != min_p_select::kXtcSecondDrawAbsent) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                !std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0 ||
                !std::isfinite(dyna_range) || dyna_range < 0.0 ||
                !std::isfinite(dyna_exponent) || dyna_exponent <= 0.0 ||
                !std::isfinite(temperature + dyna_range) || !std::isfinite(xtc_probability) || xtc_probability < 0.0 || xtc_probability > 1.0 || !std::isfinite(xtc_threshold) || xtc_threshold < 0.0 || xtc_threshold > 1.0) {
                error_reply("sample_mask_v5 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v5 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v5 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v5 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v5 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v5 mask read failed"); continue; }

            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = dry_active ?
                std::min<uint64_t>(actual_consumed, dry_last_n_arg) : 0;
            std::vector<uint32_t> history, breakers;
            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max()) || breakers.size() > history.size()) {
                error_reply("sample_mask_v5 history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_v5 history is not the native session tail"); continue; }
            min_p_select::V5Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select_v5(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53, draw53_second,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    typical_p, dyna_range, dyna_exponent, xtc_probability, xtc_threshold);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_v5_ns += elapsed_ns;
            ++sampling_calls; ++sampling_v5_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes; sampling_v5_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.base.mask_bytes; sampling_v5_mask_bytes += selection.base.mask_bytes;
            if (selection.base.typical_computed) ++typical_filter_calls_v5;
            if (selection.base.dynatemp_computed) ++dynatemp_filter_calls_v5;
            dry_penalized_candidates += selection.base.dry_penalized_candidates;
            dry_penalized_candidates_v5 += selection.base.dry_penalized_candidates;
            sampling_v5_activation_draws += selection.activation_draws;
            sampling_v5_activated_calls += selection.activated ? 1 : 0;
            sampling_v5_removed_candidates += selection.removed_count;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"full-vocab-xtc-min-p-u53-v5\",\"token\":" << selection.base.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.base.mask_bytes << ",\"legal_count\":" << selection.base.legal_count
                << ",\"min_p_kept_count\":" << selection.base.min_p_kept_count
                << ",\"top_k_kept_count\":" << selection.base.top_k_kept_count
                << ",\"typical_kept_count\":" << selection.base.typical_kept_count
                << ",\"kept_count\":" << selection.base.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.base.dry_penalized_candidates
                << ",\"typical_p\":" << typical_p << ",\"dynatemp_range\":" << dyna_range
                << ",\"dynatemp_exponent\":" << dyna_exponent
                << ",\"effective_temperature\":" << selection.base.effective_temperature << ",\"total_weight\":" << selection.base.total_weight << ",\"draw53_second\":" << selection.draw53_second << ",\"draw_count\":" << selection.draw_count << ",\"activation_draws\":" << selection.activation_draws << ",\"activated\":" << (selection.activated ? "true" : "false") << ",\"removed_count\":" << selection.removed_count << ",\"pre_xtc_count\":" << selection.pre_xtc_count << ",\"xtc_probability\":" << xtc_probability << ",\"xtc_threshold\":" << xtc_threshold << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
                } else if (command == "sample_mask_mirostat2") {
            int64_t expected_consumed = 0;
            uint64_t draw53 = 0, dry_allowed_arg = 0, dry_last_n_arg = 0;
            double temperature = 0.0, tau = 0.0, eta = 0.0, mu = 0.0;
            double dry_multiplier = 0.0, dry_base = 1.0;
            std::string dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> draw53 >> std::quoted(mask_path)
                    >> tau >> eta >> mu >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)) || (in >> extra)) {
                error_reply("sample_mask_mirostat2 requires consumed T draw53 mask_path tau eta mu dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path"); continue;
            }
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_mirostat2 DRY integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_mirostat2 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(tau) || tau <= 0.0 || !std::isfinite(eta) || eta < 0.0 ||
                !std::isfinite(mu) || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0) {
                error_reply("sample_mask_mirostat2 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_mirostat2 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_mirostat2 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_mirostat2 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_mirostat2 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_mirostat2 mask read failed"); continue; }
            const size_t used_bits = vocab % 8;
            if (used_bits != 0 && (mask.back() & static_cast<uint8_t>(~((1u << used_bits) - 1u))) != 0) {
                error_reply("sample_mask_mirostat2 mask has nonzero padding bits"); continue;
            }

            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = dry_active ?
                std::min<uint64_t>(actual_consumed, dry_last_n_arg) : 0;
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_mirostat2 history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_mirostat2 history is not the native session tail"); continue; }

            min_p_select::DryConfig dry{dry_multiplier, dry_base,
                static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)};
            size_t dry_penalized_candidates = 0;
            std::vector<float> adjusted;
            std::vector<uint32_t> legal_ids;
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                adjusted = min_p_select::apply_dry_penalties(frontier.data(), vocab, history, breakers,
                    dry, &dry_penalized_candidates);
                legal_ids.reserve(vocab);
                for (size_t token = 0; token < vocab; ++token) {
                    if (!min_p_select::finite_binary32(frontier[token]))
                        throw std::invalid_argument("Mirostat2 logits contain a nonfinite score");
                    if (min_p_select::legal(mask.data(), token)) legal_ids.push_back(static_cast<uint32_t>(token));
                }
                const auto selection = mirostat2_select::choose(adjusted, legal_ids,
                    temperature, draw53, mu, tau, eta);
                const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - selection_start).count();
                sampling_ns += elapsed_ns; sampling_mirostat2_ns += elapsed_ns;
                ++sampling_calls; ++sampling_mirostat2_calls;
                const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
                sampling_logit_bytes += logit_bytes; sampling_mirostat2_logit_bytes += logit_bytes;
                sampling_mask_bytes += expected_mask_bytes; sampling_mirostat2_mask_bytes += expected_mask_bytes;
                const auto old_precision = std::cout.precision();
                std::cout << std::setprecision(17)
                    << "{\"status\":\"ok\",\"operator\":\"full-vocab-mirostat2\","
                    << "\"protocol\":\"full-vocab-mirostat2-u53-v6\","
                    << "\"feedback_protocol\":\"mirostat2-state-v1\",\"feedback_version\":1"
                    << ",\"token\":" << selection.token << ",\"consumed\":" << actual_consumed
                    << ",\"logits_count\":" << vocab << ",\"mask_bytes\":" << expected_mask_bytes
                    << ",\"legal_count\":" << legal_ids.size() << ",\"kept_count\":" << selection.kept_ids.size()
                    << ",\"legal_total_weight\":" << selection.legal_total_weight
                    << ",\"total_weight\":" << selection.total_weight << ",\"draw53\":" << draw53
                    << ",\"draw_count\":1,\"temperature\":" << temperature << ",\"tau\":" << tau
                    << ",\"eta\":" << eta << ",\"mu\":" << mu << ",\"next_mu\":" << selection.next_mu
                    << ",\"selected_log_probability\":" << selection.selected_log_probability
                    << ",\"selected_probability\":" << selection.selected_probability
                    << ",\"surprise_bits\":" << selection.surprise_bits
                    << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                    << ",\"dry_allowed_length\":" << dry_allowed_arg
                    << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                    << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                    << ",\"dry_penalized_candidates\":" << dry_penalized_candidates << "}";
                std::cout.precision(old_precision);
                std::cout << std::endl;
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
                } else if (command == "sample_mask_adaptive_p") {
            int64_t expected_consumed = 0;
            uint64_t draw53 = 0, top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 0.0;
            double dry_multiplier = 0.0, dry_base = 1.0, typical_p = 1.0;
            double dyna_range = 0.0, dyna_exponent = 1.0, adaptive_target = 0.0;
            double adaptive_decay = 0.0, state_weighted_sum = 0.0, state_total_weight = 0.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path;
            std::string history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path) >> typical_p >> dyna_range
                    >> dyna_exponent >> adaptive_target >> adaptive_decay >> state_weighted_sum >> state_total_weight)
                    || (in >> extra)) {
                error_reply("sample_mask_adaptive_p requires consumed T min_p top_k top_p draw53 mask_path dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path typical_p dyna_range dyna_exponent target decay weighted_sum total_weight"); continue;
            }
            const auto parse_u64 = [](const std::string& text, uint64_t& value) {
                if (text.empty() || text[0] == '-') return false;
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
                return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
            };
            if (!parse_u64(top_k_text, top_k_arg) || top_k_arg > std::numeric_limits<size_t>::max() ||
                !parse_u64(dry_allowed_text, dry_allowed_arg) || dry_allowed_arg > UINT32_MAX ||
                !parse_u64(dry_last_n_text, dry_last_n_arg) || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_adaptive_p integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_adaptive_p consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                !std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0 ||
                !std::isfinite(dyna_range) || dyna_range < 0.0 ||
                !std::isfinite(dyna_exponent) || dyna_exponent <= 0.0 ||
                !std::isfinite(temperature + dyna_range) ||
                !std::isfinite(adaptive_target) || adaptive_target > 1.0 ||
                !std::isfinite(adaptive_decay) || adaptive_decay < 0.0 || adaptive_decay > 0.99 ||
                !std::isfinite(state_weighted_sum) || !std::isfinite(state_total_weight) || state_total_weight <= 0.0) {
                error_reply("sample_mask_adaptive_p parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_adaptive_p frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_adaptive_p could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_adaptive_p truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_adaptive_p extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_adaptive_p mask read failed"); continue; }
            const size_t used_bits = vocab % 8;
            if (used_bits != 0 && (mask.back() & static_cast<uint8_t>(~((1u << used_bits) - 1u))) != 0) {
                error_reply("sample_mask_adaptive_p mask has nonzero padding bits"); continue;
            }
            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = adaptive_p_select::expected_dry_history_count(
                actual_consumed, dry_multiplier, dry_last_n_arg);
            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                return true;
            };
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_adaptive_p history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_adaptive_p history is not the native session tail"); continue; }
            if (dry_active) {
                uint32_t prior = 0; bool first_breaker = true;
                for (uint32_t breaker : breakers) {
                    if (breaker >= vocab || (!first_breaker && breaker <= prior)) { history_matches = false; break; }
                    first_breaker = false; prior = breaker;
                }
                if (!history_matches) { error_reply("sample_mask_adaptive_p breakers are not sorted unique vocabulary IDs"); continue; }
            }

            min_p_select::DryConfig dry{dry_multiplier, dry_base,
                static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)};
            adaptive_p_select::State state{state_weighted_sum, state_total_weight};
            size_t dry_penalized_candidates = 0;
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                const auto prepared = min_p_select::select_v4(frontier.data(), vocab, mask.data(),
                    expected_mask_bytes, temperature, min_p, static_cast<size_t>(top_k_arg), top_p,
                    draw53, history, breakers, dry, typical_p, dyna_range, dyna_exponent, true);
                dry_penalized_candidates = prepared.dry_penalized_candidates;
                const auto selection = adaptive_p_select::select_prepared(
                    prepared.v5_post_top_p_weights, prepared.v5_post_top_p_ids, state,
                    adaptive_target, adaptive_decay, draw53);
                const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - selection_start).count();
                sampling_ns += elapsed_ns; sampling_adaptive_p_ns += elapsed_ns;
                ++sampling_calls; ++sampling_adaptive_p_calls;
                const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
                sampling_logit_bytes += logit_bytes; sampling_adaptive_p_logit_bytes += logit_bytes;
                sampling_mask_bytes += expected_mask_bytes; sampling_adaptive_p_mask_bytes += expected_mask_bytes;
                const auto old_precision = std::cout.precision();
                std::cout << std::setprecision(17)
                    << "{\"status\":\"ok\",\"operator\":\"full-vocab-adaptive-p\","
                    << "\"protocol\":\"full-vocab-adaptive-p-u53-v7\","
                    << "\"feedback_protocol\":\"adaptive-p-state-v1\",\"feedback_version\":1"
                    << ",\"token\":" << selection.token << ",\"consumed\":" << actual_consumed
                    << ",\"logits_count\":" << vocab << ",\"mask_bytes\":" << expected_mask_bytes
                    << ",\"legal_count\":" << prepared.legal_count
                    << ",\"min_p_kept_count\":" << prepared.min_p_kept_count
                    << ",\"top_k_kept_count\":" << prepared.top_k_kept_count
                    << ",\"typical_kept_count\":" << prepared.typical_kept_count
                    << ",\"kept_count\":" << selection.legal_count
                    << ",\"categorical_total_weight\":" << selection.total_weight
                    << ",\"draw53\":" << draw53 << ",\"draw_count\":1"
                    << ",\"temperature\":" << temperature << ",\"effective_temperature\":" << prepared.effective_temperature
                    << ",\"min_p\":" << min_p << ",\"top_k\":" << top_k_arg << ",\"top_p\":" << top_p
                    << ",\"typical_p\":" << typical_p << ",\"dynatemp_range\":" << dyna_range
                    << ",\"dynatemp_exponent\":" << dyna_exponent
                    << ",\"adaptive_target\":" << adaptive_target << ",\"adaptive_decay\":" << adaptive_decay
                    << ",\"original_probability\":" << selection.original_probability
                    << ",\"adapted_target\":";
                if (selection.adapted_target) std::cout << *selection.adapted_target;
                else std::cout << "null";
                std::cout << ",\"state_weighted_sum\":" << selection.source_weighted_sum
                    << ",\"state_total_weight\":" << selection.source_total_weight
                    << ",\"next_weighted_sum\":" << selection.next_weighted_sum
                    << ",\"next_total_weight\":" << selection.next_total_weight
                    << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                    << ",\"dry_allowed_length\":" << dry_allowed_arg
                    << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                    << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                    << ",\"dry_penalized_candidates\":" << dry_penalized_candidates << "}";
                std::cout.precision(old_precision);
                std::cout << std::endl;
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
        } else if (command == "generate") {
            // One resident, greedy request. Progress is fully consumed but is
            // not a canonical snapshot. The caller owns durable publication.
            int64_t limit = 0, eos_count = 0;
            std::vector<int32_t> eos;
            bool valid = bool(in >> limit >> eos_count) && limit > 0 && eos_count >= 0 &&
                eos_count <= forward.vocab() && !session.ids().empty() &&
                limit <= state.max_cells - (int64_t)session.ids().size();
            for (int64_t i = 0; valid && i < eos_count; ++i) {
                int64_t token;
                valid = bool(in >> token) && token >= 0 && token < forward.vocab();
                if (valid) eos.push_back((int32_t)token);
            }
            std::string extra;
            if (!valid || (in >> extra)) { error_reply("invalid greedy request"); continue; }
            const auto start_position = session.ids().size();
            std::vector<int64_t> completed_positions;
            bool ok = true;
            for (int64_t i = 0; i < limit; ++i) {
                const auto& frontier = session.logits();
                if (frontier.empty() || !std::all_of(frontier.begin(), frontier.end(), [](float v) { return std::isfinite(v); })) {
                    error = "nonfinite greedy frontier"; ok = false; break;
                }
                const auto token = (int32_t)std::distance(frontier.begin(), std::max_element(frontier.begin(), frontier.end()));
                const auto position = (int64_t)session.ids().size();
                int32_t unused = 0;
                pool_failed = false;
                auto stage_timer = std::chrono::steady_clock::now();
                const bool prepared = session.begin(position, 1, error);
                decode_prepare_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-stage_timer).count();
                if (!prepared) { ok = false; break; }
                stage_timer = std::chrono::steady_clock::now();
                const bool computed = forward.run(1, &token, position, pool, user, &unused, error) && !pool_failed &&
                    forward.commit(1, error) && forward.copy_logits(0, logits.data());
                decode_compute_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-stage_timer).count();
                if (!computed) { ok = false; break; }
                stage_timer = std::chrono::steady_clock::now();
                const bool finished = session.finish({token}, logits, error);
                decode_finish_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-stage_timer).count();
                if (!finished) { ok = false; break; }
                ++decode_timed_tokens;
                completed_positions.push_back(position);
                std::cout << "{\"status\":\"token\",\"token\":" << token
                          << ",\"consumed\":" << session.ids().size() << "}" << std::endl;
                // ACK is control only, not another model request. It provides a
                // deterministic cancellation boundary without a second owner.
                std::string acknowledgment;
                if (!std::getline(std::cin, acknowledgment) ||
                    (acknowledgment != "advance" && acknowledgment != "cancel")) {
                    error_reply("invalid greedy request acknowledgment"); return 1;
                }
                if (acknowledgment == "cancel" || std::find(eos.begin(), eos.end(), token) != eos.end()) break;
            }
            if (!ok) { error_reply(error.empty() ? "native greedy forward failed" : error); continue; }
            std::cout << "{\"status\":\"ok\",\"operator\":\"normal-verifier-decode\",\"position\":"
                      << start_position << ",\"count\":" << completed_positions.size()
                      << ",\"generated\":" << completed_positions.size()
                      << ",\"consumed\":" << session.ids().size()
                      << ",\"forward_tokens\":" << (forward.completed_forward_tokens + prefill_tokens)
                      << ",\"chunks\":[";
            for (size_t i = 0; i < completed_positions.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << '[' << completed_positions[i] << ",1]";
            }
            std::cout << "]}" << std::endl;
        } else if (command == "decode") {
            int32_t token; std::string extra;
            if (!(in >> token) || (in >> extra) || token < 0 || token >= forward.vocab()) {
                error_reply("decode requires one selected token"); continue;
            }
            const int64_t position = session.ids().size(); int32_t unused = 0; pool_failed = false;
            auto timer = std::chrono::steady_clock::now();
            const bool prepared = session.begin(position, 1, error);
            decode_prepare_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-timer).count();
            if (!prepared) { error_reply(error); continue; }
            timer = std::chrono::steady_clock::now();
            const bool computed = forward.run(1, &token, position, pool, user, &unused, error) &&
                !pool_failed && forward.commit(1, error) && forward.copy_logits(0, logits.data());
            decode_compute_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-timer).count();
            if (!computed) { error_reply(error); continue; }
            timer = std::chrono::steady_clock::now();
            const bool finished = session.finish({token}, logits, error);
            decode_finish_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-timer).count();
            if (!finished) { error_reply(error); continue; }
            ++decode_timed_tokens;
            std::cout << "{\"status\":\"ok\",\"operator\":\"normal-verifier-decode\",\"position\":" << position << ",\"count\":1,\"consumed\":" << session.ids().size() << ",\"chunks\":[[" << position << ",1]]}" << std::endl;
        } else if (command == "capture") {
            auto image = session.freeze(error);
            if (!image) { error_reply(error); continue; }
            try {
                const auto handle = images.insert(std::move(image));
                std::cout << "{\"status\":\"ok\",\"handle\":" << handle
                          << ",\"consumed\":" << session.ids().size() << "}" << std::endl;
            } catch (const std::exception& e) { error_reply(e.what()); }
        } else if (command == "restore" || command == "drop") {
            uint64_t index;
            std::string extra;
            if (!(in >> index) || (in >> extra)) {
                error_reply("missing derived checkpoint handle"); continue;
            }
            if (command == "drop") {
                if (!images.drop(index)) { error_reply("missing derived checkpoint handle"); continue; }
            } else {
                const auto* image = images.find(index);
                if (!image) { error_reply("missing derived checkpoint handle"); continue; }
                if (!session.restore(*image, error)) { error_reply(error); continue; }
            }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "logits") {
            std::string path, extra;
            if (!(in >> path) || (in >> extra) || session.logits().empty()) { error_reply("missing path or frontier"); continue; }
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write((const char*)session.logits().data(), (std::streamsize)session.logits().size()*sizeof(float));
            out.close();
            if (!out) { error_reply("frontier write failed"); continue; }
            std::cout << "{\"status\":\"ok\",\"bytes\":" << session.logits().size()*sizeof(float) << "}" << std::endl;
        } else if (command == "park") {
            std::string extra;
            size_t estimate = 0;
            if ((in >> extra) || !session.host_capture_bytes(estimate, error)) {
                error_reply(error.empty() ? "park takes no arguments" : error); continue;
            }
            if (!ram_cache.make_room(estimate)) { error_reply("RAM checkpoint admission refused"); continue; }
            SavedConversation image;
            if (!session.capture_host(image, error)) { error_reply(error); continue; }
            if (!ram_cache.put(std::move(image))) { error_reply("RAM checkpoint admission refused"); continue; }
            const auto metrics = ram_cache.storage_metrics();
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"bytes\":" << ram_cache.bytes() << ",\"payload_bytes\":" << metrics.payload_bytes
                      << ",\"entries\":" << ram_cache.size() << "}" << std::endl;
        } else if (command == "activate") {
            std::string path, extra;
            if (!(in >> std::quoted(path)) || (in >> extra)) { error_reply("activate requires exact history path"); continue; }
            std::ifstream input(path);
            std::vector<int32_t> ids;
            int64_t token;
            bool valid = bool(input);
            while (input >> token) {
                if (token < 0 || token >= forward.vocab() || ids.size() >= (size_t)state.max_cells) {
                    valid = false; break;
                }
                ids.push_back((int32_t)token);
            }
            if (!valid || !input.eof() || ids.empty()) { error_reply("invalid exact RAM history"); continue; }
            const auto* image = ram_cache.exact(ids, {}, false);
            if (!image) {
                ++ram_misses;
                std::cout << "{\"status\":\"ok\",\"hit\":false}" << std::endl;
                continue;
            }
            if (image->next_logits.size() != (size_t)forward.vocab() ||
                !session.restore_host(*image, ids, image->next_logits, error)) {
                error_reply(error.empty() ? "invalid cached frontier extent" : error); continue;
            }
            ++ram_hits;
            std::cout << "{\"status\":\"ok\",\"hit\":true,\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "ram_clear") {
            const auto count = ram_cache.clear();
            std::cout << "{\"status\":\"ok\",\"removed\":" << count << "}" << std::endl;
        } else if (command == "ram_stats") {
            const auto metrics = ram_cache.storage_metrics();
            std::cout << "{\"status\":\"ok\",\"bytes\":" << ram_cache.bytes()
                      << ",\"entries\":" << ram_cache.size() << ",\"hits\":" << ram_hits << ",\"misses\":" << ram_misses
                      << ",\"payload_bytes\":" << metrics.payload_bytes << ",\"payload_allocations\":" << metrics.allocations
                      << ",\"payload_owners\":" << metrics.owners << ",\"copied_bytes\":" << metrics.copied_bytes << "}" << std::endl;
        } else if (command == "write") {
            std::string path, extra;
            if (!(in >> std::quoted(path)) || (in >> extra)) { error_reply("missing checkpoint path"); continue; }
            size_t bytes = 0;
            if (!session.write(path, bytes, error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"bytes\":" << bytes << "}" << std::endl;
        } else if (command == "load") {
            std::string path, ids_path, logits_path, extra;
            if (!(in >> std::quoted(path) >> std::quoted(ids_path) >> std::quoted(logits_path)) || (in >> extra)) {
                error_reply("cold load requires state, exact history and frontier paths"); continue;
            }
            std::ifstream inputs(ids_path);
            std::vector<int32_t> ids;
            int64_t token;
            bool valid = bool(inputs);
            while (inputs >> token) {
                if (token < 0 || token >= forward.vocab() || ids.size() >= (size_t)state.max_cells) {
                    valid = false; break;
                }
                ids.push_back((int32_t)token);
            }
            std::ifstream frontier(logits_path, std::ios::binary | std::ios::ate);
            const auto bytes = (std::streamsize)forward.vocab()*sizeof(float);
            if (!valid || !inputs.eof() || ids.empty() || !frontier || frontier.tellg() != bytes) {
                error_reply("invalid cold history or frontier extent"); continue;
            }
            std::vector<float> next_logits((size_t)forward.vocab());
            frontier.seekg(0);
            frontier.read((char*)next_logits.data(), bytes);
            if (!frontier) { error_reply("incomplete cold frontier"); continue; }
            if (!session.load(path, ids, next_logits, error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "audit") {
            uint64_t index;
            std::string path, extra;
            if (!(in >> index >> std::quoted(path)) || (in >> extra)) {
                error_reply("audit requires a retained handle and output path"); continue;
            }
            const auto* image = images.find(index);
            if (!image) { error_reply("audit requires a retained handle and output path"); continue; }
            BranchSessionAudit result;
            if (!session.audit(*image, path, result, error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"bytes\":" << result.bytes << ",\"consumed\":" << result.consumed
                      << ",\"qsa_page_cells\":" << result.qsa_page_cells
                      << ",\"index_block_cells\":" << result.index_block_cells
                      << ",\"allocation_page_bytes\":" << vmm_granularity() << ",\"regions\":[";
            bool first = true;
            for (const auto& r : result.regions) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"kind\":\"" << r.kind << "\",\"offset\":" << r.offset << ",\"bytes\":" << r.bytes
                          << ",\"pool\":" << r.pool << ",\"qsa_ordinal\":" << r.qsa_ordinal
                          << ",\"pool_offset\":" << r.pool_offset << ",\"bytes_per_cell\":" << r.bytes_per_cell
                          << ",\"filled_prefix_bytes\":" << r.filled_prefix_bytes
                          << ",\"allocation_id\":" << r.page.allocation_id << ",\"driver_handle\":" << r.page.driver_handle
                          << ",\"owners\":" << r.page.owners << '}';
            }
            std::cout << "]}" << std::endl;
        } else if (command == "pages") {
            std::cout << "{\"status\":\"ok\",\"pages\":[";
            bool first = true;
            for (const auto& page : session.ownership()) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << page.allocation_id << ',' << page.driver_handle << ',' << page.owners << ',' << page.bytes
                          << ",\"" << page.kind << "\"," << page.pool << ']';
            }
            std::cout << "]}" << std::endl;
        } else if (command == "stats") {
            const auto p = BranchPages::counters();
            const auto s = session.counters();
            const auto e = vmm_failure_counters();
            const auto h = branch_digest_counters();
            const auto snapshots = BranchSession::snapshot_counters();
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"forward_tokens\":" << (forward.completed_forward_tokens + prefill_tokens)
                      << ",\"prefill_tokens\":" << prefill_tokens
                      << ",\"prefill_chunks\":" << prefill_chunks << ",\"grammar_selection_calls\":" << grammar_selection_calls << ",\"grammar_selection_logit_bytes\":" << grammar_selection_logit_bytes << ",\"grammar_selection_mask_bytes\":" << grammar_selection_mask_bytes << ",\"sampling_calls\":" << sampling_calls << ",\"sampling_logit_bytes\":" << sampling_logit_bytes << ",\"sampling_mask_bytes\":" << sampling_mask_bytes << ",\"sampling_ns\":" << sampling_ns << ",\"sampling_v2_calls\":" << sampling_v2_calls << ",\"sampling_v2_logit_bytes\":" << sampling_v2_logit_bytes << ",\"sampling_v2_mask_bytes\":" << sampling_v2_mask_bytes << ",\"sampling_v2_ns\":" << sampling_v2_ns << ",\"sampling_v3_calls\":" << sampling_v3_calls << ",\"sampling_v3_logit_bytes\":" << sampling_v3_logit_bytes << ",\"sampling_v3_mask_bytes\":" << sampling_v3_mask_bytes << ",\"sampling_v3_ns\":" << sampling_v3_ns << ",\"sampling_mixed_dry_calls\":" << sampling_mixed_dry_calls << ",\"sampling_mixed_dry_ns\":" << sampling_mixed_dry_ns << ",\"sampling_mixed_dry_logit_bytes\":" << sampling_mixed_dry_logit_bytes << ",\"sampling_mixed_dry_mask_bytes\":" << sampling_mixed_dry_mask_bytes << ",\"sampling_v4_calls\":" << sampling_v4_calls << ",\"sampling_v4_logit_bytes\":" << sampling_v4_logit_bytes << ",\"sampling_v4_mask_bytes\":" << sampling_v4_mask_bytes << ",\"sampling_v4_ns\":" << sampling_v4_ns << ",\"sampling_image_dry_combination_calls\":" << sampling_image_dry_combination_calls << ",\"sampling_v5_calls\":" << sampling_v5_calls << ",\"sampling_v5_logit_bytes\":" << sampling_v5_logit_bytes << ",\"sampling_v5_mask_bytes\":" << sampling_v5_mask_bytes << ",\"sampling_v5_ns\":" << sampling_v5_ns << ",\"sampling_mirostat2_calls\":" << sampling_mirostat2_calls << ",\"sampling_mirostat2_logit_bytes\":" << sampling_mirostat2_logit_bytes << ",\"sampling_mirostat2_mask_bytes\":" << sampling_mirostat2_mask_bytes << ",\"sampling_mirostat2_ns\":" << sampling_mirostat2_ns << ",\"sampling_adaptive_p_calls\":" << sampling_adaptive_p_calls << ",\"sampling_adaptive_p_logit_bytes\":" << sampling_adaptive_p_logit_bytes << ",\"sampling_adaptive_p_mask_bytes\":" << sampling_adaptive_p_mask_bytes << ",\"sampling_adaptive_p_ns\":" << sampling_adaptive_p_ns << ",\"sampling_v5_activation_draws\":" << sampling_v5_activation_draws << ",\"sampling_v5_activated_calls\":" << sampling_v5_activated_calls << ",\"sampling_v5_removed_candidates\":" << sampling_v5_removed_candidates << ",\"typical_filter_calls_v5\":" << typical_filter_calls_v5 << ",\"dynatemp_filter_calls_v5\":" << dynatemp_filter_calls_v5 << ",\"dry_penalized_candidates_v5\":" << dry_penalized_candidates_v5 << ",\"typical_filter_calls\":" << typical_filter_calls << ",\"dynatemp_filter_calls\":" << dynatemp_filter_calls << ",\"dry_penalized_candidates_v4\":" << dry_penalized_candidates_v4 << ",\"dry_penalized_candidates\":" << dry_penalized_candidates
                      << ",\"decode_prepare_ns\":" << decode_prepare_ns
                      << ",\"decode_compute_ns\":" << decode_compute_ns
                      << ",\"decode_finish_ns\":" << decode_finish_ns
                      << ",\"decode_timed_tokens\":" << decode_timed_tokens
                      << ",\"windows\":" << forward.windows
              << ",\"resident_expert_entries\":" << dispatch.cache_hits
              << ",\"cpu_expert_entries\":" << dispatch.cache_refused
              << ",\"pcie_expert_groups\":" << dispatch.pcie_experts
                      << ",\"allocations\":" << p.allocations << ",\"releases\":" << p.releases
                      << ",\"live_bytes\":" << p.live_bytes << ",\"cow_pages\":" << p.cow_pages
                      << ",\"copied_bytes\":" << p.copied_bytes << ",\"mappings\":" << p.mappings
                      << ",\"unmappings\":" << p.unmappings
                      << ",\"address_reservations\":" << p.address_reservations
                      << ",\"address_releases\":" << p.address_releases
                      << ",\"vmm_driver_failures\":" << e.driver_failures
                      << ",\"vmm_out_of_memory_failures\":" << e.out_of_memory_failures
                      << ",\"vmm_last_driver_error\":" << e.last_error
                      << ",\"vmm_last_failed_operation\":\"" << e.last_operation << "\""
                      << ",\"audit_read_bytes\":" << p.audit_read_bytes
                      << ",\"integrity_checked_pages\":" << p.integrity_checked_pages
                      << ",\"integrity_mismatches\":" << p.integrity_mismatches
                      << ",\"integrity_last_mismatch_tile\":" << p.last_integrity_mismatch_tile
                      << ",\"integrity_seal_allocations\":" << p.seal_allocations
                      << ",\"integrity_seal_releases\":" << p.seal_releases
                      << ",\"integrity_seal_live_bytes\":" << p.seal_live_bytes
                      << ",\"integrity_seal_upload_bytes\":" << p.seal_upload_bytes
                      << ",\"integrity_device_hashed_bytes\":" << h.device_hashed_bytes
                      << ",\"integrity_digest_copy_bytes\":" << h.digest_copy_bytes
                      << ",\"integrity_plan_copy_bytes\":" << h.plan_copy_bytes
                      << ",\"integrity_expected_seal_plan_copy_bytes\":" << h.expected_seal_plan_copy_bytes
                      << ",\"integrity_compare_flag_init_bytes\":" << h.compare_flag_init_bytes
                      << ",\"integrity_compare_result_copy_bytes\":" << h.compare_result_copy_bytes
                      << ",\"integrity_plan_upload_ns\":" << h.plan_upload_ns
                      << ",\"integrity_page_hash_ns\":" << h.page_hash_ns
                      << ",\"integrity_page_digest_download_ns\":" << h.page_digest_download_ns
                      << ",\"integrity_seal_compare_ns\":" << h.seal_compare_ns
                      << ",\"integrity_seal_result_download_ns\":" << h.seal_result_download_ns
                      << ",\"integrity_compare_workspace_live_bytes\":" << h.compare_workspace_live_bytes
                      << ",\"integrity_host_workspace_allocations\":" << h.host_workspace_allocations
                      << ",\"integrity_host_workspace_live_bytes\":" << h.host_workspace_live_bytes
                      << ",\"integrity_kernel_calls\":" << h.kernel_calls
                      << ",\"integrity_workspace_allocations\":" << h.workspace_allocations
                      << ",\"integrity_workspace_releases\":" << h.workspace_releases
                      << ",\"integrity_workspace_live_bytes\":" << h.workspace_live_bytes
                      << ",\"binding_copy_bytes\":" << s.binding_copy_bytes
                      << ",\"saved_running_bytes\":" << s.saved_running_bytes
                      << ",\"restored_running_bytes\":" << s.restored_running_bytes
                      << ",\"written_file_bytes\":" << s.written_file_bytes
                      << ",\"loaded_file_bytes\":" << s.loaded_file_bytes
                      << ",\"cold_restore_copy_bytes\":" << s.cold_restore_copy_bytes
                      << ",\"cold_restore_copy_calls\":" << s.cold_restore_copy_calls
                      << ",\"captures\":" << s.captures << ",\"restores\":" << s.restores
                      << ",\"handle_allocations\":" << handle_work.allocations
                      << ",\"handle_releases\":" << handle_work.releases
                      << ",\"handle_live_entries\":" << handle_work.live_entries
                      << ",\"handle_live_bytes\":" << handle_work.live_bytes
                      << ",\"handle_peak_bytes\":" << handle_work.peak_bytes
                      << ",\"handle_issued\":" << handle_work.issued
                      << ",\"handle_lookup_steps\":" << handle_work.lookup_steps
                      << ",\"snapshot_allocations\":" << snapshots.allocations
                      << ",\"snapshot_releases\":" << snapshots.releases
                      << ",\"snapshot_live_images\":" << snapshots.live_images
                      << ",\"snapshot_payload_capacity_bytes\":" << snapshots.payload_capacity_bytes
                      << ",\"snapshot_peak_payload_capacity_bytes\":" << snapshots.peak_payload_capacity_bytes
                      << ",\"host_heap_allocated_bytes\":";
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 33)
            const auto heap = mallinfo2();
            std::cout << heap.uordblks + heap.hblkhd;
#else
            std::cout << "null";
#endif
#else
            std::cout << "null";
#endif
            std::cout << "}" << std::endl;
        } else error_reply("unknown command");
    }
    return 0;
}
} // namespace strata::core
