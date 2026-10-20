"""Build an independent fresh multi-turn native oracle in a new output directory.

The executable follows this path without importing/restoring BranchSession or
ConversationCache state: first prompt Prefill, normal verifier generation to EOS,
next-user suffix Prefill, then normal verifier generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os

import build_fast_prefill_probe as base
import build_batched_resume_probe as batched_base


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_stopped_turn(tokens, eos_tokens, max_tokens):
    if not eos_tokens:
        raise ValueError('EOS set is empty')
    if not tokens:
        raise ValueError('no first-turn tokens')
    if len(tokens) > max_tokens or tokens[-1] not in eos_tokens:
        raise ValueError('first turn did not stop on EOS')
    return True


EOS_INIT = r'''    std::vector<int32_t> eos_tokens;
    const char* eos_text = std::getenv("STRATA_EOS_IDS");
    if (!eos_text) { std::cerr << "STRATA_EOS_IDS is required" << std::endl; return 2; }
    std::string eos_words(eos_text); std::replace(eos_words.begin(), eos_words.end(), ',', ' ');
    std::istringstream eos_input(eos_words); int64_t eos_id = -1;
    while (eos_input >> eos_id) {
        if (eos_id < 0 || eos_id >= (int64_t)logits.size()) return 2;
        eos_tokens.push_back((int32_t)eos_id);
    }
    if (!eos_input.eof() || eos_tokens.empty()) { std::cerr << "invalid EOS list"; return 2; }
'''

MULTITURN = r'''    const bool first_stopped_on_eos = !generated.empty() &&
        std::find(eos_tokens.begin(), eos_tokens.end(), generated.back()) != eos_tokens.end();
    const char* suffix_path = std::getenv("STRATA_NEXT_SUFFIX_FILE");
    if (!suffix_path) { std::cerr << "STRATA_NEXT_SUFFIX_FILE is required" << std::endl; return 2; }
    // Do not admit a synthetic EOS or a max-bound partial turn as a new turn.
    if (!first_stopped_on_eos) {
        std::cerr << "EOS_GUARD_REJECT first_generated_tokens=" << generated.size()
                  << " first_prefill_work_tokens=" << first_prefill_work_tokens
                  << " prefill_work_since_first_prompt="
                  << batched.stats().tokens-first_prefill_work_tokens << std::endl;
        std::cerr << "first turn did not stop on EOS; suffix prefill refused" << std::endl; return 3;
    }
    bool first_outputs_verified = false;
    if (const char* expected_path = std::getenv("STRATA_EXPECT_FIRST_FILE")) {
        std::ifstream expected_file(expected_path);
        std::vector<int32_t> expected_tokens; int64_t expected_id = -1;
        while (expected_file >> expected_id) {
            if (expected_id < 0 || expected_id >= (int64_t)logits.size()) return 2;
            expected_tokens.push_back((int32_t)expected_id);
        }
        if (!expected_file.eof() || expected_tokens != generated) {
            std::cerr << "first-turn tokens differ from fixture; suffix prefill refused" << std::endl; return 4;
        }
        first_outputs_verified = true;
    }
    std::vector<int32_t> suffix_tokens, second_generated;
    std::ifstream suffix_file(suffix_path);
    int64_t suffix_id = -1;
    while (suffix_file >> suffix_id) {
        if (suffix_id < 0 || suffix_id >= (int64_t)logits.size() ||
            initial_tokens.size()+generated.size()+suffix_tokens.size() >= (size_t)state.max_cells) {
            std::cerr << "invalid suffix token or context capacity" << std::endl; return 2;
        }
        suffix_tokens.push_back((int32_t)suffix_id);
    }
    if (!suffix_file.eof() || suffix_tokens.empty()) {
        std::cerr << "invalid or empty suffix file" << std::endl; return 2;
    }
    const int64_t suffix_position = (int64_t)(initial_tokens.size()+generated.size());
    const size_t first_chunk_count = prefill_chunk_positions.size();
    const uint64_t prefill_tokens_before = batched.stats().tokens;
    const uint64_t prefill_chunks_before = batched.stats().chunks;
    std::vector<int64_t> suffix_wide(suffix_tokens.begin(), suffix_tokens.end());
    if (!batched.run(suffix_wide.data(), (int64_t)suffix_wide.size(), suffix_position, error)) {
        std::cerr << "second user suffix prefill failed: " << error << std::endl; return 1;
    }
    const uint64_t second_prefill_tokens = batched.stats().tokens-prefill_tokens_before;
    const uint64_t second_prefill_chunks = batched.stats().chunks-prefill_chunks_before;
    std::ofstream prefill_schedule(output + ".prefill-schedule.json");
    prefill_schedule << "{\"first\":[";
    for (size_t i=0;i<first_chunk_count;++i) {
        if(i) prefill_schedule << ',';
        prefill_schedule << '[' << prefill_chunk_positions[i].first << ',' << prefill_chunk_positions[i].second << ']';
    }
    prefill_schedule << "],\"second\":[";
    for (size_t i=first_chunk_count;i<prefill_chunk_positions.size();++i) {
        if(i>first_chunk_count) prefill_schedule << ',';
        prefill_schedule << '[' << prefill_chunk_positions[i].first << ',' << prefill_chunk_positions[i].second << ']';
    }
    prefill_schedule << "]}"; prefill_schedule.close(); if (!prefill_schedule) return 1;
    const int64_t second_limit = 128;
    if ((uint64_t)second_limit > state.max_cells-initial_tokens.size()-generated.size()-suffix_tokens.size()) {
        std::cerr << "second generation exceeds context capacity" << std::endl; return 2;
    }
    std::ofstream second_frontier(output + ".second.frontier.f32", std::ios::binary);
    second_frontier.write((const char*)logits.data(), logits.size()*sizeof(float));
    second_frontier.close(); if (!second_frontier) return 1;

    // Reuse the ordinary Verifier object across turns; only the suffix input
    // is processed by native batched Prefill between the two decode segments.
    dispatch.plan = decode.plan_sink();
    std::ofstream second_trace(output + ".second.decode.f32", std::ios::binary);
    for (int64_t i=0; i<second_limit; ++i) {
        if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); })) {
            std::cerr << "nonfinite second-turn frontier" << std::endl; return 1;
        }
        int32_t token = 0;
        for (int32_t t=1; t<(int32_t)logits.size(); ++t)
            if (logits[t] > logits[token]) token = t;
        int32_t unused = 0; dispatch.failed = false;
        const int64_t position = suffix_position+suffix_tokens.size()+i;
        if (!decode.run(1, &token, position, pool, user, &unused, error) || dispatch.failed ||
            !decode.commit(1, error) || !decode.copy_logits(0, logits.data())) {
            std::cerr << "second-turn native generation failed: " << error << std::endl; return 1;
        }
        second_generated.push_back(token);
        second_trace.write((const char*)logits.data(), logits.size()*sizeof(float));
        if (std::find(eos_tokens.begin(), eos_tokens.end(), token) != eos_tokens.end()) break;
    }
    second_trace.close(); dispatch.plan = nullptr; if (!second_trace) return 1;
    std::ofstream final_frontier(output + ".second.final.f32", std::ios::binary);
    final_frontier.write((const char*)logits.data(), logits.size()*sizeof(float));
    final_frontier.close(); if (!final_frontier) return 1;
    const bool second_stopped_on_eos = !second_generated.empty() &&
        std::find(eos_tokens.begin(), eos_tokens.end(), second_generated.back()) != eos_tokens.end();
    std::ofstream turn_report(output + ".multiturn.json");
    turn_report << "{\"oracle\":\"fresh-prefill-then-normal-decode\",\"first_prefill_tokens\":"
                << initial_tokens.size() << ",\"first_prefill_work_tokens\":" << first_prefill_work_tokens
                << ",\"first_prefill_work_chunks\":" << first_prefill_work_chunks
                << ",\"first_decode_tokens\":" << generated.size()
                << ",\"first_generated\":[";
    for (size_t i=0;i<generated.size();++i) { if(i) turn_report << ','; turn_report << generated[i]; }
    turn_report << "],\"first_stopped_on_actual_eos\":" << (first_stopped_on_eos ? "true" : "false")
                << ",\"eos_ids\":[";
    for (size_t i=0;i<eos_tokens.size();++i) { if(i) turn_report << ','; turn_report << eos_tokens[i]; }
    turn_report << "],\"first_consumed\":" << initial_tokens.size()+generated.size()
                << ",\"second_suffix_position\":" << suffix_position
                << ",\"second_suffix_tokens\":" << suffix_tokens.size()
                << ",\"second_prefill_tokens\":" << second_prefill_tokens
                << ",\"second_prefill_chunks\":" << second_prefill_chunks
                << ",\"second_decode_tokens\":" << second_generated.size()
                << ",\"total_decode_tokens\":" << generated.size()+second_generated.size()
                << ",\"total_prefill_tokens\":" << batched.stats().tokens
                << ",\"total_prefill_chunks\":" << batched.stats().chunks
                << ",\"normal_forward_tokens\":" << decode.completed_forward_tokens
                << ",\"total_forward_tokens\":" << decode.completed_forward_tokens+batched.stats().tokens
                << ",\"second_generated\":[";
    for (size_t i=0;i<second_generated.size();++i) { if(i) turn_report << ','; turn_report << second_generated[i]; }
    turn_report << "],\"second_stopped_on_actual_eos\":" << (second_stopped_on_eos ? "true" : "false")
                << ",\"first_outputs_verified\":" << (first_outputs_verified ? "true" : "false")
                << ",\"second_consumed\":" << suffix_position+suffix_tokens.size()+second_generated.size()
                << "}";
    turn_report.close(); if (!turn_report) return 1;
'''


def multiturn_source(source):
    source = batched_base.matching_schedule_source(source)
    source = '#include <cstdlib>\n#include <sstream>\n' + source
    source = base.replace_once(source, '    batched.on_chunk =',
        '    std::vector<std::pair<int64_t,int64_t>> prefill_chunk_positions;\n    batched.on_chunk =')
    source = base.replace_once(source, '    std::vector<int32_t> generated;\n', EOS_INIT+'\n    std::vector<int32_t> generated;\n')
    source = base.replace_once(source,
        '    const auto end_prefill = std::chrono::steady_clock::now();',
        '    const uint64_t first_prefill_work_tokens = batched.stats().tokens;\n'
        '    const uint64_t first_prefill_work_chunks = batched.stats().chunks;\n'
        '    const auto end_prefill = std::chrono::steady_clock::now();')
    source = base.replace_once(source, 'if (ok) completed_chunks.emplace_back(position, count);',
        'if (ok) { completed_chunks.emplace_back(position, count); prefill_chunk_positions.emplace_back(position, count); }')
    source = base.replace_once(source, 'if (token == 248044 || token == 248046) break;',
        'if (std::find(eos_tokens.begin(), eos_tokens.end(), token) != eos_tokens.end()) break;')
    source = base.replace_once(source, '    trace.close(); dispatch.plan = nullptr;\n    if (!trace) return 1;\n',
                               '    trace.close(); dispatch.plan = nullptr;\n    if (!trace) return 1;\n'+MULTITURN)
    old = 'generated.back()==248044 || generated.back()==248046'
    if source.count(old) != 1:
        raise ValueError('Expected unique original EOS summary expression')
    source = source.replace(old,
        'std::find(eos_tokens.begin(), eos_tokens.end(), generated.back()) != eos_tokens.end()')
    start = source.index('    std::cout << "{\\"consumed\\":')
    end = source.index('    return 0;', start)
    summary = r'''    std::cout << "{\"status\":\"ok\",\"oracle\":\"fresh-multiturn\",\"first_consumed\":"
              << initial_tokens.size()+generated.size()
              << ",\"first_decode_tokens\":" << generated.size()
              << ",\"second_consumed\":" << suffix_position+suffix_tokens.size()+second_generated.size()
              << ",\"second_decode_tokens\":" << second_generated.size()
              << ",\"total_prefill_tokens\":" << batched.stats().tokens
              << ",\"total_prefill_chunks\":" << batched.stats().chunks
              << ",\"normal_forward_tokens\":" << decode.completed_forward_tokens
              << ",\"total_forward_tokens\":" << decode.completed_forward_tokens+batched.stats().tokens << "}" << std::endl;
'''
    return source[:start]+summary+source[end:]


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'build output must be new: {output}')
    original_fresh = base.FRESH
    base.FRESH = multiturn_source(original_fresh)
    base.build(root, output, long_demo=True, restore_counters=False)
    report_path = output/'build.json'
    report = json.loads(report_path.read_text())
    report.update(
        oracle='fresh-prefill-then-normal-verifier-decode',
        multi_turn='normal first generation must consume configured EOS; then native Prefill suffix; then normal generation up to 128',
        cache_or_branch_restore='none',
        generated_fresh_source_sha256=sha(output/'fresh_prefill.cpp'),
        runtime='UNRUN; wait for root GPU lock release',
    )
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='pinned source-digest-v1 tree; read-only')
    parser.add_argument('output', type=Path, help='unique external build output directory')
    args = parser.parse_args()
    build(args.root, args.output)
