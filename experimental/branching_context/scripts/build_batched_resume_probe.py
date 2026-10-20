"""Build the batches-only resume experiment; retain normal-decode evidence separately."""
import argparse
import json
from pathlib import Path

import build_fast_prefill_probe as base


def fresh_batched_source(source):
    source = base.replace_once(source,
        '    const auto end_prefill = std::chrono::steady_clock::now();',
        '    const uint64_t prompt_prefill_tokens = batched.stats().tokens;\n'
        '    const auto end_prefill = std::chrono::steady_clock::now();')
    start = source.index('    Verifier decode;')
    end = source.index('    std::vector<int32_t> generated;', start)
    source = source[:start] + source[end:]
    start = source.index('        int32_t unused = 0; dispatch.failed = false;')
    end = source.index('        generated.push_back(token);', start)
    source = source[:start] + '''        // Autoregression selects one token at a time, but consumption is
        // the native Prefill operator, not the verifier/debug step operator.
        const int64_t selected = token;
        if (!batched.run(&selected, 1, initial_tokens.size()+i, error)) {
            std::cerr << error << std::endl; return 1;
        }
''' + source[end:]
    source = base.replace_once(source, '    trace.close(); dispatch.plan = nullptr;', '    trace.close();')
    source = base.replace_once(source,
        '<< ",\\\"prefill_tokens\\\":" << batched.stats().tokens',
        '<< ",\\\"prefill_tokens\\\":" << prompt_prefill_tokens\n'
        '              << ",\\\"total_prefill_tokens\\\":" << batched.stats().tokens\n'
        '              << ",\\\"continuation_operator\\\":\\\"batched-prefill\\\""')
    return source


def matching_schedule_source(source):
    """Fresh native state, with explicit request boundaries and chunk witnesses."""
    source = '#include <sstream>\n#include <cstdlib>\n' + source
    source = base.replace_once(source, '    batched.on_chunk =',
        '    std::vector<std::pair<int64_t,int64_t>> completed_chunks;\n    batched.on_chunk =')
    source = base.replace_once(source, '        return ok;',
        '        if (ok) completed_chunks.emplace_back(position, count);\n        return ok;')
    source = base.replace_once(source,
        '''    if (!batched.run(initial_tokens.data(), initial_tokens.size(), 0, error)) {
        std::cerr << error << std::endl; return 1;
    }''', r'''    // Requests are lengths, not a second token history. All tokens come from
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
    request_frontiers.close();''')
    source = base.replace_once(source, '    std::cout << "{\\"consumed\\":"',
        '''    std::ofstream schedule(output + ".schedule.json");
    schedule << "{\\"operator\\":\\"batched-prefill\\",\\"requests\\":[";
    for (size_t i=0; i<request_counts.size(); ++i) { if(i) schedule << ','; schedule << request_counts[i]; }
    schedule << "],\\"completed_chunks\\":[";
    for (size_t i=0; i<completed_chunks.size(); ++i) {
        if(i) schedule << ',';
        schedule << '[' << completed_chunks[i].first << ',' << completed_chunks[i].second << ']';
    }
    schedule << "]}"; schedule.close(); if (!schedule) return 1;
    std::cout << "{\\"consumed\\":"''')
    return source


def build(root, output):
    base.FRESH = matching_schedule_source(fresh_batched_source(base.FRESH))
    base.INIT = base.INIT.replace('    batched.on_chunk =',
        '    std::vector<std::pair<int64_t,int64_t>> request_chunks;\n    batched.on_chunk =')
    base.INIT = base.INIT.replace('int64_t count, int64_t, std::string& e)',
        'int64_t count, int64_t position, std::string& e)')
    base.INIT = base.INIT.replace(
        '        return head_rows(rows + (count - 1) * geometry.hc * geometry.n_embd, logits.data(), e);',
        '        const bool ok = head_rows(rows + (count - 1) * geometry.hc * geometry.n_embd, logits.data(), e);\n'
        '        if (ok) request_chunks.emplace_back(position, count);\n        return ok;')
    base.FEED = '            request_chunks.clear();\n' + base.FEED + r'''
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"operator\":\"batched-prefill\",\"position\":" << position
                      << ",\"count\":" << tokens.size() << ",\"chunks\":[";
            for (size_t i=0;i<request_chunks.size();++i) {
                if(i) std::cout << ',';
                std::cout << '[' << request_chunks[i].first << ',' << request_chunks[i].second << ']';
            }
            std::cout << "]}" << std::endl;
            continue;
'''
    # Preserve the original bounded cache. The ordinary builder's long-demo
    # variant enlarged it; this experiment does not change its retention budget.
    original_replace = base.replace_once
    def replace(text, old, new):
        if old.startswith('ConversationCache ram_cache('): return text
        return original_replace(text, old, new)
    base.replace_once = replace
    base.build(root, output, long_demo=True, restore_counters=True)
    path = Path(output)/'build.json'
    receipt = json.loads(path.read_text())
    receipt['continuation_operator'] = 'batched-prefill'
    receipt['resume_contract'] = 'identical-native-batches-v1'
    receipt['scope'] = 'batches-only durable-resume experiment; normal decode excluded'
    receipt['builder_sha256'] = base.sha(Path(__file__))
    path.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    build(args.root,args.output)
