"""Compile the resident COW request primitive beside an uncached native reference.

This is a private probe, not an integration claim for ordinary --serve. Its
Verifier and batched input operator are real; numerical qualification is separate.
"""
import argparse
import json
from pathlib import Path

import build_fast_prefill_probe as base
from build_batched_resume_probe import matching_schedule_source
from native_build_provenance import collect_build_provenance


def attach_provenance(root, output, receipt, generator_entrypoints, build_dir=None):
    root = Path(root).resolve()
    try:
        build_dir = base.load_build_recipe(root, build_dir)[1]
        receipt['build_dir'] = str(build_dir)
        receipt['link_working_directory'] = str(build_dir)
        source_paths = set(receipt.get('source_sha256', {}))
        source_paths.update(receipt.get('generated_sha256', {}))
        receipt['build_provenance'] = collect_build_provenance(
            source_root=root,
            build_dir=build_dir,
            commands=receipt.get('commands', []),
            generator_entrypoints=generator_entrypoints,
            declared_sources=source_paths,
            link_working_directory=build_dir,
        )
    except Exception as error:
        # The base builder has already written its preliminary receipt. Replace
        # its apparent success with a terminal status before surfacing failure.
        receipt.pop('build_provenance', None)
        receipt['status'] = 'BUILD_PROVENANCE_FAILED'
        receipt['provenance_error'] = f'{type(error).__name__}: {error}'
        Path(output, 'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
        raise
    Path(output, 'build.json').write_text(json.dumps(receipt, indent=2)+'\n')


def instrument_request_timing(source):
    marker = '} else if (command == "generate")'
    end_marker = '} else if (command == "capture")'
    if source.count(marker) != 1 or source.count(end_marker) != 1:
        raise ValueError('Require one actual native generation handler')
    start,end = source.index(marker),source.index(end_marker)
    block = source[start:end]
    old = '''                if (!session.begin(position, 1, error) ||
                    !forward.run(1, &token, position, pool, user, &unused, error) || pool_failed ||
                    !forward.commit(1, error) || !forward.copy_logits(0, logits.data()) ||
                    !session.finish({token}, logits, error)) { ok = false; break; }'''
    new = '''                auto stage_timer = std::chrono::steady_clock::now();
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
                ++decode_timed_tokens;'''
    # Current production Strata already includes this exact instrumentation.
    # Accept only the complete known block, never a partial/changed operator.
    if block.count(new) == 1 and block.count(old) == 0:
        return source
    if block.count(old) != 1:
        raise ValueError('Native operator changed; do not instrument an assumed handler')
    return source[:start]+block.replace(old,new)+source[end:]


def build(root, output, extra_link_objects=(), build_dir=None):
    root, build_dir, *_ = base.load_build_recipe(root, build_dir)
    original = base.replace_once
    previous = (base.replace_once, base.FRESH, base.INIT, base.FEED)

    def replace(source, old, new):
        if old.startswith('ConversationCache ram_cache('):
            return source  # Keep the original bounded host cache.
        source = original(source, old, new)
        if old == '    std::string line;':
            source = instrument_request_timing(source)
        return source

    try:
        base.replace_once = replace
        base.FRESH = original(base.FRESH, '<< ",\\\"decode_tokens\\\":" << generated.size()',
            '<< ",\\\"decode_tokens\\\":" << generated.size()\n'
            '              << ",\\\"normal_forward_tokens\\\":" << decode.completed_forward_tokens\n'
            '              << ",\\\"normal_decode_windows\\\":" << decode.windows')
        base.FRESH = matching_schedule_source(base.FRESH)
        # Record actual successful input chunks, just as the batch-only resume
        # builder does. These are observed native boundaries, not a repacked history.
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
        base.build(root, output, long_demo=True, restore_counters=True,
                   extra_link_objects=extra_link_objects, build_dir=build_dir)
    finally:
        base.replace_once, base.FRESH, base.INIT, base.FEED = previous
    path = Path(output)/'build.json'
    record = json.loads(path.read_text())
    record.update(native_request_protocol='consumed-greedy-ack-v1',
        resume_contract='identical-native-operators-v1',
        continuation_operator='normal-verifier-decode', builder_sha256=base.sha(Path(__file__)),
        build_dir=str(build_dir), link_working_directory=str(build_dir),
        scope='private resident COW request probe; not ordinary --serve integration')
    record['request_timing'] = 'successful consumed native windows; prepare/compute/finish blocking wall time'
    attach_provenance(root, output, record, [
        Path(__file__), Path(base.__file__), Path(__import__('build_batched_resume_probe').__file__),
    ], build_dir=build_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--build-dir', type=Path,
                        help='selected CMake build directory (default: <root>/build)')
    args = parser.parse_args(argv)
    return build(args.root, args.output, build_dir=args.build_dir)


if __name__ == '__main__':
    main()
