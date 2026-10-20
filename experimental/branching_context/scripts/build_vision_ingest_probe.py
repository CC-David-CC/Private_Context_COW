"""Build image-ingest COW and fresh reference probes from a checked link receipt.

The native session and two driver translation units are rebuilt against one
header overlay. Image disk checkpoints use the existing session serializer,
canonical outer bindings and metadata-only position reconstruction.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from build_min_p_selector_probe import _verify_inputs, sha256


VISION_COMMAND = r'''        } else if (command == "vision_info") {
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
                      << ",\"validated_embedding_bytes\":" << vision_validated_bytes
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
        } else if (command == "feed") {'''


def replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one inspected source anchor: '+before[:100])
    return source.replace(before, after, 1)


def instrument(source):
    source = replace(source, '#include "strata/prefill/prefill.hpp"',
                     '#include "strata/prefill/prefill.hpp"\n#include "vision_ingest.h"')
    source = replace(source, '    std::string line;',
        '    uint64_t vision_validated_bytes = 0;\n'
        '    std::string line;')
    source = replace(source, '        error.clear();', '''        error.clear();
        if (!session.input_images().empty() && (command == "park" || command == "activate")) {
            error_reply("legacy native host cache has no canonical image lookup"); continue;
        }''')
    return replace(source, '        } else if (command == "feed") {', VISION_COMMAND)


def build(receipt_path, output, root, session_receipt_path):
    receipt_path, root = receipt_path.resolve(strict=True), root.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    unchanged = _verify_inputs(receipt)
    original = Path(receipt['generated_driver']['path'])
    if sha256(original) != receipt['generated_driver']['sha256']:
        raise ValueError('Executed driver differs from its pinned receipt')
    compiles = [row for row in receipt['commands'] if '-c' in row['argv'] and row['exit_code'] == 0]
    links = [row for row in receipt['commands'] if '-c' not in row['argv'] and row['exit_code'] == 0]
    if len(compiles) != 1 or len(links) != 1:
        raise ValueError('Expected one driver compile and one successful link')
    output.mkdir(parents=True, exist_ok=False)
    session_receipt_path = session_receipt_path.resolve(strict=True)
    session_recipe = json.loads(session_receipt_path.read_text())
    session_compile = [row for row in session_recipe['commands']
                       if row['label'] == 'compile-branch_session' and row['exit_code'] == 0]
    if len(session_compile) != 1:
        raise ValueError('Expected the pinned native session compile recipe')
    session_compile = session_compile[0]
    old_session_obj = session_compile['argv'][session_compile['argv'].index('-o')+1]
    if sha256(Path(old_session_obj)) != unchanged[old_session_obj]:
        raise ValueError('Native session recipe does not match the pinned link')
    headers = ['vision_layout.hpp', 'vision_payload.hpp', 'branch_sha256.hpp', 'branch_digest.hpp',
               'branch_session.hpp']
    overlay = output/'include/strata/core'
    overlay.mkdir(parents=True)
    for name in headers:
        (overlay/name).write_bytes((root/'include/strata/core'/name).read_bytes())
    (output/'vision_ingest.h').write_bytes((root/'experimental/branching_context/native/vision_ingest.h').read_bytes())
    (output/'branch_driver.cpp').write_text(instrument(original.read_text()), newline='\n')
    (output/'reference_driver.cpp').write_bytes(
        (root/'experimental/branching_context/native/vision_reference_driver.cpp').read_bytes())
    commands, binaries, dependencies = [], {}, {}
    result = dict(status='BUILDING', source_receipt=dict(path=str(receipt_path), sha256=sha256(receipt_path)),
        base_driver=dict(path=str(original), sha256=sha256(original)), commands=commands,
        verified_unchanged_inputs=unchanged, binaries=binaries, dependencies=dependencies,
        session_recipe=dict(path=str(session_receipt_path), sha256=sha256(session_receipt_path)),
        session_source=dict(path=str(root/'src/core/branch_session.cpp'), sha256=sha256(root/'src/core/branch_session.cpp')),
        numerical_qualification='UNRUN', scope='VC03-VC05 image ingest, hot snapshots and bound disk restore')
    def save():
        (output/'build.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        session_obj, session_dep = output/'branch_session.o', output/'branch_session.d'
        session_args = list(session_compile['argv'])
        session_args[session_args.index('-c')+1] = str(root/'src/core/branch_session.cpp')
        session_args[session_args.index('-o')+1] = str(session_obj)
        session_args[1:1] = ['-I'+str(output/'include')]
        session_args += ['-MMD', '-MF', str(session_dep)]
        started = time.monotonic()
        compiled = subprocess.run(session_args, cwd=session_compile['cwd'], capture_output=True)
        for stream in ('stdout', 'stderr'):
            (output/('session-compile.'+stream)).write_bytes(getattr(compiled, stream))
        commands.append(dict(label='session-compile', argv=session_args, cwd=session_compile['cwd'],
            exit_code=compiled.returncode, seconds=time.monotonic()-started))
        save()
        if compiled.returncode:
            raise RuntimeError('Session compilation failed; inspect retained output')
        tokens = shlex.split(session_dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
        dependencies['session'] = {str(Path(path).resolve()): sha256(Path(path)) for path in tokens}
        for name, filename in (('cow', 'branch_driver'), ('reference', 'reference_driver')):
            cpp, obj = output/(filename+'.cpp'), output/(filename+'.o')
            binary, dep = output/('strata-vision-'+name), output/(filename+'.d')
            compile_args = list(compiles[0]['argv'])
            compile_args[compile_args.index(str(original))] = str(cpp)
            compile_args[compile_args.index('-o')+1] = str(obj)
            compile_args[1:1] = ['-I'+str(output/'include'), '-I'+str(output), '-I'+str(original.parent)]
            compile_args += ['-MMD', '-MF', str(dep)]
            link_args = list(links[0]['argv'])
            old_obj = receipt['generated_object']['path']
            if link_args.count(old_obj) != 1:
                raise ValueError('The recorded link does not contain exactly one driver object')
            link_args[link_args.index(old_obj)] = str(obj)
            if link_args.count(old_session_obj) != 1:
                raise ValueError('Expected one pinned native session object in the link')
            link_args[link_args.index(old_session_obj)] = str(session_obj)
            link_args[link_args.index('-o')+1] = str(binary)
            for label, argv, cwd in ((name+'-compile', compile_args, compiles[0]['cwd']),
                                      (name+'-link', link_args, links[0]['cwd'])):
                start = time.monotonic()
                proc = subprocess.run(argv, cwd=cwd, capture_output=True)
                for stream in ('stdout', 'stderr'):
                    (output/(label+'.'+stream)).write_bytes(getattr(proc, stream))
                commands.append(dict(label=label, argv=argv, cwd=cwd, exit_code=proc.returncode,
                                     seconds=time.monotonic()-start))
                save()
                if proc.returncode:
                    raise RuntimeError(label+' failed; inspect retained compiler output')
            tokens = shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
            dependencies[name] = {str(Path(path).resolve()): sha256(Path(path)) for path in tokens}
            binaries[name] = dict(path=str(binary), sha256=sha256(binary), bytes=binary.stat().st_size)
        if _verify_inputs(receipt) != unchanged:
            raise ValueError('Pinned link inputs changed during build')
        result['status'] = 'BUILT_NOT_EXECUTED'
    except BaseException as error:
        result.update(status='BUILD_FAILED', error=repr(error))
        raise
    finally:
        save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--session-receipt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.receipt, args.output, args.root, args.session_receipt)))
