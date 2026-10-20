"""Build isolated batched-prefill probes from existing native objects (no production edits)."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EXTRA = """, ExpertSource* source, const ExpertCache* cache, const int32_t* residency,
                  int64_t chunk, void* stream,
                  const std::function<bool(const float*, float*, std::string&)>& head_rows,
                  const std::vector<int64_t>& initial_tokens, const std::string& output,
                  int64_t decode_steps"""

INIT = """
    strata::prefill::Prefill batched;
    if (chunk <= 0 || !batched.init(weights, geometry, state, source, cache, residency, chunk, stream, error)) {
        error_reply(error.empty() ? "missing real batched prefill" : error); return 1;
    }
    batched.on_chunk = [&](const float* rows, int64_t count, int64_t, std::string& e) {
        return head_rows(rows + (count - 1) * geometry.hc * geometry.n_embd, logits.data(), e);
    };
    uint64_t prefill_tokens = 0, prefill_chunks = 0;
    uint64_t decode_prepare_ns = 0, decode_compute_ns = 0, decode_finish_ns = 0, decode_timed_tokens = 0;
"""

FEED = """            std::vector<int64_t> wide(tokens.begin(), tokens.end());
            const auto position = (int64_t)session.ids().size();
            if (!session.begin(position, (int64_t)tokens.size(), error) ||
                !batched.run(wide.data(), (int64_t)wide.size(), position, error) ||
                !session.finish(tokens, logits, error)) {
                error_reply(error.empty() ? "batched prefill failed" : error); continue;
            }
            prefill_tokens = batched.stats().tokens;
            prefill_chunks = batched.stats().chunks;
"""

FRESH = r'''
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
    batched.on_chunk = [&](const float* rows, int64_t count, int64_t position, std::string& e) {
        const bool ok = head_rows(rows + (count - 1) * geometry.hc * geometry.n_embd, logits.data(), e);
        std::cerr << "fresh prefill chunk position=" << position << " count=" << count << " ok=" << ok << std::endl;
        return ok;
    };
    const auto begin_prefill = std::chrono::steady_clock::now();
    if (!batched.run(initial_tokens.data(), initial_tokens.size(), 0, error)) {
        std::cerr << error << std::endl; return 1;
    }
    std::ofstream file(output, std::ios::binary);
    file.write((const char*)logits.data(), logits.size()*sizeof(float));
    file.close();
    if (!file) return 1;
    const auto end_prefill = std::chrono::steady_clock::now();
    Verifier decode;
    if (!decode.init(weights, geometry, state, hits, head, 2, error)) { std::cerr << error; return 1; }
    decode.set_pcie_mode(pcie_mode); decode.set_head_sampling(false);
    dispatch.plan = decode.plan_sink();
    std::vector<int32_t> generated;
    std::ofstream trace(output + ".decode.f32", std::ios::binary);
    const auto begin_decode = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < decode_steps; ++i) {
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
        if ((i+1)%16 == 0) std::cerr << "fresh consumed outputs=" << i+1 << std::endl;
        if (token == 248044 || token == 248046) break;
    }
    trace.close(); dispatch.plan = nullptr;
    if (!trace) return 1;
    std::cout << "{\"consumed\":" << initial_tokens.size()+generated.size()
              << ",\"prompt_consumed\":" << initial_tokens.size()
              << ",\"decode_tokens\":" << generated.size()
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
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected unique source anchor: ' + old[:80])
    return text.replace(old, new)


def load_cpp_flags(flags_path):
    flags_path = Path(flags_path)
    if not flags_path.is_file():
        raise FileNotFoundError(f'missing selected CMake build input: {flags_path}')
    flags = {}
    for line in flags_path.read_text().splitlines():
        if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
            key, value = line.split('=', 1)
            flags[key.strip()] = shlex.split(value)
    required = ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS')
    missing_flags = [key for key in required if key not in flags]
    if missing_flags:
        raise ValueError(f'missing required CMake compile flags in {flags_path}: {missing_flags}')
    return flags


def load_build_recipe(root, build_dir=None):
    """Resolve the selected CMake tree and its compile/link recipes before work starts."""
    root = Path(root).resolve()
    build_dir = Path(build_dir).resolve() if build_dir is not None else (root/'build').resolve()
    if not build_dir.is_dir():
        raise FileNotFoundError(f'missing CMake build directory: {build_dir}')
    cache_path = build_dir/'CMakeCache.txt'
    flags_path = build_dir/'CMakeFiles/strata-branch.dir/flags.make'
    link_path = build_dir/'CMakeFiles/strata-branch.dir/link.txt'
    for path in (cache_path, flags_path, link_path):
        if not path.is_file():
            raise FileNotFoundError(f'missing selected CMake build input: {path}')
    flags = load_cpp_flags(flags_path)
    original_link = shlex.split(link_path.read_text())
    if not original_link or original_link.count('-o') != 1 or original_link.count('strata-branch') != 1:
        raise ValueError(f'invalid strata-branch link recipe: {link_path}')
    expected_objects = (
        'CMakeFiles/strata-branch.dir/src/program/generate.cpp.o',
        'CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o',
    )
    for item in expected_objects:
        if original_link.count(item) != 1:
            raise ValueError(f'link recipe must contain exactly one {item}: {link_path}')
    return root, build_dir, cache_path, flags_path, link_path, flags, original_link


def build(root, output, long_demo=False, restore_counters=False, extra_link_objects=(), build_dir=None):
    root, build_dir, cache_path, flags_path, link_path, flags, original_link = load_build_recipe(root, build_dir)
    output = Path(output).resolve()
    extra_link_objects = [Path(path).resolve() for path in extra_link_objects]
    if any(not path.is_file() or path.suffix != '.o' for path in extra_link_objects):
        raise ValueError('Require compiled core override objects')
    output.mkdir(parents=True, exist_ok=False)
    header_path = root/'include/strata/core/branch_driver.hpp'
    header = header_path.read_text()
    header = replace_once(header, 'ExpertDispatch&, int pcie_mode);', 'ExpertDispatch&, int pcie_mode'+EXTRA+');')
    header = '#include <functional>\n#include <vector>\n#include <string>\n'+header
    generated_header = output/'include/strata/core/branch_driver.hpp'
    generated_header.parent.mkdir(parents=True)
    generated_header.write_text(header)
    extra_sources = []
    if restore_counters:
        # Instrumented core objects replace their archive members; default serving
        # executables and original build outputs remain untouched.
        session_header = output/'include/strata/core/branch_session.hpp'
        session_header.write_bytes((root/'include/strata/core/branch_session.hpp').read_bytes())
        extra_sources = [root/'src/core'/name for name in
            ('branch_session.cpp','conversation_snapshot.cpp','conversation_state.cpp',
             'conversation_file.cpp')]
    program_path = root/'src/program/generate.cpp'
    program = program_path.read_text()
    program = replace_once(program, '&drive_pool_multi, &drive, drive.d, pcie_mode);', '''&drive_pool_multi, &drive, drive.d, pcie_mode,
            srcp, o.expert_cache > 0 ? &xcache : nullptr, host_res.empty() ? nullptr : host_res.data(),
            o.prefill_chunk, main_cs,
            [&](const float* row, float* host, std::string& e) {
                if (cudaMemcpyAsync(ss.R, row, (size_t)g.hc*g.n_embd*sizeof(float), cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)main_cs) != cudaSuccess || !run_head(main_cs) ||
                    cudaMemcpyAsync(host, d_logits, (size_t)n_vocab*sizeof(float), cudaMemcpyDeviceToHost,
                                    (cudaStream_t)main_cs) != cudaSuccess ||
                    cudaStreamSynchronize((cudaStream_t)main_cs) != cudaSuccess) {
                    e = err.empty() ? "prefill frontier projection/copy failed" : err; return false;
                }
                return true;
            }, o.tokens, std::getenv("STRATA_DUMP_FIRST_LOGITS") ? std::getenv("STRATA_DUMP_FIRST_LOGITS") : "", o.max_new);''')
    generated_program = output/'generate.cpp'
    generated_program.write_text(program)
    driver_path = root/'src/core/branch_driver.cpp'
    driver = driver_path.read_text()
    driver = '#include "strata/prefill/prefill.hpp"\n#include <chrono>\n'+driver
    driver = replace_once(driver, 'ExpertDispatch& dispatch, int pcie_mode) {', 'ExpertDispatch& dispatch, int pcie_mode'+EXTRA+') {')
    driver = replace_once(driver, '    std::string line;', INIT+'\n    std::string line;')
    start = driver.index('            bool ok = true;')
    end = driver.index('            std::cout << "{\\"status\\":\\"ok\\",\\"consumed\\":"', start)
    driver = driver[:start]+FEED+driver[end:]
    if long_demo:
        driver = replace_once(driver, 'ConversationCache ram_cache(size_t(1024) * 1024 * 1024, 8);',
                              'ConversationCache ram_cache(size_t(16384) * 1024 * 1024, 8);')
        decode_code = r'''        } else if (command == "decode") {
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
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
'''
        driver = replace_once(driver, '        } else if (command == "capture") {',decode_code+'        } else if (command == "capture") {')
    driver = replace_once(driver, '<< ",\\"forward_tokens\\":" << forward.completed_forward_tokens',
                          '<< ",\\"forward_tokens\\":" << (forward.completed_forward_tokens + prefill_tokens)\n'
                          '                      << ",\\"prefill_tokens\\":" << prefill_tokens\n'
                          '                      << ",\\"prefill_chunks\\":" << prefill_chunks\n'
                          '                      << ",\\"decode_prepare_ns\\":" << decode_prepare_ns\n'
                          '                      << ",\\"decode_compute_ns\\":" << decode_compute_ns\n'
                          '                      << ",\\"decode_finish_ns\\":" << decode_finish_ns\n'
                          '                      << ",\\"decode_timed_tokens\\":" << decode_timed_tokens')
    # Stats and request completion must expose the same cumulative work count.
    # Prefill uses a separate operator, so its completed work must be included
    # before the consumer can compare request-boundary deltas.
    driver = replace_once(driver,
                          '<< ",\\"forward_tokens\\":" << (forward.completed_forward_tokens)',
                          '<< ",\\"forward_tokens\\":" << (forward.completed_forward_tokens + prefill_tokens)')
    generated_driver = output/'branch_driver.cpp'
    generated_driver.write_text(driver)
    fresh_path = output/'fresh_prefill.cpp'
    fresh_source = FRESH
    if not long_demo:
        fresh_source = replace_once(fresh_source, 'for (int64_t i = 0; i < decode_steps; ++i)',
                                   'for (int64_t i = 0; i < 0; ++i)')
    fresh_path.write_text(fresh_source)
    compiler = ['/usr/bin/c++', '-I'+str(output/'include'), '-I'+str(root/'include'), *flags['CXX_DEFINES'],
                *flags['CXX_INCLUDES'], *flags['CXX_FLAGS']]
    report = dict(status='BUILDING',scope='isolated batched-prefill experiment; not an admitted execution profile',long_demo=long_demo,
                  build_dir=str(build_dir), link_working_directory=str(build_dir),
                  source_sha256={str(p):sha(p) for p in [header_path, program_path, driver_path]},
                  generated_sha256={str(p):sha(p) for p in [generated_header,generated_program,generated_driver,fresh_path]},
                  flags_sha256=sha(flags_path),link_sha256=sha(link_path),commands=[])
    report['restore_counters'] = restore_counters
    report['generated_sha256'].update({str(path):sha(path) for path in extra_link_objects})
    if restore_counters:
        report['source_sha256'].update({str(path):sha(path) for path in extra_sources+
            [root/'src/core/conversation_copy.hpp',root/'include/strata/core/branch_session.hpp']})
        report['generated_sha256'][str(session_header)] = sha(session_header)
    commands = [compiler+['-c',str(p),'-o',str(output/(p.stem+'.o'))]
                for p in [generated_program,generated_driver,fresh_path,*extra_sources]]
    for label, obj in [('strata-batched-branch','branch_driver.o'), ('strata-batched-fresh','fresh_prefill.o')]:
        link = list(original_link)
        for old, new in [('CMakeFiles/strata-branch.dir/src/program/generate.cpp.o',str(output/'generate.o')),
                         ('CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o',str(output/obj)),
                         ('strata-branch',str(output/label))]:
            if link.count(old)!=1: raise ValueError('Unexpected link recipe: '+old)
            link[link.index(old)] = new
        if extra_sources:
            # Place definitions before static archives so their old objects are
            # not extracted. Both executable variants use the same measured core.
            link[1:1] = [str(output/(p.stem+'.o')) for p in extra_sources]
        link[1:1] = [str(path) for path in extra_link_objects]
        commands.append(link)
    for index, cmd in enumerate(commands):
        with (output/(str(index)+'.stdout')).open('wb') as out, (output/(str(index)+'.stderr')).open('wb') as err:
            result=subprocess.run(cmd,cwd=build_dir,stdout=out,stderr=err)
        report['commands'].append(dict(argv=cmd,cwd=str(build_dir),exit_code=result.returncode))
        (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError('Compile/link failed: '+str(output/(str(index)+'.stderr')))
    report.update(status='BUILT_NOT_EXECUTED',binaries={p.name:sha(p) for p in output.glob('strata-batched-*')})
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--long-demo',action='store_true')
    parser.add_argument('--restore-counters',action='store_true')
    parser.add_argument('--build-dir',type=Path,
                        help='selected CMake build directory (default: <root>/build)')
    args=parser.parse_args(argv)
    return build(args.root,args.output,args.long_demo,args.restore_counters,build_dir=args.build_dir)


if __name__=='__main__':
    main()
