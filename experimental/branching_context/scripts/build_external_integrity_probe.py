"""Build an explicitly caller-managed write-integrity experiment.

Production sources/defaults and the registered integrity tests stay unchanged.
Only the isolated branching executable receives the replacement object. Fresh
inference uses the ordinary object library and remains byte-identical.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from build_fast_prefill_probe import build, replace_once, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reuse-private-access', action='store_true',
                        help='Do not reset access on a unique unsealed page that is already writable')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    build(root, output, True)
    source = root / 'src/core/branch_pages.cpp'
    text = source.read_text()
    start = text.index('        if (!inputs.empty()) {', text.index('bool BranchPages::prepare_writes('))
    end = text.index('        for (const auto& write:writes)\n', start)
    text = text[:start] + '''        // EXPERIMENT ONLY: the caller guarantees unchanged sealed GPU bytes.
        // Preserve ordered metadata/range validation; no device-byte verification.
        const auto per_page=g/branch_digest_tile_bytes;
        for (const auto* seal:seals) if (seal->size()!=per_page) return false;
        (void)scratch;
''' + text[end:]
    if args.reuse_private_access:
        text = replace_once(text, '                if (!mapping_.protect(i,true)) return false;',
                            '                if (!table_->pages[i]->seal.empty() && !mapping_.protect(i,true)) return false;')
    generated = output / 'external_branch_pages.cpp'
    generated.write_text(text)
    report = json.loads((output / 'build.json').read_text())
    report.update(status='BUILDING', write_integrity_policy='externally-managed',
                  private_page_access='reuse-writable' if args.reuse_private_access else 'set-each-write',
                  external_promise='Caller prevents mutations of sealed physical pages through every other alias '
                                   'between restore/capture and continuation; GPU preparation does not detect violations.')
    report['source_sha256'][str(source)] = sha(source)
    report['generated_sha256'][str(generated)] = sha(generated)
    flags = {}
    for line in (root/'build/CMakeFiles/strata-branch.dir/flags.make').read_text().splitlines():
        if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
            key, value = line.split('=', 1); flags[key.strip()] = shlex.split(value)
    compiler = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS']]
    override = output / 'external_branch_pages.o'
    branch_link = next(item['argv'] for item in report['commands']
                       if '-o' in item['argv'] and item['argv'][item['argv'].index('-o')+1] == str(output/'strata-batched-branch'))
    # Before the static archive: this object supplies the same BranchPages symbols.
    branch_link = list(branch_link)
    branch_link.insert(branch_link.index('libstrata_engine.a'), str(override))
    test_source = root/'tests/core/branch_external_integrity_test.cpp'
    test_object = output/'external_contract_test.o'
    test_link = shlex.split((root/'build/CMakeFiles/branch_prepare_test.dir/link.txt').read_text())
    test_link[test_link.index('CMakeFiles/branch_prepare_test.dir/tests/core/branch_prepare_test.cpp.o')] = str(test_object)
    test_link[test_link.index('branch_prepare_test')] = str(output/'external-contract-default-test')
    external_test_link = list(test_link)
    external_test_link[external_test_link.index(str(output/'external-contract-default-test'))] = str(output/'external-contract-test')
    external_test_link.insert(external_test_link.index('libstrata_engine.a'), str(override))
    commands = [compiler+['-c',str(generated),'-o',str(override)], branch_link,
                compiler+['-c',str(test_source),'-o',str(test_object)],test_link,external_test_link]
    report['source_sha256'][str(test_source)] = sha(test_source)
    for index, command in enumerate(commands):
        with (output/f'external-{index}.stdout').open('wb') as out, (output/f'external-{index}.stderr').open('wb') as err:
            done = subprocess.run(command,cwd=root/'build',stdout=out,stderr=err)
        report['commands'].append(dict(argv=command,exit_code=done.returncode))
        (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        if done.returncode:raise RuntimeError('Build failed: '+str(output/f'external-{index}.stderr'))
    report.update(status='BUILT_NOT_EXECUTED',binaries={p.name:sha(p) for p in output.glob('strata-batched-*')},
                  test_binaries={p.name:sha(p) for p in output.glob('external-contract-*-test')})
    # The ordinary baseline executable is named external-contract-default-test.
    report['test_binaries']['external-contract-test'] = sha(output/'external-contract-test')
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],write_integrity_policy=report['write_integrity_policy'],
                          private_page_access=report['private_page_access'],binaries=report['binaries'])))


if __name__ == '__main__':
    main()
