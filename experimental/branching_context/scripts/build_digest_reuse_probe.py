"""Rebuild all changed COW class users beside the pinned inference archives.

The staged root must contain the changed digest/header/pages sources. Rebuild
BranchPages and digest here; the request builder already rebuilds BranchSession,
both main programs and the driver. Never link new class layouts to old objects.
"""
import argparse
import json
from pathlib import Path
import subprocess
import build_fast_prefill_probe as base
from build_fast_prefill_probe import sha
from build_native_request_probe import attach_provenance, build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--build-dir',type=Path,
                        help='selected CMake build directory (default: <root>/build)')
    args = parser.parse_args(argv)
    root,output = args.root.resolve(),args.output.resolve()
    # Validate the selected CMake recipe before the first compile. Do not
    # compile replacement core objects without the matching branch link recipe.
    root,build_dir,cache_path,flags_path,link_path,flags,original_link = base.load_build_recipe(
        root,args.build_dir)
    engine_flags_path = build_dir/'CMakeFiles/strata_engine.dir/flags.make'
    engine_flags = base.load_cpp_flags(engine_flags_path)
    cuda_compilers = [line.split('=', 1)[1] for line in cache_path.read_text().splitlines()
                      if line.startswith('CMAKE_CUDA_COMPILER:') and '=' in line]
    if len(cuda_compilers) != 1 or not cuda_compilers[0]:
        raise ValueError('Require one CMAKE_CUDA_COMPILER in the selected build cache')
    cuda_compiler = str(Path(cuda_compilers[0]).resolve())
    core = output.with_name(output.name+'-core')
    core.mkdir(parents=True,exist_ok=False)
    commands = [
        [cuda_compiler,'-std=c++17','-arch=sm_120','-I'+str(root/'include'),
         '-c',str(root/'src/core/branch_digest.cu'),'-o',str(core/'branch_digest.o')],
    ]
    # VmmMapping/VmmRange carry device ownership in the current headers. The
    # original archive predates that layout, so rebuild every concrete consumer,
    # not just the newly referenced VMM functions. generate.cpp is rebuilt below.
    cpu_sources = [root/'src/core'/(name+'.cpp') for name in
                   ('branch_pages', 'vmm', 'layer', 'expert_cache')]
    commands += [
        ['/usr/bin/c++','-I'+str(root/'include'),*engine_flags['CXX_DEFINES'],
         *engine_flags['CXX_INCLUDES'],*engine_flags['CXX_FLAGS'],
         '-c',str(path),'-o',str(core/(path.stem+'.o'))] for path in cpu_sources]
    record = dict(status='BUILDING',build_dir=str(build_dir),
                  link_working_directory=str(build_dir),commands=[])
    for index,command in enumerate(commands):
        with (core/(str(index)+'.stdout')).open('wb') as out,(core/(str(index)+'.stderr')).open('wb') as err:
            process = subprocess.run(command,cwd=build_dir,stdout=out,stderr=err)
        record['commands'].append(dict(argv=command,cwd=str(build_dir),exit_code=process.returncode))
        (core/'build.json').write_text(json.dumps(record,indent=2)+'\n')
        if process.returncode: raise RuntimeError('Core build failed: '+str(core/(str(index)+'.stderr')))
    objects = [core/'branch_digest.o',*[core/(path.stem+'.o') for path in cpu_sources]]
    build(root,output,extra_link_objects=objects,build_dir=build_dir)
    receipt_path = output/'build.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['commands'] = record['commands']+receipt['commands']
    paths = [root/'src/core/branch_digest.cu',*cpu_sources,
        root/'include/strata/core/branch_digest.hpp',root/'include/strata/core/branch_sha256.hpp',
        root/'include/strata/core/branch_pages.hpp',root/'include/strata/core/vmm.hpp',
        root/'include/strata/core/layer.hpp',root/'include/strata/core/expert_cache.hpp',
        Path(__file__).resolve(),flags_path,engine_flags_path]
    receipt['source_sha256'].update({str(path):sha(path) for path in paths})
    receipt['integrity_implementation'] = 'fresh per-write SHA-256; implementation pinned by core source hashes'
    attach_provenance(root, output, receipt, [
        Path(__file__), Path(__import__('build_native_request_probe').__file__),
        Path(__import__('build_fast_prefill_probe').__file__),
        Path(__import__('build_batched_resume_probe').__file__),
    ], build_dir=build_dir)
    print(json.dumps(dict(status=receipt['status'],binaries=receipt['binaries'])))


if __name__ == '__main__': main()
