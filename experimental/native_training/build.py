"""Build PCC native GGML training; no Python tensor/autograd runtime is linked."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

PIN='972d2313bc0bf0a45f634f77d95c9fb03aeab12c'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--llama',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ll=args.llama.resolve();out=args.output.resolve()
    assert subprocess.check_output(['git','-C',str(ll),'rev-parse','HEAD'],text=True).strip()==PIN
    out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('native_train.cpp')
    # A private dependency build leaves every serving executable untouched.
    # GGML's backward can send a strided adjoint into scale. Materialize that
    # input without changing inplace semantics or the numerical operation.
    dependency=out/'ggml-source'
    if not dependency.exists():
        shutil.copytree(ll/'ggml',dependency,ignore=shutil.ignore_patterns('build','.git','__pycache__'))
    core=dependency/'src/ggml.c'
    original=core.read_text()
    old='''        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
'''
    new='''        bool                  inplace) {
    // PCC training: preserve strided adjoints through an explicit copy.
    if (!ggml_is_padded_1d(a)) {
        GGML_ASSERT(!inplace);
        a = ggml_cont(ctx, a);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
'''
    if old in original:
        assert original.count(old)==1
        core.write_text(original.replace(old,new),newline='\n')
    else:
        assert new in original,'Unrecognized dependency source'
    # Embed GGML as the original engine does. Its vendored tree does not ship
    # the pkg-config template required by GGML's standalone install project.
    (out/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\nproject(PCCNativeTrainingBackend LANGUAGES C CXX ASM)\nadd_subdirectory(ggml-source)\n')
    build=out/'training-backend-build'
    subprocess.run(['cmake','-S',str(out),'-B',str(build),'-DBUILD_SHARED_LIBS=ON',
                    '-DGGML_CUDA=OFF','-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF',
                    '-DGGML_BUILD_TESTS=OFF','-DGGML_BUILD_EXAMPLES=OFF'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','ggml','ggml-cpu','--parallel','2'],check=True)
    libraries=[]
    for name in ('libggml.so','libggml-base.so','libggml-cpu.so'):
        matches=list(build.rglob(name));assert len(matches)==1,(name,matches)
        libraries.append(matches[0])
    runtime=':'.join(dict.fromkeys(str(p.parent) for p in libraries));engine=ll/'build/bin'
    cmd=['c++','-std=c++17','-O2','-g','-rdynamic','-I'+str(ll/'include'),'-I'+str(ll/'ggml/include'),
         '-I'+str(ll/'vendor'),'-DPCC_CUDA_BACKEND="'+str(engine/'libggml-cuda.so')+'"',
         str(source),'-Wl,-rpath,'+runtime,
         '-o',str(out/'pcc-native-train'),*map(str,libraries),'-ldl','-pthread']
    subprocess.run(cmd,check=True)
    linked=subprocess.check_output(['ldd',str(out/'pcc-native-train')],text=True)
    assert 'torch' not in linked.lower()
    receipt=dict(pin=PIN,command=cmd,linked=linked,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 dependency_patch='materialize-strided-noninplace-scale-v1',
                 dependency_core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
