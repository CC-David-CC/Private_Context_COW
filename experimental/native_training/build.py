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
    p.add_argument('--cpu-only',action='store_true',help='Build without a CUDA backend; CUDA device requests then fail explicitly')
    p.add_argument('--cuda-out-prod-singletons',action='store_true',help='Require and record the imported llama.cpp PR 30142 CUDA source patch')
    args=p.parse_args();ll=args.llama.resolve();out=args.output.resolve()
    assert subprocess.check_output(['git','-C',str(ll),'rev-parse','HEAD'],text=True).strip()==PIN
    cuda_source=ll/'ggml/src/ggml-cuda/ggml-cuda.cu'
    cuda_binary=ll/'build/bin/libggml-cuda.so'
    if not args.cpu_only:
        assert 'GGML_CUDA_F32_ACTIVATIONS' in cuda_source.read_text(), 'Build against the inspected --f32-activations engine tree'
        assert b'GGML_CUDA_F32_ACTIVATIONS' in cuda_binary.read_bytes(), 'Rebuild the FP32 CUDA backend before training'
    out_prod_patch=None
    if args.cuda_out_prod_singletons:
        if args.cpu_only:raise ValueError('The CUDA singleton patch requires a CUDA build')
        out_prod=ll/'ggml/src/ggml-cuda/out-prod.cu'
        actual=hashlib.sha256(out_prod.read_bytes()).hexdigest()
        if actual!='a0b8fb1e07b8a8d1caa33680367f4ba36c7f38df55e22bf03c280779b4dcbae8':
            raise ValueError('Apply the pinned PR 30142 patch in a new isolated dependency tree before building')
        out_prod_patch=dict(upstream_pr='https://github.com/ggml-org/llama.cpp/pull/30142',
            upstream_author='edenfunf',upstream_commit='fdb4a0553f6ecb00450f04cc8da5b2b38dd9146d',
            source_sha256=actual,cuda_library_sha256=hashlib.sha256(cuda_binary.read_bytes()).hexdigest(),
            qualification='Source and library identities only; operator and rank-one runtime checks still required')
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
    # PCC long-fixture assembly uses balanced concatenation. The pinned GGML
    # lacks its backward rule. Slice the incoming adjoint without detaching it.
    concat_backward = r"""        case GGML_OP_CONCAT: {
            const int dim = ggml_get_op_params_i32(tensor, 0);
            GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
            struct ggml_tensor * dense = ggml_cont(ctx, grad);
            if (src0_needs_grads) {
                struct ggml_tensor * slice = ggml_view_4d(ctx, dense,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    dense->nb[1], dense->nb[2], dense->nb[3], 0);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cont(ctx, slice));
            }
            if (src1_needs_grads) {
                struct ggml_tensor * slice = ggml_view_4d(ctx, dense,
                    src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    dense->nb[1], dense->nb[2], dense->nb[3], src0->ne[dim]*dense->nb[dim]);
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_cont(ctx, slice));
            }
        } break;
"""
    patched=core.read_text()
    anchor='        case GGML_OP_SET: {'
    if concat_backward not in patched:
        assert patched.count(anchor)==1, 'Unrecognized pinned backward switch'
        core.write_text(patched.replace(anchor,concat_backward+anchor),newline='\n')
    from allocator_guard import apply as apply_allocator_guard
    allocator_patch=apply_allocator_guard(dependency)
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
    cmd=['c++','-std=c++17','-O2','-g','-rdynamic','-I'+str(ll/'include'),'-I'+str(dependency/'include'),
         '-I'+str(ll/'vendor'),'-I'+str(ll/'vendor/hash'),'-DPCC_CUDA_BACKEND="'+('' if args.cpu_only else str(engine/'libggml-cuda.so'))+'"',
         str(source),str(ll/'vendor/hash/sha256/sha256.c'),'-Wl,-rpath,'+runtime,
         '-o',str(out/'pcc-native-train'),*map(str,libraries),'-ldl','-pthread']
    subprocess.run(cmd,check=True)
    linked=subprocess.check_output(['ldd',str(out/'pcc-native-train')],text=True)
    assert 'torch' not in linked.lower()
    receipt=dict(pin=PIN,command=cmd,linked=linked,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 dependency_patch='materialize-strided-noninplace-scale-v1+concat-adjoint-v1+bounded-assigned-allocator-v1',
                 dependency_allocator=allocator_patch,
                  optimizer_source_sha256=hashlib.sha256(source.with_name('native_optimizer.h').read_bytes()).hexdigest(),
                  optimizer_state_source_sha256=hashlib.sha256(source.with_name('optimizer_state.h').read_bytes()).hexdigest(),
                  sha256_dependency_sha256=hashlib.sha256((ll/'vendor/hash/sha256/sha256.c').read_bytes()).hexdigest(),
                  sha256_rotate_dependency_sha256=hashlib.sha256((ll/'vendor/hash/rotate-bits/rotate-bits.h').read_bytes()).hexdigest(),
                 dependency_core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
                 cpu_only=args.cpu_only,
                 cuda_sha256=None if args.cpu_only else hashlib.sha256(cuda_binary.read_bytes()).hexdigest(),
                 cuda_source_sha256=None if args.cpu_only else hashlib.sha256(cuda_source.read_bytes()).hexdigest(),
                 cuda_math_profile=None if args.cpu_only else 'GGML_CUDA_F32_ACTIVATIONS=1; NVIDIA_TF32_OVERRIDE=0')
    if out_prod_patch is not None:receipt['cuda_out_prod_patch']=out_prod_patch
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
