"""Build the explicit dense Llama bridge and independent fresh-prefill oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='972d2313bc0bf0a45f634f77d95c9fb03aeab12c'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama',required=True,type=Path)
    p.add_argument('--engine-build',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--build-engine',action='store_true')
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--f32-activations',action='store_true')
    a=p.parse_args()
    from prepare_wen_llama import validate_checkout
    patch=validate_checkout(a.llama,f32_activations=a.f32_activations)
    a.output.mkdir(parents=True,exist_ok=False)
    commands=[]
    if a.build_engine:
        if a.engine_build.exists():raise ValueError('A new owned engine build is required')
        command=['cmake','-S',str(a.llama),'-B',str(a.engine_build),'-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_CUDA='+('OFF' if a.cpu_only else 'ON'),'-DLLAMA_BUILD_COMMON=OFF',
            '-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_TESTS=OFF',
            '-DLLAMA_BUILD_EXAMPLES=OFF','-DBUILD_SHARED_LIBS=ON']
        commands.append(command);subprocess.run(command,check=True)
        command=['cmake','--build',str(a.engine_build),'--target','llama','-j','4']
        commands.append(command);subprocess.run(command,check=True)
    cache=(a.engine_build/'CMakeCache.txt').read_text()
    expected='GGML_CUDA:BOOL='+('OFF' if a.cpu_only else 'ON')
    if expected not in cache:raise ValueError('Engine backend differs from requested build')
    lib=(a.engine_build/'bin').resolve()
    common=['c++','-std=c++17','-O2','-I'+str(a.llama/'include'),'-I'+str(a.llama/'ggml/include'),
            '-L'+str(lib),'-Wl,-rpath,'+str(lib)]
    for command in [common+['-shared','-fPIC',str(ROOT/'native/llama_dense_bridge.cpp'),
                    '-o',str(a.output/'libllamadensectt.so'),'-lllama','-lggml','-lggml-base'],
                    common+[str(ROOT/'native/llama_dense_fresh_oracle.cpp'),
                    '-o',str(a.output/'llama-dense-fresh-oracle'),'-lllama','-lggml','-lggml-base']]:
        commands.append(command);subprocess.run(command,check=True)
    files=[ROOT/'native'/n for n in ('llama_dense_bridge.cpp','llama_dense_fresh_oracle.cpp',
        'qwen35_bridge.cpp','qwen35_fresh_oracle.cpp','qwen35_sampler_bridge.cpp','qwen35_device.h',
        'qwen35_math.h','qwen35_lora.h','min_p_select.h','grammar_select.h')]
    files.extend([a.output/'libllamadensectt.so',a.output/'llama-dense-fresh-oracle',
        a.llama/'include/llama.h',a.engine_build/'CMakeCache.txt'])
    files.extend(sorted(lib.glob('*.so')))
    hashes={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    build=dict(backend_revision=PIN,architecture_contract='llama-dense-v1',
        device_binding='cpu-only-v1' if a.cpu_only else 'cuda-single-device-v1',
        lora_binding='unqualified',attention_precision_binding='fp32-attention-v1',
        commands=commands,sha256=hashes,**patch)
    (a.output/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(a.output/'build.json')

if __name__=='__main__':main()
