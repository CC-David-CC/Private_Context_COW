"""Build versioned mixed-input adapter and separate fresh oracle at one llama pin."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from build_qwen35 import PIN
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--llama',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if subprocess.check_output(['git','-C',str(a.llama),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.llama),'status','--porcelain']):raise SystemExit('Requires clean inspected llama.cpp pin')
 a.output.mkdir(parents=True,exist_ok=True);lib=a.llama/'build/bin';vendor=a.llama/'vendor';sha=vendor/'hash/sha256/sha256.c';obj=a.output/'sha256.o'
 common=['c++','-std=c++17','-O2','-I'+str(a.llama/'include'),'-I'+str(a.llama/'ggml/include'),'-I'+str(vendor),'-I'+str(vendor/'hash'),'-L'+str(lib),'-Wl,-rpath,'+str(lib)]
 commands=[['cc','-O2','-fPIC','-I'+str(vendor/'hash'),'-c',str(sha),'-o',str(obj)],common+['-shared','-fPIC',str(ROOT/'native/qwen35_mixed_sampler_bridge.cpp'),str(obj),'-o',str(a.output/'libqwen35ctt.so'),'-lllama','-lggml','-lggml-base'],common+[str(ROOT/'native/qwen35_mixed_oracle.cpp'),str(obj),'-o',str(a.output/'qwen35-mixed-oracle'),'-lllama','-lggml','-lggml-base']]
 for command in commands:subprocess.run(command,check=True)
 files=[a.output/'libqwen35ctt.so',a.output/'qwen35-mixed-oracle',sha,vendor/'hash/sha256/sha256.h',vendor/'hash/rotate-bits/rotate-bits.h',vendor/'nlohmann/json.hpp',a.llama/'include/llama.h',a.llama/'src/llama-memory-hybrid.cpp',a.llama/'src/llama-memory-recurrent.cpp',a.llama/'src/llama-context.cpp',ROOT/'native/qwen35_mixed_bridge.cpp',ROOT/'native/qwen35_mixed_sampler_bridge.cpp',ROOT/'native/qwen35_mixed_oracle.cpp',ROOT/'native/min_p_select.h',ROOT/'native/grammar_select.h']+list(lib.glob('*.so'))
 result=dict(backend_revision=PIN,state_format='qwen35-hybrid-mrope-frontier-v2',native_binding_protocol='qwen35-streamed-cells-mrope-v2',commands=commands,sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 (a.output/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(a.output/'build.json')
if __name__=='__main__':main()
