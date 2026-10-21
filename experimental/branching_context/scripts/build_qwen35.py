"""Build the isolated Qwen adapter/oracle against a clean, inspected llama pin."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PIN='972d2313bc0bf0a45f634f77d95c9fb03aeab12c'
def main():
 p=argparse.ArgumentParser();p.add_argument('--llama',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if subprocess.check_output(['git','-C',str(a.llama),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.llama),'status','--porcelain']):raise SystemExit('Requires clean inspected llama.cpp pin')
 a.output.mkdir(parents=True,exist_ok=True);lib=a.llama/'build/bin'
 common=['c++','-std=c++17','-O2','-I'+str(a.llama/'include'),'-I'+str(a.llama/'ggml/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib)]
 commands=[common+['-shared','-fPIC',str(ROOT/'native/qwen35_bridge.cpp'),'-o',str(a.output/'libqwen35ctt.so'),'-lllama','-lggml','-lggml-base'],common+[str(ROOT/'native/qwen35_fresh_oracle.cpp'),'-o',str(a.output/'qwen35-fresh-oracle'),'-lllama','-lggml','-lggml-base']]
 for c in commands:subprocess.run(c,check=True)
 files=[a.output/'libqwen35ctt.so',a.output/'qwen35-fresh-oracle',ROOT/'native/qwen35_bridge.cpp',ROOT/'native/qwen35_fresh_oracle.cpp',a.llama/'include/llama.h',a.llama/'src/llama-memory-hybrid.cpp',a.llama/'src/llama-memory-recurrent.cpp',a.llama/'src/llama-context.cpp']+list(lib.glob('*.so'))
 hashes={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 (a.output/'build.json').write_text(json.dumps(dict(backend_revision=PIN,commands=commands,sha256=hashes),indent=2)+'\n');print(a.output/'build.json')
if __name__=='__main__':main()
