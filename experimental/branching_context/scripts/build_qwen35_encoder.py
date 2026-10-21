"""Build the COW image-producer adapter against the same clean llama.cpp pin."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from build_qwen35 import PIN
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--llama',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if subprocess.check_output(['git','-C',str(a.llama),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.llama),'status','--porcelain']):raise SystemExit('Requires clean inspected llama.cpp pin')
 a.output.mkdir(parents=True,exist_ok=True);lib=a.llama/'build/bin';source=ROOT/'native/qwen35_vision_encoder.cpp';binary=a.output/'qwen35-vision-encoder'
 cmd=['g++','-std=c++17','-O3',str(source),'-I'+str(a.llama/'include'),'-I'+str(a.llama/'ggml/include'),'-I'+str(a.llama/'tools/mtmd'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lmtmd','-lllama','-lggml','-o',str(binary)]
 subprocess.run(cmd,check=True)
 files=[binary,source,a.llama/'include/llama.h',a.llama/'tools/mtmd/mtmd.h',a.llama/'tools/mtmd/mtmd-helper.h',a.llama/'tools/mtmd/mtmd.cpp',a.llama/'tools/mtmd/mtmd-helper-common.h']+list(lib.glob('*.so'))
 result=dict(backend_revision=PIN,commands=[cmd],sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 (a.output/'encoder-build.json').write_text(json.dumps(result,indent=2)+'\n');print(a.output/'encoder-build.json')
if __name__=='__main__':main()
