"""Real tiny native conversions and immutable source/output refusal checks."""
from pathlib import Path
import argparse,tempfile,json
from experimental.native_training.make_llama_fixture import make_fixture
from tools.gguf_reader import GGUFFile,GGML_TYPES,TensorInfo
from tools.pcc_model_quantize import convert,sha

def run(executable,output):
 root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False);model=make_fixture(root/'base.gguf');base=sha(model)
 assert GGML_TYPES[41]=='Q1_0' and TensorInfo('matrix',[128,2],41,'Q1_0',0).expected_bytes()==36
 results=[]
 for format in ['F16','F32']:
  result=convert(executable,model,base,root/format,format,threads=2,timeout=60);assert result['status']=='CONVERTED_NOT_EVALUATED' and result['behavior_measured'] is False;results.append(result)
 for args in [(executable,model,'0'*64,root/'bad-hash','F16'),(executable,model,base,root/'F16','F16')]:
  try:convert(*args)
  except (ValueError,FileExistsError):pass
  else:raise AssertionError('Unpinned source or reused destination admitted')
 assert sha(model)==base
 value=dict(status='PASS',scope='Actual tiny GGUF F16/F32 conversion; Q1_0 header geometry; immutable-source/output refusal',cases=results)
 (root/'result.json').write_text(json.dumps(value,indent=2)+'\n');return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.executable,a.output)))
