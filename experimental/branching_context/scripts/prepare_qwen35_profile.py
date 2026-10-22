"""Fingerprint a Qwen3.5-9B text execution profile, never an expert pack."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parents[1])]
from context_time_travel.qwen35_backend import validate_geometry
from tools.gguf_reader import GGUFFile

def make_profile(model,build_dir,output,capacity,batch,threads,gpus=None):
 class Args:pass
 a=Args();a.model=Path(model);a.build=Path(build_dir);a.output=Path(output);a.context=capacity;a.batch=batch;a.threads=threads;a.gpus=gpus
 return _make(a)

def _make(a):
 md=GGUFFile(a.model).metadata;geometry=validate_geometry(md)
 build=json.loads((a.build/'build.json').read_text())
 gpus=getattr(a,'gpus',None)
 if gpus is not None:
  from context_time_travel.pools import normalize_gpu_selection
  gpus=normalize_gpu_selection(gpus)
  if any(type(gpu) is not int for gpu in gpus):raise ValueError('Dense grouped workers require multi-device checkpoint support')
  if build.get('device_binding')!='cuda-single-device-v1':raise ValueError('Rebuild dense adapter for explicit GPU selection')
 for path,sha in build['sha256'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise SystemExit('Build changed: '+path)
 print('Hashing model',flush=True)
 with a.model.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 runtime_root=ROOT.parents[1];prefixes=['serve','tools','experimental/branching_context/src','experimental/branching_context/scripts']
 files={str(f.relative_to(runtime_root)):hashlib.sha256(f.read_bytes()).hexdigest() for prefix in prefixes for f in (runtime_root/prefix).rglob('*.py') if '__pycache__' not in f.parts}
 build['selected_runtime_source_closure']=dict(source_commit=subprocess.check_output(['git','-C',str(runtime_root),'rev-parse','HEAD'],text=True).strip(),files_sha256=files)
 hardware=dict(host=os.uname().nodename,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip())
 eos=[md['tokenizer.ggml.tokens'].index(t) for t in ('<|endoftext|>','<|im_end|>') if t in md['tokenizer.ggml.tokens']]
 execution=dict(environment={'CUDA_VISIBLE_DEVICES':'0'},eos_tokens=eos,text_template_kwargs={'enable_thinking':True},generated_operator='llama-forward-consumed-v1',selected_token_protocol='consumed-selected-receipt-v1',sampling_protocol='full-vocab-min-p-u53-v2',sampling_location='native-cpu-frontier',capacity=a.context,prefill_chunk=a.batch,threads=a.threads,kv_dtype='FP16',flash_attention=True,state_format='qwen35-hybrid-frontier-v1',resume_contract='identical-native-batches-v1',cuda_cublas_compute_type=os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE','auto'),nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE','unset'),GPU=[0],physical_sharing=False,embedding_width=md[md['general.architecture']+'.embedding_length'])
 mixed=build.get('state_format')=='qwen35-hybrid-mrope-frontier-v2'
 if gpus is not None:
  execution.update(GPU=list(gpus),device_binding='cuda-single-device-v1',
      environment={'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))})
 if mixed:
  execution.update(state_format=build['state_format'],resume_contract='identical-mixed-native-operators-v1',vision_protocol='explicit-image-rows-mrope-v3',native_binding_protocol=build['native_binding_protocol'])
 eid=hashlib.sha256(json.dumps(dict(build=build,execution=execution,hardware=hardware),sort_keys=True).encode()).hexdigest()
 contract=dict(weights_sha256=sha,architecture=md['general.architecture'],state_kind='qwen35-hybrid-frontier-v1',positions='absolute-zero-based',kv_dtype='FP16',input_kind='token_ids')
 if mixed:contract.update(state_kind=build['state_format'],positions='qwen-image-grid-mrope-v1',input_kind='canonical-mixed-spans-v1')
 cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
 profile=dict(backend='qwen35-llama.cpp',model_path=str(a.model.resolve()),model_weights_sha256=sha,model_contract_id=cid,execution_profile_id=eid,native_build=build,execution=execution,hardware=hardware,geometry=geometry,bridge_library=str((a.build/'libqwen35ctt.so').resolve()),vocab_size=len(md['tokenizer.ggml.tokens']),oracle_binary=str((a.build/('qwen35-mixed-oracle' if mixed else 'qwen35-fresh-oracle')).resolve()))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(profile,indent=2)+'\n');a.output.chmod(0o600);print(a.output);return profile
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,default=16384);p.add_argument('--batch',type=int,default=512);p.add_argument('--threads',type=int,default=os.cpu_count());a=p.parse_args()
 _make(a)

if __name__=='__main__':main()
