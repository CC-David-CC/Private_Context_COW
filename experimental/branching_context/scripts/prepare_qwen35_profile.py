"""Fingerprint a Qwen3.5-9B text execution profile, never an expert pack."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parents[1])]
from context_time_travel.qwen35_backend import validate_geometry
from tools.gguf_reader import GGUFFile

def make_profile(model,build_dir,output,capacity,batch,threads,gpus=None,lora=None,lora_scale=1.,math_profile='native'):
 class Args:pass
 a=Args();a.model=Path(model);a.build=Path(build_dir);a.output=Path(output);a.context=capacity;a.batch=batch;a.threads=threads;a.gpus=gpus;a.lora=lora;a.lora_scale=lora_scale;a.math_profile=math_profile
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
 math_profile=getattr(a,'math_profile','native')
 if math_profile not in ('native','f32-activations','f32-reference'):raise ValueError('Unsupported dense math profile')
 if math_profile in ('f32-activations','f32-reference'):
  if not build.get('f32_activations_patch_sha256'):raise ValueError('Rebuild native runtime with the explicit F32 activation patch')
  execution.update(f32_activations=True,cuda_cublas_compute_type='f32',nvidia_tf32_override='0')
  execution['environment'].update(GGML_CUDA_F32_ACTIVATIONS='1',GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32',NVIDIA_TF32_OVERRIDE='0')
 if math_profile=='f32-reference':
  if build.get('attention_precision_binding')!='fp32-attention-v1':raise ValueError('Rebuild native adapter for FP32 attention')
  execution.update(f32_attention=True,kv_dtype='FP32',flash_attention=False)
  execution['environment'].update(COW_QWEN35_F32_ATTENTION='1',GGML_CUDA_DISABLE_FUSION='1')
 contract=dict(weights_sha256=sha,architecture=md['general.architecture'],state_kind='qwen35-hybrid-frontier-v1',positions='absolute-zero-based',kv_dtype=execution['kv_dtype'],input_kind='token_ids')
 if mixed:contract.update(state_kind=build['state_format'],positions='qwen-image-grid-mrope-v1',input_kind='canonical-mixed-spans-v1')
 adapter=None
 if getattr(a,'lora',None) is not None:
  path=Path(a.lora).absolute()
  if path.is_symlink() or not path.is_file() or path.resolve()!=path:raise ValueError('Absolute regular LoRA file required')
  with path.open('rb') as stream:adapter_sha=hashlib.file_digest(stream,'sha256').hexdigest()
  identity=dict(sha256=adapter_sha,scale=float(a.lora_scale))
  execution['adapter']=identity;contract['adapters']=[identity]
  adapter=dict(path=str(path),**identity)
 eid=hashlib.sha256(json.dumps(dict(build=build,execution=execution,hardware=hardware),sort_keys=True).encode()).hexdigest()
 cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
 profile=dict(backend='qwen35-llama.cpp',model_path=str(a.model.resolve()),model_weights_sha256=sha,model_contract_id=cid,execution_profile_id=eid,native_build=build,execution=execution,hardware=hardware,geometry=geometry,bridge_library=str((a.build/'libqwen35ctt.so').resolve()),vocab_size=len(md['tokenizer.ggml.tokens']),oracle_binary=str((a.build/('qwen35-mixed-oracle' if mixed else 'qwen35-fresh-oracle')).resolve()))
 if adapter is not None:
  profile.update(adapter=adapter,model_contract=contract)
  from context_time_travel.qwen35_lora import validate_adapter
  validate_adapter(profile)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(profile,indent=2)+'\n');a.output.chmod(0o600);print(a.output);return profile
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,default=16384);p.add_argument('--batch',type=int,default=512);p.add_argument('--threads',type=int,default=os.cpu_count());p.add_argument('--lora',type=Path);p.add_argument('--lora-scale',type=float,default=1.);p.add_argument('--math-profile',choices=('native','f32-activations','f32-reference'),default='native');a=p.parse_args()
 _make(a)

if __name__=='__main__':main()
