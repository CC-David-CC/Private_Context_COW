"""Qualify tiny ordinary Llama F16/full/LoRA mechanics with independent math.

No real language-model training, GPU claim, task-learning claim or promotion.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from .make_llama_fixture import make_fixture
from .make_cpu_fixture import make_fixture as make_hybrid_fixture
from .llama_reference import verify_chain,verify_layers,read_gguf,model_value
from tools.pcc_native_training import execute_native

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(executable,output,devices=None):
 devices=['CPU'] if devices is None else list(devices)
 if not devices or (devices!=['CPU'] and any(not x.startswith('CUDA') or not x[4:].isdigit() for x in devices)):
  raise ValueError('Explicit CPU or CUDA device names required')
 executable=Path(executable).resolve(strict=True);output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
 def command(label,args,should_fail=False):
  result=subprocess.run([str(executable),*args],capture_output=True,text=True,timeout=90)
  (output/(label+'.stdout')).write_text(result.stdout);(output/(label+'.stderr')).write_text(result.stderr)
  if should_fail:
   assert result.returncode!=0,(label,result.stdout);return result.stderr
  assert result.returncode==0,(label,result.stderr[-1800:])
  return json.loads(result.stdout)
 def native(label,request,should_fail=False):
  path=output/(label+'-request.json');path.write_text(json.dumps(request)+'\n')
  return command(label,['--train',str(path)],should_fail)
 checks=[]
 for label,option,verify in [('layer','--llama-gradient-diagnostic',verify_layers),('complete','--llama-chain-diagnostic',verify_chain)]:
  proof=verify(command(label,[option,','.join(devices) if label=='complete' else devices[0]]))
  assert proof['status']=='PASS',proof
  (output/(label+'-independent.json')).write_text(json.dumps(proof,indent=2)+'\n');checks.append(dict(name=label,independent=proof))
 example=dict(id='owned_a',tokens=[0,1,2,3],targets=[1,2,3,4],loss_mask=[False,True,True,True],verified=True,provenance_sha256='a'*64)
 examples=[example,dict(example,id='owned_b',tokens=[3,2,1,0],targets=[2,1,0,6]),dict(example,id='owned_c',tokens=[1,3,5,2],targets=[3,5,2,0])]
 models={dtype:make_fixture(output/('tiny-'+dtype+'.gguf'),dtype=dtype) for dtype in ['f16','f32']}
 base=dict(devices=devices,rank=2,sequence_profile='dense-llama-short-v1',device_budget_bytes={name:4<<30 for name in devices},reserve_bytes=128<<20,
   host_reserve_bytes=128<<20,steps=2,batch_size=2,learning_rate=.001,examples=examples)
 for dtype,model in models.items():
  logits_file=output/(dtype+'-logits.f32')
  request=dict(base,model=str(model),base_sha256=sha(model),mode='full',evaluate_only=True,examples=[example],logits_output=str(logits_file),candidate=str(output/'unused.gguf'))
  actual=native(dtype+'-forward',request)
  weights,h=read_gguf(model);loss,logits=model_value(example,weights,h)
  observed=np.fromfile(logits_file,dtype='<f4').reshape(logits.shape[1],logits.shape[0]).T
  assert actual['architecture_contract']=='llama-dense-v1'
  assert abs(loss-actual['losses'][0])<5e-6
  assert np.allclose(logits,observed,rtol=5e-4,atol=5e-5)
  checks.append(dict(name=dtype+'-decode-and-forward',max_absolute_logit_error=float(np.max(np.abs(logits-observed))),loss_error=abs(loss-actual['losses'][0])))
 model=models['f16'];original=sha(model)
 for mode in ['full','lora']:
  request=dict(base,model=str(model),base_sha256=original,mode=mode)
  candidate=execute_native(executable,request,output/(mode+'-update'),timeout=120)
  receipt=candidate['receipt'];assert receipt['architecture_contract']=='llama-dense-v1'
  assert candidate['promoted'] is False and receipt['promoted'] is False
  assert [s['examples'] for s in receipt['steps']]==[2,1]
  assert receipt['example_cursor']==3
  expected_parameters=21 if mode=='full' else 30
  assert all(len(s['gradients'])==expected_parameters for s in receipt['steps'])
  assert all(np.isfinite(g['l2']) for s in receipt['steps'] for g in s['gradients'])
  reload=dict(request,evaluate_only=True,examples=[examples[2]],candidate=str(output/'unused.gguf'))
  if mode=='full':reload.update(model=candidate['candidate_path'],base_sha256=candidate['candidate_sha256'])
  else:reload.update(initial_adapter=candidate['candidate_path'],initial_adapter_sha256=candidate['candidate_sha256'])
  loaded=native(mode+'-reload',reload)
  assert abs(loaded['losses'][0]-receipt['steps'][-1]['after'])<1e-5
  assert sha(model)==original
  row=dict(name=mode+'-batch-update-and-reload',parameters=expected_parameters,partial_final_batch=True,reload_loss_error=abs(loaded['losses'][0]-receipt['steps'][-1]['after']),base_unchanged=True)
  if mode=='full':
   old,h=read_gguf(model);new,_=read_gguf(candidate['candidate_path'])
   row['tensors_with_changed_values']=sum(not np.array_equal(old[n],new[n]) for n in old)
   assert row['tensors_with_changed_values']>0
  checks.append(row)
  # Existing run directories refuse another update; there is no hidden replay.
  before=sha(candidate['candidate_path'])
  try:execute_native(executable,request,output/(mode+'-update'),timeout=120)
  except FileExistsError:pass
  else:raise AssertionError('Completed run directory admitted a second update')
  assert sha(candidate['candidate_path'])==before
 deep=make_fixture(output/'tiny-42-layer-f16.gguf',layers=42)
 request=dict(base,model=str(deep),base_sha256=sha(deep),mode='full',gradient_probe_only=True,examples=[example],candidate=str(output/'never-written.gguf'))
 gradient=native('forty-two-layer-gradient',request)
 assert gradient['model_layers']==42 and len(gradient['gradients'])==381
 assert gradient['weights_updated'] is False and not Path(request['candidate']).exists()
 checks.append(dict(name='forty-two-layer-standard-backprop',parameter_tensors=len(gradient['gradients']),weights_updated=False))
 request=dict(base,model=str(model),base_sha256=original,mode='full',evaluate_only=True,candidate=str(output/'refused.gguf'))
 for label,change,fragment in [
  ('wrong-profile',dict(sequence_profile='dense-short-v1'),'explicit bounded sequence'),
  ('overlong',dict(examples=[dict(example,tokens=[0]*513,targets=[0]*513,loss_mask=[False]*512+[True])]),'Sequence admission'),
  ('too-many-targets',dict(examples=[dict(example,tokens=[0]*512,targets=[0]*512,loss_mask=[True]*512)]),'256 selected')]:
  error=native(label,dict(request,**change),True);assert fragment in error,(label,error)
  assert not Path(request['candidate']).exists()
 hybrid=make_hybrid_fixture(output/'unsupported-hybrid-f16.gguf',dtype='f16')
 error=native('hybrid-f16-remains-refused',dict(request,model=str(hybrid),base_sha256=sha(hybrid),sequence_profile='dense-short-v1'),True)
 assert 'Unsupported trainable source dtype' in error,error
 result=dict(status='PASS',scope='Tiny ordinary Llama native mechanics, not real MiniCPM learning',checks=checks,negative_cases=6,gpu_used=devices!=['CPU'],devices=devices,promoted=False)
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',nargs='+',default=['CPU']);a=p.parse_args()
 print(json.dumps(run(a.executable,a.output,a.devices)))
