"""Longer native ordinary-Llama qualification on generated tiny weights.

Actual CPU/CUDA graphs, FP64 forward and finite differences, full/LoRA updates,
optimizer-state continuation and reload. No language-model learning claim.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess
import numpy as np
from .make_llama_fixture import make_fixture
from .llama_reference import read_gguf,model_value
from tools.gguf_reader import GGUFFile
from tools.pcc_native_training import execute_native

PROFILE='dense-llama-action-8k-v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def example(n):
 values=[(i*3+i//11)%7 for i in range(n+1)]
 return dict(id='owned-long-'+str(n),tokens=values[:-1],targets=values[1:],
     loss_mask=[i in (n-2,n-1) for i in range(n)],verified=True,provenance_sha256='a'*64)

def tensor_arrays(path):
 file=GGUFFile(path);arrays={}
 with Path(path).open('rb') as stream:
  for tensor in file.tensors:
   assert tensor.type_name=='F32'
   stream.seek(file.data_start+tensor.offset)
   arrays[tensor.name]=np.frombuffer(stream.read(tensor.expected_bytes()),dtype='<f4').astype(np.float64).reshape(
       tensor.shape[1] if len(tensor.shape)>1 else 1,tensor.shape[0])
 return arrays

def run(executable,output,devices):
 executable=Path(executable).resolve(strict=True);output=Path(output)
 output.mkdir(mode=0o700,parents=True,exist_ok=False)
 devices=list(devices)
 if devices not in (['CPU'],['CUDA0']):raise ValueError('This fixture qualifies CPU or one explicitly mapped CUDA0')
 model=make_fixture(output/'tiny-f32.gguf',dtype='f32',layers=1,context_length=8192)
 base=sha(model);weights,h=read_gguf(model)
 common=dict(model=str(model),base_sha256=base,mode='full',rank=1,devices=devices,
     sequence_profile=PROFILE,memory_profile='allocator-v1',steps=1,batch_size=1,learning_rate=.01,
     device_budget_bytes={d:12<<30 for d in devices},reserve_bytes=128<<20,host_reserve_bytes=128<<20)
 checks=[];negative=0
 def native(label,request,*,fragment=None):
  path=output/(label+'-request.json');path.write_text(json.dumps(request)+'\n')
  process=subprocess.run([str(executable),'--train',str(path)],capture_output=True,text=True,timeout=600)
  (output/(label+'-stdout.json')).write_text(process.stdout)
  (output/(label+'-stderr.log')).write_text(process.stderr)
  if fragment is not None:
   assert process.returncode!=0 and fragment in process.stderr,(label,process.stderr[-2000:])
   assert not Path(request['candidate']).exists()
   return None
  assert process.returncode==0,(label,process.stderr[-2000:])
  value=json.loads(process.stdout)
  assert value['sequence_profile']==request['sequence_profile']
  return value
 def forward(label,e,profile=PROFILE,source=model):
  log=output/(label+'-logits.f32')
  request=dict(common,model=str(source),base_sha256=sha(source),examples=[e],sequence_profile=profile,
      evaluate_only=True,logits_output=str(log),candidate=str(output/(label+'-unused.gguf')))
  value=native(label,request)
  w,geometry=read_gguf(source);expected,logits=model_value(e,w,geometry)
  observed=np.fromfile(log,dtype='<f4').reshape(logits.shape[1],logits.shape[0]).T
  assert abs(expected-value['losses'][0])<1e-5,(label,expected,value['losses'])
  assert np.allclose(logits,observed,rtol=5e-4,atol=5e-5),(label,float(np.max(np.abs(logits-observed))))
  checks.append(dict(name=label,tokens=len(e['tokens']),loss_error=abs(expected-value['losses'][0]),
      maximum_absolute_logit_error=float(np.max(np.abs(logits-observed))),
      graph_peaks=value['allocation_peaks']))
  return value,log.read_bytes()
 # Explicit old/new routes must retain identical existing short forward bytes.
 old,old_bytes=forward('old-short-profile',example(4),profile='dense-llama-short-v1')
 new,new_bytes=forward('new-profile-short-input',example(4))
 assert old_bytes==new_bytes and old['losses']==new['losses']
 checks.append(dict(name='old-short-forward-byte-equivalence',status='PASS'))
 e=example(513)
 forward('beyond-old-bound-forward',e)
 # Read a real full update's complete F32 artifact to obtain the native SGD
 # derivative without adding a diagnostic gradient-output implementation.
 candidate=execute_native(executable,dict(common,examples=[e]),output/'full-sgd',timeout=600)
 updated,_=read_gguf(candidate['candidate_path']);rate=float(np.float32(common['learning_rate']))
 errors=[];sensitivity=[]
 for name,values in weights.items():
  index=(0,0);saved=values[index];estimates=[]
  for epsilon in (1e-5,2e-5):
   values[index]=saved+epsilon;plus=model_value(e,weights,h)[0]
   values[index]=saved-epsilon;minus=model_value(e,weights,h)[0]
   values[index]=saved;estimates.append((plus-minus)/(2*epsilon))
  observed=(saved-updated[name][index])/rate
  errors.append(abs(observed-estimates[0]));sensitivity.append(abs(estimates[0]-estimates[1]))
  assert np.isclose(observed,estimates[0],rtol=.005,atol=5e-5),(name,observed,estimates)
  assert np.isclose(estimates[0],estimates[1],rtol=1e-4,atol=1e-7)
 checks.append(dict(name='513-token-full-backprop-vs-independent-fp64-finite-differences',
     checked_tensor_directions=len(errors),maximum_absolute_gradient_error=max(errors),step_sensitivity=max(sensitivity)))
 forward('full-update-independent-reload',e,source=Path(candidate['candidate_path']))
 # Rank one exercises the size-one outer-product route as well as long context.
 adapted=execute_native(executable,dict(common,mode='lora',rank=1,examples=[e]),output/'lora-r1-sgd',timeout=600)
 arrays=tensor_arrays(adapted['candidate_path']);effective=copy.deepcopy(weights)
 b_errors=[]
 for name in weights:
  if name+'.lora_a' not in arrays:continue
  a=arrays[name+'.lora_a'];b=arrays[name+'.lora_b'];effective[name]+=b@a
  estimates=[]
  for epsilon in (1e-5,2e-5):
   weights[name][0,:]+=epsilon*a[0,:];plus=model_value(e,weights,h)[0]
   weights[name][0,:]-=2*epsilon*a[0,:];minus=model_value(e,weights,h)[0]
   weights[name][0,:]+=epsilon*a[0,:];estimates.append((plus-minus)/(2*epsilon))
  observed=-b[0,0]/rate;b_errors.append(abs(observed-estimates[0]))
  assert np.isclose(observed,estimates[0],rtol=.005,atol=5e-5),(name,observed,estimates)
  assert np.isclose(estimates[0],estimates[1],rtol=1e-4,atol=1e-7)
 expected=model_value(e,effective,h)[0];after=adapted['receipt']['steps'][0]['after']
 assert abs(expected-after)<1e-5
 checks.append(dict(name='513-token-rank-one-backprop-vs-independent-fp64',checked_b_directions=len(b_errors),
     maximum_absolute_gradient_error=max(b_errors),independent_after_loss_error=abs(expected-after)))
 # Actual maximum-size forward AND backward; only the generated tiny fixture.
 largest=example(8192);forward('8192-token-forward',largest)
 largest_update=execute_native(executable,dict(common,examples=[largest]),output/'8192-token-full-update',timeout=600)
 assert largest_update['receipt']['sequence_profile']==PROFILE
 checks.append(dict(name='8192-token-standard-full-backprop',parameter_tensors=len(largest_update['receipt']['steps'][0]['gradients']),
     graph_peaks=largest_update['receipt']['allocation_peaks'],weights_updated=True))
 # Adaptive states must remain usable for the exact long-example stream.
 for mode,kind in [('full','adamw'),('full','muon'),('lora','adamw')]:
  label=mode+'-'+kind;request=dict(common,mode=mode,optimizer=kind,optimizer_state_mode='fresh',examples=[e])
  first=execute_native(executable,request,output/(label+'-first'),timeout=600)
  checkpoint=first['receipt']['optimizer_checkpoint']
  resumed=dict(request,optimizer_state_mode='resume',initial_optimizer_state=checkpoint['path'],
      initial_optimizer_state_sha256=checkpoint['sha256'],example_cursor=1)
  if mode=='full':resumed.update(model=first['candidate_path'],base_sha256=first['candidate_sha256'])
  else:resumed.update(initial_adapter=first['candidate_path'],initial_adapter_sha256=first['candidate_sha256'])
  second=execute_native(executable,resumed,output/(label+'-resumed'),timeout=600)
  assert second['receipt']['optimizer_checkpoint']['completed_steps']==2 and second['receipt']['example_cursor']==2
  checks.append(dict(name=label+'-long-context-state-continuation',completed_updates=2,example_cursor=2))
 for label,change,fragment in [
     ('old-short-still-refuses-513',dict(sequence_profile='dense-llama-short-v1',examples=[e]),'Sequence admission refused'),
     ('new-profile-refuses-8193',dict(examples=[example(8193)]),'Sequence admission refused'),
     ('new-profile-refuses-257-targets',dict(examples=[dict(e,loss_mask=[i<257 for i in range(513)])]),'256 selected'),
     ('new-profile-refuses-context-overflow',dict(model=str(make_fixture(output/'tiny-short-context.gguf',layers=1,context_length=512)),examples=[e]),'context metadata'),
 ]:
  request=dict(common,evaluate_only=True,examples=[e],candidate=str(output/(label+'-never.gguf')))
  request.update(change)
  request['base_sha256']=sha(request['model'])
  native(label,request,fragment=fragment);negative+=1
 assert sha(model)==base
 result=dict(schema=1,status='PASS',scope='Generated tiny ordinary-Llama 8k native mechanics; not real MiniCPM learning or model quality',
     profile=PROFILE,devices=devices,gpu_used=devices!=['CPU'],checks=checks,negative_cases=negative,
     source_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable),original_base_unchanged=True,promoted=False)
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--devices',nargs='+',default=['CPU'])
 args=parser.parse_args();print(json.dumps(run(args.executable,args.output,args.devices)))
