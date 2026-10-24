"""Completed native optimizer continuation against uninterrupted full backprop.

Uses only generated tiny ordinary Llama weights; no MiniCPM learning claim.
"""
from pathlib import Path
import argparse,copy,hashlib,json,struct
import numpy as np
from .make_llama_fixture import make_fixture
from .llama_reference import read_gguf
from tools.pcc_native_training import execute_native,read_optimizer_state

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_arrays(path):
 metadata=read_optimizer_state(path,sha(path));arrays={}
 with Path(path).open('rb') as stream:
  assert stream.read(8)==b'PCCOPT1\n';length=int.from_bytes(stream.read(8),'little');stream.read(length)
  for tensor in metadata['tensors']:
   raw=stream.read(tensor['bytes']);assert len(raw)==tensor['bytes'];arrays[tensor['name']]=np.frombuffer(raw,dtype='<f4').copy()
  assert stream.read(1)==b''
 return metadata,arrays

def run(executable,output,devices):
 output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
 model=make_fixture(output/'base-f16.gguf');base=sha(model)
 e=dict(id='owned_a',tokens=[0,1,2,3],targets=[1,2,3,4],loss_mask=[False,True,True,True],verified=True,provenance_sha256='a'*64)
 examples=[e,dict(e,id='owned_b',tokens=[3,2,1,0],targets=[2,1,0,6]),dict(e,id='owned_c',tokens=[1,3,5,2],targets=[3,5,2,0])]
 common=dict(model=str(model),base_sha256=base,mode='full',devices=devices,device_budget_bytes={d:4<<30 for d in devices},reserve_bytes=128<<20,host_reserve_bytes=128<<20,sequence_profile='dense-llama-short-v1',steps=1,batch_size=2,learning_rate=.001,examples=examples,optimizer_state_mode='fresh')
 arms=[('adamw',{}),('muon',{}),('muon-rms',dict(scaling='rms',weight_decay=.1)),('muon-refine',dict(projection='fast-refine-v1',refine_start_step=2,refine_steps=5,scaling='rms',weight_decay=.1))]
 results=[];negative=0;first_pair=None
 for label,settings in arms:
  folder=output/label;folder.mkdir(mode=0o700);request=dict(common,optimizer='adamw' if label=='adamw' else 'muon',optimizer_settings=dict(schema=1,**settings))
  first=execute_native(executable,request,folder/'first',timeout=120)
  checkpoint=first['receipt']['optimizer_checkpoint']
  resumed_request=dict(request,model=first['candidate_path'],base_sha256=first['candidate_sha256'],steps=2,optimizer_state_mode='resume',initial_optimizer_state=checkpoint['path'],initial_optimizer_state_sha256=checkpoint['sha256'],example_cursor=first['receipt']['example_cursor'])
  resumed=execute_native(executable,resumed_request,folder/'resumed',timeout=120)
  straight=execute_native(executable,dict(request,steps=3),folder/'uninterrupted',timeout=120)
  actual,_=read_gguf(resumed['candidate_path']);expected,_=read_gguf(straight['candidate_path'])
  assert actual.keys()==expected.keys()
  assert all(np.array_equal(actual[n],expected[n]) for n in actual),label
  am,aa=state_arrays(resumed['receipt']['optimizer_checkpoint']['path']);em,ea=state_arrays(straight['receipt']['optimizer_checkpoint']['path'])
  assert aa.keys()==ea.keys() and all(np.array_equal(aa[n],ea[n]) for n in aa),label
  assert am['completed_steps']==em['completed_steps']==3 and am['example_cursor']==em['example_cursor']==5
  assert resumed['promoted'] is False and straight['promoted'] is False and sha(model)==base
  # Completed-directory reuse cannot apply another update.
  try:execute_native(executable,request,folder/'first',timeout=120)
  except FileExistsError:negative+=1
  else:raise AssertionError('Completed directory reused')
  results.append(dict(arm=label,status='PASS',weight_tensors=len(actual),state_tensors=len(aa),resumed_equals_uninterrupted_weights=True,resumed_equals_uninterrupted_moments=True,completed_steps=3,example_cursor=5,partial_final_batch=True,optimizer_refinement_active=resumed['receipt']['steps'][-1]['optimizer_step']['projection_refinement_active']))
  if first_pair is None:first_pair=(first,resumed_request,request)
 first,resume,request=first_pair
 for label,change in [('wrong-model',dict(model=str(model),base_sha256=base)),('wrong-order',dict(examples=list(reversed(examples)))),('wrong-cursor',dict(example_cursor=1))]:
  try:execute_native(executable,dict(resume,**change),output/label,timeout=120)
  except ValueError:negative+=1
  else:raise AssertionError(label+' admitted')
 corrupted=output/'corrupted-state.bin';data=bytearray(Path(resume['initial_optimizer_state']).read_bytes());data[-1]^=1;corrupted.write_bytes(data)
 try:execute_native(executable,dict(resume,initial_optimizer_state=str(corrupted)),output/'corrupted',timeout=120)
 except ValueError:negative+=1
 else:raise AssertionError('Corrupt state admitted')
 # Canonical optimizer settings are also checked by the native loader.
 for label,change,fragment in [('changed-rate',dict(learning_rate=.002),'configuration mismatch'),('changed-decay',dict(optimizer_settings=dict(schema=1,weight_decay=.1)),'configuration mismatch')]:
  try:execute_native(executable,dict(resume,**change),output/label,timeout=120)
  except RuntimeError:
   error=(output/label/'stderr.log').read_text();assert fragment in error,error[-1500:];negative+=1
  else:raise AssertionError(label+' state admitted')
  assert not (output/label/'candidate.gguf').exists()
 # Native parser must reject malformed payload/layout even with a matching SHA.
 original_state=Path(resume['initial_optimizer_state']).read_bytes();header_size=int.from_bytes(original_state[8:16],'little');header=json.loads(original_state[16:16+header_size]);payload=original_state[16+header_size:]
 malformed_header=copy.deepcopy(header);malformed_header['tensors'].pop()
 encoded=json.dumps(malformed_header,sort_keys=True,separators=(',',':')).encode()
 malformed={
  'missing-tensor':(b'PCCOPT1\n'+len(encoded).to_bytes(8,'little')+encoded+payload,'tensor coverage/roles mismatch'),
  'truncated-state':(original_state[:-4],'Truncated optimizer resume state'),
  'trailing-state':(original_state+b'extra','Unexpected trailing optimizer state bytes'),
  'nonfinite-state':(original_state[:16+header_size]+struct.pack('<f',float('nan'))+payload[4:],'Nonfinite optimizer resume state'),
 }
 for label,(body,fragment) in malformed.items():
  bad=output/(label+'.bin');bad.write_bytes(body)
  changed=dict(resume,initial_optimizer_state=str(bad),initial_optimizer_state_sha256=sha(bad))
  try:execute_native(executable,changed,output/label,timeout=120)
  except RuntimeError:
   error=(output/label/'stderr.log').read_text();assert fragment in error,error[-1500:];negative+=1
  else:raise AssertionError(label+' state admitted')
  assert not (output/label/'candidate.gguf').exists()
 result=dict(status='PASS',scope='Generated tiny ordinary-Llama full backprop and completed optimizer continuation; not MiniCPM learning',devices=devices,cases=results,negative_cases=negative,promoted=False)
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',nargs='+',default=['CPU']);a=p.parse_args();print(json.dumps(run(a.executable,a.output,a.devices)))
