"""Executed matching-schedule Qwen branching/restart probe. No mocked engine."""
import argparse,hashlib,json,os,struct,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from context_time_travel.qwen35_backend import Qwen35Backend,Qwen35FreshOracle
from context_time_travel.real import BranchingModel
from context_time_travel.execution_receipt import tokens_digest

def write(p,v):
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def raw(values):return struct.pack('<'+str(len(values))+'f',*values)
def event(ids,pos,chunk):return dict(operator='batched-prefill',position=pos,count=len(ids),tokens_sha256=tokens_digest(ids),chunks=[[pos+i,min(chunk,len(ids)-i)] for i in range(0,len(ids),chunk)])
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['reference','prepare','resume','campaign']);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=int,default=128);p.add_argument('--suffix',type=int,default=4);p.add_argument('--outputs',type=int,default=8);a=p.parse_args()
 os.umask(0o077);a.output.mkdir(parents=True,exist_ok=True);profile=json.loads(a.profile.read_text());chunk=profile['execution']['prefill_chunk'];started=time.monotonic()
 result=dict(phase=a.phase,status='RUNNING',pid=os.getpid(),prefix=a.prefix,suffix=a.suffix,output_limit=a.outputs)
 def progress(**kw):result.update(kw);write(a.output/(a.phase+'-result.json'),result)
 progress()
 if a.phase=='campaign':
  for phase in ['reference','prepare','resume']:
   cmd=[sys.executable,__file__,phase,'--profile',str(a.profile),'--output',str(a.output),'--prefix',str(a.prefix),'--suffix',str(a.suffix),'--outputs',str(a.outputs)]
   with (a.output/(phase+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);last=0
    while child.poll() is None:
     now=time.monotonic()
     if now-last>30:progress(child_pid=child.pid,child_phase=phase,seconds=now-started);print(phase,child.pid,round(now-started,1),flush=True);last=now
     time.sleep(1)
    if child.returncode:raise RuntimeError('Phase failed: '+phase+'; inspect retained log')
  progress(status='PASS',seconds=time.monotonic()-started);return
 if a.phase=='reference':
  oracle=Qwen35FreshOracle(profile)
  try:
   seed=oracle.tokenize('The project has a blue website and a tested Python program. ')
   prefix=tuple((seed*((a.prefix+len(seed)-1)//len(seed)))[:a.prefix]);schedule=[event(prefix,0,chunk)]
   expected=oracle.forward_schedule(prefix,schedule);repeat=oracle.forward_schedule(prefix,schedule)
   if raw(expected)!=raw(repeat):raise AssertionError('Fresh repeat disagrees; exactness prerequisite failed')
   (a.output/'prefix.f32').write_bytes(raw(expected));fixtures=dict(prefix=list(prefix),branches={})
   for name,text in [('A','Now write a small dog website. '),('B','Instead explain the math result. ')]:
    suffix_seed=oracle.tokenize(text);suffix=tuple((suffix_seed*((a.suffix+len(suffix_seed)-1)//len(suffix_seed)))[:a.suffix]);ids=prefix+suffix;sc=schedule+[event(suffix,len(prefix),chunk)]
    frontier=oracle.forward_schedule(ids,sc);(a.output/(name+'-0.f32')).write_bytes(raw(frontier));tokens=[]
    for i in range(a.outputs):
     t=max(range(len(frontier)),key=frontier.__getitem__);tokens.append(t);sc.append(event((t,),len(ids),chunk));ids+=(t,)
     frontier=oracle.forward_schedule(ids,sc);(a.output/(name+'-'+str(i+1)+'.f32')).write_bytes(raw(frontier));progress(branch=name,generated=i+1)
    fixtures['branches'][name]=dict(suffix=list(suffix),tokens=tokens)
   if fixtures['branches']['A']['suffix']==fixtures['branches']['B']['suffix']:raise AssertionError('Suffixes must differ')
   write(a.output/'fixtures.json',fixtures);progress(status='PASS',fresh_forwards=oracle.requests,fresh_forward_tokens=oracle.tokens_computed,seconds=time.monotonic()-started)
  finally:oracle.close()
  return
 fixture=json.loads((a.output/'fixtures.json').read_text());backend=Qwen35Backend(profile)
 native_acquire=backend._acquire
 def observed_acquire():
  gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True).strip()
  try:return native_acquire()
  except BaseException:
   print('CONTEXT_ALLOCATION_BEFORE',gpu_before,flush=True)
   print(subprocess.check_output(['nvidia-smi'],text=True),flush=True);raise
 backend._acquire=observed_acquire
 model=BranchingModel(a.output/'store',backend,profile);comparisons=0;max_error=0;receipts=[]
 def compare(sid,file):
  nonlocal comparisons,max_error
  actual=raw(model.next_logits(sid));expected=file.read_bytes()
  if actual!=expected:
   x=struct.unpack('<'+str(len(actual)//4)+'f',actual);y=struct.unpack('<'+str(len(expected)//4)+'f',expected);err=max(abs(a-b) for a,b in zip(x,y));max_error=max(max_error,err)
   raise AssertionError(('Exact frontier mismatch',file.name,err))
  comparisons+=1
 try:
  if a.phase=='prepare':
   root=model.create(tuple(fixture['prefix']));compare(root,a.output/'prefix.f32');model.store.fork('prefix',root);model.checkpoint(root);points={'root':root,'generated':{},'controllers':{}}
   for name,row in fixture['branches'].items():
    sid=model.feed(root,tuple(row['suffix']));compare(sid,a.output/(name+'-0.f32'));sid=model.start_generation(sid,{'temperature':0},'fixture-'+name)
    for i in range(min(4,a.outputs)):
     g=model.generate(sid,1);sid=g.snapshot_id
     if g.accepted_tokens!=(row['tokens'][i],):raise AssertionError('Generated token differs')
     compare(sid,a.output/(name+'-'+str(i+1)+'.f32'))
    model.checkpoint(sid);points['generated'][name]=sid;points['controllers'][name]=model.controller(sid)
   write(a.output/'points.json',points)
  else:
   points=json.loads((a.output/'points.json').read_text());root=points['root'];before=backend.work_counters();receipts.append(model.restore_checkpoint(root));compare(root,a.output/'prefix.f32')
   if backend.work_counters()['consumed_tokens']!=before['consumed_tokens']:raise AssertionError('Prefix replayed')
   payloads=list(model.checkpoint_directory.glob('*.state'));hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in payloads}
   for name,row in fixture['branches'].items():
    sid=model.feed(root,tuple(row['suffix']));compare(sid,a.output/(name+'-0.f32'));sid=model.start_generation(sid,{'temperature':0},'fixture-'+name)
    for i in range(a.outputs):
     g=model.generate(sid,1);sid=g.snapshot_id
     if g.accepted_tokens!=(row['tokens'][i],):raise AssertionError('Generated token differs')
     compare(sid,a.output/(name+'-'+str(i+1)+'.f32'));progress(branch=name,generated=i+1,comparisons=comparisons)
    restored=points['generated'][name];receipts.append(model.restore_checkpoint(restored));sid=restored
    if model.controller(sid)!=points['controllers'][name]:raise AssertionError('Canonical controller changed across restart')
    for i in range(min(4,a.outputs),a.outputs):
     g=model.generate(sid,1);sid=g.snapshot_id
     if g.accepted_tokens!=(row['tokens'][i],):raise AssertionError('Generated restart duplicated or omitted token')
     compare(sid,a.output/(name+'-'+str(i+1)+'.f32'))
   if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise AssertionError('Saved parent/sibling bytes changed')
  progress(status='PASS',comparisons=comparisons,max_logit_error=max_error,restore_receipts=receipts,native_work=backend.work_counters(),seconds=time.monotonic()-started,model_contract_id=profile['model_contract_id'],execution_profile_id=profile['execution_profile_id'])
 finally:backend.close()
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if '--output' in sys.argv:
   folder=Path(sys.argv[sys.argv.index('--output')+1]);phase=sys.argv[1];path=folder/(phase+'-result.json')
   result=json.loads(path.read_text()) if path.exists() else {}
   result.update(status='FAIL',error=repr(error));write(path,result)
  traceback.print_exc();raise
