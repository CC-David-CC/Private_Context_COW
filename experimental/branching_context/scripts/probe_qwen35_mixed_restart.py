"""Independent fresh mixed forwards vs canonical branches and process-separated restore."""
import argparse,hashlib,json,os,struct,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT.parents[1])]
from context_time_travel.qwen35_mixed_backend import Qwen35MixedBackend,Qwen35MixedOracle
from context_time_travel.real import BranchingModel
from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_embeddings import parse_sve
from context_time_travel.vision_execution import input_binding
from context_time_travel.execution_receipt import tokens_digest
from context_time_travel.store import IntegrityError
from context_time_travel.vision_artifacts import ImageArtifacts
from context_time_travel.pools import SharedCheckpointPool,DiskCheckpointPool

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def raw(v):return struct.pack('<'+str(len(v))+'f',*v)
def event(prior,delta,chunk):
 h=prior+delta;chunks=[];offset=len(prior)
 for span in delta.spans:
  n=len(span['tokens']) if span['kind']=='text' else span['rows']
  chunks.extend([[offset+i,min(chunk,n-i)] for i in range(0,n,chunk)]);offset+=n
 return dict(operator='batched-prefill',position=len(prior),count=len(delta),tokens_sha256=tokens_digest(delta.native_tokens),chunks=chunks,inputs_sha256=h.digest,**input_binding(h))

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['reference','prepare','resume','campaign']);p.add_argument('--profile',type=Path,required=True);p.add_argument('--embedding',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--encoder-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=int,default=128);p.add_argument('--suffix',type=int,default=4);p.add_argument('--outputs',type=int,default=8);a=p.parse_args();os.umask(0o077);a.output.mkdir(parents=True,exist_ok=True)
 profile=json.loads(a.profile.read_text());chunk=profile['execution']['prefill_chunk'];started=time.monotonic();result=dict(status='RUNNING',phase=a.phase,pid=os.getpid(),prefix_text_tokens=a.prefix,suffix_tokens=a.suffix,output_limit=a.outputs)
 def progress(**kw):result.update(kw);write(a.output/(a.phase+'-result.json'),result)
 progress()
 if a.phase=='campaign':
  for phase in ['reference','prepare','resume']:
   cmd=[sys.executable,__file__,phase,*sys.argv[2:]]
   with (a.output/(phase+'.log')).open('w') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None:
    progress(child_phase=phase,child_pid=child.pid,seconds=time.monotonic()-started);print(phase,child.pid,round(time.monotonic()-started,1),flush=True);time.sleep(5)
   if child.returncode:raise RuntimeError('Retained phase failure: '+phase)
  progress(status='PASS',seconds=time.monotonic()-started);return
 payload=a.embedding.read_bytes();embedding=parse_sve(payload,expected_sha256=hashlib.sha256(payload).hexdigest(),expected_width=4096);original=a.image.read_bytes();image_hash=hashlib.sha256(original).hexdigest()
 if a.phase=='reference':
  oracle=Qwen35MixedOracle(profile,embeddings={embedding.sha256:embedding})
  try:
   seed=oracle.tokenize('The project has a blue website and a tested Python program. ');prefix=list((seed*((a.prefix+len(seed)-1)//len(seed)))[:a.prefix]);tail=list(oracle.tokenize('Now describe this image. ')[:8])
   span=dict(kind='image',original_sha256=image_hash,normalized_sha256=image_hash,encoder_profile_id=a.encoder_id,embedding_sha256=embedding.sha256,embedding_bytes=embedding.nbytes,rows=embedding.rows,width=embedding.width,grid_width=embedding.grid_width,grid_height=embedding.grid_height,pad_token_id=248056)
   # Token ID comes from this actual GGUF, not a cross-model constant.
   from tools.strata_tokenizer import Tokenizer
   span['pad_token_id'],=Tokenizer.from_gguf(profile['model_path']).encode('<|image_pad|>',parse_special=True)
   root=InputHistory([dict(kind='text',tokens=prefix),span,dict(kind='text',tokens=tail)]);empty=InputHistory([]);sc=[event(empty,root,chunk)]
   f=oracle.forward_inputs(root,sc);repeat=oracle.forward_inputs(root,sc)
   assert raw(f)==raw(repeat),'Fresh mixed repeat differs';(a.output/'prefix.f32').write_bytes(raw(f));fixture=dict(root_spans=root.spans,branches={})
   for name,text in [('A','Now write a small dog website. '),('B','Instead explain the math result. ')]:
    source=oracle.tokenize(text);suffix=list((source*((a.suffix+len(source)-1)//len(source)))[:a.suffix]);delta=InputHistory([dict(kind='text',tokens=suffix)]);history=root+delta;schedule=sc+[event(root,delta,chunk)];frontier=oracle.forward_inputs(history,schedule);(a.output/(name+'-0.f32')).write_bytes(raw(frontier));tokens=[]
    for i in range(a.outputs):
     t=max(range(len(frontier)),key=frontier.__getitem__);tokens.append(t);delta=InputHistory([dict(kind='text',tokens=[t])]);schedule.append(event(history,delta,chunk));history=history+delta;frontier=oracle.forward_inputs(history,schedule);(a.output/(name+'-'+str(i+1)+'.f32')).write_bytes(raw(frontier));progress(branch=name,generated=i+1)
     if t in profile['execution']['eos_tokens']:break
    fixture['branches'][name]=dict(suffix=suffix,tokens=tokens)
   assert fixture['branches']['A']['suffix']!=fixture['branches']['B']['suffix'];write(a.output/'fixtures.json',fixture);progress(status='PASS',root_cells=len(root),image_rows=embedding.rows,fresh_forwards=oracle.requests,fresh_forward_tokens=oracle.tokens_computed,seconds=time.monotonic()-started)
  finally:oracle.close()
  return
 fixture=json.loads((a.output/'fixtures.json').read_text());backend=Qwen35MixedBackend(profile);model=BranchingModel(a.output/'store',backend,profile);comparisons=0;receipts=[];pool=SharedCheckpointPool(32*1024**2);disk=DiskCheckpointPool(model.store,64*1024**2);artifacts=ImageArtifacts(model.store,pool,disk,encoder_profile_id=a.encoder_id,expected_width=4096,max_rows=1024,original_bytes_limit=16*1024**2)
 def compare(sid,path):
  nonlocal comparisons
  actual=raw(model.next_logits(sid));wanted=path.read_bytes()
  if actual!=wanted:
   x=struct.unpack('<'+str(len(actual)//4)+'f',actual);y=struct.unpack('<'+str(len(wanted)//4)+'f',wanted);raise AssertionError(('Mixed frontier mismatch',path.name,max(abs(a-b) for a,b in zip(x,y))))
  comparisons+=1
 try:
  if a.phase=='prepare':
   # This is the recorded output of the separately executed real GPU encoder,
   # reused as an immutable fixture; no encoder performance is claimed here.
   def recorded_encoder(raw_image):
    assert raw_image==original;return payload
   with artifacts.prepare(original=original,normalized=original,encode=recorded_encoder,pad_token_id=next(s['pad_token_id'] for s in fixture['root_spans'] if s['kind']=='image')) as prepared:
    assert prepared.span==next(s for s in fixture['root_spans'] if s['kind']=='image')
    root=model.create_inputs(fixture['root_spans'],artifacts=artifacts)
   compare(root,a.output/'prefix.f32');model.store.fork('prefix',root);model.checkpoint(root);points=dict(root=root,generated={},controllers={})
   for name,row in fixture['branches'].items():
    sid=model.feed(root,tuple(row['suffix']));compare(sid,a.output/(name+'-0.f32'));sid=model.start_generation(sid,{'temperature':0},'mixed-'+name)
    for i,t in enumerate(row['tokens'][:4]):
     g=model.generate(sid,1);sid=g.snapshot_id;assert g.accepted_tokens==(t,);compare(sid,a.output/(name+'-'+str(i+1)+'.f32'))
    model.checkpoint(sid);points['generated'][name]=sid;points['controllers'][name]=model.controller(sid)
   write(a.output/'points.json',points)
   # Same number of cells and same frontier are insufficient if the native
   # image grid changes. Failed import must poison this lease before reuse.
   history=model.history(root);spans=history.spans
   for span in spans:
    if span['kind']=='image':span['grid_width'],span['grid_height']=span['grid_height'],span['grid_width']
   wrong=InputHistory(spans);saved=model.cache[root]
   with backend.lease() as lease:
    before=backend.work_counters()['forward_tokens']
    try:backend.restore(lease,saved.blob,history=wrong,frontier=saved.frontier)
    except IntegrityError:pass
    else:raise AssertionError('Transposed image grid was accepted')
    assert backend.work_counters()['forward_tokens']==before
    try:backend.logits(lease)
    except IntegrityError:pass
    else:raise AssertionError('Failed mixed import left usable state')
  else:
   points=json.loads((a.output/'points.json').read_text());root=points['root'];receipts.append(model.restore_checkpoint(root));compare(root,a.output/'prefix.f32');hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in model.checkpoint_directory.glob('*.state')}
   for name,row in fixture['branches'].items():
    sid=model.feed(root,tuple(row['suffix']));compare(sid,a.output/(name+'-0.f32'));sid=model.start_generation(sid,{'temperature':0},'mixed-'+name)
    for i,t in enumerate(row['tokens']):
     g=model.generate(sid,1);sid=g.snapshot_id;assert g.accepted_tokens==(t,);compare(sid,a.output/(name+'-'+str(i+1)+'.f32'));progress(branch=name,generated=i+1,comparisons=comparisons)
    sid=points['generated'][name];receipts.append(model.restore_checkpoint(sid));assert model.controller(sid)==points['controllers'][name]
    for i in range(min(4,len(row['tokens'])),len(row['tokens'])):
     g=model.generate(sid,1);sid=g.snapshot_id;assert g.accepted_tokens==(row['tokens'][i],);compare(sid,a.output/(name+'-'+str(i+1)+'.f32'))
   assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
  progress(status='PASS',comparisons=comparisons,max_logit_error=0,root_cells=len(model.history(root)),restore_receipts=receipts,native_work=backend.work_counters(),seconds=time.monotonic()-started,model_contract_id=profile['model_contract_id'],execution_profile_id=profile['execution_profile_id'])
 finally:backend.close();pool.close()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if '--output' in sys.argv:
   p=Path(sys.argv[sys.argv.index('--output')+1])/(sys.argv[1]+'-result.json');v=json.loads(p.read_text()) if p.exists() else {};v.update(status='FAIL',error=repr(e));write(p,v)
  traceback.print_exc();raise
