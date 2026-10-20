"""VC08 real CLI disconnect at consumed image/token boundaries and writer handoff.

Deterministic observer gates pause after real work; only actual HTTP disconnect
sets the cancellation flag. Writer handoff may finish after disconnect, so the
original operation must reconcile without repeated inference. No fake counters,
mutated inputs, alternative history or relaxed numerical comparison is used.
"""
import argparse,hashlib,json,math,os,signal,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

class ProbeError(RuntimeError):pass

def head_identity(head):
 values=tuple(getattr(head,k) for k in ('name','revision','tip','root'))
 if not isinstance(values[0],str) or type(values[1]) is not int or values[1]<0 or any(not isinstance(v,str) or len(v)!=64 for v in values[2:]):
  raise ProbeError('invalid branch head fields')
 return values

def _need(row,key,value):
 if key not in row or type(row[key]) is not type(value) or row[key]!=value:raise ProbeError('missing or failed witness: '+key)

def _exact(row):
 for k in ['parent_unchanged','parent_payload_unchanged','tokens_exact','controller_exact','frontier_exact']:_need(row,k,True)
 if type(row.get('max_abs_error')) not in (int,float) or row['max_abs_error']!=0:raise ProbeError('missing/nonzero frontier error')
 for k in ['reservations','pins']:_need(row,k,0)
 _need(row,'cli_exit',-15);_need(row,'disconnect_observed',True)

def validate_cancel(row):
 _exact(row)
 for k,v in dict(state='cancelled',durable=False,branch_unchanged=True,queued=0,worker_healthy=True,control_prefix_forward=0).items():_need(row,k,v)
 if row.get('stage')=='four':
  for k,v in dict(forward_tokens=4,prefill_tokens=0,progress_tokens=4).items():_need(row,k,v)
 elif row.get('stage')=='root':
  n=row.get('forward_tokens')
  if type(n) is not int or n<=0:raise ProbeError('root did not consume real prefill')
  _need(row,'prefill_tokens',n);_need(row,'progress_tokens',0)
 else:raise ProbeError('unknown consumed cancellation stage')

def validate_publication(row):
 _exact(row)
 for k,v in dict(state='durable',durable=True,revision_delta=1,receipt_exact=True,retry_native_unchanged=True).items():_need(row,k,v)

def install_hook(body,root):
 marker="importlib.import_module('serve.cow_http').main()\n"
 if body.count(marker)!=1:raise ProbeError('ambiguous real server entrypoint')
 hook=r"""import os,time,threading,struct
from context_time_travel.coordinator import ContextCoordinator,Request
from context_time_travel.real import BranchingModel
_ROOT=Path(ROOT_PATH);_LOCAL=threading.local();_COUNT={}
def _json(path,row):
 temporary=path.with_suffix('.tmp')
 with temporary.open('w') as f:json.dump(row,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,path)
def _gate(stage,model,job,*,snapshot=None,prepared=None):
 directory=_ROOT/stage;directory.mkdir(mode=0o700,exist_ok=True)
 row=dict(stage=stage,operation=job.request.operation,branch=job.branch,server_pid=os.getpid(),
  snapshot=snapshot,progress_tokens=_COUNT.get(job.request.operation,0),snapshot_future_done=job.request.snapshot.done(),
  durable_future_done=job.request.durable.done(),monotonic=time.monotonic())
 if prepared is None:
  after=model.backend.work_counters();before=_LOCAL.before
  row.update(forward_tokens=after['forward_tokens']-before['forward_tokens'],prefill_tokens=after['prefill_tokens']-before['prefill_tokens'])
 else:row.update(consumed=prepared.checkpoint.consumed,prepared_snapshot=prepared.snapshot)
 _json(directory/'gate.json',row)
 observed=False;deadline=time.monotonic()+60
 while not (directory/'release.json').exists():
  if job.cancelled is not None and job.cancelled() and not observed:
   _json(directory/'disconnect.json',dict(actual_cancelled_callback=True,monotonic=time.monotonic()));observed=True
  if time.monotonic()>deadline:raise TimeoutError('real disconnect/release not observed at '+stage)
  time.sleep(.02)
_original_create=BranchingModel.create_with_reuse
def _create(self,*args,**kwargs):
 result=_original_create(self,*args,**kwargs)
 job=getattr(_LOCAL,'job',None)
 if job is not None and job.request.operation=='vc08-deep-root':
  if result[1]:raise RuntimeError('root cancellation fixture unexpectedly reused computation')
  _gate('root',self,job,snapshot=result[0])
 return result
BranchingModel.create_with_reuse=_create
_original_emit=Request.emit
def _emit(self,event,**fields):
 result=_original_emit(self,event,**fields)
 if self.operation=='vc08-deep-four' and event=='token':
  _COUNT[self.operation]=_COUNT.get(self.operation,0)+1
  if _COUNT[self.operation]==4:_gate('four',_LOCAL.model,_LOCAL.job)
 return result
Request.emit=_emit
_original_run=ContextCoordinator._run
def _run(self,model,job):
 _LOCAL.model=model;_LOCAL.job=job;_LOCAL.before=model.backend.work_counters()
 row=dict(operation=job.request.operation,before=_LOCAL.before)
 try:
  result=_original_run(self,model,job);row.update(outcome='complete',snapshot=result['snapshot'])
  with model.cache._lock:cp=model.cache._entries[result['snapshot']]
  raw=struct.pack('<%df'%len(cp.frontier),*cp.frontier);name=hashlib.sha256(job.request.operation.encode()).hexdigest()+'.f32'
  (_ROOT/name).write_bytes(raw);row.update(frontier_file=name,frontier_sha256=hashlib.sha256(raw).hexdigest(),logits=len(cp.frontier))
  return result
 except BaseException as exc:row.update(outcome=type(exc).__name__,message=str(exc));raise
 finally:
  after=model.backend.work_counters();row.update(after=after,forward_tokens=after['forward_tokens']-row['before']['forward_tokens'],prefill_tokens=after['prefill_tokens']-row['before']['prefill_tokens'])
  with (_ROOT/'jobs.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  _LOCAL.job=None;_LOCAL.model=None
ContextCoordinator._run=_run
_original_publish=ContextCoordinator._publish
def _publish(self,model,job,prepared,*args,**kwargs):
 if job.request.operation=='vc08-deep-publish':_gate('publication',model,job,prepared=prepared)
 return _original_publish(self,model,job,prepared,*args,**kwargs)
ContextCoordinator._publish=_publish
"""
 return body.replace(marker,hook.replace('ROOT_PATH',repr(str(root)))+'\n'+marker)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ['runtime','branchwrite','profile','dependency-root','vision-config','fixtures','output']:parser.add_argument('--'+n,type=Path,required=True)
 a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False)
 class Owner(image_probe_type(GrammarProbe,a.vision_config)):
  def _bootstrap(self,registry):
   path=super()._bootstrap(registry);path.write_text(install_hook(path.read_text(),out),newline='\n');return path
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
 probe.record.update(status='RUNNING',scope=__doc__,cases=[],numerical_oracle='BEFORE_AFTER_MATCHING_PROFILE_CONTROL_NOT_FRESH_FULL_HISTORY',
  numerical_max_error=None,chat_server='OFF',q8='DEFERRED_BY_OWNER',harness_sha256=sha(Path(__file__)))
 error=None;pending=None
 def idle():
  deadline=time.monotonic()+30
  while time.monotonic()<deadline:
   state=client.status().capacity
   if not state['queued'] and not state['pending_branches'] and not any(state['active_per_gpu'].values()):return state
   time.sleep(.05)
  raise TimeoutError('worker/writer did not settle')
 def job(operation):
  matches=[json.loads(s) for s in (out/'jobs.jsonl').read_text().splitlines() if json.loads(s)['operation']==operation]
  if len(matches)!=1:raise ProbeError('missing/duplicate actual work witness: '+operation)
  return matches[0]
 def cli(args,label,asynchronous=False):
  command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','180','--state-dir',str(out/'cli-state'),'--machine-result',*args]
  with (out/(label+'.stdout')).open('wb') as so,(out/(label+'.stderr')).open('wb') as se:
   process=subprocess.Popen(command,env={**os.environ,'COW_API_KEY':probe.secret},stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
  if asynchronous:return process
  if process.wait(240)!=0:raise ProbeError('CLI failed '+label)
  return json.loads((out/(label+'.stdout')).read_bytes())
 def answer_args(branch,revision,operation):return ['answer','--branch',branch,'--revision',str(revision),'--operation-id',operation,'--max-tokens','1024','--temperature','0','--seed','11','--grammar-state','VC08_REPLY']
 def control(name):
  fork=client.fork('kept',parent.tip,name,'vc08-'+name+'-fork');value=cli(answer_args(name,fork.revision,'vc08-'+name),name)
  obj=SimpleNamespace(answer=value['answer'],receipt=SimpleNamespace(snapshot=value['receipt']['snapshot']))
  if completed_thinking_text(obj,profile['execution']['eos_tokens'])!='OK':raise ProbeError('control final not OK')
  measurement=job('vc08-'+name);return value,measurement,(out/measurement['frontier_file']).read_bytes()
 def exact(value,measurement,frontier):
  actual=struct.unpack('<%df'%(len(frontier)//4),frontier);expected=struct.unpack('<%df'%(len(reference_frontier)//4),reference_frontier)
  if len(actual)!=len(expected) or len(actual)!=248320 or not all(map(math.isfinite,actual+expected)):raise ProbeError('incomplete/nonfinite full control frontier')
  return dict(parent_unchanged=probe.store.head('kept')==parent and probe.store.get_snapshot(parent.tip)==parent_snapshot,
   parent_payload_unchanged=sha(payload)==parent_sha,tokens_exact=value['answer']['tokens']==reference['answer']['tokens'],
   controller_exact=probe.store.get_snapshot(value['receipt']['snapshot'])==reference_snapshot,
   frontier_exact=frontier==reference_frontier,max_abs_error=max(abs(x-y) for x,y in zip(actual,expected)),
   control_prefix_forward=measurement['forward_tokens']-len(value['answer']['tokens']))
 def gate(stage):
  directory=out/stage;deadline=time.monotonic()+60
  while time.monotonic()<deadline:
   if (directory/'gate.json').exists():return json.loads((directory/'gate.json').read_bytes())
   if pending.poll() is not None:raise ProbeError('CLI ended before intended '+stage+' gate')
   time.sleep(.02)
  raise TimeoutError('no '+stage+' gate')
 def disconnect(stage):
  os.killpg(pending.pid,signal.SIGTERM);code=pending.wait(10);directory=out/stage;deadline=time.monotonic()+30
  while time.monotonic()<deadline:
   if (directory/'disconnect.json').exists():
    observed=json.loads((directory/'disconnect.json').read_bytes());_atomic_json(directory/'release.json',dict(reason='real disconnect observed'));return code,observed
   time.sleep(.02)
  raise TimeoutError('server did not observe real disconnect at '+stage)
 try:
  probe._imports();profile=json.loads(a.profile.read_bytes());probe.runtime_closure(profile,runtime)
  assert profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  probe.record.update(execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],profile_sha256=sha(a.profile),native_sha256=profile['native_build']['sha256'])
  registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(out,'VC08-deep-disconnect').start();probe.start_service(registry)
  url=probe.ready_url(probe.service.ready);client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=out/'cli-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=hashlib.sha256(str(out).encode()).hexdigest())
  ns=client.create_namespace('Consumed cancellation boundaries','vc08-deep-namespace')['namespace_id']
  upload=client.upload_image(ns,(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes())
  prefix=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-deep-kept',images=(upload,))
  parent=probe.store.head('kept');parent_snapshot=probe.store.get_snapshot(parent.tip);payload=out/'store/checkpoints'/(prefix.receipt.raw['payload_sha256']+'.state');parent_sha=sha(payload)
  state=idle();native_pid=state['workers']['0']['pid'];probe.tracker.native_started(native_pid)
  reference,reference_job,reference_frontier=control('reference');reference_snapshot=probe.store.get_snapshot(reference['receipt']['snapshot']);idle()
  probe.record.update(reference=reference,reference_job=reference_job,reference_snapshot=reference_snapshot);probe.publish()
  for stage in ['root','four']:
   operation='vc08-deep-'+stage;branch='cancel-'+stage
   if stage=='root':
    before=None;args=['prepare','--branch',branch,'--namespace-id',ns,'--operation-id',operation,'--text','Keep this separate image context.',
     '--image',str(a.fixtures/'branchwrite-h05-cli-gpu-v1/before/capture.png')]
   else:before=client.fork('kept',parent.tip,branch,operation+'-fork');args=answer_args(branch,before.revision,operation)
   heads=probe.store.branches();probe.phase('disconnect-after-'+stage);pending=cli(args,stage,asynchronous=True)
   witness=gate(stage);code,observed=disconnect(stage);pending=None;state=idle();observation=client.operation(operation);measurement=job(operation)
   row=dict(stage=stage,gate=witness,cli_exit=code,disconnect_observed=observed['actual_cancelled_callback'],state=observation.state,durable=observation.durable,
    branch_unchanged=probe.store.branches()==heads,forward_tokens=measurement['forward_tokens'],prefill_tokens=measurement['prefill_tokens'],progress_tokens=witness['progress_tokens'],
    reservations=state['ram']['reserved_bytes']+state['disk']['reserved_bytes'],pins=state['ram']['pins'],queued=state['queued']+state['pending_branches'],
    worker_healthy=state['workers']['0']['health']=='healthy' and state['workers']['0']['pid']==native_pid,operation=dict(observation.raw),model_job=measurement)
   value,control_job,frontier=control('control-'+stage);row.update(exact(value,control_job,frontier),control=value,control_job=control_job)
   probe.record['cases'].append(row);probe.publish();validate_cancel(row);row['status']='PASS';probe.publish();idle()
  stage='publication';operation='vc08-deep-publish';head=client.fork('kept',parent.tip,'publish',operation+'-fork')
  probe.phase('disconnect-after-publication-handoff');pending=cli(answer_args('publish',head.revision,operation),stage,asynchronous=True)
  witness=gate(stage)
  current_head=head_identity(probe.store.head('publish'));expected_head=head_identity(head)
  probe.record['publication_gate_heads']=dict(actual=current_head,expected=expected_head);probe.publish()
  if current_head!=expected_head:raise ProbeError('branch moved before writer release')
  code,observed=disconnect(stage);pending=None;state=idle();completed=client.operation(operation)
  trace=out/'native-1.jsonl';trace_before=trace.read_bytes();receipts=[]
  for i in range(2):receipts.append(cli(['retry-request','--operation-id',operation],'reconcile-'+str(i)))
  unchanged=trace.read_bytes()==trace_before;measurement=job(operation);value=probe.store.get_snapshot(completed.receipt.snapshot)
  frontier=(out/measurement['frontier_file']).read_bytes();control=receipts[0]
  if receipts[0]['answer']!=receipts[1]['answer']:raise ProbeError('reconciled answers differ')
  row=dict(stage=stage,gate=witness,cli_exit=code,disconnect_observed=observed['actual_cancelled_callback'],state=completed.state,durable=completed.durable,
   revision_delta=probe.store.head('publish').revision-head.revision,receipt_exact=all(x['receipt']==dict(completed.receipt.raw) for x in receipts),
   retry_native_unchanged=unchanged,reservations=state['ram']['reserved_bytes']+state['disk']['reserved_bytes'],pins=state['ram']['pins'],
   original_operation=dict(completed.raw),reconciled=receipts,model_job=measurement)
  row.update(exact(control,measurement,frontier));row['controller_exact']=value==reference_snapshot
  probe.record['cases'].append(row);probe.publish();validate_publication(row);row['status']='PASS'
  probe.record.update(status='VC08_BOUNDED_Q4_DEEP_DISCONNECT_PASS',numerical_max_error=max(c['max_abs_error'] for c in probe.record['cases']));idle()
 except BaseException as exc:
  error=exc;probe.record['status']='VC08_BOUNDED_Q4_DEEP_DISCONNECT_FAIL';probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  for stage in ['root','four','publication']:
   directory=out/stage
   if directory.exists():_atomic_json(directory/'release.json',dict(reason='campaign-finally'))
  if pending is not None and pending.poll() is None:
   os.killpg(pending.pid,signal.SIGTERM)
   try:pending.wait(10)
   except subprocess.TimeoutExpired:os.killpg(pending.pid,signal.SIGKILL);pending.wait(5)
  if probe.service is not None:
   try:probe.stop_service()
   except BaseException as exc:error=error or exc;probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  if error:probe.record['status']='VC08_BOUNDED_Q4_DEEP_DISCONNECT_FAIL'
  probe.publish()
  if probe.tracker:probe.tracker.finish(probe.record['status']);probe.tracker.close()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
