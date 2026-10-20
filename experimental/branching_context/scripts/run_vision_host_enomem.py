"""Real encoder host ENOMEM under child-only RLIMIT_AS; parent Q4 worker unrestricted.

This is bounded address-space allocation failure, not physical host-RAM OOM.
strace proves successful exec of the pinned encoder and kernel ENOMEM. No model
or image bytes, return values, allocator errors or numerical tolerances are mocked.
"""
import argparse,hashlib,json,math,os,re,shutil,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

class ProbeError(RuntimeError):pass

def validate(row):
 expected=dict(uid=1000,limit_bytes=134217728,encoder_exec_observed=True,cli_exit=2,state='failed',durable=False,
  new_branch_absent=True,parent_unchanged=True,parent_payload_unchanged=True,native_forward=0,ram_reserved=0,disk_reserved=0,pins=0,
  staging_empty=True,child_group_absent=True,unbounded_retry_encoder_calls=1,unbounded_retry_saved=True,worker_unchanged=True,
  tokens_exact=True,controller_exact=True,frontier_exact=True,control_prefix_forward=0)
 for k,v in expected.items():
  if k not in row or type(row[k]) is not type(v) or row[k]!=v:raise ProbeError('missing or failed witness: '+k)
 if type(row.get('ENOMEM_syscalls')) is not int or row['ENOMEM_syscalls']<1:raise ProbeError('no kernel ENOMEM witness')
 if type(row.get('child_exit')) is not int or row['child_exit']==0:raise ProbeError('bounded child did not fail')
 if type(row.get('max_abs_error')) not in (int,float) or row['max_abs_error']!=0:raise ProbeError('missing/nonzero numerical error')

def trace_witness(trace,encoder):
 executed=[];failures=[]
 for line in trace.splitlines():
  if 'execve('+json.dumps(encoder) in line and re.search(r'= 0\s*$',line):executed.append(line)
  if re.search(r'\b(?:mmap|mremap|brk)\(.*= -1 ENOMEM\b',line):failures.append(line)
 return dict(encoder_exec_observed=bool(executed),encoder_exec_lines=executed,ENOMEM_syscalls=len(failures),ENOMEM_lines=failures)

def install_hook(body,root,encoder):
 marker="importlib.import_module('serve.cow_http').main()\n"
 if body.count(marker)!=1:raise ProbeError('ambiguous server entrypoint')
 hook=r"""import os,time,struct
from context_time_travel.coordinator import ContextCoordinator
import serve.cow_vision_encoder as _ENC
_ROOT=Path(ROOT_PATH);_ENCODER=ENCODER_PATH
_original_popen=_ENC.subprocess.Popen
def _popen(argv,*args,**kwargs):
 if isinstance(argv,(tuple,list)) and str(argv[0])==_ENCODER and (_ROOT/'limit-next-encoder').exists():
  (_ROOT/'limit-next-encoder').unlink()
  trace=_ROOT/'encoder-host.trace'
  command=['/usr/bin/prlimit','--as=134217728:134217728','--','/usr/bin/strace','-f','-qq','-e',
   'trace=execve,mmap,mremap,brk,munmap','-o',str(trace),'--',*argv]
  process=_original_popen(command,*args,**kwargs)
  witness=dict(argv=command,uid=os.getuid(),pid=process.pid,limit_bytes=134217728,started_monotonic=time.monotonic())
  (_ROOT/'limited-launch.json').write_text(json.dumps(witness,sort_keys=True)+'\n')
  communicate=process.communicate
  def _communicate(*a,**kw):
   result=communicate(*a,**kw)
   for field in ['stdout','stderr']:
    path=Path(kwargs[field].name)
    (_ROOT/('encoder-'+field)).write_bytes(path.read_bytes())
   witness.update(exit=process.returncode,ended_monotonic=time.monotonic())
   (_ROOT/'limited-launch.json').write_text(json.dumps(witness,sort_keys=True)+'\n')
   return result
  process.communicate=_communicate
  return process
 return _original_popen(argv,*args,**kwargs)
_ENC.subprocess.Popen=_popen
_original_run=ContextCoordinator._run
def _run(self,model,job):
 before=model.backend.work_counters();row=dict(operation=job.request.operation)
 try:
  result=_original_run(self,model,job);row['outcome']='complete';row['snapshot']=result['snapshot']
  with model.cache._lock:cp=model.cache._entries[result['snapshot']]
  raw=struct.pack('<%df'%len(cp.frontier),*cp.frontier);name=hashlib.sha256(job.request.operation.encode()).hexdigest()+'.f32'
  (_ROOT/name).write_bytes(raw);row.update(frontier_file=name,frontier_sha256=hashlib.sha256(raw).hexdigest(),logits=len(cp.frontier))
  return result
 except BaseException as exc:row.update(outcome=type(exc).__name__,message=str(exc));raise
 finally:
  after=model.backend.work_counters();row.update(before=before,after=after,forward_tokens=after['forward_tokens']-before['forward_tokens'])
  with (_ROOT/'jobs.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
ContextCoordinator._run=_run
"""
 return body.replace(marker,hook.replace('ROOT_PATH',repr(str(root))).replace('ENCODER_PATH',repr(str(encoder)))+'\n'+marker)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ('runtime','branchwrite','profile','dependency-root','vision-config','fixtures','output'):parser.add_argument('--'+n,type=Path,required=True)
 a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False);encoder=Path(json.loads(a.vision_config.read_bytes())['executable']).resolve()
 class Owner(image_probe_type(GrammarProbe,a.vision_config)):
  def _bootstrap(self,registry):
   path=super()._bootstrap(registry);path.write_text(install_hook(path.read_text(),out,encoder),newline='\n');return path
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,
  dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
 probe.record.update(status='RUNNING',scope=__doc__,numerical_oracle='UNINTERRUPTED_MATCHING_PROFILE_CONTROL_NOT_FRESH_HISTORY',numerical_max_error=None,
  q8='DEFERRED_BY_OWNER',chat_server='OFF',harness_sha256=sha(Path(__file__)),tool_fingerprints={n:sha(Path(n)) for n in ['/usr/bin/prlimit','/usr/bin/strace']})
 error=None
 def idle(client):
  deadline=time.monotonic()+30
  while time.monotonic()<deadline:
   s=client.status().capacity
   if not s['queued'] and not s['pending_branches'] and not any(s['active_per_gpu'].values()):return s
   time.sleep(.05)
  raise TimeoutError('worker did not settle')
 def job(operation):
  values=[json.loads(s) for s in (out/'jobs.jsonl').read_text().splitlines()]
  matches=[r for r in values if r['operation']==operation]
  if len(matches)!=1:raise ProbeError('missing/duplicate native job witness')
  return matches[0]
 def cli(args,label,expected=0):
  command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','180','--state-dir',str(out/'cli-state'),'--machine-result',*args]
  r=subprocess.run(command,env={**os.environ,'COW_API_KEY':probe.secret},stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
  (out/(label+'.stdout')).write_bytes(r.stdout);(out/(label+'.stderr')).write_bytes(r.stderr)
  if r.returncode!=expected:raise ProbeError('CLI exit '+label+': '+str(r.returncode))
  return r,json.loads(r.stdout)
 def control(name,operation):
  head=client.fork('kept',parent.tip,name,operation+'-fork')
  _,value=cli(['answer','--branch',name,'--revision',str(head.revision),'--operation-id',operation,'--max-tokens','1024',
   '--temperature','0','--seed','11','--grammar-state','VC08_REPLY'],name)
  obj=SimpleNamespace(answer=value['answer'],receipt=SimpleNamespace(snapshot=value['receipt']['snapshot']))
  if completed_thinking_text(obj,profile['execution']['eos_tokens'])!='OK':raise ProbeError('control final not OK')
  j=job(operation);return value,j,(out/j['frontier_file']).read_bytes()
 try:
  probe._imports();profile=json.loads(a.profile.read_bytes());probe.runtime_closure(profile,runtime)
  assert os.getuid()==1000 and profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  probe.record.update(execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],profile_sha256=sha(a.profile),native_sha256=profile['native_build']['sha256'])
  registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(out,'VC08-host-ENOMEM').start();probe.start_service(registry)
  url=probe.ready_url(probe.service.ready);pin=hashlib.sha256(str(out).encode()).hexdigest()
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=out/'cli-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
  ns=client.create_namespace('Host allocation failure','vc08-host-namespace')['namespace_id']
  first=(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes()
  uploaded=client.upload_image(ns,first);root=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-host-kept',images=(uploaded,))
  parent=probe.store.head('kept');snapshot=probe.store.get_snapshot(parent.tip)
  payload=out/'store/checkpoints'/(root.receipt.raw['payload_sha256']+'.state');parent_sha=sha(payload)
  state=idle(client);native_pid=state['workers']['0']['pid'];probe.tracker.native_started(native_pid)
  baseline,baseline_job,baseline_frontier=control('reference','vc08-host-reference')
  baseline_snapshot=probe.store.get_snapshot(baseline['receipt']['snapshot']);idle(client)
  picture=a.fixtures/'branchwrite-h05-cli-gpu-v1/before/capture.png'
  if sha(picture)==hashlib.sha256(first).hexdigest():raise ProbeError('fault image must require a distinct encoding')
  probe.phase('real-child-address-space-enomem');(out/'limit-next-encoder').write_text('actual child RLIMIT_AS128MiB only\n')
  response,value=cli(['prepare','--branch','fault','--namespace-id',ns,'--operation-id','vc08-host-fault',
   '--text','Keep the new image context.','--image',str(picture)],'fault',expected=2)
  state=idle(client);operation=client.operation('vc08-host-fault');failure=job('vc08-host-fault')
  trace=(out/'encoder-host.trace').read_text();witness=trace_witness(trace,str(encoder));launch=json.loads((out/'limited-launch.json').read_bytes())
  try:os.killpg(launch['pid'],0);group_absent=False
  except ProcessLookupError:group_absent=True
  row=dict(witness,uid=launch['uid'],limit_bytes=launch['limit_bytes'],child_exit=launch['exit'],cli_exit=response.returncode,
   state=operation.state,durable=operation.durable,new_branch_absent='fault' not in [h.name for h in probe.store.branches()],
   parent_unchanged=probe.store.head('kept')==parent and probe.store.get_snapshot(parent.tip)==snapshot,parent_payload_unchanged=sha(payload)==parent_sha,
   native_forward=failure['forward_tokens'],ram_reserved=state['ram']['reserved_bytes'],disk_reserved=state['disk']['reserved_bytes'],pins=state['ram']['pins'],
   staging_empty=not any((out/'store/image-staging').iterdir()),child_group_absent=group_absent)
  if 'error' not in value or 'receipt' in value:raise ProbeError('failure incorrectly acknowledged durable success')
  probe.record.update(host_fault=row,launch=launch,failure_operation=dict(operation.raw),failure_model_job=failure,metrics_after_fault=state);probe.publish()
  probe.phase('ordinary-encoder-retry-with-new-operation')
  _,success=cli(['prepare','--branch','fault','--namespace-id',ns,'--operation-id','vc08-host-recovered',
   '--text','Keep the new image context.','--image',str(picture)],'recovered')
  encodings=success['answer']['input_preparation']['encodings']
  row.update(unbounded_retry_encoder_calls=len(encodings),unbounded_retry_saved=client.operation('vc08-host-recovered').durable)
  after,after_job,after_frontier=control('after','vc08-host-control')
  state=idle(client)
  v=struct.unpack('<%df'%(len(baseline_frontier)//4),baseline_frontier);w=struct.unpack('<%df'%(len(after_frontier)//4),after_frontier)
  if len(v)!=248320 or len(w)!=len(v) or not all(map(math.isfinite,v+w)):raise ProbeError('incomplete/nonfinite control vectors')
  row.update(worker_unchanged=state['workers']['0']['pid']==native_pid and state['workers']['0']['restarts']==0,
   tokens_exact=after['answer']['tokens']==baseline['answer']['tokens'],controller_exact=probe.store.get_snapshot(after['receipt']['snapshot'])==baseline_snapshot,
   frontier_exact=after_frontier==baseline_frontier,max_abs_error=max(abs(x-y) for x,y in zip(v,w)),
   control_prefix_forward=after_job['forward_tokens']-len(after['answer']['tokens']))
  row['parent_unchanged']=row['parent_unchanged'] and probe.store.head('kept')==parent and probe.store.get_snapshot(parent.tip)==snapshot
  row['parent_payload_unchanged']=row['parent_payload_unchanged'] and sha(payload)==parent_sha
  probe.record.update(host_fault=row,baseline_answer=baseline,baseline_job=baseline_job,control_answer=after,control_job=after_job,
   recovered=success,metrics_after_recovery=state,numerical_max_error=row['max_abs_error'])
  validate(row);probe.record['status']='VC08_BOUNDED_Q4_ENCODER_HOST_ENOMEM_PASS'
 except BaseException as exc:
  error=exc;probe.record['status']='VC08_BOUNDED_Q4_ENCODER_HOST_ENOMEM_FAIL';probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if probe.service is not None:
   try:probe.stop_service()
   except BaseException as exc:error=error or exc;probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  if error:probe.record['status']='VC08_BOUNDED_Q4_ENCODER_HOST_ENOMEM_FAIL'
  probe.publish()
  if probe.tracker:probe.tracker.finish(probe.record['status']);probe.tracker.close()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
