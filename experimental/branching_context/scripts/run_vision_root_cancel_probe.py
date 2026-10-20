"""VC08 real CLI disconnect during GPU-allocated initial image preparation.

No fault mocks, injected sleeps, alternate history or relaxed model oracle.
A read-only preparer wrapper records actual native work counters on its owner
thread. It changes no request, result, controller, timing gate or model input.
"""
import argparse
from contextlib import contextmanager
import hashlib,json,os
from pathlib import Path
import signal,subprocess,sys,time
from types import SimpleNamespace


def sha(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('runtime','branchwrite','profile','dependency-root','vision-config','output','fixtures'):
  parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args();runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=[str(x) for x in (bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root)]
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False)
 trace=out/'input-preparation.jsonl'
 class Owner(image_probe_type(GrammarProbe,a.vision_config)):
  def _bootstrap(self,registry):
   path=super()._bootstrap(registry);body=path.read_text()
   marker="importlib.import_module('serve.cow_http').main()\n"
   hook="""from contextlib import contextmanager
import threading,time,os
from serve.cow_vision_preparer import CowVisionPreparer
_real_prepare=CowVisionPreparer.prepare
@contextmanager
def measured_prepare(self,model,*args,**kwargs):
 before=model.backend.work_counters()
 row=dict(thread=threading.get_ident(),server_pid=os.getpid(),phase='input-preparation',outcome='entered')
 try:
  with _real_prepare(self,model,*args,**kwargs) as prepared:yield prepared
 except BaseException as exc:
  row.update(outcome=type(exc).__name__)
  raise
 else:row['outcome']='complete'
 finally:
  after=model.backend.work_counters()
  row['counter_deltas']={k:after[k]-v for k,v in before.items() if type(v) in (int,float) and type(after.get(k)) in (int,float)}
  with Path(TRACE_PATH).open('a') as stream:
   stream.write(json.dumps(row,sort_keys=True)+'\\n');stream.flush();os.fsync(stream.fileno())
CowVisionPreparer.prepare=measured_prepare
""".replace('TRACE_PATH',repr(str(trace)))
   assert body.count(marker)==1
   path.write_text(body.replace(marker,hook+marker),newline='\n');return path
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,
  dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
 probe.record.update(status='RUNNING',scope=__doc__,cases=[],numerical_max_error=None,numerical_oracle='NOT_RUN',
  chat_server='OFF',q8='DEFERRED_BY_OWNER',diagnostic_hook='Actual native counters around real preparer on owner thread; no input/result substitutions')
 error=None;cli=None
 try:
  probe._imports();profile=json.loads(a.profile.read_bytes());probe.runtime_closure(profile,runtime)
  assert profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  probe.record.update(execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
   profile_sha256=file_sha(a.profile),harness_sha256=file_sha(Path(__file__)),native_sha256=profile['native_build']['sha256'])
  registry=out/'registry.json'
  _atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,
   final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(out,'VC08-root-disconnect').start();probe.start_service(registry)
  url=probe.ready_url(probe.service.ready);pin=sha((str(out)+probe.record['profile_sha256']).encode())
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=out/'cli-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
  ns=client.create_namespace('Cancellation boundary','vc08-namespace')['namespace_id']
  image=(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes()
  upload=client.upload_image(ns,image)
  root=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-kept',images=(upload,))
  old=probe.store.head('kept');snapshot=probe.store.get_snapshot(old.tip)
  payload=out/'store/checkpoints'/(root.receipt.raw['payload_sha256']+'.state');payload_sha=file_sha(payload)
  def idle():
   deadline=time.monotonic()+30
   while time.monotonic()<deadline:
    metrics=client.status().capacity
    if not metrics['queued'] and not metrics['pending_branches'] and not any(metrics['active_per_gpu'].values()):return metrics
    time.sleep(.05)
   raise TimeoutError('worker did not become idle')
  metrics=idle();server_pid=probe.service.process.pid
  for worker in metrics['workers'].values():probe.tracker.native_started(worker['pid'])
  encoder=Path(json.loads(a.vision_config.read_bytes())['executable']).resolve()
  def owned_encoders():
   if probe.service.process.poll() is not None:raise RuntimeError('server exited during fault injection')
   pids=set()
   for task in Path('/proc/'+str(server_pid)+'/task').iterdir():
    try:children=(task/'children').read_text().split()
    except FileNotFoundError:continue
    for pid in children:
     try:
      if Path('/proc/'+pid+'/exe').resolve()==encoder:pids.add(int(pid))
     except FileNotFoundError:pass
   return pids
  for index,(command,relative) in enumerate((('context','branchwrite-h05-cli-gpu-v1/before/capture.png'),('prepare','branchwrite-h05-cli-gpu-v1/after/capture.png'))):
   probe.phase(command+'-start-cli')
   before_heads=probe.store.branches();start_rows=len(trace.read_text().splitlines())
   picture=a.fixtures/relative;operation='vc08-cancel-'+command;branch='cancel-'+command
   argv=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','120','--state-dir',str(out/'interrupted-cli-state'),
    '--machine-result',command,'--branch',branch,'--namespace-id',ns,'--operation-id',operation,
    '--text','Preserve this screenshot context.','--image',str(picture)]
   env={**os.environ,'COW_API_KEY':probe.secret}
   stdout=out/(command+'.stdout');stderr=out/(command+'.stderr')
   with stdout.open('wb') as o,stderr.open('wb') as e:
    cli=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True)
   row=dict(command=command,operation_id=operation,branch=branch,cli_pid=cli.pid,picture_sha256=file_sha(picture),argv=argv)
   probe.record['cases'].append(row);probe.publish()
   deadline=time.monotonic()+60;observed=None
   while time.monotonic()<deadline:
    if cli.poll() is not None:raise AssertionError('CLI finished before GPU cancellation injection')
    owned=owned_encoders()
    if owned:
     samples=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
     for sample in samples:
      pid,mib=[v.strip() for v in sample.split(',')]
      if int(pid) in owned and float(mib)>0:observed=dict(pid=int(pid),gpu_memory_mib=float(mib),sample=sample)
    if observed:break
    time.sleep(.01)
   if observed is None:raise TimeoutError('did not observe an owned GPU-allocated encoder')
   row['gpu_observation']=observed;probe.tracker.native_started(observed['pid']);probe.phase(command+'-disconnect-cli')
   os.killpg(cli.pid,signal.SIGTERM);code=cli.wait(10);row['cli_exit']=code;assert code==-signal.SIGTERM
   cli=None
   after=idle();observation=client.operation(operation)
   row.update(operation=dict(observation.raw),metrics_after=after,encoder_absent=not Path('/proc/'+str(observed['pid'])).exists(),
    branch_absent=branch not in [h.name for h in probe.store.branches()],parent_heads_unchanged=probe.store.branches()==before_heads,
    payload_unchanged=file_sha(payload)==payload_sha,controller_unchanged=probe.store.get_snapshot(old.tip)==snapshot,
    preparation=json.loads(trace.read_text().splitlines()[start_rows]))
   assert observation.state=='cancelled' and not observation.durable and observation.receipt is None
   assert row['encoder_absent'] and row['branch_absent'] and row['parent_heads_unchanged'] and row['payload_unchanged'] and row['controller_unchanged']
   assert after['ram']['reserved_bytes']==after['disk']['reserved_bytes']==0
   assert row['preparation']['outcome']=='CancelledError'
   assert row['preparation']['counter_deltas']['forward_tokens']==0
   assert row['preparation']['counter_deltas']['prefill_tokens']==0
   assert not list((out/'store/image-staging').iterdir())
   probe.publish()
  probe.phase('retained-root-remains-usable')
  head=client.fork('kept',old.tip,'control','vc08-control-fork')
  answer=client.answer('control',head.revision,'vc08-control-answer',1024,settings=dict(temperature=0.),grammar_state='VC08_REPLY')
  final=completed_thinking_text(answer,profile['execution']['eos_tokens']);assert final=='OK'
  assert answer.answer['counter_deltas']['forward_tokens']==len(answer.answer['tokens'])
  assert answer.answer['counter_deltas']['prefill_tokens']==0
  assert probe.store.head('kept')==old and file_sha(payload)==payload_sha
  probe.record.update(control=dict(answer=dict(answer.answer),receipt=dict(answer.receipt.raw)),
   retained_root=dict(receipt=dict(root.receipt.raw),payload_sha256=payload_sha),status='VC08_ROOT_DISCONNECT_GPU_PASS')
 except BaseException as exc:
  error=exc;probe.record['status']='VC08_ROOT_DISCONNECT_FAIL';probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if cli is not None and cli.poll() is None:
   os.killpg(cli.pid,signal.SIGTERM)
   try:cli.wait(10)
   except subprocess.TimeoutExpired:os.killpg(cli.pid,signal.SIGKILL);cli.wait(5)
  if probe.service is not None:
   try:probe.stop_service()
   except BaseException as exc:
    error=error or exc;probe.record['status']='CLEANUP_FAIL';probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  probe.publish()
  if probe.tracker:
   probe.tracker.finish(probe.record['status'],None if error is None else type(error).__name__);probe.tracker.close()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
