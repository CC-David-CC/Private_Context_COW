"""VC08 real CUDA allocation exhaustion during consumed image continuation.

Test-only observers gate the actual native decode command; no allocator failure,
model state or numerical result is fabricated. Linux process recovery only.
"""
import argparse,hashlib,json,math,os,re,shutil,signal,struct,subprocess,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
class ProbeError(RuntimeError):pass

def image_source_binding(store,snapshot):
 from context_time_travel.input_spans import InputHistory,history_for_snapshot
 history=history_for_snapshot(store,snapshot)
 if not isinstance(history,InputHistory):raise ProbeError('Expected canonical mixed image history')
 images=[x for x in history.spans if x['kind']=='image']
 if not images:raise ProbeError('Image allocation fixture requires an actual image span')
 return dict(snapshot=snapshot,input_identity=history.digest,consumed=len(history),text_tokens=len(history.text_tokens),image_rows=sum(x['rows'] for x in images),image_count=len(images))

def validate_image_gate(row,binding,*,operation,branch):
 if (row.get('operation_id')!=operation or row.get('branch')!=branch or row.get('lease_active') is not True
  or type(row.get('native_pid')) is not int or row['native_pid']<=0
  or re.fullmatch(r'decode [0-9]+',str(row.get('command'))) is None or row.get('input_binding')!=binding):raise ProbeError('Wrong image decode gate identity/lease')
 capture=row.get('fixture_capture',{})
 if (capture.get('native_capture') is not True or capture.get('hot_restore') is not True or type(capture.get('restores_delta')) is not int or capture['restores_delta']!=1 or type(capture.get('captures_delta')) is not int or capture['captures_delta']!=1
  or type(capture.get('forward_delta')) is not int or capture['forward_delta']!=0
  or re.fullmatch('[0-9a-f]{64}',str(capture.get('frontier_before_sha256'))) is None
  or capture.get('frontier_before_sha256')!=capture.get('frontier_after_sha256')):raise ProbeError('Read-only native prefix capture was not proved')
 stats=row.get('stats',{})
 if any(type(stats.get(k)) is not int or stats[k]!=v for k,v in dict(forward_tokens=0,prefill_tokens=0,consumed=binding['consumed']).items()):raise ProbeError('Prefix was recomputed or decoded before the allocation gate')
 pages=row.get('pages')
 if type(pages) is not list or not any(type(x) is list and len(x)==6 and type(x[2]) is int and x[2]>=2 and type(x[3]) is int and x[3]>0 for x in pages):raise ProbeError('No real shared-page ownership witness')

def validate_outcome(row):
 for key in ['oom_verified','helper_released','old_native_absent','replacement_before_followup','branch_unchanged_on_error','parent_unchanged','no_error_ack']:
  if row.get(key) is not True:raise ProbeError('Missing real recovery witness: '+key)
 for side in ['same_service','restart']:
  value=row.get(side,{})
  for key in ['tokens_exact','controller_exact','frontier_exact']:
   if value.get(key) is not True:raise ProbeError('Continuation disagreement: '+side+'/'+key)
  if type(value.get('max_abs_error')) is not float or value['max_abs_error']!=0. or type(value.get('prefill_tokens')) is not int or value['prefill_tokens']!=0 or type(value.get('output_tokens')) is not int or value['output_tokens']<=0 or type(value.get('forward_tokens')) is not int or value['forward_tokens']!=value['output_tokens']:raise ProbeError('Continuation work/frontier mismatch: '+side)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ['runtime','branchwrite','profile','dependency-root','vision-config','fixtures','pressure-helper','output']:parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();runtime=args.runtime.resolve(strict=True);bw=args.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',args.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json,redact_tree
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 from context_time_travel.store import Store,_json_bytes
 import run_q4_cuda_oom as old
 from run_q4_publication_fault import validate_idle_metrics
 from run_q4_cuda_recovery import validate_worker_transition_trace,_service_stop_receipt
 out=args.output.absolute();out.mkdir(mode=0o700,exist_ok=False)
 profile=json.loads(args.profile.read_bytes());helper_path=args.pressure_helper.resolve(strict=True)
 helper_expected='a3d2690d104f192ca0a25ee224db4b3f7fcef45d0cad4de4a491c755937cd61b'
 if sha(helper_path)!=helper_expected:raise ProbeError('Pressure helper is not the reviewed actual CUDA allocator')
 record=dict(schema=1,status='RUNNING',scope=__doc__,execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],profile_sha256=sha(args.profile),native_sha256=profile['native_build']['sha256'],pressure_helper=dict(path=str(helper_path),sha256=helper_expected),harness_sha256=sha(Path(__file__)),failures=[],chat_server='OFF',q8='DEFERRED_BY_OWNER')
 def publish():_atomic_json(out/'result.json',record)
 def make_probe(name,store,gate=False):
  run=out/name;run.mkdir(mode=0o700);control=run/'control';control.mkdir(mode=0o700)
  class Owner(image_probe_type(GrammarProbe,args.vision_config)):
   def _bootstrap(self,registry):
    path=super()._bootstrap(registry);body=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
    hook=r"""import os,threading,time,hashlib,struct
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.strata_backend import StrataBackend
from context_time_travel.input_spans import InputHistory,history_for_snapshot
_control=Path(CONTROL);_gated=ENABLED;_local=threading.local()
def _save(path,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def _wait(name):
 deadline=time.monotonic()+180
 while not (_control/name).exists():
  if (_control/'abort').exists():raise RuntimeError('parent aborted image OOM gate')
  if time.monotonic()>deadline:raise TimeoutError('image OOM gate timed out')
  time.sleep(.02)
_original_run=ContextCoordinator._run
_original_request=StrataBackend._request
_original_read=StrataBackend._read
def _run(self,model,job):
 _local.job=(job.request.operation,job.branch);_local.model=model;_local.source=(job.source_head or job.head).tip if job.head is not None else None
 try:return _original_run(self,model,job)
 finally:_local.fixture_image=None;_local.job=None
ContextCoordinator._run=_run
def _request(self,command):
 if _gated and getattr(_local,'job',None)==('vc08-oom-fault','trial') and command.startswith('decode ') and not getattr(_local,'gate_sent',False):
  _local.gate_sent=True
  history=history_for_snapshot(_local.model.store,_local.source)
  if not isinstance(history,InputHistory):raise RuntimeError('image gate lost canonical input history')
  images=[x for x in history.spans if x['kind']=='image'];lease=self._active
  if not isinstance(getattr(lease,'inputs',None),InputHistory) or lease.inputs.digest!=history.digest:raise RuntimeError('active native lease input differs from canonical image history')
  binding=dict(snapshot=_local.source,input_identity=history.digest,consumed=len(history),text_tokens=len(history.text_tokens),image_rows=sum(x['rows'] for x in images),image_count=len(images))
  # Pin a native image, then hot-restore it into a distinct page table
  # at this consumed boundary to exercise observable physical COW OOM.
  # This is an explicit test workload step, not production-path instrumentation.
  before=_original_request(self,'stats');frontier_before=self.logits(lease)
  _local.fixture_image=self.export(lease)
  # Freeze aliases a table; restore creates an independent page table that
  # shares physical pages, making the direct page-owner count observable.
  self.restore(lease,_local.fixture_image,history=history,frontier=frontier_before)
  frontier_after=self.logits(lease);stats=_original_request(self,'stats');pages=_original_request(self,'pages')
  def digest(values):return hashlib.sha256(struct.pack('<'+'f'*len(values),*values)).hexdigest()
  capture=dict(native_capture=True,hot_restore=True,restores_delta=stats['restores']-before['restores'],captures_delta=stats['captures']-before['captures'],forward_delta=stats['forward_tokens']-before['forward_tokens'],frontier_before_sha256=digest(frontier_before),frontier_after_sha256=digest(frontier_after))
  _local.gate_stats=stats
  _save(_control/'gate.json',dict(operation_id='vc08-oom-fault',branch='trial',native_pid=self.process.pid,server_pid=os.getpid(),command=command,
   lease_active=lease is not None and lease.state=='ACTIVE' and lease._owner_thread==threading.get_ident(),input_binding=binding,fixture_capture=capture,stats=stats,pages=pages['pages']))
  _wait('pressure-ready')
 return _original_request(self,command)
def _read(self):
 value=_original_read(self)
 if _gated and getattr(_local,'job',None)==('vc08-oom-fault','trial') and getattr(_local,'gate_sent',False) and hasattr(_local,'gate_stats') and value.get('status')=='error' and not getattr(_local,'error_sent',False):
  _local.error_sent=True
  self.process.stdin.write('stats\n');self.process.stdin.flush();after=_original_read(self)
  _save(_control/'native-error.json',dict(response=value,stats_before=_local.gate_stats,stats_after=after,native_pid=self.process.pid))
  _wait('pressure-released')
 return value
StrataBackend._request=_request;StrataBackend._read=_read
"""
    hook=hook.replace('CONTROL',repr(str(control))).replace('ENABLED',repr(gate));assert body.count(marker)==1;path.write_text(body.replace(marker,hook+marker),newline='\n');return path
  probe=Owner(SimpleNamespace(runtime_root=runtime,profile=args.profile,store=store,run_dir=run,dependency_root=args.dependency_root,ram_gib=96,disk_gib=128))
  probe._imports();probe.runtime_closure(profile,runtime)
  registry=run/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(run,'VC08-IMAGE-OOM-'+name).start()
  try:probe.start_service(registry)
  except BaseException:
   if probe.service is not None and probe.service.process is not None:probe.service.stop()
   probe.tracker.finish('SETUP_FAIL');probe.tracker.close();raise
  url=probe.ready_url(probe.service.ready);client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=run/'cli-state',journal=OperationJournal(run/'journal'),api_key=probe.secret,endpoint_pin=hashlib.sha256(str(run).encode()).hexdigest())
  return probe,client
 def finish(probe,status):
  probe.record['status']=status
  if probe.service is not None:
   probe.phase('service-stop-'+str(probe.ordinal));pid=probe.service.process.pid
   receipt=_service_stop_receipt(probe.service)
   if receipt.get('exit_code')!=0 or not receipt['pid_absent_after_stop']:raise ProbeError('Owned HTTP process did not exit zero and disappear')
   row=probe.record['services'][-1];row['stop']=redact_tree(receipt,probe.secret)
   trace_path=probe.run_dir/('pool-'+str(probe.ordinal)+'.json');trace=json.loads(trace_path.read_text())
   closes=[x for x in trace['events'] if x.get('event')=='native_worker_closed']
   row['worker_close_events']=closes
   if not closes or any(type(x.get('returncode')) is not int or Path('/proc',str(x['pid'])).exists() for x in closes):raise ProbeError('Owned native process terminal evidence incomplete')
   if status=='SAME_SERVICE_PASS':
    row['worker_transition']=validate_worker_transition_trace(trace,old_pid=record['old_native_identity']['pid'],new_pid=record['new_native_identity']['pid'],http_pid=pid)
   elif status!='CAMPAIGN_ABORTED':row['worker_shutdown']=probe.trace_worker(trace_path,pid)
   probe.service=None
  probe.publish();probe.tracker.finish(status);probe.tracker.close()
 def vector(store,snapshot,label):
  envelope=json.loads((store.root/'checkpoints'/(snapshot+'.json')).read_text());manifest=envelope['manifest']
  assert hashlib.sha256(_json_bytes(manifest)).hexdigest()==envelope['checksum']
  path=store.root/'checkpoints'/(manifest['payload_sha256']+'.state');assert sha(path)==manifest['payload_sha256']
  with path.open('rb') as f:f.seek(manifest['state_bytes']);raw=f.read()
  assert len(raw)==4*manifest['logits_count'];values=struct.unpack('<'+'f'*manifest['logits_count'],raw);assert all(map(math.isfinite,values))
  dst=out/(label+'.f32');dst.write_bytes(raw);return dst
 def comparison(probe,answer,expected,side):
  assert completed_thinking_text(answer,profile['execution']['eos_tokens'])=='OK'
  actual=vector(probe.store,answer.receipt.snapshot,side);before=Path(expected['frontier']).read_bytes();after=actual.read_bytes();assert len(before)==len(after)
  count=len(before)//4;a=struct.unpack('<'+'f'*count,before);c=struct.unpack('<'+'f'*count,after);work=answer.answer['counter_deltas']
  return dict(tokens_exact=answer.answer['tokens']==expected['tokens'],controller_exact=probe.store.get_snapshot(answer.receipt.snapshot)==expected['canonical'],frontier_exact=before==after,max_abs_error=max(abs(x-y) for x,y in zip(a,c)),prefill_tokens=work['prefill_tokens'],forward_tokens=work['forward_tokens'],output_tokens=len(answer.answer['tokens']),counter_deltas=work,frontier=str(actual),snapshot=answer.receipt.snapshot)
 def wait_file(probe,path,child):
  deadline=time.monotonic()+180
  while not path.exists():
   if child.poll() is not None:raise ProbeError('CLI exited before actual native OOM gate')
   if time.monotonic()>deadline:raise TimeoutError('Image allocation gate timed out')
   time.sleep(.03)
 def status(client):return dict(client.status().capacity)
 active=None;helper=None;helper_id=None;cli=None;error=None
 try:
  producer=out/'producer-store';fault_store=out/'fault-store';active,client=make_probe('producer',producer)
  ns=client.create_namespace('Image allocation recovery','vc08-oom-namespace')['namespace_id'];image=(args.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes();upload=client.upload_image(ns,image)
  root=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-oom-root',images=(upload,));parent=active.store.head('kept');canonical=active.store.get_snapshot(parent.tip);binding=image_source_binding(active.store,parent.tip)
  payload=producer/'checkpoints'/(root.receipt.raw['payload_sha256']+'.state');parent_sha=sha(payload)
  def copy_fixture(src,dst):
   if Path(src).parent.name=='checkpoints' and Path(src).suffix=='.state':os.link(src,dst);return dst
   return shutil.copy2(src,dst)
  with active.store._locked():shutil.copytree(producer,fault_store,copy_function=copy_fixture)
  expected={}
  for branch,op in [('trial','vc08-oom-recovery'),('later','vc08-oom-restart')]:
   h=client.fork('kept',parent.tip,branch,'vc08-fork-'+branch);answer=client.answer(branch,h.revision,op,1024,settings=dict(temperature=0),grammar_state='VC08_REPLY')
   assert completed_thinking_text(answer,profile['execution']['eos_tokens'])=='OK'
   expected[branch]=dict(tokens=answer.answer['tokens'],canonical=active.store.get_snapshot(answer.receipt.snapshot),frontier=str(vector(active.store,answer.receipt.snapshot,'reference-'+branch)),work=answer.answer['counter_deltas'])
  record.update(input_binding=binding,parent_payload_sha256=parent_sha,reference=expected)
  finish(active,'REFERENCE_PASS');record['producer_services']=active.record['services'];active=None;publish()
  active,client=make_probe('fault',fault_store,True)
  heads={branch:client.fork('kept',parent.tip,branch,'vc08-fork-'+branch) for branch in ['trial','later']};old_head=active.store.head('trial');old_root=active.store.head('kept');source_file=fault_store/'checkpoints'/(root.receipt.raw['payload_sha256']+'.state')
  assert sha(source_file)==parent_sha;run=active.run_dir;control=run/'control';url=active.ready_url(active.service.ready);env={**os.environ,'COW_API_KEY':active.secret}
  command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','240','--state-dir',str(run/'cli-state'),'--machine-result','answer','--branch','trial','--revision',str(old_head.revision),'--operation-id','vc08-oom-fault','--max-tokens','1024','--temperature','0','--grammar-state','VC08_REPLY']
  with (run/'fault.stdout').open('wb') as so,(run/'fault.stderr').open('wb') as se:cli=subprocess.Popen(command,env=env,stdout=so,stderr=se,stdin=subprocess.DEVNULL,start_new_session=True)
  active.phase('wait-first-image-decode');wait_file(active,control/'gate.json',cli);gate=json.loads((control/'gate.json').read_text());validate_image_gate(gate,binding,operation='vc08-oom-fault',branch='trial')
  old_pid=gate['native_pid'];old_identity=old._proc_identity(old_pid);assert old_identity['parent_pid']==active.service.process.pid
  active.phase('fill-real-device-memory');helper,helper_id,ready=old._helper_start(helper_path,run/'helper.stderr');pressure=old._helper_fill(helper,60);old.validate_pressure_frame(pressure);record['pressure']=dict(identity=helper_id,ready=ready,fill=pressure);publish();(control/'pressure-ready').touch()
  active.phase('wait-actual-native-oom');wait_file(active,control/'native-error.json',cli);event=json.loads((control/'native-error.json').read_text());assert event['native_pid']==old_pid
  oom=old.validate_native_oom(event['response'],event['stats_before'],event['stats_after'],expected_forward=0);record['native_oom']=oom;record['gate']=gate;publish()
  release,terminal=old._helper_release(helper,helper_id,60);old.validate_helper_pressure(pressure,release);old.validate_owned_process_exit(terminal,expected_pid=helper_id['pid'],expected_start_time=helper_id['start_time'],expected_exe=helper_id['exe']);record['pressure'].update(release=release,terminal=terminal);helper=None;(control/'pressure-released').touch()
  code=cli.wait(60);cli=None;failure=json.loads((run/'fault.stdout').read_bytes());assert code==2 and 'error' in failure and 'receipt' not in failure and 'answer' not in failure
  assert event['response']['message'] in json.dumps(failure)
  record.update(oom_verified=True,helper_released=True,no_error_ack=True,branch_unchanged_on_error=active.store.head('trial')==old_head)
  active.phase('replacement-ready-before-followup');deadline=time.monotonic()+180;observed=None
  while time.monotonic()<deadline:
   observed=status(client);workers=observed['workers'];worker=workers['0']
   if worker['health']=='healthy' and worker['restarts']==1 and worker['pid']!=old_pid and worker['pid'] is not None and not Path('/proc',str(old_pid)).exists():break
   time.sleep(.1)
  else:raise ProbeError('Same HTTP service did not replace poisoned image worker')
  validate_idle_metrics(observed);new_identity=old._proc_identity(worker['pid']);assert new_identity['parent_pid']==active.service.process.pid and new_identity['exe']==old_identity['exe']
  record.update(old_native_absent=not Path('/proc',str(old_pid)).exists(),replacement_before_followup=True,replacement_status=observed,old_native_identity=old_identity,new_native_identity=new_identity)
  active.phase('same-service-image-continuation');answer=client.answer('trial',heads['trial'].revision,'vc08-oom-recovery',1024,settings=dict(temperature=0),grammar_state='VC08_REPLY');record['same_service']=comparison(active,answer,expected['trial'],'same-service')
  record['parent_unchanged']=active.store.head('kept')==old_root and active.store.get_snapshot(parent.tip)==canonical and sha(source_file)==parent_sha
  finish(active,'SAME_SERVICE_PASS');record['fault_services']=active.record['services'];active=None
  trace=json.loads((run/'pool-1.json').read_text());record['worker_transition']=validate_worker_transition_trace(trace,old_pid=old_pid,new_pid=new_identity['pid'],http_pid=gate['server_pid']);publish()
  active,client=make_probe('restart',fault_store)
  answer=client.answer('later',heads['later'].revision,'vc08-oom-restart',1024,settings=dict(temperature=0),grammar_state='VC08_REPLY');record['restart']=comparison(active,answer,expected['later'],'restart')
  assert active.store.head('kept')==old_root and active.store.get_snapshot(parent.tip)==canonical and sha(source_file)==parent_sha
  finish(active,'RESTART_PASS');record['restart_services']=active.record['services'];active=None
  validate_outcome(record);record['status']='VC08_IMAGE_REAL_CUDA_OOM_RECOVERY_PASS'
 except BaseException as exc:error=exc;record['status']='VC08_IMAGE_REAL_CUDA_OOM_RECOVERY_FAIL';record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if helper is not None:
   cleanup=old._stop_owned_helper(helper,helper_id,30);record['helper_cleanup']=cleanup;helper=None
  if active is not None:
   (active.run_dir/'control/abort').touch()
   if cli is not None and cli.poll() is None:
    os.killpg(cli.pid,signal.SIGTERM)
    try:cli.wait(10)
    except subprocess.TimeoutExpired:os.killpg(cli.pid,signal.SIGKILL);cli.wait(5)
   try:finish(active,'CAMPAIGN_ABORTED');record['aborted_services']=active.record['services']
   except BaseException as exc:record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  publish()
 return 1 if error else 0
if __name__=='__main__':raise SystemExit(main())
