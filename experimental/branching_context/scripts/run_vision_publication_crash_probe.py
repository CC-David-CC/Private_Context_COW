"""VC08 owned SIGKILL at real image checkpoint/reference publication boundaries.

Process crashes on Linux/local filesystem, not host power-loss simulation.
New processes replay the exact original CLI intent against the same endpoint.
"""
import argparse,ctypes,hashlib,json,math,os,shutil,signal,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import urllib.request,urllib.error
POINTS=('payload_during_write','branch_committed_before_ack','payload_before_publish','payload_durable',
 'before_manifest_publish','manifest_during_write','manifest_before_publish','manifest_durable',
 'before_branch_publish','after_refs_stage_fsync','after_refs_rename_before_dir_fsync')
class ProbeError(RuntimeError):pass

def committed(point):
 if point not in POINTS:raise ProbeError('unknown crash point')
 return point in ('branch_committed_before_ack','after_refs_rename_before_dir_fsync')

def validate_case(row,*,outputs):
 expected=dict(server_exit=-9,native_exit=-9,server_absent=True,native_absent=True,no_ack=True,
  visible_new=committed(row.get('point')),source_intact=True,receipt_revision_delta=1,move_count=1,
  tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,
  recovery_forward_tokens=0 if committed(row['point']) else outputs,recovery_prefill_tokens=0,
  repeat_native_trace_unchanged=True,changed_request_rejected=True,checkpoint_stages_after=[],reference_stages_after=[])
 for k,v in expected.items():
  if k not in row or type(row[k]) is not type(v) or row[k]!=v:raise ProbeError('failed crash/recovery witness: '+k)

def owned_crash(service,gate):
 import run_q4_postcommit_restart as kill
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'cannot become owned-child subreaper')
 native=kill._read_proc(Path('/proc'),gate['native_pid'])
 if native['parent_pid']!=gate['server_pid'] or Path(native['exe']).resolve()!=Path(gate['native_exe']).resolve():
  raise ProbeError('native identity differs from the exact gate-owned child')
 raw=kill._kill_owned_group(service,native,Path('/proc'))
 reaped,status=os.waitpid(gate['native_pid'],0)
 if reaped!=gate['native_pid']:raise ProbeError('did not reap the exact owned native process')
 result={k:v for k,v in raw.items() if k!='prekill_members'}
 result['prekill_members']=[{k:x[k] for k in ['pid','parent_pid','process_group','start_time','exe']} for x in raw['prekill_members']]
 result['native_exit_code']=os.waitstatus_to_exitcode(status)
 return result

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ['runtime','branchwrite','profile','dependency-root','vision-config','fixtures','output']:parser.add_argument('--'+n,type=Path,required=True)
 a=parser.parse_args();runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 from context_time_travel.store import Store
 from cow_cli import RequestJournal
 import run_q4_postcommit_restart as kill
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'cannot become subreaper for owned crash children')
 value=ctypes.c_int();assert libc.prctl(37,ctypes.byref(value),0,0,0)==0 and value.value==1
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False);profile=json.loads(a.profile.read_bytes())
 record=dict(schema=1,status='RUNNING',scope=__doc__,points=list(POINTS),cases=[],failures=[],
  execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],profile_sha256=sha(a.profile),
  native_sha256=profile['native_build']['sha256'],harness_sha256=sha(Path(__file__)),q8='DEFERRED_BY_OWNER',chat_server='OFF')
 def publish():_atomic_json(out/'result.json',record)
 def create_probe(name,store,point=None):
  run=out/name;run.mkdir(mode=0o700)
  class Owner(image_probe_type(GrammarProbe,a.vision_config)):
   fixed_port=None
   def _bootstrap(self,registry):
    path=super()._bootstrap(registry);body=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
    hook=r"""import os,threading,time,struct
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store
from context_time_travel.coordinator import ContextCoordinator
_control=Path(CONTROL_PATH);_local=threading.local();_point=CRASH_POINT;_target_operation=TARGET_OPERATION
if FIXED_PORT is not None:sys.argv[sys.argv.index('--port')+1]=str(FIXED_PORT)
def _save_json(path,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def _gate(point):
 if (_control/'crash-completed').exists() or point!=_point or getattr(_local,'operation',None)!=_target_operation:return
 prepared=_local.prepared
 _save_json(_control/'gate.json',dict(point=point,operation_id=_local.operation,branch=_local.branch,
  snapshot=prepared.snapshot,server_pid=os.getpid(),native_pid=_local.model.backend.process.pid,
  durable_done=_local.request.durable.done(),native_exe=str(Path('/proc/'+str(_local.model.backend.process.pid)+'/exe').resolve())))
 deadline=time.monotonic()+300
 while not (_control/'abort').exists() and time.monotonic()<deadline:time.sleep(.02)
 raise RuntimeError('test parent aborted or timed out the publication crash gate')
_original_run=ContextCoordinator._run
def _run(self,model,job):
 result=_original_run(self,model,job)
 if job.request.operation==_target_operation and not (_control/'crash-completed').exists():_save_json(_control/'computed-answer.json',result)
 return result
ContextCoordinator._run=_run
_original_publish=ContextCoordinator._publish
def _publish(self,model,job,prepared,*args,**kwargs):
 if job.request.operation!=_target_operation or (_control/'crash-completed').exists():return _original_publish(self,model,job,prepared,*args,**kwargs)
 _local.operation=job.request.operation;_local.branch=job.branch;_local.prepared=prepared;_local.model=model;_local.request=job.request
 raw=struct.pack('<'+'f'*len(prepared.checkpoint.frontier),*prepared.checkpoint.frontier)
 with (_control/'expected-frontier.f32').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 _save_json(_control/'expected-snapshot.json',model.store.get_snapshot(prepared.snapshot))
 try:return _original_publish(self,model,job,prepared,*args,**kwargs)
 finally:_local.operation=None
ContextCoordinator._publish=_publish
_original_init=BranchingModel.__init__
def _init(self,*args,**kwargs):
 _original_init(self,*args,**kwargs);previous=self._checkpoint_fault
 def fault(point):_gate(point);return previous(point)
 self._checkpoint_fault=fault
BranchingModel.__init__=_init
_original_store_fault=Store._fault
def _store_fault(self,point):_gate(point);return _original_store_fault(self,point)
Store._fault=_store_fault
"""
    control=run/'control';control.mkdir(mode=0o700,exist_ok=True)
    hook=hook.replace('CONTROL_PATH',repr(str(control))).replace('CRASH_POINT',repr(point)).replace('TARGET_OPERATION',repr('vc08-crash-original')).replace('FIXED_PORT',repr(self.fixed_port))
    assert body.count(marker)==1;path.write_text(body.replace(marker,hook+marker),newline='\n');return path
  probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=store,run_dir=run,dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
  probe._imports();probe.runtime_closure(profile,runtime)
  registry=run/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(run,'VC08-'+name).start()
  try:probe.start_service(registry)
  except BaseException:
   if probe.service is not None and probe.service.process is not None:probe.service.stop()
   probe.tracker.finish('SETUP_FAIL');probe.tracker.close();raise
  url=probe.ready_url(probe.service.ready);probe.fixed_port=probe.service.ready['address'][1]
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=run/'cli-state',journal=OperationJournal(run/'journal'),api_key=probe.secret,endpoint_pin=hashlib.sha256(str(run).encode()).hexdigest())
  return probe,client,registry
 def finish(probe,status):
  probe.record['status']=status;probe.stop_service();probe.publish();probe.tracker.finish(status);probe.tracker.close()
  rows=[json.loads(x) for x in (probe.run_dir/f'native-{probe.ordinal}.jsonl').read_text().splitlines()]
  closed=[x for x in rows if x.get('closed') is True];assert len(closed)==1
  return closed[0]['counters']
 active=None;cli=None;error=None
 try:
  base=out/'base-store';active,client,registry=create_probe('producer',base)
  ns=client.create_namespace('Publication crash','vc08-crash-namespace')['namespace_id']
  image=(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes();upload=client.upload_image(ns,image)
  root=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-kept-root',images=(upload,))
  parent=active.store.head('kept');source=active.store.get_snapshot(parent.tip);payload=base/'checkpoints'/(root.receipt.raw['payload_sha256']+'.state');parent_sha=sha(payload)
  record['producer']=dict(receipt=dict(root.receipt.raw),services=active.record['services'],counters=finish(active,'PRODUCER_PASS'));active=None;publish()
  for point in POINTS:
   store=out/('store-'+point)
   def copy_fixture(src,dst):
    # The prepared immutable root payload is shared only by these disposable fixtures.
    # Publication always creates a new target file; every arm audits the root hash.
    if Path(src).parent.name=='checkpoints' and Path(src).suffix=='.state':os.link(src,dst);return dst
    return shutil.copy2(src,dst)
   shutil.copytree(base,store,copy_function=copy_fixture)
   active,client,registry=create_probe(point,store,point);run=active.run_dir;url=active.ready_url(active.service.ready)
   fork=client.fork('kept',parent.tip,'fault','vc08-crash-fork');old=active.store.head('fault')
   op='vc08-crash-original';command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','180','--state-dir',str(run/'cli-state'),'--machine-result']
   env={**os.environ,'COW_API_KEY':active.secret}
   with (run/'crash.stdout').open('wb') as so,(run/'crash.stderr').open('wb') as se:
    cli=subprocess.Popen(command+['answer','--branch','fault','--revision',str(old.revision),'--operation-id',op,'--max-tokens','1024','--temperature','0','--grammar-state','VC08_REPLY'],env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
   active.phase('waiting-for-'+point);deadline=time.monotonic()+180;gatefile=run/'control/gate.json'
   while not gatefile.exists():
    if cli.poll() is not None:raise ProbeError('CLI exited before crash gate')
    if time.monotonic()>deadline:raise TimeoutError('selected publication stage was not reached')
    time.sleep(.05)
   gate=json.loads(gatefile.read_bytes());expected=json.loads((run/'control/computed-answer.json').read_bytes());expected_snapshot=json.loads((run/'control/expected-snapshot.json').read_bytes())
   assert gate['point']==point and gate['operation_id']==op and gate['branch']=='fault' and gate['durable_done'] is False
   original=RequestJournal(run/'cli-state').get(op);assert original['state']=='pending'
   before_ref=(store/'references/fault.json').read_bytes()
   cp_stages=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in (store/'checkpoints').glob('.stage-*')]
   crash=owned_crash(active.service,gate);native_exit=crash['native_exit_code']
   active.record['services'][-1]['crash']=crash;active.service=None
   code=cli.wait(30);cli=None;assert code==2
   pending_after=RequestJournal(run/'cli-state').get(op)['state']=='pending';assert pending_after
   failed_cli=json.loads((run/'crash.stdout').read_bytes())
   (run/'control/crash-completed').touch()
   after=Store(store).head('fault');row=dict(point=point,gate=gate,crash=crash,cli_exit=code,
    server_exit=crash['server_exit_code'],native_exit=native_exit,server_absent=not Path('/proc',str(gate['server_pid'])).exists(),native_absent=not Path('/proc',str(gate['native_pid'])).exists(),
    no_ack=(original['state']=='pending' and pending_after and gate['durable_done'] is False and 'receipt' not in failed_cli and 'error' in failed_cli),visible_new=after.tip==gate['snapshot'],checkpoint_stages_at_crash=cp_stages,
    reference_bytes_sha256_at_gate=hashlib.sha256(before_ref).hexdigest(),expected_tokens=expected['tokens'])
   assert after.tip==(gate['snapshot'] if committed(point) else parent.tip)
   assert after.revision==(old.revision+1 if committed(point) else old.revision)
   record['cases'].append(row);publish()
   active.phase('restart-and-original-cli-retry');active.start_service(registry)
   assert active.ready_url(active.service.ready)==url
   retry=subprocess.run(command+['retry-request','--operation-id',op],env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
   (run/'retry.stdout').write_bytes(retry.stdout);(run/'retry.stderr').write_bytes(retry.stderr);assert retry.returncode==0,retry.stderr.decode(errors='replace')
   result=json.loads(retry.stdout);answer=SimpleNamespace(answer=result['answer'],receipt=SimpleNamespace(snapshot=result['receipt']['snapshot']))
   assert completed_thinking_text(answer,profile['execution']['eos_tokens'])=='OK'
   assert result['receipt']['snapshot']==gate['snapshot']
   current=active.store.head('fault');canonical=active.store.get_snapshot(current.tip)
   manifest=json.loads((store/'checkpoints'/(current.tip+'.json')).read_bytes())['manifest'];path=store/'checkpoints'/(manifest['payload_sha256']+'.state')
   with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['payload_sha256']
   with path.open('rb') as f:f.seek(manifest['state_bytes']);actual=f.read()
   predicted=(run/'control/expected-frontier.f32').read_bytes();assert len(actual)==len(predicted)==4*manifest['logits_count']
   av=struct.unpack('<'+'f'*manifest['logits_count'],actual);bv=struct.unpack('<'+'f'*manifest['logits_count'],predicted);assert all(map(math.isfinite,av)) and all(map(math.isfinite,bv))
   (run/'recovered-frontier.f32').write_bytes(actual)
   trace=run/'native-2.jsonl';before=trace.read_bytes()
   def post(body):
    request=urllib.request.Request(url+'/cow/text/answer',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+active.secret,'Content-Type':'application/json'})
    try:
     with urllib.request.urlopen(request,timeout=180) as response:return response.status,response.read()
    except urllib.error.HTTPError as e:
     with e:return e.code,e.read()
   body=json.loads(original['body_json']);status,data=post(body);assert status==200
   repeats=[json.loads(x) for x in data.splitlines() if x.strip()];complete=[x for x in repeats if x.get('event')=='complete'];assert len(complete)==1 and complete[0]['receipt']==result['receipt']
   bad_status,bad=post(dict(body,max_tokens=body['max_tokens']+1))
   branch=active.store._read_branch('fault');move_count=sum(x.get('operation_id')==op for x in branch['events'])
   row.update(source_intact=active.store.head('kept')==parent and active.store.get_snapshot(parent.tip)==source and sha(payload)==parent_sha,
    receipt_revision_delta=current.revision-old.revision,move_count=move_count,tokens_exact=result['answer']['tokens']==expected['tokens'],controller_exact=canonical==expected_snapshot,
    frontier_exact=actual==predicted,max_abs_error=max(abs(x-y) for x,y in zip(av,bv)),
    repeat_native_trace_unchanged=trace.read_bytes()==before,changed_request_rejected=bad_status==400 and b'operation id reused' in bad.lower(),
    checkpoint_stages_after=[p.name for p in (store/'checkpoints').glob('.stage-*')],reference_stages_after=[p.name for p in (store/'references').glob('.stage-*')],
    expected_frontier_sha256=hashlib.sha256(predicted).hexdigest(),recovered_frontier_sha256=hashlib.sha256(actual).hexdigest(),receipt=result['receipt'])
   counters=finish(active,'RECOVERY_RETURNED');row['services']=active.record['services'];active=None
   row.update(recovery_forward_tokens=counters['forward_tokens'],recovery_prefill_tokens=counters['prefill_tokens'],recovery_native_counters=counters)
   validate_case(row,outputs=len(expected['tokens']));row['status']='PASS';publish()
  record['status']='VC08_IMAGE_PUBLICATION_PROCESS_CRASH_PASS'
 except BaseException as exc:
  error=exc;record['status']='VC08_IMAGE_PUBLICATION_PROCESS_CRASH_FAIL';record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if active is not None:
   (active.run_dir/'control/abort').touch()
   if cli is not None and cli.poll() is None:
    os.killpg(cli.pid,signal.SIGTERM)
    try:cli.wait(10)
    except subprocess.TimeoutExpired:os.killpg(cli.pid,signal.SIGKILL);cli.wait(5)
   if active.service is not None:
    try:finish(active,'CAMPAIGN_ABORTED')
    except BaseException as exc:record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  publish()
 return 1 if error else 0
if __name__=='__main__':raise SystemExit(main())
