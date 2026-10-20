"""VC08 real checkpoint-payload ENOSPC on a private512MiB ext4 volume.

Canonical history and diagnostics are outside the full filesystem. Observers do
not replace the writer or synthesize errors; retry uses captured prepared bytes.
"""
import argparse,contextlib,hashlib,json,math,os,shutil,signal,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
class CampaignError(RuntimeError):pass

def validate_layout(row,*,checkpoint,canonical_device,checkpoint_device):
 if (row.get('fstype')!='ext4' or not str(row.get('source','')).startswith('/dev/loop')
  or row.get('mountpoint')!=str(checkpoint) or row.get('image_bytes')!=512*1024**2
  or type(row.get('capacity_bytes')) is not int or not 256*1024**2<=row['capacity_bytes']<=512*1024**2
  or row.get('mount_namespace')==row.get('init_mount_namespace') or canonical_device==checkpoint_device):raise CampaignError('Checkpoint filesystem is not the exact bounded private mount')

def validate_campaign(row):
 expected=dict(cli_exit=2,state='failed',durable=False,publication_retryable=True,errno=28,
  old_reference_unchanged=True,parent_payload_unchanged=True,parent_controller_unchanged=True,
  payload_published=False,manifest_published=False,stage_files_after=[],stage_existed_during_error=True,
  retry_same_receipt=True,retry_revision_delta=1,retry_native_trace_unchanged=True,replay_native_trace_unchanged=True,
  retry_payload_matches_prepared=True,tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,retained_root_usable=True)
 if type(row) is not dict or type(row.get('operation_id')) is not str or not row['operation_id']:raise CampaignError('Missing original operation identity')
 for key,value in expected.items():
  if key not in row or type(row[key]) is not type(value) or row[key]!=value:raise CampaignError('Missing or failed payload witness: '+key)
 for key in ['stage_logical_bytes','requested_payload_bytes','stage_device','checkpoint_device']:
  if type(row.get(key)) is not int or row[key]<=0:raise CampaignError('Missing actual staged-file geometry: '+key)
 # Delayed allocation may accept logical bytes before allocating any disk blocks.
 if type(row.get('stage_allocated_bytes')) is not int or row['stage_device']!=row['checkpoint_device'] or not 0<=row['stage_allocated_bytes']<row['requested_payload_bytes']:raise CampaignError('ENOSPC did not interrupt actual checkpoint allocation')

def install_writer_hook(body,control,branch,operation,store):
 marker="importlib.import_module('serve.cow_http').main()\n"
 if body.count(marker)!=1:raise CampaignError('Expected exactly one actual HTTP entrypoint')
 hook=r"""import os,threading,time,contextlib,struct,hashlib,json
from pathlib import Path
from context_time_travel.real import BranchingModel
from context_time_travel.coordinator import ContextCoordinator
_local=threading.local();_control=Path(CONTROL_DIR);_operation=TARGET_OP;_branch=TARGET_BRANCH;_checkpoint=Path(CHECKPOINT_PATH)
def _save(path,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
_original_run=ContextCoordinator._run
_original_publish=ContextCoordinator._publish
_original_parts=BranchingModel._publish_checkpoint_parts
_original_stage=BranchingModel._checkpoint_stage
def _run(self,model,job):
 answer=_original_run(self,model,job)
 if job.request.operation==_operation and not (_control/'computed-answer.json').exists():_save(_control/'computed-answer.json',answer)
 return answer
def _publish(self,model,job,prepared,*args,**kwargs):
 if job.request.operation!=_operation:return _original_publish(self,model,job,prepared,*args,**kwargs)
 _local.target=True;_local.snapshot=prepared.snapshot
 if _local.target and not (_control/'expected-snapshot.json').exists():
  _save(_control/'expected-snapshot.json',model.store.get_snapshot(prepared.snapshot))
  raw=struct.pack('<'+'f'*len(prepared.checkpoint.frontier),*prepared.checkpoint.frontier)
  with (_control/'expected-frontier.f32').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 try:return _original_publish(self,model,job,prepared,*args,**kwargs)
 finally:_local.target=False
@contextlib.contextmanager
def _stage(self):
 with _original_stage(self) as (stream,path):
  try:yield stream,path
  except OSError as error:
   if getattr(_local,'payload',False) and not (_control/'partial-stage.json').exists():
    st=os.fstat(stream.fileno());entry=path.lstat()
    _save(_control/'partial-stage.json',dict(errno=error.errno,path=str(path),device=st.st_dev,inode=st.st_ino,logical_bytes=st.st_size,allocated_bytes=st.st_blocks*512,
     existed=entry.st_ino==st.st_ino and entry.st_dev==st.st_dev))
   raise
def _parts(self,destination,parts):
 target=getattr(_local,'target',False) and str(destination).endswith('.state')
 if not target:return _original_parts(self,destination,parts)
 parts=tuple(parts);h=hashlib.sha256()
 for part in parts:h.update(part)
 if not (_control/'gate.json').exists():
  _save(_control/'gate.json',dict(operation_id=_operation,branch=_branch,snapshot=_local.snapshot,target_snapshot=_local.snapshot,
   destination=str(destination),destination_directory=str(Path(destination).parent),requested_payload_bytes=sum(map(len,parts)),expected_payload_sha256=h.hexdigest(),
   old_reference_sha256=hashlib.sha256((self.store.references/(_branch+'.json')).read_bytes()).hexdigest()))
  deadline=time.monotonic()+180
  while not (_control/'release.json').exists():
   if (_control/'abort.json').exists():raise RuntimeError('checkpoint ENOSPC gate aborted')
   if time.monotonic()>deadline:raise TimeoutError('checkpoint ENOSPC gate timed out')
   time.sleep(.02)
  if json.loads((_control/'release.json').read_text())['operation_id']!=_operation:raise RuntimeError('wrong release operation')
 _local.payload=True
 try:return _original_parts(self,destination,parts)
 except OSError as error:
  _save(_control/'writer-error.json',dict(operation_id=_operation,branch=_branch,snapshot=_local.snapshot,destination_kind='checkpoint-payload-state',payload_name=Path(destination).name,
   destination_directory=str(Path(destination).parent),requested_payload_bytes=sum(map(len,parts)),exception_type=type(error).__name__,errno=error.errno,message=str(error),
   published_payload_exists=Path(destination).exists(),published_manifest_exists=(self.checkpoint_directory/(_local.snapshot+'.json')).exists(),staging_files_after=[p.name for p in self.checkpoint_directory.glob('.stage-*')]))
  raise
 finally:_local.payload=False
ContextCoordinator._run=_run;ContextCoordinator._publish=_publish
BranchingModel._checkpoint_stage=_stage;BranchingModel._publish_checkpoint_parts=_parts
"""
 for key,value in [('CONTROL_DIR',str(control)),('TARGET_OP',operation),('TARGET_BRANCH',branch),('CHECKPOINT_PATH',str(Path(store)/'checkpoints'))]:
  hook=hook.replace(key,repr(value))
 return body.replace(marker,hook+marker)

def wait_gate(path,pending,timeout,progress):
 deadline=time.monotonic()+timeout
 while not path.exists():
  progress.check()
  if pending.done():raise CampaignError('CLI exited before payload gate')
  if time.monotonic()>deadline:raise TimeoutError('payload gate not reached')
  time.sleep(.03)
 return json.loads(path.read_bytes())

def completed_retry_receipt(journal,operation,url):
 record=journal.get(operation)
 if not isinstance(record,dict) or record.get('state')!='complete' or record.get('url')!=url or record.get('operation_id')!=operation:raise CampaignError('Original operation retry receipt missing')
 if type(record.get('receipt')) is not dict:raise CampaignError('Durable retry receipt missing')
 return record['receipt']

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def wait_idle(client):
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  row=client.status().capacity
  if not row['queued'] and not row['pending_branches'] and not any(row['active_per_gpu'].values()):return row
  time.sleep(.05)
 raise TimeoutError('coordinator did not settle')

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ('runtime','branchwrite','profile','dependency-root','vision-config','fixtures','output','store','loop-image'):
  parser.add_argument('--'+n,type=Path,required=True)
 parser.add_argument('--preflight-only',action='store_true')
 parser.add_argument('--parent-mount-namespace',required=True)
 parser.add_argument('--expected-image-uuid',required=True)
 parser.add_argument('--expected-image-inode',required=True,type=int)
 a=parser.parse_args(argv)
 runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 import run_q4_canonical_enospc as fault
 import run_q4_enospc as disk_fault
 import run_q4_publication_fault as fingerprints
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 if os.geteuid()==0:raise CampaignError('inference must run as the ordinary project user')
 if a.loop_image.stat().st_size!=512*1024**2:raise CampaignError('fixture is not the exact bounded512MiB image')
 checkpoint=a.store/'checkpoints'
 if a.loop_image.stat().st_ino!=a.expected_image_inode:raise CampaignError('Loop image inode changed')
 if subprocess.check_output(['blkid','-p','-s','UUID','-o','value',str(a.loop_image)],text=True).strip()!=a.expected_image_uuid:raise CampaignError('Loop image UUID changed')
 layout=disk_fault.validate_ext4_mount(checkpoint,a.loop_image,parent_mount_namespace=a.parent_mount_namespace,min_capacity=256*1024**2,max_capacity=512*1024**2)
 validate_layout(layout,checkpoint=checkpoint,canonical_device=a.store.stat().st_dev,checkpoint_device=checkpoint.stat().st_dev)
 if a.preflight_only:
  print(json.dumps(dict(status='PREPARED_NOT_RUN',mount=layout,inference_uid=os.geteuid()),sort_keys=True));return 0
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False);control=out/'control';control.mkdir(mode=0o700)
 if out.stat().st_dev==checkpoint.stat().st_dev:raise CampaignError('evidence must be outside the full checkpoint filesystem')
 operation='vc08-image-checkpoint-enospc-original';branch='fault'
 class Owner(image_probe_type(GrammarProbe,a.vision_config)):
  def _bootstrap(self,registry):
   p=super()._bootstrap(registry)
   p.write_text(install_writer_hook(p.read_text(),control,branch,operation,a.store),newline='\n');return p
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=a.store,run_dir=out,
  dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
 probe.record.update(status='RUNNING',scope=__doc__,mount=layout,numerical_oracle='SAVED_CONSUMED_FRONTIER_RETRY_COMPARISON_NOT_FRESH_FULL_HISTORY',numerical_max_error=None,
  chat_server='OFF',q8='DEFERRED_BY_OWNER',harness_sha256=sha(Path(__file__)))
 error=None;cli=None;filler=None
 try:
  probe._imports();profile=json.loads(a.profile.read_bytes());probe.runtime_closure(profile,runtime)
  assert profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  probe.record.update(execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
   profile_sha256=sha(a.profile),native_sha256=profile['native_build']['sha256'])
  registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(
   schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(out,'VC08-image-checkpoint-enospc').start();probe.start_service(registry)
  url=probe.ready_url(probe.service.ready);pin=hashlib.sha256(str(out).encode()).hexdigest()
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=out/'cli-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
  ns=client.create_namespace('Full filesystem boundary','vc08-enospc-namespace')['namespace_id']
  image=(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes()
  upload=client.upload_image(ns,image)
  root=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-kept',images=(upload,))
  parent=probe.store.head('kept');snapshot=probe.store.get_snapshot(parent.tip)
  payload=a.store/'checkpoints'/(root.receipt.raw['payload_sha256']+'.state');payload_sha=sha(payload)
  fork=client.fork('kept',parent.tip,branch,'vc08-fault-fork')
  head=probe.store.head(branch);ref=fault._branch_reference_path(a.store,branch);old_ref=ref.read_bytes()
  metrics=wait_idle(client)
  for w in metrics['workers'].values():probe.tracker.native_started(w['pid'])
  cli_command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','180',
   '--state-dir',str(out/'cli-state'),'--machine-result']
  answer_args=['answer','--branch',branch,'--revision',str(head.revision),'--operation-id',operation,
   '--max-tokens','1024','--temperature','0','--grammar-state','VC08_REPLY']
  env={**os.environ,'COW_API_KEY':probe.secret}
  probe.phase('answer-to-exact-reference-writer')
  with (out/'fault.stdout').open('wb') as so,(out/'fault.stderr').open('wb') as se:
   cli=subprocess.Popen(cli_command+answer_args,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
  class Pending:
   def done(self):return cli.poll() is not None
  gate=wait_gate(control/'gate.json',Pending(),180,probe.tracker)
  assert gate['operation_id']==operation and gate['branch']==branch and gate['old_reference_sha256']==hashlib.sha256(old_ref).hexdigest()
  target=gate['target_snapshot'];target_expected=dict(snapshot=target,payload_sha256=gate['expected_payload_sha256'],payload_bytes=gate['requested_payload_bytes'])
  target_object=fingerprints._snapshot_record_fingerprint(a.store,target)
  probe.record.update(writer_gate=gate,prepared_payload=target_expected,parent_payload_sha256=payload_sha,root_receipt=dict(root.receipt.raw));probe.publish()
  probe.phase('fill-bounded-ext4-to-kernel-enospc')
  filler=disk_fault._fill_private_mount(checkpoint,max_bytes=layout['capacity_bytes'],keep_free_bytes=4*1024**2)
  _atomic_json(out/'filler-receipt.json',filler)
  assert filler['mount_device']==str(checkpoint.stat().st_dev) and cli.poll() is None
  _atomic_json(control/'release.json',dict(operation_id=operation))
  code=cli.wait(90);cli=None;wait_idle(client)
  observation=client.operation(operation)
  failure=json.loads((control/'writer-error.json').read_bytes());partial=json.loads((control/'partial-stage.json').read_bytes())
  disk_fault.validate_enospc_writer_observation(failure,operation,branch,checkpoint_directory=checkpoint)
  fault_stdout=json.loads((out/'fault.stdout').read_bytes());assert 'error' in fault_stdout and 'receipt' not in fault_stdout and 'answer' not in fault_stdout
  witness=dict(cli_exit=code,operation_id=operation,state=observation.state,durable=observation.durable,
   publication_retryable=observation.raw.get('publication_retryable'),errno=failure['errno'],
   old_reference_unchanged=ref.read_bytes()==old_ref and probe.store.head(branch)==head,
   parent_payload_unchanged=sha(payload)==payload_sha,parent_controller_unchanged=probe.store.get_snapshot(parent.tip)==snapshot,
   payload_published=failure['published_payload_exists'],manifest_published=failure['published_manifest_exists'],
   stage_files_after=failure['staging_files_after'],stage_existed_during_error=partial['existed'],stage_device=partial['device'],checkpoint_device=checkpoint.stat().st_dev,
   stage_allocated_bytes=partial['allocated_bytes'],stage_logical_bytes=partial['logical_bytes'],requested_payload_bytes=gate['requested_payload_bytes'])
  probe.record.update(failure_operation=dict(observation.raw),fault_witness=witness,writer_failure=failure,partial_stage=partial);probe.publish()
  disk_fault._remove_owned_filler(filler);filler=None
  probe.phase('retry-original-publication-without-native-work')
  trace=out/'native-1.jsonl';trace_before=trace.read_bytes()
  retries=[]
  for i in range(2):
   response=subprocess.run(cli_command+['retry-publication','--operation-id',operation],env=env,stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
   (out/f'retry-{i}.stdout').write_bytes(response.stdout);(out/f'retry-{i}.stderr').write_bytes(response.stderr)
   assert response.returncode==0,response.stderr.decode(errors='replace')
   from cow_cli import RequestJournal
   retries.append(completed_retry_receipt(RequestJournal(out/'cli-state'),operation,url));wait_idle(client)
  durable=client.operation(operation);after=probe.store.head(branch)
  witness.update(retry_same_receipt=retries[0]==retries[1] and durable.durable and durable.receipt.raw['snapshot']==target,
   retry_revision_delta=after.revision-head.revision,retry_native_trace_unchanged=trace.read_bytes()==trace_before)
  # A fresh CLI process/journal explicitly replays the saved answer; this is not a new stale-revision dispatch.
  replay=CowClient.http(url,journal=OperationJournal(out/'replay-journal'),api_key=probe.secret,endpoint_pin=pin)
  repeat_server=replay.transport.request('POST','/cow/publication/retry',json.dumps(dict(operation_id=operation)).encode()).json_value
  assert repeat_server['event']=='complete' and repeat_server['receipt']==retries[0]
  replay_command=list(cli_command);replay_command[replay_command.index('--state-dir')+1]=str(out/'replay-cli-state')
  replay_response=subprocess.run(replay_command+['retry','--from-operation',operation,'--operation-id',operation,
   '--branch',branch,'--temperature','0'],env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
  (out/'replay.stdout').write_bytes(replay_response.stdout);(out/'replay.stderr').write_bytes(replay_response.stderr)
  assert replay_response.returncode==0,replay_response.stderr.decode(errors='replace')
  replay_value=json.loads(replay_response.stdout);assert replay_value['receipt']==durable.receipt.raw
  answer=SimpleNamespace(answer=replay_value['answer'],receipt=SimpleNamespace(snapshot=replay_value['receipt']['snapshot']))
  final=completed_thinking_text(answer,profile['execution']['eos_tokens']);wait_idle(client)
  witness.update(replay_native_trace_unchanged=trace.read_bytes()==trace_before,final_text=final)
  target_cp=fault._checkpoint_details(a.store,target,fingerprints)
  expected_frontier=(control/'expected-frontier.f32').read_bytes()
  envelope=json.loads((checkpoint/(target+'.json')).read_bytes())['manifest']
  with (checkpoint/(envelope['payload_sha256']+'.state')).open('rb') as stream:stream.seek(envelope['state_bytes']);actual_frontier=stream.read()
  assert len(actual_frontier)==len(expected_frontier)==4*envelope['logits_count']
  av=struct.unpack('<'+'f'*envelope['logits_count'],actual_frontier);bv=struct.unpack('<'+'f'*envelope['logits_count'],expected_frontier)
  assert all(map(math.isfinite,av)) and all(map(math.isfinite,bv))
  (out/'retry-frontier.f32').write_bytes(actual_frontier)
  expected_answer=json.loads((control/'computed-answer.json').read_bytes())
  witness.update(retry_payload_matches_prepared=target_cp['payload_sha256']==target_expected['payload_sha256'] and target_cp['payload_bytes']==target_expected['payload_bytes'],
   tokens_exact=answer.answer['tokens']==expected_answer['tokens'],controller_exact=probe.store.get_snapshot(target)==json.loads((control/'expected-snapshot.json').read_bytes()),
   frontier_exact=actual_frontier==expected_frontier,max_abs_error=max(abs(x-y) for x,y in zip(av,bv)))
  probe.record.update(actual_saved_checkpoint=target_cp,numerical_max_error=witness['max_abs_error'])
  probe.record.update(retries=retries,replayed_answer=dict(answer.answer),retry_operation=dict(durable.raw));probe.publish()
  probe.phase('retained-image-root-still-usable')
  control_head=client.fork('kept',parent.tip,'control','vc08-control-fork')
  control_answer=client.answer('control',control_head.revision,'vc08-control-answer',1024,settings=dict(temperature=0.),grammar_state='VC08_REPLY')
  witness['retained_root_usable']=(completed_thinking_text(control_answer,profile['execution']['eos_tokens'])=='OK'
   and probe.store.head('kept')==parent and sha(payload)==payload_sha and probe.store.get_snapshot(parent.tip)==snapshot)
  witness['parent_payload_unchanged']=witness['parent_payload_unchanged'] and sha(payload)==payload_sha
  witness['parent_controller_unchanged']=witness['parent_controller_unchanged'] and probe.store.get_snapshot(parent.tip)==snapshot
  probe.record.update(control_answer=dict(control_answer.answer),fault_witness=witness)
  validate_campaign(witness);probe.record['status']='VC08_IMAGE_CHECKPOINT_ENOSPC_PASS'
 except BaseException as exc:
  error=exc;probe.record['status']='VC08_IMAGE_CHECKPOINT_ENOSPC_FAIL';probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  _atomic_json(control/'abort.json',dict(reason='campaign-finally'))
  if cli is not None and cli.poll() is None:
   os.killpg(cli.pid,signal.SIGTERM)
   try:cli.wait(10)
   except subprocess.TimeoutExpired:os.killpg(cli.pid,signal.SIGKILL);cli.wait(5)
  if filler is not None:
   try:disk_fault._remove_owned_filler(filler)
   except BaseException as exc:error=error or exc;probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  if probe.service is not None:
   try:probe.stop_service()
   except BaseException as exc:error=error or exc;probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  if error:probe.record['status']='VC08_IMAGE_CHECKPOINT_ENOSPC_FAIL'
  retained=out/'checkpoint-metadata';retained.mkdir(mode=0o700,exist_ok=True);files=[]
  for path in checkpoint.iterdir():
   if path.is_file() and not path.is_symlink():
    row=dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path));files.append(row)
    if path.suffix!='.state' and not path.name.startswith('.stage-') and path.name!='.q4-enospc-filler':shutil.copy2(path,retained/path.name)
  probe.record['checkpoint_files_before_unmount']=files
  probe.publish()
  if probe.tracker:
   probe.tracker.finish(probe.record['status'],None if error is None else type(error).__name__);probe.tracker.close()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
