"""VC08 actual image CLI publication failure on a bounded, private ext4 volume.

The gate observes the real writer; it does not replace it or synthesize errno.
This is a process/filesystem fault experiment, not a fresh-logit qualification.
"""
import argparse
import hashlib,json,os
from pathlib import Path
import signal,subprocess,sys,time
from types import SimpleNamespace

class CampaignError(RuntimeError):pass

def validate_campaign(row):
 expected=dict(cli_exit=2,state='failed',durable=False,publication_retryable=True,
  errno=28,old_reference_unchanged=True,parent_payload_unchanged=True,
  parent_controller_unchanged=True,prepared_checkpoint_unchanged=True,
  stage_files=[],available_bytes=0,retry_same_receipt=True,retry_revision_delta=1,
  retry_native_trace_unchanged=True,replay_native_trace_unchanged=True,
  final_text='OK',retained_root_usable=True)
 if not isinstance(row,dict) or not isinstance(row.get('operation_id'),str) or not row['operation_id']:
  raise CampaignError('missing original operation identity')
 for key,value in expected.items():
  if key not in row or type(row[key]) is not type(value) or row[key]!=value:
   raise CampaignError('missing or failed witness: '+key)

def install_writer_hook(body,control,branch,operation,store):
 from run_q4_canonical_enospc import _reference_wrapper_source
 marker="importlib.import_module('serve.cow_http').main()\n"
 if body.count(marker)!=1:raise CampaignError('expected one actual HTTP entrypoint')
 hook=_reference_wrapper_source(control,branch,operation,store)
 footer='sys.argv=["serve.cow_http",*sys.argv[1:]]\nfrom serve.cow_http import main\nmain()\n'
 if not hook.endswith(footer):raise CampaignError('writer observer entrypoint changed')
 return body.replace(marker,hook[:-len(footer)]+marker)

def bind_filler_receipt(raw,mount):
 from run_q4_canonical_enospc import validate_filler_receipt
 device=mount.stat().st_dev
 if raw.get('st_dev')!=device:raise CampaignError('filler device differs from the actual canonical mount')
 receipt=dict(raw,canonical_st_dev=device,schema='vision-canonical-enospc-filler-v1')
 validate_filler_receipt(receipt,canonical_mount=str(mount),max_fill_bytes=64*1024**2,canonical_st_dev=device)
 return receipt

def completed_retry_receipt(journal,operation,url):
 record=journal.get(operation)
 if not isinstance(record,dict) or record.get('state')!='complete' or record.get('url')!=url or record.get('operation_id')!=operation:
  raise CampaignError('CLI retry has no completed original-operation journal receipt')
 receipt=record.get('receipt')
 if not isinstance(receipt,dict):raise CampaignError('CLI journal is missing its durable receipt')
 return receipt

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def wait_idle(client):
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  m=client.status().capacity
  if not m['queued'] and not m['pending_branches'] and not any(m['active_per_gpu'].values()):return m
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
 import run_q4_publication_fault as fingerprints
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 if os.geteuid()==0:raise CampaignError('inference must run as the ordinary project user')
 if a.loop_image.stat().st_size!=64*1024**2:raise CampaignError('fixture is not the exact bounded 64MiB image')
 layout=fault.live_mount_receipt(SimpleNamespace(canonical_mount=a.store,checkpoint_directory=a.store/'checkpoints',
  parent_mount_namespace=a.parent_mount_namespace,loop_image=a.loop_image,
  expected_image_inode=a.expected_image_inode,expected_image_uuid=a.expected_image_uuid))
 if a.preflight_only:
  print(json.dumps(dict(status='PREPARED_NOT_RUN',mount=layout,inference_uid=os.geteuid()),sort_keys=True));return 0
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False);control=out/'control';control.mkdir(mode=0o700)
 if out.stat().st_dev==a.store.stat().st_dev:raise CampaignError('evidence must be outside the full filesystem')
 operation='vc08-image-enospc-original';branch='fault'
 class Owner(image_probe_type(GrammarProbe,a.vision_config)):
  def _bootstrap(self,registry):
   p=super()._bootstrap(registry)
   p.write_text(install_writer_hook(p.read_text(),control,branch,operation,a.store),newline='\n');return p
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=a.store,run_dir=out,
  dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
 probe.record.update(status='RUNNING',scope=__doc__,mount=layout,numerical_oracle='NOT_RUN',numerical_max_error=None,
  chat_server='OFF',q8='DEFERRED_BY_OWNER',harness_sha256=sha(Path(__file__)))
 error=None;cli=None;filler=None
 try:
  probe._imports();profile=json.loads(a.profile.read_bytes());probe.runtime_closure(profile,runtime)
  assert profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  probe.record.update(execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
   profile_sha256=sha(a.profile),native_sha256=profile['native_build']['sha256'])
  registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(
   schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(out,'VC08-image-canonical-enospc').start();probe.start_service(registry)
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
  gate=fault._wait_gate(control/'writer-events.jsonl',Pending(),180,probe.tracker)
  assert gate['operation_id']==operation and gate['branch']==branch and gate['old_reference_sha256']==hashlib.sha256(old_ref).hexdigest()
  target=gate['target_snapshot'];target_cp=fault._checkpoint_details(a.store,target,fingerprints)
  target_object=fingerprints._snapshot_record_fingerprint(a.store,target)
  probe.record.update(writer_gate=gate,prepared_checkpoint=target_cp,parent_payload_sha256=payload_sha,root_receipt=dict(root.receipt.raw));probe.publish()
  probe.phase('fill-bounded-ext4-to-kernel-enospc')
  filler=fault._fill_to_enospc(a.store,64*1024**2,'vision-v1')
  _atomic_json(out/'filler-raw.json',filler)
  filler=bind_filler_receipt(filler,a.store)
  def release():
   fault._require_request_waiting_at_release(Pending())
   _atomic_json(control/'release.json',dict(operation_id=operation))
  fault.persist_receipt_before_release(out/'filler-receipt.json',filler,release,canonical_st_dev=a.store.stat().st_dev)
  code=cli.wait(90);cli=None;wait_idle(client)
  observation=client.operation(operation)
  rows=fault._events(control/'writer-events.jsonl');failures=[x for x in rows if x['kind']=='branch-reference-write-error']
  assert len(failures)==1
  fault.validate_reference_fault(failures[0],operation_id=operation,branch=branch,reference_path=str(ref.resolve()),
   old_reference_sha256=hashlib.sha256(old_ref).hexdigest(),target_snapshot=target)
  witness=dict(cli_exit=code,operation_id=operation,state=observation.state,durable=observation.durable,
   publication_retryable=observation.raw.get('publication_retryable'),errno=failures[0]['errno'],
   old_reference_unchanged=ref.read_bytes()==old_ref and probe.store.head(branch)==head,
   parent_payload_unchanged=sha(payload)==payload_sha,parent_controller_unchanged=probe.store.get_snapshot(parent.tip)==snapshot,
   prepared_checkpoint_unchanged=fault._checkpoint_details(a.store,target,fingerprints)==target_cp and fingerprints._snapshot_record_fingerprint(a.store,target)==target_object,
   stage_files=failures[0]['stage_files_after'],available_bytes=filler['available_bytes_after'])
  probe.record.update(failure_operation=dict(observation.raw),fault_witness=witness,writer_failure=failures[0]);probe.publish()
  fault._remove_owned_filler(filler,a.store);filler=None
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
  assert fault._checkpoint_details(a.store,target,fingerprints)==target_cp
  probe.record.update(retries=retries,replayed_answer=dict(answer.answer),retry_operation=dict(durable.raw));probe.publish()
  probe.phase('retained-image-root-still-usable')
  control_head=client.fork('kept',parent.tip,'control','vc08-control-fork')
  control_answer=client.answer('control',control_head.revision,'vc08-control-answer',1024,settings=dict(temperature=0.),grammar_state='VC08_REPLY')
  witness['retained_root_usable']=(completed_thinking_text(control_answer,profile['execution']['eos_tokens'])=='OK'
   and probe.store.head('kept')==parent and sha(payload)==payload_sha and probe.store.get_snapshot(parent.tip)==snapshot)
  witness['parent_payload_unchanged']=witness['parent_payload_unchanged'] and sha(payload)==payload_sha
  witness['parent_controller_unchanged']=witness['parent_controller_unchanged'] and probe.store.get_snapshot(parent.tip)==snapshot
  probe.record.update(control_answer=dict(control_answer.answer),fault_witness=witness)
  validate_campaign(witness);probe.record['status']='VC08_IMAGE_CANONICAL_ENOSPC_PASS'
 except BaseException as exc:
  error=exc;probe.record['status']='VC08_IMAGE_CANONICAL_ENOSPC_FAIL';probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  _atomic_json(control/'abort.json',dict(reason='campaign-finally'))
  if cli is not None and cli.poll() is None:
   os.killpg(cli.pid,signal.SIGTERM)
   try:cli.wait(10)
   except subprocess.TimeoutExpired:os.killpg(cli.pid,signal.SIGKILL);cli.wait(5)
  if filler is not None:
   try:fault._remove_owned_filler(filler,a.store)
   except BaseException as exc:error=error or exc;probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  if probe.service is not None:
   try:probe.stop_service()
   except BaseException as exc:error=error or exc;probe.record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  if error:probe.record['status']='VC08_IMAGE_CANONICAL_ENOSPC_FAIL'
  probe.publish()
  if probe.tracker:
   probe.tracker.finish(probe.record['status'],None if error is None else type(error).__name__);probe.tracker.close()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
