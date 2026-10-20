"""VC08 real-GPU artifact guards and successful image reconstruction.

The regenerated-mismatch arm is a declared test-only producer-output mutant.
It runs the actual GPU encoder, then alters one finite float. Other cases use
unchanged native encoding. This does not claim the normal encoder is defective.
"""
import argparse
import hashlib,json,math,os,shutil,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

class ProbeError(RuntimeError):pass

def validate_rejection(row,*,expected_encoder_calls):
 expected=dict(cli_exit=2,state='failed',durable=False,prefix_forward_tokens=0,
  parent_head_unchanged=True,canonical_record_unchanged=True,controller_unchanged=True,
  encoder_calls=expected_encoder_calls,integrity_error=True,expected_fault_observed=True,
  model_process_exited=True,server_process_exited=True)
 for key,value in expected.items():
  if key not in row or type(row[key]) is not type(value) or row[key]!=value:raise ProbeError('missing or failed rejection witness: '+key)

def work_accounting(total,prefix,generated):
 if any(type(value) is not int or value<0 for value in (total,prefix,generated)) or total!=prefix+generated:
  raise ProbeError('native forward total differs from prefix plus generated work')
 return dict(native_forward_tokens_total=total,prefix_forward_tokens=prefix,generated_forward_tokens=generated)

def mutate_regenerated_payload(raw):
 if len(raw)<24:raise ProbeError('encoder output is too short')
 before=struct.unpack('<f',raw[-4:])[0]
 if not math.isfinite(before):raise ProbeError('producer returned a nonfinite value before mutation')
 after=0.5 if before!=0.5 else -0.5
 changed=raw[:-4]+struct.pack('<f',after)
 return changed,dict(test_only_fault='one finite embedding float after actual GPU encoding',offset=len(raw)-4,
  before_sha256=hashlib.sha256(raw).hexdigest(),after_sha256=hashlib.sha256(changed).hexdigest(),
  before_value=before,after_value=after,finite_before=math.isfinite(before),finite_after=math.isfinite(after))

def mutate_position_manifest(raw,field):
 if field not in ('position_sha256','rotary_cursor'):raise ProbeError('unsupported position fault')
 envelope=json.loads(raw);value=envelope['manifest']
 value[field]='0'*64 if field=='position_sha256' else value[field]+1
 canonical=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')+b'\n'
 envelope['checksum']=hashlib.sha256(canonical).hexdigest()
 return (json.dumps(envelope,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['runtime','branchwrite','profile','dependency-root','vision-config','fixtures','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 from context_time_travel.store import Store
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False);profile=json.loads(a.profile.read_bytes())
 record=dict(schema=1,status='RUNNING',scope=__doc__,cases=[],failures=[],profile_sha256=sha(a.profile),
  execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],native_sha256=profile['native_build']['sha256'],
  harness_sha256=sha(Path(__file__)),numerical_max_error=None,numerical_oracle='NOT_RUN',chat_server='OFF',q8='DEFERRED_BY_OWNER')
 def publish():_atomic_json(out/'result.json',record)
 def make_probe(name,store):
  run=out/name;run.mkdir(mode=0o700)
  class Owner(image_probe_type(GrammarProbe,a.vision_config)):
   def _bootstrap(self,registry):
    path=super()._bootstrap(registry);body=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
    hook=r"""import importlib.util,os,time
from dataclasses import replace
from context_time_travel.store import IntegrityError
from context_time_travel.real import BranchingModel
import context_time_travel.vision_artifacts as va
from context_time_travel.vision_embeddings import parse_sve
from serve.cow_vision_encoder import CowNativeVisionEncoder
_fault_spec=importlib.util.spec_from_file_location('vc08_declared_fault',FAULT_MODULE)
_fault_module=importlib.util.module_from_spec(_fault_spec);_fault_spec.loader.exec_module(_fault_module)
def _observe(value):
 with Path(OBSERVATIONS).open('a') as stream:
  stream.write(json.dumps(dict(value,server_pid=os.getpid(),at_unix=time.time()),sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
def _guard(original,name):
 def checked(*args,**kwargs):
  try:return original(*args,**kwargs)
  except IntegrityError as exc:
   _observe(dict(kind='actual-integrity-guard',guard=name,error_type=type(exc).__name__,message=str(exc)));raise
 return checked
BranchingModel.checkpoint_manifest=_guard(BranchingModel.checkpoint_manifest,'checkpoint_manifest')
va.read_sve=_guard(va.read_sve,'read_sve')
va.ImageArtifacts._parse=_guard(va.ImageArtifacts._parse,'parse_regenerated')
_original_encode=CowNativeVisionEncoder.encode
def _encoded(self,*args,**kwargs):
 result=_original_encode(self,*args,**kwargs)
 row=dict(kind='actual-gpu-encoder-return',receipt=result.receipt,original_sha256=result.embedding.sha256,mutated=False)
 if Path(MUTATION_FLAG).exists():
  raw,witness=_fault_module.mutate_regenerated_payload(result.embedding.payload)
  _observe(dict(kind='declared-regeneration-mutant',**witness))
  result=replace(result,embedding=parse_sve(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),expected_width=self.width,max_rows=self.max_rows))
  row.update(mutated=True,returned_sha256=result.embedding.sha256)
 _observe(row);return result
CowNativeVisionEncoder.encode=_encoded
"""
    hook=hook.replace('FAULT_MODULE',repr(str(Path(__file__).resolve()))).replace('OBSERVATIONS',repr(str(run/'guards.jsonl'))).replace('MUTATION_FLAG',repr(str(run/'mutate-regeneration')))
    assert body.count(marker)==1;path.write_text(body.replace(marker,hook+marker),newline='\n');return path
  probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=store,run_dir=run,dependency_root=a.dependency_root,ram_gib=96,disk_gib=128))
  probe._imports();probe.runtime_closure(profile,runtime)
  registry=run/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(run,'VC08-'+name).start()
  try:probe.start_service(registry)
  except BaseException:
   if probe.service is not None and probe.service.process is not None:probe.service.stop()
   probe.tracker.finish('SETUP_FAIL');probe.tracker.close();raise
  url=probe.ready_url(probe.service.ready);pin=hashlib.sha256(str(run).encode()).hexdigest()
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=run/'cli-state',journal=OperationJournal(run/'journal'),api_key=probe.secret,endpoint_pin=pin)
  return probe,client
 def finish(probe,status):
  probe.record['status']=status;probe.stop_service();probe.publish();probe.tracker.finish(status);probe.tracker.close()
  service=probe.record['services'][-1];worker=service['worker_shutdown'];trace=probe.run_dir/'native-1.jsonl'
  rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
  closed=[row for row in rows if row.get('closed') is True]
  if len(closed)!=1:raise ProbeError('missing actual final native counter witness')
  return dict(server_pid=service['pid'],native_pid=worker['native_pid'],server_exit=service['stop']['exit_code'],native_exit=worker['native_exit_code'],
   server_process_exited=not Path('/proc',str(service['pid'])).exists(),model_process_exited=not Path('/proc',str(worker['native_pid'])).exists(),
   native_counters=closed[0]['counters'],native_trace_sha256=sha(trace))
 active=None;error=None
 try:
  assert profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  base=out/'base-store';active,client=make_probe('producer',base)
  ns=client.create_namespace('Artifact integrity','vc08-artifact-namespace')['namespace_id']
  picture=(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes();upload=client.upload_image(ns,picture)
  root=client.prepare('kept',ns,'Keep this page available. When asked, reply exactly OK.','vc08-retained-root',images=(upload,))
  parent=active.store.head('kept');snapshot=active.store.get_snapshot(parent.tip);history=active.store.get_snapshot(parent.tip)
  payload=base/'checkpoints'/(root.receipt.raw['payload_sha256']+'.state');payload_sha=sha(payload)
  native_manifest=json.loads((base/'checkpoints'/(parent.tip+'.json')).read_bytes())['manifest']
  with payload.open('rb') as stream:stream.seek(native_manifest['state_bytes']);saved_frontier_sha=hashlib.sha256(stream.read()).hexdigest()
  fork=client.fork('kept',parent.tip,'uninterrupted','vc08-reference-fork')
  reference=client.answer('uninterrupted',fork.revision,'vc08-reference-answer',1024,settings=dict(temperature=0.),grammar_state='VC08_REPLY')
  assert completed_thinking_text(reference,profile['execution']['eos_tokens'])=='OK'
  record['producer']=dict(receipt=dict(root.receipt.raw),payload_sha256=payload_sha,saved_frontier_sha256=saved_frontier_sha,
   reference_answer=dict(reference.answer),lifecycle=finish(active,'PRODUCER_PASS'));active=None;publish()
  modes=['cached-embedding-corrupt','position-digest-corrupt','rotary-cursor-corrupt','regenerated-embedding-mismatch','reencoded-control']
  for index,mode in enumerate(modes):
   store=out/('store-'+mode);shutil.copytree(base,store)
   cp=store/'checkpoints';changes=[]
   if mode in ('cached-embedding-corrupt','regenerated-embedding-mismatch','reencoded-control'):
    suffixes=('.json','.state') if mode=='cached-embedding-corrupt' else ('.json','.state','.sve')
    for path in cp.iterdir():
     if path.is_file() and path.suffix in suffixes:
      changes.append(dict(action='evict-derived',path=str(path.relative_to(store)),bytes=path.stat().st_size,sha256=sha(path)));path.unlink()
   if mode=='cached-embedding-corrupt':
    path=next(cp.glob('*.sve'));before=path.read_bytes();path.write_bytes(before[:-1]+bytes([before[-1]^1]))
    changes.append(dict(action='one-byte-disk-corruption',path=str(path.relative_to(store)),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=sha(path)))
   if mode in ('position-digest-corrupt','rotary-cursor-corrupt'):
    path=cp/(parent.tip+'.json');before=path.read_bytes();field='position_sha256' if mode=='position-digest-corrupt' else 'rotary_cursor';path.write_bytes(mutate_position_manifest(before,field))
    changes.append(dict(action='position-field-with-valid-envelope-checksum',field=field,path=str(path.relative_to(store)),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=sha(path)))
   record['cases'].append(dict(case=mode,status='RUNNING',derived_changes=changes));row=record['cases'][-1];publish()
   active,client=make_probe(mode,store)
   if mode=='regenerated-embedding-mismatch':(active.run_dir/'mutate-regeneration').write_text('test-only finite-float output fault after actual GPU encoding\n')
   op='vc08-'+str(index)+'-answer';url=active.ready_url(active.service.ready)
   cmd=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',url,'--timeout','180','--state-dir',str(active.run_dir/'cli-state'),
    '--machine-result','answer','--branch','kept','--revision',str(parent.revision),'--operation-id',op,'--max-tokens','1024','--temperature','0','--grammar-state','VC08_REPLY']
   active.phase(mode+'-actual-cli-answer')
   answer=subprocess.run(cmd,env={**os.environ,'COW_API_KEY':active.secret},stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
   (active.run_dir/'answer.stdout').write_bytes(answer.stdout);(active.run_dir/'answer.stderr').write_bytes(answer.stderr)
   observation=client.operation(op);source_intact=active.store.get_snapshot(parent.tip)==snapshot
   head_intact=active.store.head('kept')==parent
   guards=[json.loads(x) for x in (active.run_dir/'guards.jsonl').read_text().splitlines()] if (active.run_dir/'guards.jsonl').exists() else []
   life=finish(active,'CASE_RETURNED');case_dir=active.run_dir;active=None
   row.update(cli_exit=answer.returncode,state=observation.state,durable=observation.durable,operation=dict(observation.raw),
    parent_head_unchanged=head_intact,canonical_record_unchanged=source_intact,controller_unchanged=Store(store).get_snapshot(parent.tip)['controller']==snapshot['controller'],
    encoder_calls=sum(g['kind']=='actual-gpu-encoder-return' for g in guards),guard_observations=guards,
    prefix_forward_tokens=life['native_counters']['forward_tokens'],native_forward_tokens_total=life['native_counters']['forward_tokens'],lifecycle=life,
    model_process_exited=life['model_process_exited'],server_process_exited=life['server_process_exited'])
   if mode!='reencoded-control':
    expected_guard='read_sve' if mode=='cached-embedding-corrupt' else 'parse_regenerated' if mode=='regenerated-embedding-mismatch' else 'checkpoint_manifest'
    expected_message='Embedding payload digest mismatch' if mode in ('cached-embedding-corrupt','regenerated-embedding-mismatch') else 'Checkpoint manifest incompatible'
    errors=[g for g in guards if g['kind']=='actual-integrity-guard' and g['guard']==expected_guard]
    row.update(integrity_error=any(g['error_type']=='IntegrityError' for g in errors),expected_fault_observed=any(g['message']==expected_message for g in errors))
    validate_rejection(row,expected_encoder_calls=1 if mode=='regenerated-embedding-mismatch' else 0)
    if mode=='regenerated-embedding-mismatch':assert not list(cp.glob('*.sve'))
   else:
    result=json.loads(answer.stdout);complete=SimpleNamespace(answer=result['answer'],receipt=SimpleNamespace(snapshot=result['receipt']['snapshot']))
    assert answer.returncode==0 and observation.durable and completed_thinking_text(complete,profile['execution']['eos_tokens'])=='OK'
    work=result['answer']['source_recomputation'];row['recomputation']=work
    assert work['prefix_forward_tokens']==native_manifest['consumed'] and work['input_preparation']['encoder_calls']==1
    assert work['frontier_sha256']==saved_frontier_sha
    assert result['answer']['tokens']==reference.answer['tokens']
    assert row['encoder_calls']==1 and source_intact and row['controller_unchanged']
    assert life['native_counters']['forward_tokens']==native_manifest['consumed']+len(result['answer']['tokens'])
    row.update(work_accounting(life['native_counters']['forward_tokens'],work['prefix_forward_tokens'],len(result['answer']['tokens'])))
    row.update(saved_frontier_exact=True,reference_tokens_exact=True,generated_tokens=len(result['answer']['tokens']))
   assert sha(payload)==payload_sha and Store(base).head('kept')==parent and Store(base).get_snapshot(parent.tip)==snapshot
   row['status']='PASS';publish()
  record['status']='VC08_ARTIFACT_REJECTION_AND_RECONSTRUCTION_PASS'
 except BaseException as exc:
  error=exc;record['status']='VC08_ARTIFACT_REJECTION_AND_RECONSTRUCTION_FAIL';record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if active is not None:
   try:finish(active,'CAMPAIGN_ABORTED')
   except BaseException as exc:error=error or exc;record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  publish()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
