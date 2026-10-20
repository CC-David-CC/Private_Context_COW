"""Real image CLI/HTTP denial effects, authorized shared input and recovery.

Two named namespaces share numerical data only after both upload grants exist.
References import no checkpoints. This is not multi-tenant authorization or
an every-token/controller qualification campaign.
"""
import argparse,copy,hashlib,io,json,os,shutil,subprocess,sys,time
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
class ProbeFailure(RuntimeError):pass

def require_rejection(status,value,expected,fragment):
 if status!=expected or not isinstance(value,dict) or not isinstance(value.get('error'),str) or fragment.casefold() not in value['error'].casefold():
  raise ProbeFailure('Response does not demonstrate the intended rejection')
def require_unchanged(before,after):
 if before!=after:raise ProbeFailure('Rejected request changed protected state or issued work')

def http_exchange(url,path,body,*,headers=None,key,header_only=False,method='POST'):
 parsed=urlsplit(url);connection=HTTPConnection(parsed.hostname,parsed.port,timeout=10)
 values={'Authorization':'Bearer '+key,'Content-Type':'application/octet-stream','Content-Length':str(len(body)),'X-COW-Content-SHA256':hashlib.sha256(body).hexdigest()};values.update(headers or {})
 begin=time.monotonic()
 try:
  if header_only:
   connection.putrequest(method,path)
   for k,v in values.items():connection.putheader(k,v)
   connection.endheaders()
  else:connection.request(method,path,body=body,headers=values)
  response=connection.getresponse();return response.status,json.loads(response.read()),time.monotonic()-begin
 finally:connection.close()

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('runtime','branchwrite','profile','dependency-root','vision-config','reference-build','image','output'):parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);here=Path(__file__).resolve().parent
 sys.path[:0]=list(map(str,[here,a.branchwrite.resolve(strict=True),runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from run_vision_pool_pressure import install_observer
 from run_thinking_suffix_flow import install_image_observer,reference_ingest
 from run_vision_latency import close_reference
 from run_vision_ingest_probe import Native,compare_files,fingerprint
 from context_time_travel.input_spans import history_for_snapshot
 from branchwrite.cow_client import CowClient,OperationJournal
 out=a.output.resolve();out.mkdir(mode=0o700,exist_ok=False)
 if shutil.disk_usage(out).free<1024**3:raise ProbeFailure('Need1GiB free for bounded two-root authorization fixture')
 profile=json.loads(a.profile.read_bytes());record=dict(schema=1,status='RUNNING',profile=fingerprint(a.profile),execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],scope=__doc__,rejections=[],controls={},services=[],comparisons=[],fresh_references=[],q8='DEFERRED_BY_OWNER',chat_server='OFF')
 def publish():_atomic_json(out/'result.json',record)
 class Owner(GrammarProbe):
  def publish(self):_atomic_json(self.run_dir/'service-status.json',self.record,self.secret)
  def _bootstrap(self,registry):
   path=super()._bootstrap(registry);body=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
   if body.count(marker)!=1:raise ProbeFailure('Ambiguous service entrypoint')
   body=body.replace(marker,'sys.argv.extend('+repr(['--vision-config',str(a.vision_config.resolve(strict=True))])+')\n'+marker)
   path.write_text(install_image_observer(install_observer(body,out),out),newline='\n');return path
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,dependency_root=a.dependency_root,ram_gib=96,disk_gib=2))
 registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
 def rows(name):
  path=out/name;return list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
 def job(op):
  found=[x for x in rows('jobs.jsonl') if x['operation']==op]
  if len(found)!=1:raise ProbeFailure('Missing/repeated native job '+op)
  return found[0]
 def request_http(path,body,*,headers=None,key=None,header_only=False,method='POST'):
  return http_exchange(probe.ready_url(probe.service.ready),path,body,headers=headers,key=probe.secret if key is None else key,header_only=header_only,method=method)
 def effects():
  head=probe.store.head('auth-A');snap=probe.store.get_snapshot(head.tip)
  inventory={}
  for pattern in ('objects/*.json','image-originals/*.input','image-upload-grants/*.json'):
   for path in (out/'store').glob(pattern):inventory[str(path.relative_to(out/'store'))]=hashlib.sha256(path.read_bytes()).hexdigest()
  return dict(head=dict(name=head.name,root=head.root,tip=head.tip,revision=head.revision),snapshot=copy.deepcopy(snap),jobs=len(rows('jobs.jsonl')),encoders=len(rows('encoder-entries.jsonl')),inventory=inventory)
 def deny(name,path,body,expected,fragment,**options):
  probe.phase(name);before=effects();status,value,elapsed=request_http(path,body,**options);require_rejection(status,value,expected,fragment);require_unchanged(before,effects())
  if options.get('header_only') and elapsed>=3:raise ProbeFailure('Header-only rejection waited for an absent body')
  record['rejections'].append(dict(id=name,path=path,status=status,response=value,seconds=elapsed,body_sent=not options.get('header_only',False),native_jobs_delta=0,encoder_calls_delta=0,protected_state_unchanged=True));publish()
 def cli(label,*args):
  row,_=probe.cli(label,'--machine-result',*args);return json.loads(row['stdout_text'])
 try:
  probe._imports();probe.runtime_closure(profile,runtime);probe.tracker=probe.ProgressTracker(out,'vision-authorization').start();probe.start_service(registry)
  pin=hashlib.sha256((str(out)+record['profile']['sha256']).encode()).hexdigest();client=CowClient.cli(probe.ready_url(probe.service.ready),runtime/'tools/cow_cli.py',state_dir=out/'client-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
  initial=client.status().capacity['workers'];native_pids=[x['pid'] for x in initial.values()]
  if len(native_pids)!=1:raise ProbeFailure('Expected one real native worker')
  probe.tracker.native_started(native_pids[0]);nsA=client.create_namespace('Authorized A','auth-ns-A')['namespace_id'];nsB=client.create_namespace('Authorized B','auth-ns-B')['namespace_id'];raw=a.image.read_bytes();upload=client.upload_image(nsA,raw)
  parts=out/'parts.json';_atomic_json(parts,[dict(type='image',image_id=upload.image_id),dict(type='text',text='Keep this image as shared context. A question follows.')])
  root=cli('auth-context-A','context','--branch','auth-A','--namespace-id',nsA,'--parts-file',str(parts),'--operation-id','auth-context-A');sid=root['receipt']['snapshot'];history=history_for_snapshot(probe.store,sid);rootjob=job('auth-context-A')
  if rootjob['forward_tokens']!=len(history):raise ProbeFailure('Positive control did not evaluate its real prefix')
  manifest=json.loads((out/'store/checkpoints'/(sid+'.json')).read_bytes())['manifest'];parent_payload=out/'store/checkpoints'/(manifest['payload_sha256']+'.state');parent_before=fingerprint(parent_payload)
  record['controls']['initial']=dict(result=root,cells=len(history),frontier=fingerprint(out/rootjob['frontier_file']),manifest=manifest,native_pid=native_pids[0],namespace=nsA);publish()
  uploadpath='/cow/images?namespace='+nsA
  deny('VA-001-wrong-key-before-body',uploadpath,raw,401,'API key',key='definitely-wrong-private-test-key',header_only=True)
  deny('VA-002-foreign-origin-before-body',uploadpath,raw,403,'Foreign Origin',headers={'Origin':'http://foreign.invalid'},header_only=True)
  deny('VA-003-unknown-namespace-before-body','/cow/images?namespace='+'f'*32,raw,404,'namespace',header_only=True)
  from PIL import Image
  stream=io.BytesIO();Image.new('RGB',(3,2),(17,29,193)).save(stream,format='PNG');other=stream.getvalue()
  deny('VA-004-digest-mismatch',uploadpath,other,400,'digest does not match',headers={'X-COW-Content-SHA256':hashlib.sha256(raw).hexdigest()})
  deny('VA-005-truncated-image',uploadpath,b'\x89PNG\r\n\x1a\n',400,'Invalid')
  document=dict(branch='auth-ungranted',namespace_id=nsB,operation_id='auth-ungranted',stream=False,messages=[dict(role='user',content=[dict(type='image',image_id=upload.image_id),dict(type='text',text='Keep this image as shared context. A question follows.')])])
  deny('VA-006-image-not-granted-to-namespace','/cow/text/context',json.dumps(document).encode(),409,'not uploaded in this namespace',headers={'Content-Type':'application/json'})
  document=dict(document,branch='auth-too-many',namespace_id=nsA,operation_id='auth-too-many',messages=[dict(role='user',content=[*[dict(type='image',image_id=upload.image_id) for _ in range(9)],dict(type='text',text='Keep these images.')])])
  deny('VA-007-nine-placements','/cow/text/context',json.dumps(document).encode(),409,'At most eight',headers={'Content-Type':'application/json'})
  # A new upload grant permits immutable data reuse; it does not reuse the
  # previous operation receipt or change either namespace's branch reference.
  second_upload=client.upload_image(nsB,raw)
  if second_upload.image_id!=upload.image_id:raise ProbeFailure('Authorized duplicate original identity differs')
  before_jobs=rows('jobs.jsonl');before_encoders=len(rows('encoder-entries.jsonl'))
  second=cli('auth-context-B','context','--branch','auth-B','--namespace-id',nsB,'--parts-file',str(parts),'--operation-id','auth-context-B');new_jobs=rows('jobs.jsonl')[len(before_jobs):]
  second_work=sum(x['forward_tokens'] for x in new_jobs)
  if second['receipt']['snapshot']!=sid or second_work!=0 or len(rows('encoder-entries.jsonl'))!=before_encoders:raise ProbeFailure('Authorized identical input did not reuse the prepared computation')
  code,headB,_=request_http('/cow/head?branch=auth-B',b'',method='GET')
  if code!=200 or headB['namespace_id']!=nsB or headB['tip']!=sid:raise ProbeFailure('Duplicate context reused wrong branch/namespace metadata')
  if probe.store.head('auth-A').tip!=sid:raise ProbeFailure('Second namespace moved first branch')
  record['controls']['authorized_duplicate']=dict(result=second,namespace=nsB,model_forward_tokens=second_work,encoder_calls=0,native_jobs=len(new_jobs),separate_namespace_head=headB);publish()
  answer=cli('auth-valid-question','ask','--branch','auth-B','--text','Consider the image briefly and finish with OK.','--max-tokens','1024','--temperature','0','--seed','17','--grammar-state','REPLY','--operation-id','auth-valid-question')
  tokens=answer['answer']['tokens'];final=history_for_snapshot(probe.store,answer['receipt']['snapshot']);suffix=list(final.native_tokens[len(history):len(final)-len(tokens)]);askjob=job('auth-valid-question')
  if askjob['forward_tokens']!=len(suffix)+len(tokens) or not answer['answer']['thinking']['complete'] or answer['answer']['thinking']['final_text']!='OK':raise ProbeFailure('Valid continuation failed after rejected requests')
  if fingerprint(parent_payload)!=parent_before or probe.store.head('auth-A').tip!=sid:raise ProbeFailure('Protected parent changed')
  if [x['pid'] for x in client.status().capacity['workers'].values()]!=native_pids:raise ProbeFailure('Rejected request replaced the native worker')
  final_manifest=json.loads((out/'store/checkpoints'/(answer['receipt']['snapshot']+'.json')).read_bytes())['manifest'];record['controls']['post_denial_continuation']=dict(result=answer,suffix=suffix,forward_tokens=askjob['forward_tokens'],frontier=fingerprint(out/askjob['frontier_file']),schedule=final_manifest['execution_schedule'],controller=probe.store.get_snapshot(answer['receipt']['snapshot'])['controller']);record['parent_payload_unchanged']=parent_before
  embeddings={}
  for span in history.spans:
   if span['kind']=='image':
    source=out/'store/checkpoints'/(span['embedding_sha256']+'.sve');dest=out/source.name;shutil.copy2(source,dest);embeddings[span['embedding_sha256']]=dest
  probe.stop_service();record['services']=probe.record['services'];publish()
  binary=json.loads(a.reference_build.read_bytes())['binaries']['reference']
  if fingerprint(binary['path'])['sha256']!=binary['sha256']:raise ProbeFailure('Fresh executable changed')
  command=list(profile['execution']['sut_command']);command[0]=binary['path'];native=Native(command,out/'reference');ref=dict(pid=native.proc.pid,binary=binary);record['fresh_references'].append(ref)
  try:
   if native.ready.get('reference')!='fresh-no-checkpoints':raise ProbeFailure('Reference imports state')
   packet=reference_ingest(history,0,embeddings,profile['execution']['capacity'],out,'reference-root');response=native.request(packet['command'])
   if response['chunks']!=manifest['execution_schedule'][0]['chunks']:raise ProbeFailure('Fresh prefix arrangement differs')
   path,_=native.frontier('prefix');check=dict(label='authorized-prefix',**compare_files(record['controls']['initial']['frontier']['path'],path));record['comparisons'].append(check)
   if not check['exact']:raise ProbeFailure('Fresh prefix differs')
   response=native.request('feed '+' '.join(map(str,suffix)));matches=[x for x in final_manifest['execution_schedule'] if x['position']==len(history) and x['count']==len(suffix)]
   if len(matches)!=1 or response['chunks']!=matches[0]['chunks']:raise ProbeFailure('Fresh suffix arrangement differs')
   for token in tokens:native.request('decode '+str(token))
   path,_=native.frontier('final');check=dict(label='post-denial-final',**compare_files(record['controls']['post_denial_continuation']['frontier']['path'],path));record['comparisons'].append(check)
   if not check['exact']:raise ProbeFailure('Fresh continuation differs')
   ref['work']=native.request('stats')
   if ref['work']['forward_tokens']!=len(final):raise ProbeFailure('Fresh reference reused prefix')
  finally:ref['exit']=close_reference(native);publish()
  if ref['exit']['exit_code']!=0 or not ref['exit']['process_absent']:raise ProbeFailure('Fresh reference did not retire')
  record.update(status='BOUNDED_Q4_REAL_HTTP_REJECTIONS_AUTHORIZED_REUSE_AND_RECOVERY_PASS',max_abs_error=0,limits=['One API key owns both namespaces; upload grants are not a multi-tenant security boundary.','Seven actual HTTP denials, three header-only without sending the promised body. Positive prefix/duplicate/continuation go through actual CLI. No mocked GPU.','Authorized duplicate input uses a different operation/branch/namespace but same snapshot and zero encoder/prefix forwards.','Fresh reference forces observed tokens, comparing full prefix/final vectors; not every token/sampler choice, general authorization, restart, Q8 or fullVC09/G qualification. Final OK constrained; image-answer quality unscored.'])
 except BaseException as error:record.update(status='FAIL',error=repr(error));raise
 finally:
  try:
   if probe.service is not None:probe.stop_service()
  except BaseException as error:record.update(status='FAIL',cleanup_error=repr(error))
  record['services']=probe.record['services']
  if probe.tracker is not None:probe.tracker.finish(record['status']);probe.tracker.close()
  publish()
 return 0
if __name__=='__main__':raise SystemExit(main())
