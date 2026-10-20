"""Actual CLI/HTTP/GPU PNG/JPEG orientation and ordered-image matrix.

Final OK is constrained; this checks identity, computation and restoration
mechanics, not semantic image-answer quality. References force observed tokens.
"""
import argparse,hashlib,json,os,shutil,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('runtime','branchwrite','profile','dependency-root','vision-config','reference-build','output'):parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True);here=Path(__file__).resolve().parent
 sys.path[:0]=list(map(str,[here,bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from run_vision_pool_pressure import install_observer
 from run_thinking_suffix_flow import install_image_observer,reference_ingest
 from run_vision_latency import close_reference
 from run_vision_ingest_probe import Native,compare_files,fingerprint
 from context_time_travel.input_spans import history_for_snapshot
 from branchwrite.cow_client import CowClient,OperationJournal
 from serve.cow_vision import CowImageNormalizer
 from vision_format_fixture import prepare,check_normalized,case_plan
 out=a.output.resolve();out.mkdir(mode=0o700,exist_ok=False)
 if shutil.disk_usage(out).free<2500*1024**2:raise RuntimeError('Need2500MiB free before format campaign')
 (out/'embeddings').mkdir(mode=0o700);fixtures=prepare(out/'fixtures');plan=case_plan(fixtures);by_name={x['name']:x for x in fixtures};profile=json.loads(a.profile.read_bytes())
 record=dict(schema=1,status='RUNNING',scope=__doc__,profile=fingerprint(a.profile),execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],normalization=[],cases=[],comparisons=[],fresh_references=[],rejections=[],services=[],q8='DEFERRED_BY_OWNER',chat_server='OFF',harness=fingerprint(__file__))
 def publish():_atomic_json(out/'result.json',record)
 class Owner(GrammarProbe):
  def publish(self):_atomic_json(self.run_dir/'service-status.json',self.record,self.secret)
  def _bootstrap(self,registry):
   path=super()._bootstrap(registry);s=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
   if s.count(marker)!=1:raise RuntimeError('Ambiguous service entrypoint')
   s=s.replace(marker,'sys.argv.extend('+repr(['--vision-config',str(a.vision_config.resolve(strict=True))])+')\n'+marker)
   path.write_text(install_image_observer(install_observer(s,out),out),newline='\n');return path
 probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,dependency_root=a.dependency_root,ram_gib=96,disk_gib=2))
 registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
 def jobs():
  file=out/'jobs.jsonl';return list(map(json.loads,file.read_text().splitlines())) if file.exists() else []
 def job(op):
  rows=[x for x in jobs() if x['operation']==op]
  if len(rows)!=1:raise RuntimeError('Missing/extra native job '+op)
  return rows[0]
 def checkpoint(sid):return json.loads((out/'store/checkpoints'/(sid+'.json')).read_bytes())['manifest']
 client=None;embeddings={}
 try:
  probe._imports();probe.runtime_closure(profile,runtime);probe.tracker=probe.ProgressTracker(out,'vision-format-matrix').start();probe.start_service(registry)
  url=probe.ready_url(probe.service.ready);pin=hashlib.sha256((str(out)+record['profile']['sha256']).encode()).hexdigest()
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=out/'cli-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
  for worker in client.status().capacity['workers'].values():probe.tracker.native_started(worker['pid'])
  namespace=client.create_namespace('Format and orientation matrix','format-namespace')['namespace_id'];uploads={};normalizer=CowImageNormalizer()
  for row in fixtures:
   raw=Path(row['original']).read_bytes();normalized=normalizer.normalize(raw);check=check_normalized(row,normalized.payload)
   uploaded=client.upload_image(namespace,raw);uploads[row['name']]=uploaded.image_id
   if uploaded.image_id!=row['original_sha256']:raise RuntimeError('Upload identity differs')
   record['normalization'].append(dict(name=row['name'],original_sha256=row['original_sha256'],normalized_sha256=normalized.normalized_sha256,**check));publish()
  for case in plan:
   probe.phase('case-'+case['name']);name=case['name'];rootop='format-root-'+name;askop='format-ask-'+name
   parts=out/(name+'-parts.json');_atomic_json(parts,[*[dict(type='image',image_id=uploads[n]) for n in case['images']],dict(type='text',text='Keep these ordered images as shared context. A question follows.')])
   row,_=probe.cli(rootop,'--machine-result','context','--branch',name,'--namespace-id',namespace,'--parts-file',str(parts),'--operation-id',rootop);root=json.loads(row['stdout_text']);sid=root['receipt']['snapshot'];history=history_for_snapshot(probe.store,sid)
   spans=[s for s in history.spans if s['kind']=='image'];expected=[by_name[n]['original_sha256'] for n in case['images']]
   if [s['original_sha256'] for s in spans]!=expected:raise RuntimeError('Image ordering/identity differs')
   for n,span in zip(case['images'],spans):
    norm=next(x for x in record['normalization'] if x['name']==n)
    if span['normalized_sha256']!=norm['normalized_sha256']:raise RuntimeError('Server normalized identity differs from independently checked pixels')
    source=out/'store/checkpoints'/(span['embedding_sha256']+'.sve');destination=out/'embeddings'/source.name
    if not destination.exists():shutil.copy2(source,destination)
    if fingerprint(destination)['sha256']!=span['embedding_sha256']:raise RuntimeError('Accepted embedding differs')
    embeddings[span['embedding_sha256']]=destination
   rootjob=job(rootop);manifest=checkpoint(sid);parent=probe.store.get_snapshot(sid);head=probe.store.head(name);payload=out/'store/checkpoints'/(manifest['payload_sha256']+'.state');parent_hash=fingerprint(payload)
   if rootjob['forward_tokens']!=len(history):raise RuntimeError('Root prefix reused or omitted')
   question='Consider the colors and arrangement briefly. Finish with OK.'
   row,_=probe.cli(askop,'--machine-result','ask','--branch',name,'--revision',str(head.revision),'--text',question,'--max-tokens','1024','--temperature','0','--seed','17','--grammar-state','REPLY','--operation-id',askop);answer=json.loads(row['stdout_text']);final=history_for_snapshot(probe.store,answer['receipt']['snapshot']);tokens=answer['answer']['tokens'];suffix=list(final.native_tokens[len(history):len(final)-len(tokens)]);aj=job(askop)
   if (not answer['answer']['thinking']['complete'] or answer['answer']['thinking']['final_text']!='OK' or not tokens or tokens[-1] not in profile['execution']['eos_tokens']):raise RuntimeError('Incomplete consumed thinking/final answer')
   if aj['forward_tokens']!=len(suffix)+len(tokens):raise RuntimeError('Question replayed prefix or omitted output')
   # Optional disk copies may legitimately expire once the source lease ends.
   # Never call absence corruption or count it as a completed byte audit.
   try:parent_after=fingerprint(payload)
   except FileNotFoundError:parent_after=None
   if parent_after is not None and parent_after!=parent_hash:raise RuntimeError('Parent checkpoint bytes changed')
   if probe.store.get_snapshot(sid)!=parent:raise RuntimeError('Parent immutable history changed')
   record['cases'].append(dict(name=name,images=case['images'],originals=expected,root_snapshot=sid,final_snapshot=answer['receipt']['snapshot'],prefix_cells=len(history),suffix=suffix,tokens=tokens,root_frontier=fingerprint(out/rootjob['frontier_file']),final_frontier=fingerprint(out/aj['frontier_file']),root_schedule=manifest['execution_schedule'],final_schedule=checkpoint(answer['receipt']['snapshot'])['execution_schedule'],root_spans=history.spans,root_work=rootjob['forward_tokens'],question_work=aj['forward_tokens'],controller=probe.store.get_snapshot(answer['receipt']['snapshot'])['controller'],parent_payload=parent_hash,parent_payload_after=parent_after,answer=answer['answer']));publish()
  if len(record['cases'])!=18:raise RuntimeError('Missing format cases')
  probe.stop_service();record['services']=probe.record['services'];publish()
  build=json.loads(a.reference_build.read_bytes());binary=build['binaries']['reference']
  if fingerprint(binary['path'])['sha256']!=binary['sha256']:raise RuntimeError('Reference executable changed')
  for case in record['cases']:
   probe.phase('fresh-'+case['name']);command=list(profile['execution']['sut_command']);command[0]=binary['path'];native=Native(command,out/('reference-'+case['name']));reference=dict(name=case['name'],pid=native.proc.pid,binary=binary,forced_observed_tokens=True);record['fresh_references'].append(reference);publish()
   try:
    if native.ready.get('reference')!='fresh-no-checkpoints':raise RuntimeError('Reference imports checkpoints')
    history=history_for_snapshot(probe.store,case['root_snapshot']);packet=reference_ingest(history,0,embeddings,profile['execution']['capacity'],out,'reference-'+case['name'])
    response=native.request(packet['command'])
    if response['chunks']!=case['root_schedule'][0]['chunks']:raise RuntimeError('Reference prefix arrangement differs')
    path,_=native.frontier('prefix');comparison=dict(label=case['name']+'-prefix',**compare_files(case['root_frontier']['path'],path));record['comparisons'].append(comparison)
    if not comparison['exact']:raise RuntimeError('Fresh prefix mismatch')
    response=native.request('feed '+' '.join(map(str,case['suffix'])));matching=[x for x in case['final_schedule'] if x['position']==case['prefix_cells'] and x['count']==len(case['suffix'])]
    if len(matching)!=1 or response['chunks']!=matching[0]['chunks']:raise RuntimeError('Reference suffix arrangement differs')
    for token in case['tokens']:native.request('decode '+str(token))
    path,_=native.frontier('final');comparison=dict(label=case['name']+'-final',**compare_files(case['final_frontier']['path'],path));record['comparisons'].append(comparison)
    if not comparison['exact']:raise RuntimeError('Fresh final mismatch')
    reference['work']=native.request('stats')
    if reference['work']['forward_tokens']!=case['prefix_cells']+len(case['suffix'])+len(case['tokens']):raise RuntimeError('Fresh reference omitted prefix work')
   finally:reference['exit']=close_reference(native);publish()
   if reference['exit']['exit_code']!=0 or not reference['exit']['process_absent']:raise RuntimeError('Reference did not retire')
  record.update(status='BOUNDED_Q4_PNG_JPEG_ORIENTATION_AND_IMAGE_ORDER_PASS',max_abs_error=0,limits=['Real CLI/HTTP/GPU18 cases:PNG/JPEG EXIF1..8 and two ordered-image controls. Independent pixel permutations and raw checkpoint-free prefix/final comparisons; forced observed tokens, not every generated frontier or semantic answer scoring. No restart/authorization/pressure/performance/Q8/fullVC09/G qualification in this matrix.'])
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
