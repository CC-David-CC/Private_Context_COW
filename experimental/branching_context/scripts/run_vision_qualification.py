"""VC09 matching-profile numerical/restart campaign, separate from performance.

Keeps the existing source-bound profile unchanged. Raw references use the
separate checkpoint-free driver with independently supplied rotary positions.
The API consumes generated tokens under resident request leases, captures after
four consumed tokens and at completion. Tiny/medium retain every generated
frontier; target retains its declared sampled frontiers. The coverage policy is
measurement metadata, not a different model operator or checkpoint format.
"""
import argparse,hashlib,json,math,os,shutil,struct,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path

class QualificationError(RuntimeError):pass

def sample_points(count,*,policy='sampled'):
 if type(count) is not int or count<=0:raise QualificationError('invalid actual output count')
 if policy not in ('sampled','every-generated'):raise QualificationError('unknown frontier coverage policy')
 if policy=='every-generated':return list(range(1,count+1))
 return sorted({n for n in [1,4,16,64,count] if n<=count})

def validate_fixture_counts(row):
 for key in ['text_control_tokens','image_rows','total_cells','suffix_tokens','outputs']:
  if type(row.get(key)) is not int or row[key]<=0:raise QualificationError('invalid fixture extent '+key)
 if row['text_control_tokens']+row['image_rows']!=row['total_cells'] or len(row['input_tokens'])!=row['total_cells']:
  raise QualificationError('text and image extents disagree')
 if set(row['suffixes'])!={'A','B'} or any(len(v)!=row['suffix_tokens'] for v in row['suffixes'].values()) or row['suffixes']['A']==row['suffixes']['B']:
  raise QualificationError('forced suffix identities/counts disagree')

def validate_restore(row,*,consumed,producer_pids):
 for key in ['forward_tokens','consumed','bytes_loaded','native_pid','python_pid']:
  if type(row.get(key)) is not int:raise QualificationError('missing actual restore counter '+key)
 if row['forward_tokens']!=0 or row['consumed']!=consumed or row['bytes_loaded']<=0 or row['native_pid'] in producer_pids or row['python_pid'] in producer_pids:
  raise QualificationError('restore performed work, lost position, or reused producer')

def validate_generation(row,*,limit,policy='sampled'):
 n=len(row.get('tokens',[]))
 if not 0<n<=limit or type(row.get('forward_tokens')) is not int or row['forward_tokens']!=n or type(row.get('prefill_tokens')) is not int or row['prefill_tokens']!=0:
  raise QualificationError('generated-token native work differs')
 if not set(map(str,sample_points(n,policy=policy)))<=set(row.get('frontiers',{})):raise QualificationError('missing sampled or actual final frontier')

def same_batches(a,b):
 if not isinstance(a,list) or not isinstance(b,list) or not a or a!=b:raise QualificationError('native batch schedules differ')
 if any(not isinstance(x,list) or len(x)!=2 or any(type(v) is not int for v in x) or x[0]<0 or x[1]<=0 for x in a):raise QualificationError('invalid native chunks')

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
 temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',newline='\n');temporary.replace(path)

def worker(config,stage,branch,out):
 from run_vision_ingest_probe import Native
 from vision_fixture import fixture_parts,boundary_coverage
 from context_time_travel.real import BranchingModel
 from context_time_travel.strata_backend import StrataBackend
 from context_time_travel.pools import SharedCheckpointPool,DiskCheckpointPool
 from context_time_travel.vision_artifacts import ImageArtifacts
 profile=json.loads(Path(config['profile']).read_bytes())
 if sha(config['profile'])!=config['profile_sha256']:raise QualificationError('profile changed')
 record=dict(status='RUNNING',stage=stage,python_pid=os.getpid(),profile_sha256=config['profile_sha256'],execution_profile_id=profile['execution_profile_id'])
 backend=ram=None
 def frontier(values,name):
  raw=struct.pack('<%df'%len(values),*values);p=out/(name+'.f32');p.write_bytes(raw);return str(p)
 try:
  if stage=='reference':
   fixture=json.loads(Path(config['fixture']).read_bytes());argv=list(profile['execution']['sut_command']);argv[0]=config['reference_binary']
   native=Native(argv,out/'native');record['native_pid']=native.proc.pid
   try:
    if native.ready.get('reference')!='fresh-no-checkpoints':raise QualificationError('not independent checkpoint-free driver')
    prefix=native.request('feed_image '+json.dumps(str(Path(config['fixture']).parent/'input.manifest'))+' '+json.dumps(str(Path(config['fixture']).parent/'reference-positions.i32')))
    prefix_frontier,_=native.frontier('prefix');suffix=native.request('feed '+' '.join(map(str,fixture['suffixes'][branch])))
    suffix_frontier,values=native.frontier('suffix');tokens=[];frontiers={}
    for k in range(1,config['outputs']+1):
     token=max(range(len(values)),key=values.__getitem__);tokens.append(token)
     consumed=native.request('decode '+str(token));path,values=native.frontier('scratch')
     if consumed['count']!=1 or consumed['chunks']!=[[consumed['position'],1]]:raise QualificationError('reference did not decode one consumed token')
     if k in sample_points(config['outputs'],policy=config['frontier_policy']) or token in profile['execution']['eos_tokens']:
      kept=out/f'output-{k}.f32';shutil.copy2(path,kept);frontiers[str(k)]=str(kept)
     if token in profile['execution']['eos_tokens']:break
    stats=native.request('stats')
    if stats['forward_tokens']!=fixture['total_cells']+fixture['suffix_tokens']+len(tokens):raise QualificationError('fresh reference reused or omitted prefix work')
    record.update(tokens=tokens,frontiers=frontiers,prefix_frontier=str(prefix_frontier),suffix_frontier=str(suffix_frontier),prefix_chunks=prefix['chunks'],suffix_chunks=suffix['chunks'],work=stats)
   finally:record['native_exit']=native.close()
  else:
   os.environ['CTT_NATIVE_TRACE']=str(out/'native.jsonl');backend=StrataBackend(profile,gpu_id=0);api=BranchingModel(config['store'],backend,profile);record['native_pid']=backend.process.pid
   def cp_frontier(sid,name):return frontier(api.cache[sid].frontier,name)
   def audit(image):
    x=backend.audit(image);return {k:x[k] for k in ['bytes','consumed','sha256']}
   def generate(sid,name,limit,*,offset=0,resume=False):
    before=backend.work_counters();tokens=[];frontiers={};original=backend.consume_generated;points=set(sample_points(config['outputs'],policy=config['frontier_policy']))
    def consumed(lease,ids):
     result=original(lease,ids);tokens.extend(ids);count=offset+len(tokens)
     if count in points or backend.is_eog(tokens[-1]):frontiers[str(count)]=frontier(backend.logits(lease),name+'-output-'+str(count))
     return result
    backend.consume_generated=consumed
    try:
     value=api.run_state_request(sid,(),limit,engine=None,state=None,stream='vc09-greedy-'+name[0],resume=resume,
      sampling=None if resume else dict(temperature=0.,min_p=0.,top_k=0,top_p=1.))
    finally:backend.consume_generated=original
    if value['tokens']!=tokens:raise QualificationError('observed consumed tokens differ from canonical answer')
    actual=offset+len(tokens);frontiers[str(actual)]=cp_frontier(value['snapshot'],name+'-final-'+str(actual))
    after=backend.work_counters()
    return dict(snapshot=value['snapshot'],tokens=tokens,frontiers=frontiers,forward_tokens=after['forward_tokens']-before['forward_tokens'],prefill_tokens=after['prefill_tokens']-before['prefill_tokens'],answer=value)
   if stage=='prepare':
    fixture=json.loads(Path(config['fixture']).read_bytes());parts=fixture_parts(config['fixture'],[Path(x['original']['path']) for x in fixture['images']])
    ram=SharedCheckpointPool(4*1024**3);disk=DiskCheckpointPool(api.store,32*1024**3)
    artifacts=ImageArtifacts(api.store,ram,disk,encoder_profile_id=fixture['encoder_profile_id'],expected_width=parts[0]['width'],original_bytes_limit=256*1024**2)
    spans=[];cursor=0
    with ExitStack() as held:
     for part in parts:
      original=Path(part['original']['path']).read_bytes();payload=Path(part['embedding']['path'])
      image=held.enter_context(artifacts.prepare(original=original,normalized=original,encode=lambda _,p=payload:p.read_bytes(),pad_token_id=part['pad']))
      spans.extend([dict(kind='text',tokens=fixture['input_tokens'][cursor:part['start']]),image.span]);cursor=part['start']+part['rows']
     spans.append(dict(kind='text',tokens=fixture['input_tokens'][cursor:]));before=backend.work_counters();t=time.monotonic();prefix=api.create_inputs(spans,artifacts=artifacts);after=backend.work_counters()
    if after['forward_tokens']-before['forward_tokens']!=fixture['total_cells']:raise QualificationError('prefix work differs')
    api.store.fork('prefix',prefix);record['prefix_prefill_seconds']=time.monotonic()-t
    t=time.monotonic();record['prefix_save']=api.save_branch('prefix',0,prefix,'vc09-save-prefix');record['save_seconds']=time.monotonic()-t
    record.update(prefix=prefix,prefix_frontier=cp_frontier(prefix,'prefix'),prefix_chunks=list(api.cache[prefix].execution_schedule[0]['chunks']))
    parent_image=api.cache[prefix].blob;expected_audit=audit(parent_image);record['parent_audit']=expected_audit;record['parent_audits']=[]
    record['prefix_consumed']=len(api.history(prefix));record['branches']={}
   else:
    prepared=json.loads(Path(config['prepare_result']).read_bytes());prefix=prepared['prefix']
    forbidden={str(Path(config['fixture']).resolve()),*[str(Path(x).resolve()) for x in config['originals']]}
    def no_inputs(event,arguments):
     if event=='open' and isinstance(arguments[0],(str,bytes)):
      name=os.fsdecode(arguments[0])
      if name.endswith('.sve') or str(Path(name).resolve()) in forbidden:raise QualificationError('strict restart read an embedding/original/fixture')
     if event=='subprocess.Popen':raise QualificationError('strict restart started encoder/helper')
    sys.addaudithook(no_inputs)
    snapshot=prepared['generated_four'] if stage=='generated' else prefix
    receipt=api.restore_checkpoint(snapshot);projection=dict(forward_tokens=receipt['forward_tokens'],consumed=receipt['consumed'],bytes_loaded=receipt['counter_deltas']['loaded_file_bytes'],native_pid=backend.process.pid,python_pid=os.getpid())
    expected_consumed=config['total_cells']+(config['suffix_tokens']+4 if stage=='generated' else 0)
    validate_restore(projection,consumed=expected_consumed,producer_pids=set(config['producer_pids']))
    record.update(restore=receipt,restore_projection=projection,restored_frontier=cp_frontier(snapshot,'restored'))
    if stage=='generated':
     record['continuation']=generate(snapshot,'A-generated',config['outputs']-4,offset=4,resume=True)
     record['controller']=api.store.get_snapshot(record['continuation']['snapshot']);record['status']='PASS'
    else:
     parent_image=api.cache[prefix].blob;expected_audit=audit(parent_image);record.update(prefix=prefix,prefix_frontier=record['restored_frontier'],parent_audit=expected_audit,parent_audits=[],branches={})
   if stage in ['prepare','resume']:
    for name in ['A','B']:
     before=backend.work_counters();sid=api.feed(prefix,config['suffixes'][name]);after=backend.work_counters();suffix_chunks=list(api.cache[sid].execution_schedule[-1]['chunks'])
     if after['forward_tokens']-before['forward_tokens']!=config['suffix_tokens']:raise QualificationError('suffix replayed prefix or omitted work')
     suffix_frontier=cp_frontier(sid,name+'-suffix');first=generate(sid,name,4)
     if len(first['tokens'])!=4 or backend.is_eog(first['tokens'][-1]):raise QualificationError('fixture ended before required four-token checkpoint')
     if stage=='prepare' and name=='A':
      api.store.fork('generated-four',first['snapshot']);record['generated_four']=first['snapshot'];record['generated_save']=api.save_branch('generated-four',0,first['snapshot'],'vc09-save-four')
     rest=generate(first['snapshot'],name,config['outputs']-4,offset=4,resume=True)
     result=dict(tokens=first['tokens']+rest['tokens'],frontiers=dict(first['frontiers'],**rest['frontiers']),forward_tokens=first['forward_tokens']+rest['forward_tokens'],prefill_tokens=first['prefill_tokens']+rest['prefill_tokens'],
      snapshot=rest['snapshot'],controller=api.store.get_snapshot(rest['snapshot']),suffix_chunks=suffix_chunks,suffix_frontier=suffix_frontier)
     validate_generation(result,limit=config['outputs'],policy=config['frontier_policy']);api.store.fork(name+'-'+stage+'-'+str(os.getpid()),rest['snapshot'])
     record['branches'][name]=result;observed=audit(parent_image);record['parent_audits'].append(observed)
     if observed!=expected_audit:raise QualificationError('retained native prefix bytes mutated')
     save(out/'result.json',record)
    if stage=='prepare':record['saved_files']=[dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q)) for q in sorted(api.checkpoint_directory.iterdir()) if q.suffix in ['.json','.state']]
   record['work']=backend.work_counters()
  record['status']='PASS'
 except BaseException as error:record.update(status='FAIL',error=repr(error));raise
 finally:
  if ram is not None:ram.close()
  if backend is not None:
   backend.close();record['native_exit']=dict(exit_code=backend.process.returncode,process_absent=not Path('/proc',str(backend.process.pid)).exists())
  save(out/'result.json',record)

def campaign(args):
 from run_vision_ingest_probe import compare_files
 fixture=json.loads(args.fixture.read_bytes());validate_fixture_counts(fixture);profile=json.loads(args.profile.read_bytes())
 if fixture['execution_profile_id']!=profile['execution_profile_id']:raise QualificationError('fixture/profile identity differs')
 out=args.output.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
 if shutil.disk_usage(out).free<2500*1024**2:raise QualificationError('need2500MiB headroom for numerical campaign')
 ref=json.loads(args.reference_build.read_bytes());current=json.loads(args.current_build.read_bytes());reference=ref['binaries']['reference']
 if sha(reference['path'])!=reference['sha256']:raise QualificationError('reference binary changed')
 common=set(ref['verified_unchanged_inputs'])&set(current['verified_unchanged_inputs']);verified={}
 for name in sorted(common):
  if ref['verified_unchanged_inputs'][name]!=current['verified_unchanged_inputs'][name] or sha(name)!=ref['verified_unchanged_inputs'][name]:raise QualificationError('shared computational build changed: '+name)
  verified[name]=ref['verified_unchanged_inputs'][name]
 if len(verified)<20:raise QualificationError('insufficient computational source/build linkage')
 config=dict(profile=str(args.profile.resolve()),profile_sha256=sha(args.profile),fixture=str(args.fixture.resolve()),store=str(out/'store'),reference_binary=reference['path'],outputs=fixture['outputs'],frontier_policy='every-generated' if fixture['size'] in ('tiny','medium') else 'sampled',
  suffixes=fixture['suffixes'],suffix_tokens=fixture['suffix_tokens'],total_cells=fixture['total_cells'],originals=[i['original']['path'] for i in fixture['images']],prepare_result=str(out/'prepare/result.json'))
 path=out/'config.json';save(path,config)
 result=dict(status='RUNNING',scope=__doc__,profile_sha256=config['profile_sha256'],execution_profile_id=profile['execution_profile_id'],fixture_sha256=sha(args.fixture),
  reference_binary=reference,verified_shared_build_inputs=verified,stages=[],comparisons=[],controller_comparisons=[],max_abs_error=None,
  frontier_policy=config['frontier_policy'],audit_scope=('prefix/suffix/every generated frontier' if config['frontier_policy']=='every-generated' else 'prefix/suffix/generated1,4,16,64,actual-final when reached')+'; raw references choose every token from full logits',performance='NOT_MEASURED_BY_THIS_AUDIT',q8='DEFERRED_BY_OWNER',chat_server='OFF')
 def run(stage,name,branch='A'):
  directory=out/name;directory.mkdir(mode=0o700);cmd=[sys.executable,'-B',__file__,'--worker',stage,'--config',str(path),'--branch',branch,'--output',str(directory)]
  with (directory/'stdout.log').open('wb') as so,(directory/'stderr.log').open('wb') as se:
   process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);started=time.monotonic()
   while True:
    try:code=process.wait(30);break
    except subprocess.TimeoutExpired:
     save(out/'heartbeat.json',dict(stage=name,python_pid=process.pid,elapsed_seconds=time.monotonic()-started,status='RUNNING'));print(json.dumps(dict(stage=name,pid=process.pid,elapsed=time.monotonic()-started)),flush=True)
  row=dict(stage=name,python_pid=process.pid,exit_code=code,seconds=time.monotonic()-started,process_absent=not Path('/proc',str(process.pid)).exists());result['stages'].append(row);save(out/'result.json',result)
  value=json.loads((directory/'result.json').read_bytes())
  if code!=0 or value['status']!='PASS' or not value['native_exit']['process_absent'] or value['native_exit']['exit_code']!=0 or Path('/proc',str(value['native_pid'])).exists():raise QualificationError('failed stage '+name)
  return value
 def exact(left,right,label):
  check=dict(label=label,**compare_files(left,right));result['comparisons'].append(check);save(out/'result.json',result)
  if not check['exact']:raise QualificationError('nonzero numerical error: '+label)
 def branch_exact(expected,actual,label):
  if expected['tokens']!=actual['tokens']:raise QualificationError('token mismatch '+label)
  for k in sample_points(len(expected['tokens']),policy=config['frontier_policy']):exact(expected['frontiers'][str(k)],actual['frontiers'][str(k)],label+'-'+str(k))
 try:
  prepared=run('prepare','prepare');config['producer_pids']=[prepared['python_pid'],prepared['native_pid']];save(path,config)
  removed=[]
  for p in (out/'store/checkpoints').glob('*.sve'):removed.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)));p.unlink()
  if not removed:raise QualificationError('no produced embeddings to evict')
  result['removed_optional_embeddings']=removed
  for name in ['A','B']:
   first=run('reference','reference-'+name+'-0',name);second=run('reference','reference-'+name+'-1',name);expected=prepared['branches'][name]
   for value,label in [(first,'fresh0'),(second,'fresh1')]:
    same_batches(prepared['prefix_chunks'],value['prefix_chunks']);same_batches(expected['suffix_chunks'],value['suffix_chunks'])
    exact(prepared['prefix_frontier'],value['prefix_frontier'],name+'-'+label+'-prefix');exact(expected['suffix_frontier'],value['suffix_frontier'],name+'-'+label+'-suffix');branch_exact(expected,value,name+'-'+label)
  for repetition in range(3):
   resumed=run('resume','resume-'+str(repetition));exact(prepared['prefix_frontier'],resumed['restored_frontier'],'restart-'+str(repetition)+'-prefix')
   for name in ['A','B']:
    expected=prepared['branches'][name];actual=resumed['branches'][name];same_batches(expected['suffix_chunks'],actual['suffix_chunks']);exact(expected['suffix_frontier'],actual['suffix_frontier'],name+'-restart-suffix-'+str(repetition));branch_exact(expected,actual,name+'-restart-'+str(repetition))
    if expected['controller']!=actual['controller']:raise QualificationError('controller mismatch '+name)
    result['controller_comparisons'].append(name+'-restart-'+str(repetition))
  generated=run('generated','generated-restart');actual=generated['continuation'];expected=prepared['branches']['A']
  exact(expected['frontiers']['4'],generated['restored_frontier'],'generated-restored-four')
  if actual['tokens']!=expected['tokens'][4:] or generated['controller']!=expected['controller']:raise QualificationError('generated-boundary continuation/controller differs')
  for k in sample_points(len(expected['tokens']),policy=config['frontier_policy']):
   if k>4:exact(expected['frontiers'][str(k)],actual['frontiers'][str(k)],'generated-restart-'+str(k))
  for item in prepared['saved_files']:
   if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:raise QualificationError('saved payload mutated')
  result.update(status='VC09_BOUNDED_MATCHING_PROFILE_NUMERICAL_RESTART_PASS',max_abs_error=0.,controller_scope='greedy plain sampler; exact current controller across3 prefix restarts and generated-four restart',remaining='Performance paths, current text control and complete supported matrix remain; not complete VC09 or G1-G5')
 except BaseException as error:result.update(status='FAIL',error=repr(error));raise
 finally:save(out/'result.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=['prepare','reference','resume','generated']);p.add_argument('--branch',default='A');p.add_argument('--config',type=Path)
 for name in ['runtime','profile','fixture','reference-build','current-build','output']:p.add_argument('--'+name,type=Path)
 a=p.parse_args()
 if a.worker:worker(json.loads(a.config.read_bytes()),a.worker,a.branch,a.output)
 else:
  runtime=a.runtime.resolve(strict=True);paths=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),str(runtime/'experimental/branching_context/scripts')]
  sys.path[:0]=paths;os.environ['PYTHONPATH']=os.pathsep.join(paths+[os.environ.get('PYTHONPATH','')])
  campaign(a)
