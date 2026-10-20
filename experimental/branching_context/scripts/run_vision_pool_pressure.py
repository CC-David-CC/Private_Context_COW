"""VC08 real Q4 capacity LRU and wall-clock idle expiry with active image leases.

Cache budgets are deliberately bounded in this pressure fixture; native engine
resources/profile are unchanged. Idle expiry covers the shared RAM/disk pools.
Native images have a separate byte-limited cache: process exit, not disk expiry,
is the final resident-image release boundary in the idle-reconstruction arm.
"""
import argparse,hashlib,json,math,os,shutil,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

class ProbeError(RuntimeError):pass

def _need(row,key,value):
 if key not in row or type(row[key]) is not type(value) or row[key]!=value:
  raise ProbeError('missing or failed witness: '+key)

def _integer(row,key,minimum=0):
 if type(row.get(key)) is not int or row[key]<minimum:raise ProbeError('invalid native/resource counter: '+key)
 return row[key]

def _exact(row):
 for k in ('recomputation_notice','parent_unchanged','reference_payload_unchanged','tokens_exact','controller_exact','frontier_exact'):_need(row,k,True)
 if type(row.get('max_abs_error')) not in (int,float) or row['max_abs_error']!=0:raise ProbeError('nonzero or missing numerical error')
 prefix=_integer(row,'prefix_forward',1);consumed=_integer(row,'consumed',1);generated=_integer(row,'generated',1)
 if prefix!=consumed or _integer(row,'total_forward')!=prefix+generated:raise ProbeError('prefix and generated native work differ')

def validate_capacity(row):
 _exact(row)
 for k in ('old_ram_absent','old_disk_absent','old_native_absent'):_need(row,k,True)
 limit=_integer(row,'byte_limit',1)
 for k in ('ram_bytes','disk_bytes'):
  if _integer(row,k)>limit:raise ProbeError('settled pool budget exceeded: '+k)
 for k in ('ram_reserved','disk_reserved','newest_forward'):_need(row,k,0)
 if type(row.get('embedding_disk_present')) is not bool:raise ProbeError('missing pre-restart embedding presence witness')
 _need(row,'encoder_calls',0 if row['embedding_disk_present'] else 1)
 _integer(row,'lru_evictions',1)

def validate_expiry(row):
 _exact(row)
 ttl=row.get('ttl_seconds');elapsed=row.get('held_seconds')
 if type(ttl) not in (int,float) or not math.isfinite(ttl) or ttl<=0:raise ProbeError('invalid TTL')
 if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=2*ttl:raise ProbeError('real expiry interval not observed')
 _integer(row,'real_sweeps',1);_integer(row,'active_pins',1)
 digest=row.get('active_sha_before')
 if type(digest) is not str or len(digest)!=64 or row.get('active_sha_during')!=digest:raise ProbeError('active image bytes changed/disappeared')
 for k in ('active_ram_survived','old_processes_exited','new_native_process'):_need(row,k,True)
 for k in ('expired_ram_entries','expired_disk_files'):_need(row,k,0)
 _need(row,'expiry_error',None);_need(row,'encoder_calls',1)

def install_observer(body,root):
 marker="importlib.import_module('serve.cow_http').main()\n"
 if body.count(marker)!=1:raise ProbeError('ambiguous real server entrypoint')
 hook=r"""import threading,struct,time,os
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.strata_backend import StrataBackend
_ROOT=Path(ROOT_PATH);_WRITE=threading.Lock();_GATE=threading.Event();_LOCAL=threading.local()
def _write(name,row):
 with _WRITE:
  with (_ROOT/name).open('a') as f:
   f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def _images(pool):
 with pool._condition:
  return [dict(key=list(k),pins=v['pins'],sha256=hashlib.sha256(v['checkpoint'].payload).hexdigest(),bytes=len(v['checkpoint'].payload))
   for k,v in pool._entries.items() if isinstance(k,tuple) and k and k[0]=='image']
_original_collect=ContextCoordinator.collect_expired
def _collect(self):
 result=_original_collect(self)
 if _GATE.is_set():_write('sweeps.jsonl',dict(monotonic=time.monotonic(),result=result,images=_images(self.pool)))
 return result
ContextCoordinator.collect_expired=_collect
_original_consume=StrataBackend.consume_inputs
def _consume(self,lease,spans,*,embeddings):
 flag=_ROOT/'hold-next-image'
 if flag.exists():
  flag.unlink();coordinator=_LOCAL.coordinator;started=time.monotonic()
  _GATE.set();_write('gate.jsonl',dict(monotonic=started,images=_images(coordinator.pool),server_pid=os.getpid()))
  try:
   deadline=started+60
   while not (_ROOT/'release-image').exists():
    if time.monotonic()>deadline:raise TimeoutError('active-image observer gate timed out')
    time.sleep(.05)
   _write('released.jsonl',dict(monotonic=time.monotonic(),images=_images(coordinator.pool)))
  finally:_GATE.clear()
 return _original_consume(self,lease,spans,embeddings=embeddings)
StrataBackend.consume_inputs=_consume
_original_run=ContextCoordinator._run
def _run(self,model,job):
 _LOCAL.coordinator=self
 source=(job.source_head or job.head).tip if job.head is not None else None
 with model.cache._lock:
  old_native=source in model.cache._entries if source else False
 with self.pool._condition:
  old_ram=self._key(model,source) in self.pool._entries if source else False
 before=model.backend.work_counters()
 result=_original_run(self,model,job)
 after=model.backend.work_counters();sid=result['snapshot']
 with model.cache._lock:
  cp=model.cache._entries.get(sid)
  if cp is None:raise RuntimeError('observer needs actual consumed frontier')
  raw=struct.pack('<%df'%len(cp.frontier),*cp.frontier)
  native_keys=list(model.cache._entries);native_bytes=model.cache._referenced_bytes()
 name=hashlib.sha256(job.request.operation.encode()).hexdigest()+'.f32'
 (_ROOT/name).write_bytes(raw)
 _write('jobs.jsonl',dict(operation=job.request.operation,snapshot=sid,source=source,source_native_present=old_native,
  source_ram_present=old_ram,native_keys=native_keys,native_bytes=native_bytes,native_entry_evictions=model.cache.entry_evictions,
  frontier_file=name,frontier_sha256=hashlib.sha256(raw).hexdigest(),logits=len(cp.frontier),
  before=before,after=after,forward_tokens=after['forward_tokens']-before['forward_tokens']))
 return result
ContextCoordinator._run=_run
if (_ROOT/'ttl.json').exists():sys.argv.extend(['--checkpoint-idle-seconds',str(json.loads((_ROOT/'ttl.json').read_text())['seconds'])])
"""
 return body.replace(marker,hook.replace('ROOT_PATH',repr(str(root)))+'\n'+marker)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def wait_for(check,seconds=60):
 deadline=time.monotonic()+seconds
 while time.monotonic()<deadline:
  value=check()
  if value:return value
  time.sleep(.1)
 raise TimeoutError('required observation did not arrive')

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for key in ('runtime','branchwrite','profile','dependency-root','vision-config','fixtures','output'):parser.add_argument('--'+key,type=Path,required=True)
 a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
 sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
 from run_cli_grammar_probe import GrammarProbe,_atomic_json
 from validation.run_basic_loop_html import image_probe_type
 from branchwrite.cow_client import CowClient,OperationJournal
 from branchwrite.cow_actions import completed_thinking_text
 from context_time_travel.store import Store
 out=a.output.absolute();out.mkdir(mode=0o700,exist_ok=False);profile=json.loads(a.profile.read_bytes())
 if shutil.disk_usage(out).free<2500*1024**2:raise ProbeError('need at least2500MiB free for bounded pools, reference and evidence')
 record=dict(schema=1,status='RUNNING',scope=__doc__,failures=[],services=[],cases=[],
  execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
  profile_sha256=sha(a.profile),native_sha256=profile['native_build']['sha256'],harness_sha256=sha(Path(__file__)),
  numerical_oracle='UNINTERRUPTED_MATCHING_PROFILE_REFERENCE_NOT_FRESH_FULL_HISTORY',numerical_max_error=None,
  cache_budget_bytes=1024**3,engine_resources='UNCHANGED_PROFILE',chat_server='OFF',q8='DEFERRED_BY_OWNER')
 def publish():_atomic_json(out/'result.json',record)
 active=None;pending=None;error=None
 def make_probe(name,store,ttl=None,hold=False):
  run=out/name;run.mkdir(mode=0o700)
  if ttl is not None:_atomic_json(run/'ttl.json',dict(seconds=ttl))
  if hold:(run/'hold-next-image').write_text('gate only; real native consume follows release\n')
  class Owner(image_probe_type(GrammarProbe,a.vision_config)):
   def _bootstrap(self,registry):
    path=super()._bootstrap(registry);path.write_text(install_observer(path.read_text(),run),newline='\n');return path
  probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=store,run_dir=run,
   dependency_root=a.dependency_root,ram_gib=1,disk_gib=1))
  probe._imports();probe.runtime_closure(profile,runtime)
  registry=run/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(VC08_REPLY=dict(format='thinking',source=dict(
   schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
  probe.tracker=probe.ProgressTracker(run,'VC08-pool-'+name).start()
  try:probe.start_service(registry)
  except BaseException:
   if probe.service is not None:probe.service.stop()
   probe.tracker.finish('SETUP_FAIL');probe.tracker.close();raise
  url=probe.ready_url(probe.service.ready);pin=hashlib.sha256(str(run).encode()).hexdigest()
  client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=run/'cli-state',journal=OperationJournal(run/'journal'),api_key=probe.secret,endpoint_pin=pin)
  for worker in client.status().capacity['workers'].values():probe.tracker.native_started(worker['pid'])
  return probe,client
 def settled(client):
  def check():
   s=client.status().capacity
   if not s['queued'] and not s['pending_branches'] and not any(s['active_per_gpu'].values()):
    if not s['ram']['reserved_bytes'] and not s['disk']['reserved_bytes']:return s
  return wait_for(check)
 def finish(probe,status):
  probe.record['status']=status;probe.stop_service();probe.publish();probe.tracker.finish(status);probe.tracker.close()
  service=probe.record['services'][-1];worker=service['worker_shutdown']
  value=dict(name=probe.run_dir.name,server_pid=service['pid'],native_pid=worker['native_pid'],
   server_exit=service['stop']['exit_code'],native_exit=worker['native_exit_code'],
   processes_exited=not Path('/proc',str(service['pid'])).exists() and not Path('/proc',str(worker['native_pid'])).exists())
  record['services'].append(value);publish()
  if value['server_exit']!=0 or value['native_exit']!=0 or not value['processes_exited']:raise ProbeError('server/native not cleanly retired')
  return value
 def job(probe,op):
  matching=[r for r in rows(probe.run_dir/'jobs.jsonl') if r['operation']==op]
  if len(matching)!=1:raise ProbeError('missing/duplicate actual native work observation: '+op)
  return matching[0]
 def cli(probe,arguments,label,asynchronous=False):
  command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',probe.ready_url(probe.service.ready),
   '--timeout','180','--state-dir',str(probe.run_dir/'cli-state'),'--machine-result']+arguments
  stdout=(probe.run_dir/(label+'.stdout')).open('wb');stderr=(probe.run_dir/(label+'.stderr')).open('wb')
  try:process=subprocess.Popen(command,env={**os.environ,'COW_API_KEY':probe.secret},stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  finally:stdout.close();stderr.close()
  if asynchronous:return process
  if process.wait(240)!=0:raise ProbeError('CLI failed: '+label)
  return json.loads((probe.run_dir/(label+'.stdout')).read_bytes())
 def answer(probe,client):
  fork=client.fork('kept',parent.tip,'trial','vc08-pool-trial-fork')
  result=cli(probe,['answer','--branch','trial','--revision',str(fork.revision),'--operation-id','vc08-pool-trial-answer',
   '--max-tokens','1024','--temperature','0','--grammar-state','VC08_REPLY'],'answer')
  obj=SimpleNamespace(answer=result['answer'],receipt=SimpleNamespace(snapshot=result['receipt']['snapshot']))
  if completed_thinking_text(obj,profile['execution']['eos_tokens'])!='OK':raise ProbeError('expected legal final OK')
  observation=job(probe,'vc08-pool-trial-answer');raw=(probe.run_dir/observation['frontier_file']).read_bytes()
  return result,observation,raw
 def comparison(probe,store,result,observation,frontier):
  vals=struct.unpack('<%df'%(len(frontier)//4),frontier);refvals=struct.unpack('<%df'%(len(reference_frontier)//4),reference_frontier)
  if len(vals)!=len(refvals) or not all(map(math.isfinite,vals+refvals)):raise ProbeError('incomplete/nonfinite full frontier')
  work=result['answer']['source_recomputation']
  return dict(prefix_forward=work['prefix_forward_tokens'],consumed=prefix_consumed,generated=len(result['answer']['tokens']),
   total_forward=observation['forward_tokens'],encoder_calls=work['input_preparation']['encoder_calls'],
   recomputation_notice='recomput' in (probe.run_dir/'answer.stderr').read_text().lower(),
   parent_unchanged=Store(store).head('kept')==parent and Store(store).get_snapshot(parent.tip)==parent_snapshot,
   reference_payload_unchanged=sha(reference_payload)==parent_payload_sha,
   tokens_exact=result['answer']['tokens']==reference['answer']['tokens'],
   controller_exact=Store(store).get_snapshot(result['receipt']['snapshot'])==reference_snapshot,
   frontier_exact=frontier==reference_frontier,max_abs_error=max(abs(x-y) for x,y in zip(vals,refvals)),
   prefix_frontier_exact=work['frontier_sha256']==prefix_frontier_sha,recomputation=work)
 try:
  assert profile['execution']['text_template_kwargs']=={'enable_thinking':True}
  base=out/'store-capacity';refstore=out/'store-reference';ttlstore=out/'store-expiry'
  active,client=make_probe('producer',base)
  ns=client.create_namespace('Image pool pressure','vc08-pool-namespace')['namespace_id']
  image=(a.fixtures/'branchwrite-h04-cli-gpu-v1/after/capture.png').read_bytes();upload=client.upload_image(ns,image)
  text='Keep this page available. When asked, reply exactly OK.'
  root=client.prepare('kept',ns,text,'vc08-pool-prefix',images=(upload,))
  parent=active.store.head('kept');parent_snapshot=active.store.get_snapshot(parent.tip)
  root_manifest=json.loads((base/'checkpoints'/(parent.tip+'.json')).read_bytes())['manifest'];prefix_consumed=root_manifest['consumed']
  root_job=job(active,'vc08-pool-prefix');prefix_frontier_sha=root_job['frontier_sha256']
  embedding_files=list((base/'checkpoints').glob('*.sve'))
  if len(embedding_files)!=1:raise ProbeError('expected one source embedding file')
  embedding_sha=embedding_files[0].stem
  if sha(embedding_files[0])!=embedding_sha:raise ProbeError('source embedding checksum differs')
  parent_payload_sha=root_manifest['payload_sha256'];settled(client)
  record['prefix']=dict(root.receipt.raw);finish(active,'PREFIX_PASS');active=None
  def copy(source,destination):
   if Path(source).suffix=='.state':os.link(source,destination);return destination
   return shutil.copy2(source,destination)
  for target in (refstore,ttlstore):shutil.copytree(base,target,copy_function=copy)
  record['fixture_clone']='Idle producer Store copied; .state files hardlinked only as fixture storage, not native COW evidence';publish()
  reference_payload=refstore/'checkpoints'/(parent_payload_sha+'.state')
  active,client=make_probe('reference',refstore)
  reference,reference_job,reference_frontier=answer(active,client)
  reference_snapshot=active.store.get_snapshot(reference['receipt']['snapshot']);settled(client)
  record['reference']=dict(answer=reference,job=reference_job,controller=reference_snapshot)
  finish(active,'REFERENCE_PASS');active=None
  active,client=make_probe('capacity',base)
  # New upload grants authorize repeated placement; the immutable pixels remain identical.
  upload=client.upload_image(ns,image);pressure=[]
  for index in range(16):
   active.phase('capacity-distinct-root-'+str(index))
   op='vc08-pool-tree-'+str(index)
   result=client.prepare('tree-'+str(index),ns,'Tree '+str(index)+'. '+text,op,images=(upload,))
   state=settled(client);measurement=job(active,op)
   pressure.append(dict(index=index,receipt=dict(result.receipt.raw),answer=dict(result.answer),job=measurement,status=state))
   old_absent=not (base/'checkpoints'/(parent.tip+'.json')).exists()
   record['capacity_progress']=pressure;publish()
   if old_absent and index>=8:break
  else:raise ProbeError('bounded distinct roots did not evict initial source')
  # New operation identity, identical latest prefix: it should reuse computation.
  latest=client.prepare('newest-alias',ns,'Tree '+str(index)+'. '+text,'vc08-pool-newest-alias',images=(upload,))
  latest_job=job(active,'vc08-pool-newest-alias');state=settled(client)
  absent_manifest=not (base/'checkpoints'/(parent.tip+'.json')).exists()
  embedding_path=base/'checkpoints'/(embedding_sha+'.sve')
  embedding_disk_present=embedding_path.exists()
  if embedding_disk_present and sha(embedding_path)!=embedding_sha:raise ProbeError('pre-restart embedding checksum differs')
  # Shared-pool LRU does not evict separately budgeted native cache entries.
  # Preserve that observation, then retire the process before asserting full absence.
  record['capacity_before_native_retirement']=dict(status=state,last_job=latest_job,
   source_native_retained=parent.tip in latest_job['native_keys'],source_disk_absent=absent_manifest,
   embedding_disk_present=embedding_disk_present,embedding_sha256=embedding_sha)
  finish(active,'RAM_DISK_CAPACITY_PASS');active=None
  active,client=make_probe('capacity-restart',base)
  candidate,candidate_job,candidate_frontier=answer(active,client)
  row=comparison(active,base,candidate,candidate_job,candidate_frontier)
  row.update(case='capacity-lru',ram_bytes=state['ram']['resident_bytes'],disk_bytes=state['disk']['file_bytes'],byte_limit=1024**3,
   ram_reserved=state['ram']['reserved_bytes'],disk_reserved=state['disk']['reserved_bytes'],
   old_ram_absent=not candidate_job['source_ram_present'],old_native_absent=not candidate_job['source_native_present'],old_disk_absent=absent_manifest,
   embedding_disk_present=embedding_disk_present,
   lru_evictions=state['disk']['files_evicted'],newest_forward=latest_job['forward_tokens'],
   native_retirement_boundary='Clean process exit after RAM/disk LRU pressure',status='RUNNING')
  record['cases'].append(row);publish();validate_capacity(row)
  if not row['prefix_frontier_exact']:raise ProbeError('reconstructed prefix frontier differs')
  row['status']='PASS';settled(client);finish(active,'CAPACITY_PASS');active=None
  ttl=2.;active,client=make_probe('expiry',ttlstore,ttl=ttl,hold=True)
  # Startup itself exceeds TTL, so old optional disk copies should expire normally.
  wait_for(lambda:not list((ttlstore/'checkpoints').glob('*.state')))
  upload=client.upload_image(ns,image)
  # Use the real client upload receipt/request schema; the CLI transport drives the pending request.
  import concurrent.futures
  executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  pending=executor.submit(client.prepare,'lease-held',ns,text,'vc08-pool-held-prefix',images=(upload,))
  gate=wait_for(lambda:next(iter(rows(active.run_dir/'gate.jsonl')),None));held_start=gate['monotonic']
  def actual_sweep():
   candidates=[r for r in rows(active.run_dir/'sweeps.jsonl') if r['monotonic']-held_start>2*ttl]
   return candidates[-1] if candidates else None
  swept=wait_for(actual_sweep)
  _atomic_json(active.run_dir/'release-image',dict(actual_elapsed=swept['monotonic']-held_start))
  held=pending.result(120);pending=None;executor.shutdown(wait=True)
  held_job=job(active,'vc08-pool-held-prefix')
  if held_job['frontier_sha256']!=prefix_frontier_sha:raise ProbeError('held-image prefix differs from uninterrupted prefix')
  if held.receipt.snapshot!=parent.tip:raise ProbeError('fixture prefix identity changed')
  def expired():
   s=client.status().capacity
   optional=[p for p in (ttlstore/'checkpoints').iterdir() if p.is_file() and p.suffix in ('.json','.state','.sve')]
   return s if s['ram']['entries']==0 and not optional and s['idle_expiry']['runs'] else None
  state=wait_for(expired)
  before_life=finish(active,'EXPIRY_AND_LEASE_PASS');expiry_run=active.run_dir;active=None
  active,client=make_probe('expiry-restart',ttlstore)
  restart_worker=settled(client)['workers']['0']['pid']
  candidate,candidate_job,candidate_frontier=answer(active,client)
  row=comparison(active,ttlstore,candidate,candidate_job,candidate_frontier)
  before_images=gate['images'];during_images=swept['images']
  if len(before_images)!=1 or len(during_images)!=1:raise ProbeError('expected one active embedding')
  row.update(case='idle-expiry-restart',ttl_seconds=ttl,held_seconds=swept['monotonic']-held_start,
   real_sweeps=len(rows(expiry_run/'sweeps.jsonl')),active_pins=during_images[0]['pins'],active_sha_before=before_images[0]['sha256'],
   active_sha_during=during_images[0]['sha256'],active_ram_survived=before_images[0]['key']==during_images[0]['key'],
   expired_ram_entries=state['ram']['entries'],expired_disk_files=0,expiry_error=state['idle_expiry']['last_error'],
   old_processes_exited=before_life['processes_exited'],new_native_process=restart_worker!=before_life['native_pid'],
   native_retirement_boundary='Clean server/native exit after shared-pool idle expiry',status='RUNNING')
  record['cases'].append(row);publish();validate_expiry(row)
  if not row['prefix_frontier_exact']:raise ProbeError('restart-reconstructed prefix frontier differs')
  row['status']='PASS';settled(client);finish(active,'EXPIRY_RESTART_PASS');active=None
  record.update(status='VC08_BOUNDED_Q4_COMBINED_POOL_PRESSURE_PASS',numerical_max_error=max(r['max_abs_error'] for r in record['cases']))
 except BaseException as exc:
  error=exc;record['status']='VC08_BOUNDED_Q4_COMBINED_POOL_PRESSURE_FAIL';record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if active is not None:
   (active.run_dir/'release-image').write_text('release on campaign cleanup\n')
   if pending is not None:
    try:pending.result(120)
    except BaseException:pass
   try:finish(active,'CAMPAIGN_ABORTED')
   except BaseException as exc:error=error or exc;record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
  publish()
 return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
