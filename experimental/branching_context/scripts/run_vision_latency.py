"""Measure actual CLI image-context paths with controlled eligible cache eviction.

One warm-up and three measured repetitions per path. Test-only tier setup runs
on the real serialized worker before timing its ordinary request implementation.
CLI latency includes that setup; setup and worker execution are also reported
separately. No default model, operator, integrity or publication checks change.
"""
import argparse,hashlib,json,math,os,shutil,statistics,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

class LatencyError(RuntimeError):pass

def validate_tier(row):
    for key in ('prefix_cells','suffix_cells','outputs','forward_tokens','loaded_file_bytes','disk_preload_bytes','encoder_calls'):
        if type(row.get(key)) is not int or row[key]<0:raise LatencyError('missing or invalid counter '+key)
    if row['prefix_cells']<=0:raise LatencyError('missing prefix extent')
    for key in ('source_native','source_ram','source_disk','image_cached'):
        if type(row.get(key)) is not bool:raise LatencyError('missing source witness '+key)
    mode=row.get('mode');expected=row['suffix_cells']+row['outputs']
    if mode in ('cold','embedding','text-prefill'):
        expected+=row['prefix_cells']
        if any(row[k] for k in ('source_native','source_ram','source_disk')):raise LatencyError('prefill retained a numerical prefix')
        if row['outputs'] or row['suffix_cells']:raise LatencyError('root timing includes generation')
        if mode=='cold' and (row['image_cached'] or row['encoder_calls']!=1):raise LatencyError('not actual cold encoding')
        if mode=='embedding' and (not row['image_cached'] or row['encoder_calls']):raise LatencyError('embedding hit not observed')
        if mode=='text-prefill' and row['encoder_calls']:raise LatencyError('text-only root encoded an image')
    elif mode in ('resident','ram','disk'):
        if row['encoder_calls'] or row['suffix_cells']<=0 or row['outputs']<=0:raise LatencyError('question work missing or re-encoded')
        if mode=='resident' and (not row['source_native'] or row['loaded_file_bytes'] or row['disk_preload_bytes']):raise LatencyError('not resident reuse')
        if mode=='ram' and (row['source_native'] or not row['source_ram'] or row['loaded_file_bytes']<=0 or row['disk_preload_bytes']):raise LatencyError('not RAM-sourced restore')
        if mode=='disk' and (row['source_native'] or row['source_ram'] or not row['source_disk'] or row['loaded_file_bytes']<=0 or row['disk_preload_bytes']<=0):raise LatencyError('not disk-sourced restore')
    else:raise LatencyError('unknown cache path')
    if row['forward_tokens']!=expected:raise LatencyError('unexpected prefix work or omitted token')

def reference_prefix_command(tokens,manifest,coordinates,image_count):
    # Match the real adapter: consume_inputs uses ordinary feed for zero images.
    # The native image-ingest parser intentionally requires at least one image.
    if type(image_count) is not int or not 0 <= image_count <= 8:
        raise LatencyError('invalid reference image count')
    if image_count == 0:
        return 'feed '+' '.join(map(str,tokens))
    return 'feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(coordinates))

def close_reference(native):
    try:
        return native.close()
    except Exception as error:
        code=native.proc.poll()
        if code is None:
            raise
        # Retain the primary forward error when quit races a native rejection.
        # A nonzero exit or shutdown_error still fails the reference campaign.
        return dict(pid=native.proc.pid,exit_code=code,forced_kill=None,
                    process_absent=not Path('/proc',str(native.proc.pid)).exists(),
                    shutdown_error=repr(error))

def encoding_calls(receipt):
    if not isinstance(receipt,dict) or not isinstance(receipt.get('encodings'),list):
        raise LatencyError('root preparation lacks actual encoder receipts')
    return len(receipt['encodings'])

def summarize_repetitions(rows):
    warm=[r for r in rows if r.get('warmup') is True];measured=[r for r in rows if r.get('warmup') is False]
    if len(warm)!=1 or len(measured)!=3:raise LatencyError('requires one warm-up and three measurements')
    values=[r['seconds'] for r in measured]
    if any(type(n) not in (int,float) or not math.isfinite(n) or n<=0 for n in values):raise LatencyError('invalid request timing')
    return dict(individual_seconds=values,median_seconds=statistics.median(values),warmup_seconds=warm[0]['seconds'])

def install_tier_observer(body,root):
    marker="importlib.import_module('serve.cow_http').main()\n"
    if body.count(marker)!=1:raise LatencyError('ambiguous real service entrypoint')
    hook=r"""import threading,struct,time,re,os
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.store import _sync_dir
from context_time_travel.strata_backend import StrataImage
from serve.cow_vision_encoder import CowNativeVisionEncoder
_ROOT=Path(ROOT_PATH);_LOG_LOCK=threading.Lock();_ENCODER_CALLS=0
_original_encode=CowNativeVisionEncoder.encode
def _encode(self,*args,**kwargs):
 global _ENCODER_CALLS
 _ENCODER_CALLS+=1
 return _original_encode(self,*args,**kwargs)
CowNativeVisionEncoder.encode=_encode
def _metrics(pid):
 data={}
 for name in ('status','io'):
  try:
   for line in Path('/proc',str(pid),name).read_text().splitlines():
    if ':' in line:
     key,value=line.split(':',1)
     if key in ('VmRSS','VmHWM','read_bytes','write_bytes'):data[key]=value.strip()
  except FileNotFoundError:pass
 return data
_original_run=ContextCoordinator._run
def _run(self,model,job):
 path=_ROOT/'controls'/(job.request.operation+'.json')
 if not path.exists():return _original_run(self,model,job)
 control=json.loads(path.read_text());mode=control['mode'];source=(job.source_head or job.head).tip if job.head is not None else None
 target=source or control.get('target_snapshot');key=self._key(model,target) if target else None
 setup=time.monotonic()
 if mode in ('cold','embedding'):
  model.cache.clear();model.backend.release_checkpoints()
  with self.pool._condition:keys=list(self.pool._entries)
  for item in keys:
   if item[0]==model.profile['model_contract_id'] or (mode=='cold' and item[0]=='image'):self.pool.remove(item)
  with model.store._locked():
   for file in model.checkpoint_directory.iterdir():
    if file.is_file() and not file.is_symlink() and (re.fullmatch(r'[0-9a-f]{64}\.(json|state)',file.name) or (mode=='cold' and re.fullmatch(r'[0-9a-f]{64}\.sve',file.name))):file.unlink()
   _sync_dir(model.checkpoint_directory)
 elif mode in ('ram','disk'):
  model.cache.clear();model.backend.release_checkpoints()
  if mode=='disk' and key in self.pool:self.pool.remove(key)
 elif mode=='text-prefill' and target:
  model.cache.pop(target,None);model.backend.release_checkpoints()
  if key in self.pool:self.pool.remove(key)
  file=model.checkpoint_directory/(target+'.json')
  with model.store._locked():
   if file.exists():file.unlink();_sync_dir(model.checkpoint_directory)
  self.disk.collect()
 if mode in ('cold','embedding'):
  native=bool(len(model.cache))
  with self.pool._condition:ram=any(k[0]==model.profile['model_contract_id'] for k in self.pool._entries)
  disk=any(file.is_file() and not file.is_symlink() and re.fullmatch(r'[0-9a-f]{64}\.json',file.name) for file in model.checkpoint_directory.iterdir())
 else:
  native=bool(target and isinstance(getattr(model.cache.get(target),'blob',None),StrataImage))
  ram=bool(key and key in self.pool);disk=bool(target and (model.checkpoint_directory/(target+'.json')).is_file())
 with self.pool._condition:image_ram=any(k[0]=='image' for k in self.pool._entries)
 image_cached=image_ram or bool(list(model.checkpoint_directory.glob('*.sve')))
 trace=getattr(self.pool,'_transition_trace',None);sequence=trace.snapshot()['event_count'] if trace is not None else None
 before=model.backend.work_counters();encoder_calls_before=_ENCODER_CALLS
 if mode in ('cold','embedding','ram','disk') and before.get('handle_live_entries')!=0:raise RuntimeError('test tier setup retained native snapshot handles')
 setup_seconds=time.monotonic()-setup;started=time.monotonic()
 result=_original_run(self,model,job);worker_seconds=time.monotonic()-started;after=model.backend.work_counters()
 cp=model.cache[result['snapshot']];raw=struct.pack('<%df'%len(cp.frontier),*cp.frontier)
 frontier=_ROOT/(job.request.operation+'.f32');frontier.write_bytes(raw)
 events=trace.snapshot()['events'] if trace is not None else []
 reads=[x for x in events if sequence is not None and x['sequence']>=sequence and x['event']=='disk_preload_succeeded' and key is not None and x['key']==list(key)]
 row=dict(operation=job.request.operation,mode=mode,source=source,snapshot=result['snapshot'],source_native=native,source_ram=ram,source_disk=disk,encoder_calls=_ENCODER_CALLS-encoder_calls_before,
  image_cached=image_cached,setup_seconds=setup_seconds,worker_seconds=worker_seconds,forward_tokens=after['forward_tokens']-before['forward_tokens'],
  loaded_file_bytes=after['loaded_file_bytes']-before['loaded_file_bytes'],disk_preload_bytes=sum(x['bytes'] for x in reads),disk_preload_events=reads,
  before=before,after=after,native_pid=model.backend.process.pid,state_staging_storage=model.backend.state_staging_storage,
  server_resources=_metrics(os.getpid()),native_resources=_metrics(model.backend.process.pid),
  frontier_file=str(frontier),frontier_sha256=hashlib.sha256(raw).hexdigest(),logits=len(cp.frontier))
 with _LOG_LOCK:
  with (_ROOT/'jobs.jsonl').open('a') as output:output.write(json.dumps(row,sort_keys=True)+'\n');output.flush()
 return result
ContextCoordinator._run=_run
"""
    return body.replace(marker,hook.replace('ROOT_PATH',repr(str(root)))+'\n'+marker)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','branchwrite','profile','dependency-root','vision-config','image','reference-build','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prefix-text-tokens',type=int,default=1024)
    a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
    if a.prefix_text_tokens not in (1024,8192):raise LatencyError('use the initial1K or target8K fixture')
    sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
    from run_cli_grammar_probe import GrammarProbe,_atomic_json
    from branchwrite.cow_client import CowClient,OperationJournal
    from context_time_travel.input_spans import history_for_snapshot
    from tools.strata_tokenizer import Tokenizer
    from run_vision_ingest_probe import Native,compare_files,fingerprint
    from vision_fixture import reference_positions
    out=a.output.resolve();out.mkdir(mode=0o700,exist_ok=False);(out/'controls').mkdir(mode=0o700);(out/'retained-embeddings').mkdir(mode=0o700)
    if shutil.disk_usage(out).free<2500*1024**2:raise LatencyError('need2500MiB free including bounded disk pool and evidence')
    profile=json.loads(a.profile.read_bytes());tok=Tokenizer.from_gguf(profile['model_path'])
    code='\n'.join((runtime/name).read_text() for name in ['serve/cow_text_adapter.py','experimental/branching_context/src/context_time_travel/coordinator.py'])
    prefix_text='Keep this code and screenshot as shared context. When asked, think briefly and finish with OK.\n'+tok.decode(tok.encode(code,parse_special=False)[:a.prefix_text_tokens])
    (out/'prefix-text.txt').write_text(prefix_text,newline='\n')
    result=dict(schema=1,status='RUNNING',scope=__doc__,measurements=[],summaries={},chat_server='OFF',q8='DEFERRED_BY_OWNER',
        prefix_requested_text_tokens=a.prefix_text_tokens,execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
        profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),source_profile=str(a.profile),services=[],numerical_max_error=None,
        filesystem_cache='Warm/uncontrolled filesystem cache. No system-wide cache dropping. Native/file and pool reads are logical transferred bytes, not physical NVMe traffic.',
        pools=dict(ram_gib=96,disk_gib=2,native_policy=profile['execution'].get('checkpoint_cache')),
        native_command=profile['execution']['sut_command'],resource_note='Pinned98K-capacity/40GiB expert-residency/8GiB vision-headroom profile, all32 CPUs available. Disk pool bounded2GiB because host free space is limited; no2-CPU or small-VRAM execution cap.',
        performance_scope='CLI HTTP consumed/durable arrivals; no per-token snapshot/export or numerical audit. Test-only eligibility/eviction setup is separately timed and remains included in whole CLI latency.')
    def publish():_atomic_json(out/'result.json',result)
    class Owner(GrammarProbe):
        def publish(self):_atomic_json(self.run_dir/'service-status.json',self.record,self.secret)
        def _bootstrap(self,registry):
            path=super()._bootstrap(registry);body=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
            body=body.replace(marker,"sys.argv.extend(['--vision-config',"+repr(str(a.vision_config.resolve(strict=True)))+"])\n"+marker)
            path.write_text(install_tier_observer(body,out),newline='\n');return path
    probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,dependency_root=a.dependency_root,ram_gib=96,disk_gib=2))
    registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(REPLY=dict(format='thinking',source=dict(schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
    client=None;root_image=root_text=None
    def native_job(op):
        rows=[json.loads(line) for line in (out/'jobs.jsonl').read_text().splitlines()]
        rows=[x for x in rows if x['operation']==op]
        if len(rows)!=1:raise LatencyError('native job missing or repeated: '+op)
        return rows[0]
    def command(op,args,*,observed=True):
        if observed:
            trace=out/(op+'-events.json');argv=[sys.executable,'-B',str(runtime/'experimental/branching_context/scripts/observe_cow_cli.py'),
                '--runtime-root',str(runtime),'--trace-file',str(trace),'--']
        else:argv=[sys.executable,'-B',str(runtime/'tools/cow_cli.py')]
        argv+=['--url',probe.ready_url(probe.service.ready),'--timeout','180','--state-dir',str(out/'cli-state'),'--machine-result',*args]
        started=time.monotonic();run=subprocess.run(argv,env={**os.environ,'COW_API_KEY':probe.secret},capture_output=True,timeout=240);elapsed=time.monotonic()-started
        (out/(op+'.stdout')).write_bytes(run.stdout);(out/(op+'.stderr')).write_bytes(run.stderr)
        if run.returncode:raise LatencyError('CLI failure '+op+': '+run.stderr.decode(errors='replace')[-1500:])
        value=json.loads(run.stdout);observation=None
        if observed:
            data=json.loads(trace.read_bytes());rows=[x for x in data['requests'] if x['operation_id']==op]
            if len(rows)!=1 or rows[0]['summary']['validation']['status']!='valid':raise LatencyError('invalid consumed/durable event trace '+op)
            observation=rows[0]['summary']
            if observation['token_ids']!=value['answer']['tokens']:raise LatencyError('observed IDs differ from canonical answer')
        return value,elapsed,observation
    question_text='Inspect the shared code and diagram briefly in your thinking. Finish with OK.'
    def run_group(group,mode,*,question=False,observed=True,text_only=False):
        nonlocal root_image,root_text
        rows=[]
        for repetition in range(4):
            op='latency-'+group+'-'+str(repetition);probe.phase(op);root=root_text if text_only else root_image
            target=root['receipt']['snapshot'] if root is not None else None
            _atomic_json(out/'controls'/(op+'.json'),dict(mode=mode,target_snapshot=target))
            if question:
                branch=op;fork=client.fork(root['branch'],target,branch,op+'-fork')
                args=['ask','--branch',branch,'--revision',str(fork.revision),'--text',question_text,'--max-tokens','1024','--temperature','0','--grammar-state','REPLY','--operation-id',op]
            else:
                branch=op;args=['context','--branch',branch,'--namespace-id',namespace,'--operation-id',op]
                args+=['--text',prefix_text] if text_only else ['--parts-file',str(out/'context-parts.json')]
            value,elapsed,observation=command(op,args,observed=observed);job=native_job(op);snap=value['receipt']['snapshot'];history=history_for_snapshot(probe.store,snap)
            for file in (out/'store/checkpoints').glob('*.sve'):
                destination=out/'retained-embeddings'/file.name
                if not destination.exists():shutil.copyfile(file,destination)
            envelope=json.loads((out/'store/checkpoints'/(snap+'.json')).read_bytes());schedule=envelope['manifest']['execution_schedule']
            outputs=len(value['answer']['tokens']);prefix_cells=len(history) if not question else len(history_for_snapshot(probe.store,target))
            suffix_cells=0 if not question else len(history)-prefix_cells-outputs
            prep=value['answer'].get('input_preparation',{})
            if 'encodings' in prep and encoding_calls(prep)!=job['encoder_calls']:raise LatencyError('encoder receipts differ from actual encoder entry count')
            witness=dict(mode=mode,prefix_cells=prefix_cells,suffix_cells=suffix_cells,outputs=outputs,forward_tokens=job['forward_tokens'],
                source_native=job['source_native'],source_ram=job['source_ram'],source_disk=job['source_disk'],image_cached=job['image_cached'],
                loaded_file_bytes=job['loaded_file_bytes'],disk_preload_bytes=job['disk_preload_bytes'],encoder_calls=job['encoder_calls'])
            validate_tier(witness)
            if question:
                thinking=value['answer'].get('thinking',{})
                if not thinking.get('complete') or thinking.get('final_text')!='OK':raise LatencyError('incomplete thinking/final answer')
            else:
                root=dict(branch=branch,receipt=value['receipt'],spans=history.spans if hasattr(history,'spans') else None)
                if text_only:root_text=root
                else:root_image=root
            row=dict(group=group,operation=op,warmup=repetition==0,seconds=elapsed,observed=observed,text_only=text_only,witness=witness,
                setup_seconds=job['setup_seconds'],worker_seconds=job['worker_seconds'],native_job=job,
                answer=value['answer'],receipt=value['receipt'],schedule=schedule,stream=observation,whole_cli_tps=outputs/elapsed if outputs else None)
            rows.append(row);result['measurements'].append(row);publish()
        result['summaries'][group]=summarize_repetitions(rows);publish()
        return rows
    try:
        probe._imports();probe.runtime_closure(profile,runtime);probe.tracker=probe.ProgressTracker(out,'vision-latency').start();probe.start_service(registry)
        pin=hashlib.sha256((str(out)+result['profile_sha256']).encode()).hexdigest();client=CowClient.cli(probe.ready_url(probe.service.ready),runtime/'tools/cow_cli.py',
            state_dir=out/'cli-state',journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
        for worker in client.status().capacity['workers'].values():probe.tracker.native_started(worker['pid'])
        namespace=client.create_namespace('Current vision latency','latency-namespace')['namespace_id'];upload=client.upload_image(namespace,a.image.read_bytes())
        _atomic_json(out/'context-parts.json',[dict(type='image',image_id=upload.image_id),dict(type='text',text=prefix_text)])
        run_group('cold-encode-prefill','cold');run_group('embedding-prefill','embedding')
        observed=run_group('resident','resident',question=True);plain=run_group('resident-plain-cli','resident',question=True,observed=False)
        run_group('ram','ram',question=True);run_group('disk','disk',question=True)
        run_group('text-prefill','text-prefill',text_only=True);run_group('text-resident','resident',question=True,text_only=True)
        returned=run_group('image-after-text','resident',question=True)
        baseline=observed[1]
        image_rows=[x for x in result['measurements'] if x['witness']['outputs'] and not x['text_only']]
        for row in image_rows:
            if row['answer']['tokens']!=baseline['answer']['tokens'] or row['native_job']['frontier_sha256']!=baseline['native_job']['frontier_sha256']:
                raise LatencyError('cache tier or text switch changed identical image continuation')
        result['cross_tier_tokens_frontiers_exact']=True
        result['observer_overhead_percent']=100*(result['summaries']['resident']['median_seconds']/result['summaries']['resident-plain-cli']['median_seconds']-1)
        result['overhead_preference_exceeded']=result['observer_overhead_percent']>8
        result['observer_overhead_scope']='Observer wrapper only; same server tier/timing hook in both arms. Three repetitions, not a statistical significance claim.'
        probe.stop_service();result['services']=probe.record['services'];result['final_pool_transitions']=json.loads((out/'pool-1.json').read_bytes())
        build=json.loads(a.reference_build.read_bytes());binary=build['binaries']['reference']
        if fingerprint(binary['path'])['sha256']!=binary['sha256']:raise LatencyError('reference binary changed')
        result['fresh_references']=[];result['numerical_comparisons']=[]
        for label,case in [('image',baseline),('text',next(x for x in result['measurements'] if x['group']=='text-resident' and not x['warmup']))]:
            probe.phase('fresh-reference-'+label);source=case['native_job']['source'];prefix=history_for_snapshot(probe.store,source)
            full=history_for_snapshot(probe.store,case['receipt']['snapshot']);suffix=list(full.native_tokens[len(prefix):len(full)-len(case['answer']['tokens'])])
            images=[];cell=0
            for span in prefix.spans:
                if span['kind']=='text':cell+=len(span['tokens'])
                else:
                    images.append(dict(start=cell,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']));cell+=span['rows']
            coords=out/(label+'-positions.i32');coords.write_bytes(b''.join(struct.pack('<3i',*row) for row in reference_positions(len(prefix),images,profile['execution']['capacity'])))
            lines=[f'COW_VISION_INPUT_V1 0 {len(prefix)} {len(images)}',' '.join(map(str,prefix.native_tokens))];cell=0
            for span in prefix.spans:
                if span['kind']=='text':cell+=len(span['tokens'])
                else:
                    embedding=out/'retained-embeddings'/(span['embedding_sha256']+'.sve')
                    if fingerprint(embedding)['sha256']!=span['embedding_sha256']:raise LatencyError('retained embedding changed')
                    fields=[cell]+[span[k] for k in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
                    lines.append(' '.join(map(str,fields))+' '+json.dumps(str(embedding)));cell+=span['rows']
            manifest=out/(label+'-prefix.manifest');manifest.write_text('\n'.join(lines)+'\n',newline='\n')
            argv=list(profile['execution']['sut_command']);argv[0]=binary['path'];native=Native(argv,out/('reference-'+label))
            ref=dict(label=label,pid=native.proc.pid,binary=binary,forced_observed_tokens=True);result['fresh_references'].append(ref);publish()
            try:
                if native.ready.get('reference')!='fresh-no-checkpoints':raise LatencyError('reference imports cached state')
                receipt=native.request(reference_prefix_command(prefix.native_tokens,manifest,coords,len(images)))
                if receipt['chunks']!=case['schedule'][0]['chunks']:raise LatencyError('reference prefix arrangement differs')
                prefix_path,_=native.frontier('prefix');roots=[x for x in result['measurements'] if x['receipt']['snapshot']==source and not x['witness']['outputs']]
                if not roots:raise LatencyError('missing actual prepared-prefix frontier')
                comparison=dict(label=label+'-prefix',**compare_files(roots[-1]['native_job']['frontier_file'],prefix_path));result['numerical_comparisons'].append(comparison)
                if not comparison['exact']:raise LatencyError('reference prefix differs')
                receipt=native.request('feed '+' '.join(map(str,suffix)));matching=[x for x in case['schedule'] if x['position']==len(prefix) and x['count']==len(suffix)]
                if len(matching)!=1 or matching[0]['chunks']!=receipt['chunks']:raise LatencyError('reference suffix arrangement differs')
                for token in case['answer']['tokens']:native.request('decode '+str(token))
                final,_=native.frontier('final');comparison=dict(label=label+'-final',**compare_files(case['native_job']['frontier_file'],final));result['numerical_comparisons'].append(comparison)
                if not comparison['exact']:raise LatencyError('reference final differs')
                ref['work']=native.request('stats')
                if ref['work']['forward_tokens']!=len(full):raise LatencyError('fresh reference omitted prefix or output work')
            finally:ref['exit']=close_reference(native);publish()
            if ref['exit']['exit_code']!=0 or not ref['exit']['process_absent'] or ref['exit'].get('shutdown_error'):raise LatencyError('reference process did not retire')
        result.update(status='BOUNDED_CURRENT_Q4_VISION_LATENCY_MEASURED',numerical_max_error=0.0,
            numerical_scope='Identical tokens and final frontiers across image tiers/return-from-text, plus independent uncached raw image/text prefix/final references forcing observed generated tokens. Not an every-token sampler-selection oracle.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        if probe.service is not None:
            try:
                if probe.record['services']:
                    probe.stop_service()
                else:
                    # A parser/startup refusal has no ready-service row. Retire
                    # its actual process without inventing native exit evidence.
                    result['startup_failure_cleanup']=probe.service.stop()
                    probe.service=None
            except BaseException as error:result.update(status='FAIL',cleanup_error=repr(error))
        if probe.tracker is not None:probe.tracker.finish(result['status']);probe.tracker.close()
        result['services']=probe.record['services'];publish()
    return 0

if __name__=='__main__':raise SystemExit(main())
