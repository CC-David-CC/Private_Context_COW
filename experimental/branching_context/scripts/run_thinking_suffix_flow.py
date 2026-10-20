"""Real Q4 thinking-enabled context/questions, replay and producer-free restart.

Controller grammar permits final OK to isolate mechanics from answer quality.
Each generated token is consumed before publication. Raw fresh native references
force those observed tokens and compare complete logits under identical batches;
they do not validate the semantic quality of the thought or choose grammar tokens.
"""
import argparse, hashlib, json, math, os, shutil, struct, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

class FlowError(RuntimeError): pass

def validate_question(row):
    for key in ('prefix_cells','suffix_cells','generated','forward_tokens','reasoning_tokens','thinking_budget'):
        if type(row.get(key)) is not int or row[key] < (1 if key in ('prefix_cells','suffix_cells','generated') else 0):
            raise FlowError('missing or invalid witness: '+key)
    for key in ('prefix_unchanged','eos_consumed','controller_stopped','source_controller_unchanged'):
        if row.get(key) is not True: raise FlowError('failed witness: '+key)
    if row['forward_tokens'] != row['suffix_cells']+row['generated']:
        raise FlowError('request replayed prefix or omitted consumed work')
    if row['reasoning_tokens'] > row['thinking_budget']:
        raise FlowError('thinking budget exceeded')

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def reference_ingest(history,start,embeddings,capacity,directory,label):
    """Fresh-reference input packet; coordinates come from the CPU fixture.

    Only new image payloads are read. Earlier image geometry remains necessary
    for absolute rotary coordinates; cached model state is never imported.
    This is transient oracle input, not another canonical history or cache.
    """
    import re
    from context_time_travel.input_spans import InputHistory
    from context_time_travel.store import InvalidOperation
    from vision_fixture import reference_positions
    if (not isinstance(history,InputHistory) or type(start) is not int
            or type(capacity) is not int or not 0 <= start < len(history) <= capacity
            or not isinstance(label,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',label)):
        raise FlowError('invalid reference history range or label')
    try:
        history.prefix(start);delta=history.segment(start)
    except InvalidOperation as error:
        raise FlowError('reference boundary splits image rows') from error
    if not history.spans or history.spans[-1]['kind']!='text':
        raise FlowError('reference public boundary ends inside image input')
    images=[span for span in delta.spans if span['kind']=='image']
    if len(images)>8:raise FlowError('reference ingest supports at most eight new images')
    if not images:
        return dict(command='feed '+' '.join(map(str,delta.native_tokens)),image_count=0,count=len(delta))
    lines=[f'COW_VISION_INPUT_V1 {start} {len(delta)} {len(images)}',' '.join(map(str,delta.native_tokens))]
    cell=start
    for span in delta.spans:
        if span['kind']=='text':cell+=len(span['tokens']);continue
        path=embeddings.get(span['embedding_sha256'])
        if path is None:raise FlowError('missing retained reference embedding')
        path=Path(path)
        if not path.is_file() or path.is_symlink():raise FlowError('invalid retained reference embedding')
        raw=path.read_bytes()
        if len(raw)!=span['embedding_bytes'] or hashlib.sha256(raw).hexdigest()!=span['embedding_sha256']:
            raise FlowError('reference embedding checksum/length mismatch')
        header=(0x31455653,span['rows'],span['grid_width'],span['grid_height'],span['width'])
        if len(raw)<20 or struct.unpack('<5i',raw[:20])!=header:
            raise FlowError('reference embedding shape mismatch')
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack('<f',raw[20:])):
            raise FlowError('reference embedding contains nonfinite values')
        fields=[cell]+[span[k] for k in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
        lines.append(' '.join(map(str,fields))+' '+json.dumps(str(path)));cell+=span['rows']
    all_images=[];cell=0
    for span in history.spans:
        if span['kind']=='text':cell+=len(span['tokens'])
        else:
            all_images.append(dict(start=cell,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']));cell+=span['rows']
    directory=Path(directory);manifest=directory/(label+'.manifest');positions=directory/(label+'-positions.i32')
    with positions.open('xb') as output:
        output.write(b''.join(struct.pack('<3i',*row) for row in reference_positions(len(history),all_images,capacity)))
    with manifest.open('x',newline='\n') as output:output.write('\n'.join(lines)+'\n')
    return dict(command='feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(positions)),
                image_count=len(images),count=len(delta),manifest=str(manifest),positions=str(positions))

def question_command(status,append):
    if status=='idle' and append is False:return 'ask'
    if status=='stopped' and append is True:return 'append-user'
    raise FlowError('question command does not match the saved controller phase')

def validate_suffix_images(delta,originals,header):
    actual=[span['original_sha256'] for span in delta.spans if span['kind']=='image']
    if actual!=list(originals) or not originals:
        raise FlowError('new image identities/order do not match the requested uploads')
    if tuple(delta.native_tokens[:len(header)])!=tuple(header) or not header:
        raise FlowError('new image suffix repeated system context or lost its user boundary')

def install_image_observer(body,root):
    marker="importlib.import_module('serve.cow_http').main()\n"
    if body.count(marker)!=1:raise FlowError('ambiguous image observer entrypoint')
    hook=r"""import threading,struct,time
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.strata_backend import StrataBackend
from serve.cow_vision_encoder import CowNativeVisionEncoder
_TLS=threading.local();_OBS_ROOT=Path(OUTPUT_ROOT)
_original_observed_run=ContextCoordinator._run
_original_observed_prepare=ContextCoordinator._prepare_suffix_on_worker
_original_observed_encode=CowNativeVisionEncoder.encode
_original_observed_inputs=StrataBackend.consume_inputs
def _observed_run(self,model,job):
 _TLS.operation=job.request.operation
 try:return _original_observed_run(self,model,job)
 finally:_TLS.operation=None
def _observed_prepare(self,model,job):
 previous=getattr(_TLS,'operation',None);_TLS.operation=job.request.operation
 try:return _original_observed_prepare(self,model,job)
 finally:_TLS.operation=previous
def _observed_encode(self,*args,**kwargs):
 with (_OBS_ROOT/'encoder-entries.jsonl').open('a') as output:
  output.write(json.dumps(dict(operation=getattr(_TLS,'operation',None),event='actual_encode_entry',time_ns=time.monotonic_ns()))+'\n');output.flush()
 return _original_observed_encode(self,*args,**kwargs)
def _observed_inputs(self,lease,spans,**kwargs):
 value=_original_observed_inputs(self,lease,spans,**kwargs)
 if any(span['kind']=='image' for span in spans):
  operation=getattr(_TLS,'operation',None)
  if not isinstance(operation,str):raise RuntimeError('image ingest escaped request ownership')
  raw=struct.pack('<%df'%self.vocab_size,*self.logits(lease))
  (_OBS_ROOT/(operation+'-ingested.f32')).write_bytes(raw)
 return value
ContextCoordinator._run=_observed_run
ContextCoordinator._prepare_suffix_on_worker=_observed_prepare
CowNativeVisionEncoder.encode=_observed_encode
StrataBackend.consume_inputs=_observed_inputs
"""
    return body.replace(marker,hook.replace('OUTPUT_ROOT',repr(str(root)))+'\n'+marker)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','branchwrite','profile','dependency-root','vision-config','image','reference-build','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--suffix-image',type=Path,help='new image fixture; also exercise repeated placements and a completed-turn image suffix')
    a=parser.parse_args(argv);runtime=a.runtime.resolve(strict=True);bw=a.branchwrite.resolve(strict=True)
    sys.path[:0]=list(map(str,[bw,runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',a.dependency_root]))
    from run_cli_grammar_probe import GrammarProbe,_atomic_json
    from run_vision_pool_pressure import install_observer
    from run_vision_ingest_probe import Native,compare_files,fingerprint
    from run_vision_latency import close_reference
    from branchwrite.cow_client import CowClient,OperationJournal
    from branchwrite.cow_actions import completed_thinking_text
    from context_time_travel.input_spans import history_for_snapshot
    from context_time_travel.store import Store
    from vision_fixture import reference_positions
    out=a.output.resolve();out.mkdir(mode=0o700,exist_ok=False)
    if shutil.disk_usage(out).free<1500*1024**2:raise FlowError('need1500MiB free for bounded serving audit')
    profile=json.loads(a.profile.read_bytes())
    if a.suffix_image is not None:
        from tools.strata_tokenizer import Tokenizer
        tokenizer=Tokenizer.from_gguf(profile['model_path'])
    if profile['execution']['text_template_kwargs']!={'enable_thinking':True}:raise FlowError('requires thinking profile')
    record=dict(schema=1,status='RUNNING',scope=__doc__,cases=[],fresh_references=[],comparisons=[],services=[],cli_replays=[],
        profile=fingerprint(a.profile),execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
        harness=fingerprint(Path(__file__)),chat_server='OFF',q8='DEFERRED_BY_OWNER',numerical_max_error=None)
    def publish():_atomic_json(out/'result.json',record)
    class Owner(GrammarProbe):
        port=None;producer=True
        def _bootstrap(self,registry):
            path=super()._bootstrap(registry);body=path.read_text();marker="importlib.import_module('serve.cow_http').main()\n"
            if body.count(marker)!=1:raise FlowError('ambiguous service entrypoint')
            extra=[]
            if self.producer:extra+=['--vision-config',str(a.vision_config.resolve(strict=True))]
            if self.port is not None:extra+=['--port',str(self.port)]
            body=body.replace(marker,'sys.argv.extend('+repr(extra)+')\n'+marker)
            body=install_observer(body,out)
            if a.suffix_image is not None:body=install_image_observer(body,out)
            path.write_text(body,newline='\n');return path
    probe=Owner(SimpleNamespace(runtime_root=runtime,profile=a.profile,store=out/'store',run_dir=out,
        dependency_root=a.dependency_root,ram_gib=96,disk_gib=2))
    registry=out/'registry.json';_atomic_json(registry,dict(schema=1,states=dict(REPLY=dict(format='thinking',source=dict(
        schema=1,max_tokens=512,final=dict(format='gbnf',source='root ::= "OK"'))))))
    errors=[];client=None
    def rows():
        path=out/'jobs.jsonl'
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    def job(op):
        found=[x for x in rows() if x['operation']==op]
        if len(found)!=1:raise FlowError('missing or repeated native job: '+op)
        return found[0]
    def status():return dict(client.status().capacity)
    def start(producer):
        nonlocal client
        probe.producer=producer;probe.start_service(registry)
        url=probe.ready_url(probe.service.ready);probe.port=urlsplit(url).port
        pin=hashlib.sha256((str(out/'store')+record['profile']['sha256']).encode()).hexdigest()
        client=CowClient.cli(url,runtime/'tools/cow_cli.py',state_dir=out/'cli-state',
            journal=OperationJournal(out/'journal'),api_key=probe.secret,endpoint_pin=pin)
        for worker in status()['workers'].values():probe.tracker.native_started(worker['pid'])
        publish()
    def stop():
        probe.stop_service();record['services']=probe.record['services'];publish()
    def checkpoint(sid):
        envelope=json.loads((out/'store/checkpoints'/(sid+'.json')).read_bytes())
        return envelope['manifest']
    def replay(op):
        before=status();count=len(rows());probe.phase('original-request-replay-'+op)
        command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',probe.ready_url(probe.service.ready),
            '--state-dir',str(out/'cli-state'),'--machine-result','retry-request','--operation-id',op]
        done=subprocess.run(command,env={**os.environ,'COW_API_KEY':probe.secret},capture_output=True,timeout=180)
        (out/('replay-'+op+'-'+str(probe.ordinal)+'.stdout')).write_bytes(done.stdout)
        (out/('replay-'+op+'-'+str(probe.ordinal)+'.stderr')).write_bytes(done.stderr)
        if done.returncode!=0:raise FlowError('original retry CLI failed')
        after=status()
        if len(rows())!=count or before['workers']!=after['workers']:raise FlowError('retry issued native work or changed workers')
        value=json.loads(done.stdout);record['cli_replays'].append(dict(operation=op,service=probe.ordinal,result=value,native_jobs_delta=0));publish()
        return value
    case_map={}
    def question(branch,op,text,*,append=False,image_ids=()):
        probe.phase(op);store=probe.store;head=store.head(branch);source=history_for_snapshot(store,head.tip)
        old=store.get_snapshot(head.tip);command=question_command(old['controller']['status'],append)
        method=client.append_user if command=='append-user' else client.ask
        begin=time.monotonic()
        if a.suffix_image is None:
            value=method(branch,head.revision,op,text,1024,settings=dict(temperature=0.),grammar_state='REPLY')
        else:
            args=['--machine-result',command,'--branch',branch,'--revision',str(head.revision),
                  '--max-tokens','1024','--temperature','0','--seed','17','--grammar-state','REPLY','--operation-id',op]
            if image_ids:
                parts=out/(op+'-parts.json');_atomic_json(parts,[*[dict(type='image',image_id=key) for key in image_ids],dict(type='text',text=text)])
                args+=['--parts-file',str(parts)]
            else:args+=['--text',text]
            row,_=probe.cli(op,*args);raw=json.loads(row['stdout_text'])
            value=SimpleNamespace(answer=raw['answer'],receipt=SimpleNamespace(snapshot=raw['receipt']['snapshot'],raw=raw['receipt']))
        final=completed_thinking_text(value,profile['execution']['eos_tokens'])
        if final!='OK':raise FlowError('final text not accepted by immutable thinking grammar')
        current=history_for_snapshot(store,value.receipt.snapshot);tokens=list(value.answer['tokens']);ob=job(op)
        suffix=list(current.native_tokens[len(source):len(current)-len(tokens)])
        delta=current.segment(len(source),len(current)-len(tokens))
        if image_ids:
            header=('\n' if old['controller']['status']=='stopped' else '')+'<|im_start|>user\n'
            validate_suffix_images(delta,[upload_originals[key] for key in image_ids],tokenizer.encode(header,parse_special=True))
            if not (out/(op+'-ingested.f32')).is_file():raise FlowError('missing actual consumed new-image frontier')
        for path in (out/'store/checkpoints').glob('*.sve'):
            destination=out/path.name
            if not destination.exists():shutil.copy2(path,destination)
            embeddings[path.stem]=destination
        controller=store.get_snapshot(value.receipt.snapshot)['controller'];thinking=value.answer['thinking']['progress']
        witness=dict(prefix_cells=len(source),suffix_cells=len(suffix),generated=len(tokens),forward_tokens=ob['forward_tokens'],
            prefix_unchanged=current.segment(0,len(source))==source,eos_consumed=bool(tokens) and tokens[-1] in profile['execution']['eos_tokens'],
            controller_stopped=controller['status']=='stopped',reasoning_tokens=thinking['reasoning_tokens'],thinking_budget=512,
            source_controller_unchanged=store.get_snapshot(head.tip)==old)
        validate_question(witness)
        item=dict(operation=op,branch=branch,source=head.tip,snapshot=value.receipt.snapshot,tokens=tokens,suffix=suffix,
            witness=witness,answer=dict(value.answer),receipt=dict(value.receipt.raw),controller=controller,seconds=time.monotonic()-begin,
            frontier=fingerprint(out/ob['frontier_file']),schedule=checkpoint(value.receipt.snapshot)['execution_schedule'],
            suffix_spans=delta.spans,new_image_ids=list(image_ids))
        if image_ids:
            item['ingested_frontier']=fingerprint(out/(op+'-ingested.f32'))
            entries=[json.loads(line) for line in (out/'encoder-entries.jsonl').read_text().splitlines()]
            item['actual_encoder_calls']=sum(x['operation']==op for x in entries)
        record['cases'].append(item);case_map[op]=item;publish();return item
    try:
        probe._imports();probe.runtime_closure(profile,runtime)
        probe.tracker=probe.ProgressTracker(out,'thinking-suffix-flow').start();start(True)
        namespace=client.create_namespace('Thinking shared image questions','suffix-ns')['namespace_id']
        upload=client.upload_image(namespace,a.image.read_bytes());upload_originals={upload.image_id:sha(a.image)}
        new_upload=None
        if a.suffix_image is not None:
            new_upload=client.upload_image(namespace,a.suffix_image.read_bytes());upload_originals[new_upload.image_id]=sha(a.suffix_image)
            if new_upload.image_id==upload.image_id:raise FlowError('same-grid fixture must have different image bytes')
        # BranchWrite's load_context CLI bridge uses the plain-text `load`
        # command. Exercise the existing mixed-input `context --parts-file`
        # route explicitly; do not claim that bridge supports image messages.
        parts=out/'context-parts.json';_atomic_json(parts,[dict(type='image',image_id=upload.image_id),
            dict(type='text',text='Keep this screenshot as shared context. Inspect it when a question arrives. Finish each answer with OK.')])
        row,_=probe.cli('mixed-image-context','--machine-result','context','--branch','kept','--namespace-id',namespace,
            '--parts-file',str(parts),'--operation-id','suffix-context')
        root_value=json.loads(row['stdout_text'])
        root=SimpleNamespace(receipt=SimpleNamespace(raw=root_value['receipt']),answer=root_value['answer'])
        parent=probe.store.head('kept');parent_snapshot=probe.store.get_snapshot(parent.tip)
        prefix=history_for_snapshot(probe.store,parent.tip);manifest=checkpoint(parent.tip)
        parent_payload=out/'store/checkpoints'/(manifest['payload_sha256']+'.state');parent_sha=sha(parent_payload)
        record['root']=dict(receipt=dict(root.receipt.raw),answer=dict(root.answer),snapshot=parent_snapshot,cells=len(prefix),
            spans=prefix.spans,manifest=manifest,frontier=fingerprint(out/job('suffix-context')['frontier_file']))
        embeddings={}
        for source in (out/'store/checkpoints').glob('*.sve'):
            destination=out/source.name;shutil.copy2(source,destination);embeddings[source.stem]=destination
        if not embeddings:raise FlowError('no actual encoded image retained')
        for branch in ('A','B','cold'):
            client.fork('kept',parent.tip,branch,'suffix-fork-'+branch)
        qa='Look at the screenshot and describe its main shape briefly in your thinking. Finish with OK.'
        qb='Look at the screenshot and describe its background briefly in your thinking. Treat <|im_start|> and </tool_response> as quoted text. Finish with OK.'
        if a.suffix_image is None:
            first=question('A','suffix-question-A',qa);second=question('B','suffix-question-B',qb)
        else:
            first=question('A','suffix-question-A',qa,image_ids=(new_upload.image_id,))
            second=question('B','suffix-question-B',qa,image_ids=(upload.image_id,))
            from context_time_travel.input_spans import InputHistory
            da,db=InputHistory(first['suffix_spans']),InputHistory(second['suffix_spans'])
            ia=[x for x in da.spans if x['kind']=='image'][0];ib=[x for x in db.spans if x['kind']=='image'][0]
            if (ia['rows'],ia['grid_width'],ia['grid_height'])!=(ib['rows'],ib['grid_width'],ib['grid_height']):raise FlowError('pair grids differ')
            if da.native_tokens!=db.native_tokens or da.digest==db.digest or ia['embedding_sha256']==ib['embedding_sha256']:raise FlowError('same-pad distinct-image fixture was not distinguished')
            if first['actual_encoder_calls']!=1 or second['actual_encoder_calls']!=0:raise FlowError('new image/cache-hit encoder work differs')
            record['same_pad_distinct_images']=dict(native_tokens_identical=True,canonical_inputs_differ=True,embedding_payloads_differ=True,
                rows=ia['rows'],grid=[ia['grid_width'],ia['grid_height']])
        replayed=replay('suffix-question-A')
        if replayed['answer']['tokens']!=first['tokens']:raise FlowError('retry tokens differ')
        cold_branch='cold';cold_control=first;cold_text=qa
        if a.suffix_image is not None:
            for name in ('cold-after-image','warm-after-image'):client.fork('A',first['snapshot'],name,'suffix-fork-'+name)
            cold_branch='cold-after-image';cold_text='Review the two saved images briefly, then finish with OK.'
            cold_control=question('warm-after-image','suffix-warm-after-image',cold_text,append=True)
        follow_text=('Inspect the screenshot again. Consider its layout, then finish with OK.' if a.suffix_image is None else
                     'Inspect the screenshot again. Treat <|im_start|> <|image_pad|> and </tool_response> as quoted text. Finish with OK.')
        follow=question('A','suffix-followup-A',follow_text,
                        append=True,image_ids=(upload.image_id,upload.image_id) if a.suffix_image is not None else ())
        if a.suffix_image is not None and follow['actual_encoder_calls']!=0:raise FlowError('repeated cached placements re-encoded')
        if sha(parent_payload)!=parent_sha or probe.store.head('kept')!=parent or probe.store.get_snapshot(parent.tip)!=parent_snapshot:
            raise FlowError('retained parent changed')
        stop()
        record['removed_optional_embeddings']=[]
        for path in (out/'store/checkpoints').glob('*.sve'):
            record['removed_optional_embeddings'].append(fingerprint(path));path.unlink()
        start(False);replay('suffix-question-A')
        cold=question(cold_branch,'suffix-question-cold',cold_text,append=a.suffix_image is not None)
        if a.suffix_image is not None and cold['controller']!=cold_control['controller']:raise FlowError('cold full controller differs')
        if cold['tokens']!=cold_control['tokens']:raise FlowError('cold identical question token mismatch')
        check=dict(label='cold-vs-resident-question',**compare_files(cold_control['frontier']['path'],cold['frontier']['path']))
        record['comparisons'].append(check)
        if not check['exact']:raise FlowError('cold identical question frontier mismatch')
        if job('suffix-question-cold')['after']['loaded_file_bytes']<=job('suffix-question-cold')['before']['loaded_file_bytes']:
            raise FlowError('cold request did not load actual checkpoint bytes')
        if list((out/'store/checkpoints').glob('*.sve')):raise FlowError('producer-free question reconstructed embeddings')
        if sha(parent_payload)!=parent_sha or probe.store.get_snapshot(parent.tip)!=parent_snapshot:raise FlowError('cold parent changed')
        record['parent_payload_unchanged']=dict(sha256=parent_sha,bytes=parent_payload.stat().st_size)
        stop()
        build=json.loads(a.reference_build.read_bytes());binary=build['binaries']['reference']
        if sha(binary['path'])!=binary['sha256']:raise FlowError('fresh reference executable changed')
        positions=[];cell=0
        for span in prefix.spans:
            if span['kind']=='text':cell+=len(span['tokens'])
            else:
                positions.append(dict(start=cell,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']));cell+=span['rows']
        coords=out/'reference-positions.i32';coords.write_bytes(b''.join(struct.pack('<3i',*r) for r in reference_positions(len(prefix),positions,profile['execution']['capacity'])))
        lines=[f'COW_VISION_INPUT_V1 0 {len(prefix)} {len(positions)}',' '.join(map(str,prefix.native_tokens))];cell=0
        for span in prefix.spans:
            if span['kind']=='text':cell+=len(span['tokens'])
            else:
                fields=[cell]+[span[k] for k in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
                lines.append(' '.join(map(str,fields))+' '+json.dumps(str(embeddings[span['embedding_sha256']])));cell+=span['rows']
        packet=out/'prefix.manifest';packet.write_text('\n'.join(lines)+'\n',newline='\n')
        reference_cases=[('A',['suffix-question-A']),('B',['suffix-question-B']),('followup',['suffix-question-A','suffix-followup-A'])]
        if a.suffix_image is not None:reference_cases.append(('after-image',['suffix-question-A','suffix-warm-after-image']))
        for label,operations in reference_cases:
            probe.phase('independent-raw-fresh-'+label);command=list(profile['execution']['sut_command']);command[0]=binary['path']
            native=Native(command,out/('reference-'+label));ref=dict(label=label,pid=native.proc.pid,binary=binary,forced_observed_tokens=True)
            record['fresh_references'].append(ref);publish()
            try:
                if native.ready.get('reference')!='fresh-no-checkpoints':raise FlowError('reference imports state')
                prefix_receipt=native.request('feed_image '+json.dumps(str(packet))+' '+json.dumps(str(coords)))
                if prefix_receipt['chunks']!=manifest['execution_schedule'][0]['chunks']:raise FlowError('fresh prefix batch arrangement differs')
                path,_=native.frontier('prefix');comparison=dict(label=label+'-prefix',**compare_files(record['root']['frontier']['path'],path));record['comparisons'].append(comparison)
                if not comparison['exact']:raise FlowError('fresh prefix differs')
                count=len(prefix)
                for op in operations:
                    case=case_map[op]
                    consumed=history_for_snapshot(probe.store,case['snapshot']).prefix(count+len(case['suffix']))
                    operation_packet=reference_ingest(consumed,count,embeddings,profile['execution']['capacity'],out,'reference-'+label+'-'+op)
                    receipt=native.request(operation_packet['command'])
                    matching=[r for r in case['schedule'] if r['position']==count and r['count']==len(case['suffix'])]
                    if len(matching)!=1 or receipt['chunks']!=matching[0]['chunks']:raise FlowError('fresh suffix batch arrangement differs')
                    if case.get('ingested_frontier'):
                        path,_=native.frontier(op+'-ingested');comparison=dict(label=label+'-'+op+'-ingested',**compare_files(case['ingested_frontier']['path'],path));record['comparisons'].append(comparison)
                        if not comparison['exact']:raise FlowError('fresh new-image consumed frontier differs')
                    for token in case['tokens']:native.request('decode '+str(token))
                    count+=len(case['suffix'])+len(case['tokens'])
                path,_=native.frontier('final');comparison=dict(label=label+'-final',**compare_files(case['frontier']['path'],path));record['comparisons'].append(comparison)
                if not comparison['exact']:raise FlowError('fresh forced-token final frontier differs')
                ref['work']=native.request('stats')
                if ref['work']['forward_tokens']!=count:raise FlowError('fresh reference omitted or reused prefix work')
            finally:ref['exit']=close_reference(native);publish()
            if ref['exit']['exit_code']!=0 or not ref['exit']['process_absent'] or ref['exit'].get('shutdown_error'):raise FlowError('reference did not exit cleanly')
        for service in record['services']:
            for pid in (service['pid'],service['worker_shutdown']['native_pid']):
                if Path('/proc',str(pid)).exists():raise FlowError('owned service process remains')
        record.update(status='BOUNDED_Q4_THINKING_NEW_IMAGE_SUFFIX_RESTART_PASS' if a.suffix_image is not None else 'BOUNDED_Q4_THINKING_CONTEXT_SUFFIX_RESTART_PASS',numerical_max_error=0.0,
            limitations=['Final OK constrained; thought quality not scored. Fresh references force observed generated tokens; prefix/final frontiers only, not every token.',
            'Only the declared image fixtures/controllers are tested; broad format/controller/performance matrix and original G1-G5 remain unqualified. Q8 deferred.'])
    except BaseException as error:
        errors.append(repr(error));record.update(status='FAIL',errors=errors);raise
    finally:
        if probe.service is not None:
            try:stop()
            except BaseException as error:errors.append('cleanup: '+repr(error));record.update(status='FAIL',errors=errors)
        if probe.tracker is not None:
            probe.tracker.finish(record['status']);probe.tracker.close()
        record['services']=probe.record['services'];publish()
    return 0

if __name__=='__main__':raise SystemExit(main())
