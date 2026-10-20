"""VC-015: real re-encoding and explicit model reconstruction after eviction.

Imports a retained canonical store from an executed restart campaign. Rebuild
workers cannot read its old SVE fixture and must use the pinned real encoder.
Numerical references use the reconstruction's own successful batch arrangement.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import SharedCheckpointPool, DiskCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend
from context_time_travel.store import IntegrityError, NotFound, Store
from context_time_travel.vision_artifacts import ImageArtifacts
from run_vision_ingest_probe import Native, fingerprint, save, compare_files
from run_vision_restart_probe import frontier, controller_file, audit_unchanged


def audit_payload_read(event, arguments, managed_directory, allowed_outputs):
    """Forbid retained-fixture reads while permitting fresh transport writes."""
    if event != 'open' or not isinstance(arguments[0], (str, bytes)):
        return
    # CPython's open audit event supplies native open flags as its third item.
    # O_RDONLY is zero; checking truthiness would miss precisely those reads.
    if arguments[2] & os.O_ACCMODE == os.O_WRONLY:
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    if (path.suffix == '.sve' and path.parent != managed_directory.resolve()
            and str(path) not in allowed_outputs):
        raise AssertionError('Attempted to read an old external embedding: ' + str(path))


def canonical_files(store):
    return {str(path.relative_to(store.root)):fingerprint(path)['sha256']
            for name in ('objects','references','image-originals','image-encodings')
            for path in sorted((store.root/name).glob('*')) if path.is_file()}


class Encoder:
    def __init__(self, config, out, result):
        baseline=json.loads(Path(config['encoder_baseline']).read_text())
        self.argv=baseline['encodings'][0]['argv'];self.out=out;self.result=result
        self.records=result.setdefault('encoder_runs',[])
        # Verify the actual executable and projector, not just their filenames.
        for path in (self.argv[0],self.argv[self.argv.index('--mmproj')+1]):
            expected=baseline['files'][path];actual=fingerprint(path)
            assert actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes']
        self.allowed_outputs=set()

    def encode(self, original):
        directory=self.out/('encoder-'+str(len(self.records)));directory.mkdir()
        image=directory/'input.png';output=directory/'output.sve'
        image.write_bytes(original);self.allowed_outputs.add(str(output.resolve()))
        row=dict(argv=self.argv,input=fingerprint(image),started_ns=time.time_ns(),status='RUNNING')
        self.records.append(row)
        process=subprocess.Popen(self.argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
        row['pid']=process.pid;save(self.out/'result.json',self.result)
        started=time.monotonic()
        try:
            stdout,stderr=process.communicate(f'ENC {image} {output}\nQUIT\n'.encode(),timeout=120)
            (directory/'stdout.log').write_bytes(stdout);(directory/'stderr.log').write_bytes(stderr)
            assert process.returncode==0, row
            lines=stdout.decode().splitlines()
            assert len([x for x in lines if x.startswith('READY ')])==1
            assert len([x for x in lines if x.startswith('OK ')])==1,lines
            payload=output.read_bytes();row['output']=fingerprint(output);row['status']='PASS'
            return original,payload
        except BaseException as error:
            row.update(status='FAIL',error=repr(error));raise
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=30)
            row.update(exit_code=process.returncode,process_absent=not Path(f'/proc/{process.pid}').exists(),
                finished_ns=time.time_ns(),seconds=time.monotonic()-started)
            # A test-output copy must not survive the deliberate cache eviction.
            output.unlink(missing_ok=True);image.unlink(missing_ok=True)
            save(self.out/'result.json',self.result)


def worker(config, stage, branch, out):
    result=dict(status='RUNNING',stage=stage,python_pid=os.getpid())
    done=threading.Event();backend=ram=None
    def heartbeat():
        while not done.wait(30):
            print(json.dumps(dict(stage=stage,python_pid=os.getpid(),native_pid=result.get('native_pid'),
                encoder_pids=[x['pid'] for x in result.get('encoder_runs',[]) if x.get('status')=='RUNNING'])),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        profile=json.loads(Path(config['profile']).read_text());store=Store(config['store'])
        prepared=json.loads(Path(config['prepared']).read_text())
        snapshot=prepared['prefix'] if stage in ('seed','root') or (stage=='reference' and config['reference_case']=='root') else prepared['generated_four']
        history=InputHistory(store.input_lineage(snapshot)[1]);images=[x for x in history.spans if x['kind']=='image']
        assert len(images)==1,'This bounded fixture qualifies one retained image'
        span=images[0];expected_image=span['embedding_sha256']
        if stage=='reference':
            argv=list(profile['execution']['sut_command']);argv[0]=config['reference_binary']
            native=Native(argv,out/'native');result['native_pid']=native.proc.pid
            try:
                payload=store.root/'checkpoints'/(expected_image+'.sve')
                assert fingerprint(payload)['sha256']==expected_image
                tokens=history.native_tokens;start=0
                for block in history.spans:
                    if block['kind']=='image':break
                    start+=len(block['tokens'])
                manifest=out/'input.manifest'
                manifest.write_text(f'COW_VISION_INPUT_V1 0 {len(tokens)} 1\n'+' '.join(map(str,tokens))+'\n'+
                    f"{start} {span['rows']} {span['grid_width']} {span['grid_height']} {span['width']} {span['pad_token_id']} {expected_image} "+json.dumps(str(payload))+'\n')
                native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(config['reference_positions']))
                path,values=native.frontier('rebuilt');paths=[str(path)]
                if config['reference_case']=='root':
                    native.request('feed '+' '.join(map(str,config['suffixes'][branch])))
                    path,values=native.frontier('suffix');paths.append(str(path))
                tokens=[]
                for index in range(config['outputs']):
                    token=max(range(len(values)),key=values.__getitem__);tokens.append(token)
                    native.request('decode '+str(token));path,values=native.frontier('output-'+str(index+1));paths.append(str(path))
                    if token in profile['execution']['eos_tokens']:break
                result.update(tokens=tokens,frontiers=paths,work=native.request('stats'))
            finally:result['native_exit']=native.close()
        else:
            ram=SharedCheckpointPool(4*1024**3)
            artifacts=ImageArtifacts(store,ram,DiskCheckpointPool(store,32*1024**3),
                encoder_profile_id=span['encoder_profile_id'],expected_width=span['width'],
                original_bytes_limit=256*1024**2)
            encoder=Encoder(config,out,result)
            # Old fixture SVE files are outside this store and may not be read.
            def bounded_payload_reads(event,arguments):
                audit_payload_read(event,arguments,artifacts.disk.directory,encoder.allowed_outputs)
            sys.addaudithook(bounded_payload_reads)
            if stage=='seed':
                # Seed one real embedding copy so the following API eviction
                # removes actual cached bytes, not just an already empty folder.
                with artifacts.lease(span,reencode=encoder.encode) as held:
                    assert held.source=='reencoded' and held.embedding.sha256==expected_image
                    result['seeded_embedding']=dict(sha256=held.embedding.sha256,bytes=held.embedding.nbytes)
            else:
                os.environ['CTT_NATIVE_TRACE']=str(out/'native-trace.jsonl')
                started=time.monotonic();backend=StrataBackend(profile,gpu_id=0)
                result.update(native_pid=backend.process.pid,startup_seconds=time.monotonic()-started)
                api=BranchingModel(config['store'],backend,profile)
                before=canonical_files(store);initial_controller=api.controller(snapshot)
                result['evicted_files']=[fingerprint(p) for p in sorted(api.checkpoint_directory.iterdir()) if p.is_file()]
                assert any(x['path'].endswith('.sve') for x in result['evicted_files'])
                assert any(x['path'].endswith('.state') for x in result['evicted_files'])
                api.evict_all();assert not list(api.checkpoint_directory.iterdir())
                zero=backend.work_counters()
                try:api.restore_checkpoint(snapshot)
                except IntegrityError as error:result['strict_missing_error']=str(error)
                else:raise AssertionError('Strict restore silently reconstructed the prefix')
                assert backend.work_counters()['forward_tokens']==zero['forward_tokens']==0
                try:api.rebuild_materialization(snapshot,artifacts=artifacts)
                except NotFound as error:result['missing_encoder_error']=str(error)
                else:raise AssertionError('Missing embedding was reconstructed without an encoder')
                assert encoder.records==[] and backend.work_counters()['forward_tokens']==0
                if stage=='root':
                    def wrong_payload(original):
                        normalized,payload=encoder.encode(original)
                        wrong=bytearray(payload);wrong[-1]^=1
                        result['corrupted_reencoding']=dict(correct_sha256=hashlib.sha256(payload).hexdigest(),
                            altered_sha256=hashlib.sha256(wrong).hexdigest(),changed_bytes=1)
                        return normalized,bytes(wrong)
                    try:api.rebuild_materialization(snapshot,artifacts=artifacts,reencode=wrong_payload)
                    except IntegrityError as error:result['bad_reencoding_error']=str(error)
                    else:raise AssertionError('Changed encoder output was accepted')
                    assert backend.work_counters()['forward_tokens']==0
                    assert not list(api.checkpoint_directory.glob('*.sve'))
                    assert canonical_files(store)==before
                original_consume=backend.consume_inputs;consumption=[]
                def observed_consume(*args,**kwargs):
                    assert all(x['process_absent'] for x in encoder.records)
                    consumption.append(dict(started_ns=time.time_ns(),encoder_runs_completed=len(encoder.records)))
                    return original_consume(*args,**kwargs)
                backend.consume_inputs=observed_consume
                result['rebuild']=api.rebuild_materialization(snapshot,artifacts=artifacts,reencode=encoder.encode)
                assert result['rebuild']['forward_tokens']==len(history)
                assert result['rebuild']['reencode_callback_calls']==1
                assert result['rebuild']['image_artifacts'][0]['source']=='reencoded'
                assert canonical_files(store)==before and api.controller(snapshot)==initial_controller
                result['canonical_unchanged_after_rebuild']=True
                result['model_consumption']=consumption
                rebuilt=frontier(api,snapshot,out/'rebuilt.f32');result['rebuilt_frontier']=rebuilt
                result['controller_before']=controller_file(api,snapshot,out/'before-controller.json')
                old=prepared['expected']['A']['frontiers'][0 if stage=='root' else 5]
                result['historical_arrangement_comparison']=compare_files(old,rebuilt)
                if stage=='root':
                    assert result['historical_arrangement_comparison']['exact']
                    result['branches']={};parent=api.cache[snapshot].blob
                    result['parent_audit']=backend.audit(parent);result['parent_audits_after_branches']=[]
                    for name in ('A','B'):
                        tip=api.feed(snapshot,config['suffixes'][name]);paths=[rebuilt,frontier(api,tip,out/(name+'-suffix.f32'))]
                        active=api.start_generation(tip,dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),'restart-greedy')
                        tokens=[]
                        for index in range(config['outputs']):
                            batch=api.generate(active,1)
                            if not batch.accepted_tokens:break
                            active=batch.snapshot_id;tokens.extend(batch.accepted_tokens)
                            paths.append(frontier(api,active,out/(name+'-output-'+str(index+1)+'.f32')))
                            if tokens[-1] in profile['execution']['eos_tokens']:break
                        result['branches'][name]=dict(tokens=tokens,frontiers=paths,
                            controller=controller_file(api,active,out/(name+'-controller.json')))
                        assert tokens==prepared['expected'][name]['tokens']
                        assert Path(result['branches'][name]['controller']['path']).read_bytes()==Path(prepared['expected'][name]['controller']['path']).read_bytes()
                        audit_unchanged(backend,parent,result['parent_audit'],result['parent_audits_after_branches'])
                    # Rebuilding itself is not a durable save. Explicitly save
                    # here so the next worker can evict a real numerical copy.
                    api.checkpoint(snapshot)
                else:
                    active=snapshot;tokens=[];paths=[rebuilt]
                    for index in range(config['outputs']):
                        batch=api.generate(active,1)
                        if not batch.accepted_tokens:break
                        active=batch.snapshot_id;tokens.extend(batch.accepted_tokens)
                        paths.append(frontier(api,active,out/('output-'+str(index+1)+'.f32')))
                        if tokens[-1] in profile['execution']['eos_tokens']:break
                    result.update(tokens=tokens,frontiers=paths,controller=controller_file(api,active,out/'controller.json'))
                result['work']=backend.work_counters()
        result['status']='PASS'
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        if backend is not None:
            pid=backend.process.pid;backend.close()
            result['native_exit']=dict(pid=pid,exit_code=backend.process.returncode,process_absent=not Path(f'/proc/{pid}').exists())
        if ram is not None:ram.close()
        done.set();save(out/'result.json',result)


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();old=json.loads((baseline/'result.json').read_text());assert old['status']=='PASS'
    config=json.loads((baseline/'config.json').read_text())
    source_store=Path(config['store']);shutil.copytree(source_store,out/'store')
    config.update(store=str(out/'store'),prepared=str(baseline/'prepare/result.json'),
        encoder_baseline=str(args.encoder_baseline.resolve()),outputs=8,
        reference_positions=str(Path(config['fixture']).parent/'reference-positions.i32'))
    config_path=out/'config.json';save(config_path,config)
    result=dict(status='RUNNING',scope='bounded Q4 VC-015 explicit re-encoding/rebuild',
        baseline=fingerprint(baseline/'result.json'),encoder_baseline=fingerprint(args.encoder_baseline),
        stages=[],comparisons=[],prior_computation_preserved=False)
    def run(stage,name,*,branch='A',reference_case='root'):
        config['reference_case']=reference_case;save(config_path,config)
        directory=out/name;directory.mkdir()
        argv=[sys.executable,'-B',__file__,'--worker',stage,'--config',str(config_path),'--branch',branch,'--output',str(directory)]
        with (directory/'stdout.log').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:
            child=subprocess.Popen(argv,stdout=stdout,stderr=stderr);started=time.monotonic()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:print(json.dumps(dict(stage=name,python_pid=child.pid,seconds=time.monotonic()-started)),flush=True)
        row=dict(stage=name,argv=argv,python_pid=child.pid,exit_code=code,process_absent=not Path(f'/proc/{child.pid}').exists(),seconds=time.monotonic()-started)
        result['stages'].append(row);save(out/'result.json',result)
        assert code==0,row
        value=json.loads((directory/'result.json').read_text());assert value['status']=='PASS'
        if 'native_exit' in value:assert value['native_exit']['exit_code']==0 and value['native_exit']['process_absent']
        assert all(x['exit_code']==0 and x['process_absent'] for x in value.get('encoder_runs',[]))
        return value
    def exact(a,b,label):
        check=dict(label=label,**compare_files(a,b));result['comparisons'].append(check)
        save(out/'result.json',result);assert check['exact'],check
    def references(case,branch,actual):
        first=run('reference',case+'-'+branch+'-reference-0',branch=branch,reference_case=case)
        second=run('reference',case+'-'+branch+'-reference-1',branch=branch,reference_case=case)
        assert first['tokens']==second['tokens']==actual['tokens']
        assert len(first['frontiers'])==len(second['frontiers'])==len(actual['frontiers'])
        for index,(a,b,c) in enumerate(zip(first['frontiers'],second['frontiers'],actual['frontiers'])):
            exact(a,b,f'{case}-{branch}-fresh-repeat-{index}');exact(a,c,f'{case}-{branch}-rebuild-{index}')
    try:
        run('seed','seed')
        root=run('root','root')
        for branch in ('A','B'):references('root',branch,root['branches'][branch])
        generated=run('generated','generated')
        references('generated','A',generated)
        result.update(status='PASS',max_abs_error=0.0,qualification='VC-015 tiny greedy only; larger/multi-image qualification remains')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=('seed','root','generated','reference'))
    parser.add_argument('--config',type=Path);parser.add_argument('--branch',default='A')
    for key in ('baseline','encoder-baseline','output'):parser.add_argument('--'+key,type=Path)
    args=parser.parse_args()
    if args.worker:worker(json.loads(args.config.read_text()),args.worker,args.branch,args.output)
    else:campaign(args)
