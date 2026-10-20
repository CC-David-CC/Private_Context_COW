"""Real Q4 worker-owned root preparation canary, using pinned test images.

Runs the real encoder on cache misses, then coordinator Prefill/publication.
This is a fixed fixture preparer, not a production uploader or renderer.
References come from the separately executed uncached medium campaign.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import InvalidOperation, _json_bytes
from context_time_travel.strata_backend import StrataBackend
from context_time_travel.vision_artifacts import ImageArtifacts
from run_vision_ingest_probe import compare_files, fingerprint, save
from run_vision_rebuild_probe import Encoder
from vision_fixture import fixture_parts


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=args.baseline.resolve();prior=json.loads((base/'result.json').read_text())
    assert prior['status']=='PASS'
    config=json.loads((base/'config.json').read_text())
    config['encoder_baseline']=str(args.encoder_baseline.resolve())
    profile=json.loads(Path(config['profile']).read_text())
    fixture=json.loads(Path(config['fixture']).read_text())
    parts=fixture_parts(config['fixture'],config['images'])
    prior_prepare=json.loads((base/'prepare/result.json').read_text())
    result=dict(status='RUNNING',scope=__doc__,python_pid=os.getpid(),
        baseline=fingerprint(base/'result.json'),profile=fingerprint(config['profile']),
        workers=[],preparations=[],roots=[],branches={},comparisons=[],replays=[])
    encoder=Encoder(config,out,result)
    models=[];coordinator=None;done=threading.Event();phase={'name':'startup'}
    started=time.monotonic()
    def beat():
        while not done.wait(30):
            print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
                python_pid=os.getpid(),native_pids=[row['pid'] for row in result['workers']],
                encoder_runs=len(result.get('encoder_runs',[])))),flush=True)
    threading.Thread(target=beat,daemon=True).start()

    class FixturePreparer:
        identity=hashlib.sha256(_json_bytes(dict(kind='fixed-real-encoder-fixture-v1',
            fixture=fingerprint(config['fixture']),encoder_baseline=fingerprint(config['encoder_baseline']),
            model_profile=fingerprint(config['profile'])))).hexdigest()

        def validate(self,request):
            if request!={'fixture_sha256':fingerprint(config['fixture'])['sha256']}:
                raise InvalidOperation('Only the pinned fixture is admitted by this canary')
            return dict(request)

        @contextmanager
        def prepare(self,model,request,*,pool,disk,cancelled=None):
            assert threading.get_ident()==result['workers'][-1]['owner_thread']
            before=len(result.get('encoder_runs',[]));row=dict(owner_thread=threading.get_ident(),released=False)
            result['preparations'].append(row)
            artifacts=ImageArtifacts(model.store,pool,disk,encoder_profile_id=fixture['encoder_profile_id'],
                expected_width=parts[0]['width'],original_bytes_limit=256*1024**2)
            spans=[];cursor=0
            try:
                with ExitStack() as leases:
                    for part in parts:
                        if cancelled is not None and cancelled():raise InvalidOperation('Fixture preparation cancelled')
                        original=Path(part['original']['path']).read_bytes()
                        assert hashlib.sha256(original).hexdigest()==part['original']['sha256']
                        held=leases.enter_context(artifacts.prepare(original=original,normalized=original,
                            encode=lambda raw:encoder.encode(raw)[1],pad_token_id=part['pad']))
                        assert held.span['embedding_sha256']==part['embedding']['sha256']
                        spans.extend([dict(kind='text',tokens=fixture['input_tokens'][cursor:part['start']]),held.span])
                        cursor=part['start']+part['rows']
                    spans.append(dict(kind='text',tokens=fixture['input_tokens'][cursor:]))
                    history=InputHistory(spans)
                    row.update(input_sha256=history.digest,cells=len(history),encoder_runs=len(result['encoder_runs'])-before)
                    yield SimpleNamespace(history=history,artifacts=artifacts,
                        receipt=dict(kind='fixed-real-encoder-fixture-v1',**row))
            finally:
                row['released']=True
                row['release_thread']=threading.get_ident()

    preparer=FixturePreparer()
    def factory(gpu):
        start=time.monotonic();backend=StrataBackend(profile,gpu_id=gpu)
        try:model=BranchingModel(out/'store',backend,profile)
        except BaseException:backend.close();raise
        models.append(model)
        row=dict(pid=backend.process.pid,owner_thread=threading.get_ident(),
            startup_seconds=time.monotonic()-start,native_command_counts={})
        result['workers'].append(row)
        native_request=backend._request;native_close=backend.close
        def owned_request(command):
            assert threading.get_ident()==row['owner_thread'],'Native request escaped owner'
            name=command.split(' ',1)[0]
            row['native_command_counts'][name]=row['native_command_counts'].get(name,0)+1
            return native_request(command)
        def owned_close():
            assert threading.get_ident()==row['owner_thread'],'Native close escaped owner'
            if not backend._closed:row['final_native_counters']=backend.work_counters()
            return native_close()
        backend._request=owned_request;backend.close=owned_close
        return model

    def start():
        manager=ContextCoordinator(out/'store',PoolConfig(gpus=(0,),ram_bytes=8*1024**3,
            disk_bytes=32*1024**3,checkpoint_idle_seconds=1800),factory,input_preparer=preparer)
        try:manager.ready(300)
        except BaseException:manager.close();raise
        return manager

    def stop():
        nonlocal coordinator
        if coordinator is None:return
        coordinator.close();coordinator=None
        row=result['workers'][-1];backend=models[-1].backend
        row.update(exit_code=backend.process.returncode,process_absent=not Path(f"/proc/{row['pid']}").exists())
        assert row['exit_code']==0 and row['process_absent']

    def root(branch,operation):
        presentation=dict(schema=1,namespace_id=namespace,kind='root',
            messages=[dict(id='a'*32,role='user',content='Pinned two-image root fixture')],
            renderer_id=preparer.identity,output_message_id=None)
        return coordinator.prepare_root(branch,dict(fixture_sha256=fingerprint(config['fixture'])['sha256']),
            operation,presentation=presentation)

    def compare(snapshot,reference,label):
        key=(profile['model_contract_id'],profile['execution_profile_id'],snapshot)
        with coordinator.pool.pin(key) as checkpoint:values=checkpoint.frontier
        path=out/(label+'.f32');path.write_bytes(struct.pack('<'+'f'*len(values),*values))
        comparison=compare_files(reference,path);assert comparison['exact'],comparison
        result['comparisons'].append(dict(label=label,reference=fingerprint(reference),candidate=fingerprint(path),**comparison))

    try:
        coordinator=start()
        namespace=coordinator.store.create_namespace('Prepared vision fixture','vision-root-namespace').namespace_id
        for index,branch in enumerate(('served-A','served-B')):
            phase['name']='prepare-root-'+branch
            handle=root(branch,'prepare-'+branch);durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert durable['snapshot']==prior_prepare['prefix']
            assert answer['counter_deltas']['forward_tokens']==(len(fixture['input_tokens']) if index==0 else 0)
            assert answer['exact_materialization_reused']==(index==1)
            assert result['preparations'][index]['encoder_runs']==(len(parts) if index==0 else 0)
            result['roots'].append(dict(branch=branch,durable=durable,answer=answer))
            compare(durable['snapshot'],prior_prepare['expected']['A']['frontiers'][0],branch+'-root')
            save(out/'result.json',result)
        prefix=result['roots'][0]['durable']['snapshot']
        canonical=fingerprint(coordinator.store.objects/(prefix+'.json'))
        for branch in ('A','B'):
            phase['name']='question-'+branch
            presentation=dict(schema=1,namespace_id=namespace,kind='initial_turn',
                messages=[dict(id=('b' if branch=='A' else 'c')*32,role='user',content='Fixture question '+branch)],
                renderer_id=preparer.identity,output_message_id=('d' if branch=='A' else 'e')*32)
            handle=coordinator.submit('served-'+branch,config['suffixes'][branch],config['outputs'],
                'question-'+branch,expected_revision=1,presentation=presentation)
            durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert answer['tokens']==prior_prepare['expected'][branch]['tokens']
            assert answer['counter_deltas']['forward_tokens']==len(config['suffixes'][branch])+len(answer['tokens'])
            reference=json.loads((base/f'reference-{branch}-0/result.json').read_text())
            compare(durable['snapshot'],reference['frontiers'][-1],branch+'-final')
            expected=json.loads(Path(prior_prepare['expected'][branch]['controller']['path']).read_text())
            expected['stream_key']=hashlib.sha256(('question-'+branch).encode()).hexdigest()
            assert coordinator.store.get_snapshot(durable['snapshot'])['controller']==expected
            assert fingerprint(canonical['path'])==canonical
            result['branches'][branch]=dict(durable=durable,answer=answer,controller_exact=True)
            save(out/'result.json',result)
        stop();phase['name']='reopen-receipts';coordinator=start()
        calls=len(result['preparations'])
        for previous in result['roots']:
            handle=root(previous['branch'],'prepare-'+previous['branch'])
            durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert durable['branch']==previous['durable']['branch'] and answer.get('recovered')
            result['replays'].append(dict(branch=previous['branch'],receipt_recovered=True))
        assert len(result['preparations'])==calls
        stop();assert result['workers'][-1]['final_native_counters']['forward_tokens']==0
        assert all(row['released'] for row in result['preparations'])
        result.update(status='PASS',max_abs_error=0.,seconds=time.monotonic()-started,
            limitation='Fixed test-image preparation; no production upload/normalizer/renderer or CLI/HTTP image path. Python driver survives native worker restart. Final and root frontiers only, not every frontier.')
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),
        preparations=len(result['preparations']),encoder_runs=len(result['encoder_runs']),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--encoder-baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    campaign(parser.parse_args())
