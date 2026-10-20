"""Real-GPU VC06 token-adapter/coordinator canary from retained image checkpoints.

No HTTP upload, image encoder or new fresh-reference process is run here. The
executed medium campaign supplies independently measured reference frontiers.
The Python driver remains alive while the native worker exits and restarts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import history_for_snapshot
from context_time_travel.pools import PoolConfig
from context_time_travel.presentation_request import binding_for_snapshot
from context_time_travel.real import BranchingModel
from context_time_travel.store import BranchHead
from context_time_travel.strata_backend import StrataBackend
from serve.cow_adapter import CowAdapter
from run_vision_ingest_probe import compare_files, fingerprint, save


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=args.baseline.resolve();prior=json.loads((base/'result.json').read_text());assert prior['status']=='PASS'
    config=json.loads((base/'config.json').read_text())
    profile=json.loads(Path(config['profile']).read_text())
    prepared=json.loads((base/'prepare/result.json').read_text())
    shutil.copytree(config['store'],out/'store')
    result=dict(status='RUNNING',scope=__doc__,baseline=fingerprint(base/'result.json'),
        profile=fingerprint(config['profile']),workers=[],branches={},comparisons=[],replays=[])
    models=[];coordinator=None;done=threading.Event();phase={'name':'startup'}
    def beat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],python_pid=os.getpid(),
            native_workers=[dict(pid=row['pid'],live=Path(f"/proc/{row['pid']}").exists())
                            for row in result['workers']])),flush=True)
    threading.Thread(target=beat,daemon=True).start()
    def factory(gpu):
        started=time.monotonic();backend=StrataBackend(profile,gpu_id=gpu)
        try:model=BranchingModel(out/'store',backend,profile)
        except BaseException:backend.close();raise
        models.append(model)
        row=dict(pid=backend.process.pid,startup_seconds=time.monotonic()-started,
                 owner_thread=threading.get_ident(),native_command_counts={},owner_guard=True)
        result['workers'].append(row)
        native_request=backend._request;native_close=backend.close
        def owned_request(command):
            assert threading.get_ident()==row['owner_thread'], 'Native IPC escaped its owning worker'
            name=command.split(' ',1)[0]
            row['native_command_counts'][name]=row['native_command_counts'].get(name,0)+1
            return native_request(command)
        def observed_close():
            assert threading.get_ident()==row['owner_thread'], 'Native close escaped its owning worker'
            if not backend._closed:row['final_native_counters']=backend.work_counters()
            return native_close()
        backend._request=owned_request;backend.close=observed_close
        store=model.store;root=prepared['prefix']
        if not (store.references/'served-A.json').exists():
            ns=store.create_namespace('Retained image coordinator canary','vision-service-namespace').namespace_id
            request=dict(schema=1,namespace_id=ns,kind='root',renderer_id=hashlib.sha256(b'retained-medium-fixture-v1').hexdigest(),
                messages=[dict(id='a'*32,role='user',content='Retained two-image numerical fixture')],output_message_id=None)
            head=BranchHead('served-A',0,root,root)
            binding=binding_for_snapshot(store,request,operation_id='served-root',source_head=head,
                target_snapshot=root,input_ids=model.history(root))
            store.fork('served-A',root,namespace_id=ns,operation_id='served-root',presentation=binding)
        return model
    def start():
        manager=ContextCoordinator(out/'store',PoolConfig(gpus=(0,),ram_bytes=8*1024**3,
            disk_bytes=32*1024**3,checkpoint_idle_seconds=1800),factory)
        try:manager.ready(300)
        except BaseException:manager.close();raise
        return manager,CowAdapter(manager)
    def stop():
        nonlocal coordinator
        if coordinator is None:return
        coordinator.close();coordinator=None
        backend=models[-1].backend;row=result['workers'][-1]
        row.update(exit_code=backend.process.returncode,process_absent=not Path(f"/proc/{row['pid']}").exists())
        assert row['exit_code']==0 and row['process_absent']
    def presentation(branch):
        return dict(schema=1,namespace_id=coordinator.store.branch_namespace('served-'+branch),
            kind='initial_turn',messages=[dict(id=('b' if branch=='A' else 'c')*32,role='user',content='Fixture question '+branch)],
            renderer_id=hashlib.sha256(b'retained-medium-fixture-v1').hexdigest(),output_message_id=('d' if branch=='A' else 'e')*32)
    def request(adapter,branch):
        history=history_for_snapshot(coordinator.store,prepared['prefix'])
        return adapter.generate('served-'+branch,history+tuple(config['suffixes'][branch]),0,
            'served-question-'+branch,len(prepared['expected'][branch]['tokens']),presentation=presentation(branch))
    try:
        coordinator,adapter=start()
        result['canonical_prefix']=fingerprint(coordinator.store.objects/(prepared['prefix']+'.json'))
        for branch in ('A','B'):
            phase['name']='request-'+branch
            if branch=='B':adapter.fork('served-B',prepared['prefix'],source_branch='served-A',operation_id='served-fork')
            handle=request(adapter,branch);durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert answer['tokens']==prepared['expected'][branch]['tokens']
            delta=answer['counter_deltas'];assert delta['forward_tokens']==len(config['suffixes'][branch])+len(answer['tokens'])
            key=(profile['model_contract_id'],profile['execution_profile_id'],durable['snapshot'])
            with coordinator.pool.pin(key) as checkpoint:values=checkpoint.frontier
            path=out/(branch+'-final.f32');path.write_bytes(struct.pack('<'+'f'*len(values),*values))
            reference=json.loads((base/f'reference-{branch}-0/result.json').read_text())
            comparison=compare_files(reference['frontiers'][-1],path);assert comparison['exact'],comparison
            result['comparisons'].append(dict(branch=branch,reference=fingerprint(reference['frontiers'][-1]),**comparison))
            expected=json.loads(Path(prepared['expected'][branch]['controller']['path']).read_text())
            expected['stream_key']=hashlib.sha256(('served-question-'+branch).encode()).hexdigest()
            actual=coordinator.store.get_snapshot(durable['snapshot'])['controller'];assert actual==expected
            save(out/(branch+'-controller.json'),actual)
            descriptor=coordinator.committed_generation('served-question-'+branch)
            assert descriptor['generation']['schema']==8 and descriptor['receipt']['presentation']['schema']==3
            result['branches'][branch]=dict(durable=durable,answer=answer,controller=fingerprint(out/(branch+'-controller.json')),
                source_input_sha256=descriptor['generation']['source_history_sha256'])
            assert fingerprint(result['canonical_prefix']['path'])==result['canonical_prefix']
            save(out/'result.json',result)
        stop();phase['name']='reopened-worker';coordinator,adapter=start()
        assert result['workers'][0]['pid']!=result['workers'][1]['pid']
        for branch in ('A','B'):
            handle=request(adapter,branch)
            durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert durable['branch']==result['branches'][branch]['durable']['branch']
            assert answer['tokens']==result['branches'][branch]['answer']['tokens'] and answer.get('recovered')
            result['replays'].append(dict(branch=branch,branch_head=durable['branch'],recovered=True))
        stop()
        assert result['workers'][-1]['final_native_counters']['forward_tokens']==0
        result.update(status='PASS',max_abs_error=0.0,replay_forward_tokens=0,
            replay_counter_scope='Owning worker read cumulative native counters after both receipt replays')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),
        controllers=len(result['branches']),replays=len(result['replays']),workers=result['workers'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','output'):parser.add_argument('--'+name,type=Path,required=True)
    campaign(parser.parse_args())
