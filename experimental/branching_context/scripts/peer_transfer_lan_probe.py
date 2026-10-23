"""Bounded LAN transport of real checkpoints, with optional native round-trip.

Storage peers need neither weights nor a GPU. Their view validates existing COW
disk envelopes against an explicitly seeded canonical fixture; it does not claim
that their hardware can execute the source profile. Native verification must run
on the original profile's host. Raw fixtures and keys are private and temporary.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/branching_context/src')]


def disk_view(directory,fixture):
    from context_time_travel.real import BranchingModel
    from context_time_travel.store import Store,_private_directory
    # Only CPU checkpoint methods are used. There is deliberately no native
    # backend here: successful transport is not a hardware restore claim.
    view=object.__new__(BranchingModel)
    view.store=Store(directory);view.profile=fixture['profile']
    view.backend=SimpleNamespace(vocab_size=fixture['vocab_size'])
    view.checkpoint_directory=view.store.root/'checkpoints'
    _private_directory(view.checkpoint_directory)
    view._checkpoint_fault=lambda point:None
    return view


def seed(view,fixture):
    for row in fixture['roots']:
        record=row['record']
        create=view.store.create_input_root if record['schema']==2 else view.store.create_root
        identity=create(record['model_contract_id'],record['inputs'],record['controller'])
        if identity!=row['snapshot']:raise AssertionError('Canonical fixture identity differs')


def transport(stack,view):
    from context_time_travel.checkpoint_transfer import CheckpointTransfers
    from context_time_travel.pools import SharedCheckpointPool,DiskCheckpointPool
    ram=SharedCheckpointPool(100*1024**3);stack.callback(ram.close)
    return CheckpointTransfers(ram,DiskCheckpointPool(view.store,100*1024**3))


def native(stack,profile):
    from context_time_travel.strata_backend import verify_runtime_source_closure
    from context_time_travel.real import BranchingModel
    from tools.cow_server import gpu_locks
    verify_runtime_source_closure(profile,ROOT)
    for name in list(os.environ):
        if name.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')):del os.environ[name]
    os.environ.update(profile['execution']['environment'])
    stack.enter_context(gpu_locks(Path.home()/'.local/state/context-cow',profile['execution']['GPU']))
    backend=BranchingModel.load_backend(profile);stack.callback(backend.close)
    return backend


def prepare(args):
    from context_time_travel.real import BranchingModel
    from serve.cow_http import model_text_components
    profile=json.loads(args.profile.read_text())
    args.directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    fixture=dict(schema=1,profile=profile,roots=[])
    with ExitStack() as stack:
        backend=native(stack,profile);fixture['vocab_size']=backend.vocab_size
        model=BranchingModel(args.directory/'store',backend,profile)
        components=model_text_components(profile['model_path'],{'enable_thinking':False})
        for lines in (0,512):
            messages=[dict(role='system',content='Repair the Python function. Return only the complete corrected function.'),
                dict(role='user',content=('# unrelated comment\n'*lines)+
                    'Tests require double(3)==6, double(-2)==-4 and double(0)==0. Repair:\n'
                    'def double(value):\n    return value + 1\n')]
            rendered=components['render'](messages,add_generation_prompt=True)
            ids=tuple(components['encode_rendered'](messages,rendered,True))
            snapshot=model.create(ids)
            model.checkpoint(snapshot)
            baseline=model.run_greedy_request(snapshot,(),128,stream='lan-roundtrip')
            manifest=model.validate_checkpoint(snapshot)
            fixture['roots'].append(dict(snapshot=snapshot,record=model.store.get_snapshot(snapshot),
                tokens=baseline['tokens'],controller=model.store.get_snapshot(baseline['snapshot'])['controller'],
                payload_bytes=manifest['state_bytes']+4*manifest['logits_count'],
                payload_sha256=manifest['payload_sha256'],prefix_tokens=len(ids)))
            model.cache.clear()
    (args.directory/'fixture.json').write_text(json.dumps(fixture)+'\n')
    print(json.dumps(dict(status='PASS',workers_closed=True,
        roots=[{k:row[k] for k in ('snapshot','payload_bytes','payload_sha256','prefix_tokens')} for row in fixture['roots']])),flush=True)


def serve(args,fixture):
    from context_time_travel.cache_inventory import InventoryJournal
    from serve.cow_http import make_server
    key=args.key_file.read_text().strip()
    if not key or args.key_file.stat().st_mode&0o077:raise ValueError('Private key file required')
    with ExitStack() as stack:
        view=disk_view(args.directory/'store',fixture);transfers=transport(stack,view)
        profile=fixture['profile'];journal=InventoryJournal('lan-cache-probe')
        entries=[]
        for row in fixture['roots']:
            manifest=view.validate_checkpoint(row['snapshot'])
            entries.append(dict(model=profile['model_contract_id'],execution=profile['execution_profile_id'],
                snapshot=row['snapshot'],tier='disk',worker='shared',checksum=manifest['payload_sha256'],
                bytes=manifest['state_bytes']+4*manifest['logits_count']))
        journal.reconcile(entries)
        adapter=SimpleNamespace(cache_transfer_enabled=lambda:True,
            export_checkpoint=lambda sid,mid,eid:transfers.export(view,sid,mid,eid),
            cache_inventory=lambda **kw:journal.read(**kw),
            capabilities=lambda:dict(features={name:dict(supported=True,enabled=True,qualification='experimental')
                for name in ('peer_transfer','cache_inventory')}))
        server=make_server(adapter,host=args.host,port=args.port,api_key=key,
            connection_info=dict(model_contract_id=profile['model_contract_id'],
                execution_profile_id=profile['execution_profile_id'],eos_tokens=profile['execution']['eos_tokens']))
        stop=threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:stop.set())
        signal.signal(signal.SIGINT,lambda *_:stop.set())
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            (args.directory/'server-ready.json').write_text(json.dumps(dict(pid=os.getpid(),port=server.server_port)))
            print('READY',flush=True);stop.wait(args.seconds)
        finally:
            server.shutdown();server.server_close();thread.join(15)
            (args.directory/'server-closed.json').write_text(json.dumps(dict(metrics=transfers.metrics(),closed=not thread.is_alive())))


def receive(args,fixture):
    from context_time_travel.cache_peers import ConfiguredPeer,PeerCaches
    from context_time_travel.real import BranchingModel
    from context_time_travel.store import _json_bytes
    report=dict(schema=1,status='FAIL',scope='Native checkpoint LAN storage transport; not execution on storage-peer GPUs',
        model_contract_id=fixture['profile']['model_contract_id'],execution_profile_id=fixture['profile']['execution_profile_id'],roots=[])
    try:
        with ExitStack() as stack:
            view=disk_view(args.directory/'store',fixture);seed(view,fixture)
            transfers=transport(stack,view)
            peers=PeerCaches([ConfiguredPeer('source',args.url,args.key_file.read_text().strip())],transfers)
            for row in fixture['roots']:
                snapshot=row['snapshot'];before=view.snapshot_bytes(snapshot)
                result=peers.recover(view,snapshot)
                if result is None or result['transferred_bytes']!=row['payload_bytes']:
                    raise AssertionError('Expected complete cache transfer into a fresh destination')
                manifest=view.validate_checkpoint(snapshot)
                if manifest['payload_sha256']!=row['payload_sha256']:raise AssertionError('Payload identity changed')
                duplicate=peers.recover(view,snapshot)
                if duplicate is None or duplicate['transferred_bytes']!=0:raise AssertionError('Retry repeated payload work')
                if view.snapshot_bytes(snapshot)!=before:raise AssertionError('Canonical history changed')
                report['roots'].append(dict(result,canonical_unchanged=True,retry_bytes=0,
                    prefix_tokens=row['prefix_tokens'],payload_sha256=manifest['payload_sha256']))
            if args.verify_native:
                backend=native(stack,fixture['profile'])
                model=BranchingModel(args.directory/'store',backend,fixture['profile'])
                for row,observation in zip(fixture['roots'],report['roots']):
                    before=backend.work_counters()['forward_tokens']
                    restored=model.run_greedy_request(row['snapshot'],(),128,stream='lan-roundtrip')
                    after=backend.work_counters()['forward_tokens']
                    if restored['tokens']!=row['tokens']:raise AssertionError('Complete native response changed')
                    if model.store.get_snapshot(restored['snapshot'])['controller']!=row['controller']:
                        raise AssertionError('Complete controller changed')
                    if after-before!=len(restored['tokens']):raise AssertionError('Saved prefix was recomputed')
                    observation.update(complete_response_equal=True,controller_equal=True,prefix_recomputed_tokens=0,
                        response_sha256=hashlib.sha256(_json_bytes(dict(tokens=restored['tokens']))).hexdigest())
                    model.cache.clear()
            report.update(status='PASS',metrics=transfers.metrics(),native_roundtrip=args.verify_native)
        report['workers_closed']=True
    finally:
        args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','serve','receive'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--key-file',type=Path)
    parser.add_argument('--url');parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=18225)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--verify-native',action='store_true')
    args=parser.parse_args();os.umask(0o077)
    if args.mode=='prepare':return prepare(args)
    fixture=json.loads((args.directory/'fixture.json').read_text())
    if args.mode=='serve':serve(args,fixture)
    else:receive(args,fixture)


if __name__=='__main__':main()
