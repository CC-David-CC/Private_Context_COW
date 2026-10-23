"""Bounded, answer-free native qualification for configured peer recovery.

This uses two selected dense GPUs in one pinned profile. A read-only cache peer
holds the source image after its GPU has been released. The destination owns the
only branch publication authority. Both stores start with the same explicit
canonical input fixture; this does not implement remote history replication.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import sys
import tempfile
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/branching_context/src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--source-gpu',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from context_time_travel.cache_inventory import InventoryJournal
    from context_time_travel.cache_peers import ConfiguredPeer
    from context_time_travel.checkpoint_transfer import CheckpointTransfers
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig,SharedCheckpointPool,DiskCheckpointPool
    from context_time_travel.qwen35_backend import load_qwen35_backend
    from context_time_travel.real import BranchingModel
    from context_time_travel.store import Store,_json_bytes
    from context_time_travel.strata_backend import verify_runtime_source_closure
    from serve.cow_http import make_server,model_text_components
    from tools.cow_server import gpu_locks

    profile=json.loads(args.profile.read_text())
    if profile['backend']!='qwen35-llama.cpp':raise ValueError('This qualification selects the dense backend')
    gpus=profile['execution']['GPU']
    if len(gpus)!=2 or any(type(g) is not int for g in gpus) or args.source_gpu!=gpus[1]:
        raise ValueError('Use two independent GPUs, with the source GPU second in profile order')
    verify_runtime_source_closure(profile,ROOT)
    for name in list(os.environ):
        if name.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')):del os.environ[name]
    os.environ.update(profile['execution']['environment'])
    os.umask(0o077)
    report=dict(schema=1,status='FAIL',model_contract_id=profile['model_contract_id'],
        execution_profile_id=profile['execution_profile_id'],source_gpu=args.source_gpu,
        target_gpus=gpus,scope='One complete bounded code response through native peer checkpoint recovery')
    args.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    try:
        with ExitStack() as stack:
            stack.enter_context(gpu_locks(Path.home()/'.local/state/context-cow',gpus))
            temporary=stack.enter_context(tempfile.TemporaryDirectory(prefix='cow-peer-native-',dir=args.output.parent))
            work=Path(temporary)
            backend=load_qwen35_backend(profile,gpu_id=args.source_gpu)
            stack.callback(backend.close)
            source=BranchingModel(work/'source',backend,profile)
            components=model_text_components(profile['model_path'],{'enable_thinking':False})
            messages=[dict(role='system',content='Repair the Python function. Return only the complete corrected function.'),
                dict(role='user',content='The tests require double(3)==6, double(-2)==-4 and double(0)==0. '
                     'Repair this implementation:\n\ndef double(value):\n    return value + 1\n')]
            rendered=components['render'](messages,add_generation_prompt=True)
            ids=tuple(components['encode_rendered'](messages,rendered,True))
            snapshot=source.create(ids)
            source_ram=SharedCheckpointPool(100*1024**3);stack.callback(source_ram.close)
            source_disk=DiskCheckpointPool(source.store,100*1024**3)
            prepared=source.prepare_checkpoint(snapshot)
            with source_ram.reserve(2*1024**2+8*backend.vocab_size):
                with source_disk.reserve(len(prepared.envelope)+len(prepared.checkpoint.blob)+4*backend.vocab_size,
                                         payload_sha256=prepared.checkpoint.checksum):
                    source.publish_prepared_checkpoint(prepared)
            del prepared
            baseline=source.run_greedy_request(snapshot,(),128,stream='peer-native-response')
            baseline_record=source.store.get_snapshot(baseline['snapshot'])
            source.cache.clear();backend.close()
            print('SOURCE_RESPONSE_READY_GPU_RELEASED',flush=True)
            # Read-only transfer uses vocab metadata, never the closed GPU handle.
            transfers=CheckpointTransfers(source_ram,source_disk)
            manifest=source.checkpoint_manifest(snapshot)
            journal=InventoryJournal('native-cache-peer')
            journal.reconcile([dict(model=profile['model_contract_id'],execution=profile['execution_profile_id'],
                snapshot=snapshot,tier='disk',worker='shared',checksum=manifest['payload_sha256'],
                bytes=manifest['state_bytes']+4*manifest['logits_count'])])
            adapter=SimpleNamespace(cache_transfer_enabled=lambda:True,
                export_checkpoint=lambda sid,model_id,execution_id:transfers.export(source,sid,model_id,execution_id),
                cache_inventory=lambda **kw:journal.read(**kw),
                capabilities=lambda:dict(features={name:dict(supported=True,enabled=True,qualification='experimental')
                    for name in ('peer_transfer','cache_inventory')}))
            key=secrets.token_urlsafe(32)
            server=make_server(adapter,api_key=key,connection_info=dict(
                model_contract_id=profile['model_contract_id'],execution_profile_id=profile['execution_profile_id'],
                eos_tokens=profile['execution']['eos_tokens']))
            thread=threading.Thread(target=server.serve_forever);thread.start()
            stack.callback(lambda:(server.shutdown(),server.server_close(),thread.join(15)))
            destination=work/'destination';store=Store(destination)
            record=source.store.get_snapshot(snapshot)
            if record['schema']==2:
                local=store.create_input_root(record['model_contract_id'],record['inputs'],record['controller'])
            else:local=store.create_root(record['model_contract_id'],record['inputs'],record['controller'])
            if local!=snapshot:raise AssertionError('Canonical fixture differs before transfer')
            store.fork('probe',snapshot)
            canonical_before=(store.objects/(snapshot+'.json')).read_bytes()
            def factory(gpu):return BranchingModel(destination,load_qwen35_backend(profile,gpu_id=gpu),profile)
            coordinator=ContextCoordinator(destination,PoolConfig(gpus=tuple(gpus),ram_bytes=100*1024**3,
                disk_bytes=100*1024**3,min_available_bytes=16*1024**3,cache_inventory=True),factory,
                cache_peers=[ConfiguredPeer('source','http://127.0.0.1:'+str(server.server_port),key)])
            stack.callback(coordinator.close)
            coordinator.ready(timeout=600)
            handle=coordinator.submit('probe',(),128,'peer-native-response',expected_revision=0)
            receipt=handle.durable.result(600)
            events=[]
            while True:
                try:events.append(handle.events.get_nowait())
                except queue.Empty:break
            tokens=[event['token'] for event in events if event['event']=='token']
            restored=[event for event in events if event['event']=='source_peer_restored']
            started=[event for event in events if event['event']=='request_started']
            if len(restored)!=1 or restored[0]['transferred_bytes']!=journal.read()['entries'][0]['bytes']:
                raise AssertionError('Native request did not import the complete peer image')
            if len(started)!=1 or started[0]['gpu']==args.source_gpu:
                raise AssertionError('Expected a distinct GPU to consume the source image')
            if any(event['event'].startswith('source_recomputation') for event in events):
                raise AssertionError('Destination recomputed the saved prefix')
            if tokens!=baseline['tokens']:raise AssertionError('Complete native responses differ')
            final=coordinator.store.get_snapshot(receipt['snapshot'])
            if final['controller']!=baseline_record['controller']:raise AssertionError('Controller differs after transfer')
            if (store.objects/(snapshot+'.json')).read_bytes()!=canonical_before:
                raise AssertionError('Peer changed canonical source')
            work_counters={str(gpu):model['model'].backend.work_counters() for gpu,model in coordinator._workers.items()}
            if sum(row['forward_tokens'] for row in work_counters.values())!=len(tokens):
                raise AssertionError('Destination consumed more than generated tokens')
            report.update(status='PASS',prefix_tokens=len(ids),generated_tokens=len(tokens),
                target_gpu=started[0]['gpu'],transferred_bytes=restored[0]['transferred_bytes'],
                transfer_seconds=restored[0]['seconds'],complete_response_equal=True,controller_equal=True,
                canonical_source_unchanged=True,prefix_recomputed_tokens=0,
                response_sha256=hashlib.sha256(_json_bytes(dict(tokens=tokens))).hexdigest(),
                native_work_counters=work_counters,transfer_metrics=coordinator.metrics()['cache_transfers'])
        report['ephemeral_removed']=not Path(temporary).exists()
        report['workers_closed']=True
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
