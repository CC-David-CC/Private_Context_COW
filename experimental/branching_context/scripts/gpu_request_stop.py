"""Real GPU cancellation/recovery against an existing full-size campaign."""
import argparse
from concurrent.futures import CancelledError
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

from native_request_probe import publish
from pool_campaign import get_state
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    record=dict(status='RUNNING',scope='real Q8 GPU consumed-boundary cancellation and recovery',
                python_pid=os.getpid(),harness_sha256=digest(__file__),checks=[])
    destination=args.output/'result.json'
    publish(destination,record)
    done=threading.Event();started=time.monotonic()
    def heartbeat():
        while not done.wait(30):
            print(json.dumps(dict(event='heartbeat',seconds=time.monotonic()-started,
                                  checks=record['checks'])),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    coordinator=None
    try:
        # Only immutable payloads are linked. Canonical records and mutable refs
        # are independent copies; original campaign evidence stays untouched.
        def copy(src,dst):
            if str(src).endswith('.state'): os.link(src,dst);return dst
            return shutil.copy2(src,dst)
        shutil.copytree(args.source/'store',args.output/'store',copy_function=copy)
        profile=json.loads((args.source/'profile.json').read_text())
        fixture=json.loads((args.source/'fixture.json').read_text())
        prepared=json.loads((args.source/'prepare.json').read_text())
        root=prepared['roots'][0]['snapshot']
        payload=prepared['roots'][0]['payload_file'];before_hash=digest(payload)
        stop=threading.Event();trace={}
        def factory(gpu):
            model=BranchingModel(args.output/'store',StrataBackend(profile,gpu_id=gpu),profile)
            original=model.run_greedy_request
            def run(*a,**kw):
                before=model.backend.work_counters()
                callback=kw['on_token'];count=0
                def progress(token):
                    nonlocal count
                    callback(token);count+=1
                    if kw.get('cancelled') is not None and count==3: stop.set()
                kw['on_token']=progress
                try: return original(*a,**kw)
                finally:
                    trace.update(before=before,after=model.backend.work_counters(),
                                 request=getattr(model.backend,'last_native_request',None),
                                 protocol_failed=getattr(model.backend,'_protocol_failed',False),
                                 consumed_callbacks=count)
            model.run_greedy_request=run
            return model
        config=PoolConfig(gpus=(0,),ram_bytes=96*1024**3,disk_bytes=128*1024**3,
                          min_available_bytes=16*1024**3)
        coordinator=ContextCoordinator(args.output/'store',config,factory)
        coordinator.ready()
        record['startup_seconds']=time.monotonic()-started
        model=coordinator._workers[0]['model'];process=model.backend.process
        record['native_pid']=process.pid
        coordinator.fork('gpu-stop-test',root)
        before=coordinator.store.head('gpu-stop-test')
        objects=set((args.output/'store'/'objects').rglob('*'))
        request=coordinator.submit('gpu-stop-test',fixture['suffixes'][0],32,
                                   'gpu-stop-at-three',cancelled=stop.is_set)
        try: request.durable.result(900)
        except CancelledError: pass
        else: raise AssertionError('Active request was not cancelled')
        assert request.state=='cancelled' and request.snapshot.cancelled()
        assert coordinator.store.head('gpu-stop-test')==before
        assert set((args.output/'store'/'objects').rglob('*'))==objects
        assert trace['consumed_callbacks']==3 and not trace['protocol_failed']
        assert trace['request']['generated']==3
        assert trace['after']['forward_tokens']-trace['before']['forward_tokens']==len(fixture['suffixes'][0])+3
        record['cancel_trace']=dict(trace)
        record['checks'].append('GPU-STOP-001: cancel/drain at three consumed outputs; no publication or prefix replay')
        publish(destination,record)
        follow=coordinator.submit('gpu-stop-test',fixture['suffixes'][0],32,'gpu-stop-following')
        receipt=follow.durable.result(900);answer=follow.snapshot.result()
        expected=json.loads((args.source/'fresh-tree-0.stdout').read_text())['generated'][:32]
        assert answer['tokens']==expected
        state=get_state(coordinator,receipt['snapshot'])
        width=4*state['manifest']['logits_count']
        with (args.source/'fresh-tree-0.f32.decode.f32').open('rb') as stream:
            stream.seek((len(expected)-1)*width);frontier=stream.read(width)
        assert Path(state['frontier_file']).read_bytes()==frontier
        assert answer['counter_deltas']['forward_tokens']==len(fixture['suffixes'][0])+len(expected)
        assert digest(payload)==before_hash
        assert coordinator.metrics()['max_active_per_gpu']=={0:1}
        record.update(following=answer,metrics=coordinator.metrics(),numerical_max_error=0.,
                      original_head=asdict(before),parent_payload_sha256=before_hash,
                      execution_profile_id=profile['execution_profile_id'],
                      model_contract_id=profile['model_contract_id'])
        record['checks'].append('GPU-STOP-002: subsequent request exact tokens/final frontier, immutable parent, zero prefix replay')
        coordinator.close();coordinator=None
        assert process.poll() is not None
        record.update(status='PASS',native_exit=process.returncode)
    except BaseException as error:
        record.update(status='FAIL',failure=repr(error));raise
    finally:
        if coordinator is not None: coordinator.close()
        done.set();record['seconds']=time.monotonic()-started;publish(destination,record)


if __name__=='__main__': main()
