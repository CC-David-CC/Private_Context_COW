"""Real HTTP disconnect cancels GPU work before canonical publication."""
import argparse
from concurrent.futures import CancelledError
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import urllib.request

from native_request_probe import publish
from pool_campaign import get_state
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    record=dict(status='RUNNING',scope='Q8 real HTTP disconnect plus GPU recovery, greedy only',
                python_pid=os.getpid(),harness_sha256=sha(__file__),checks=[])
    result=args.output/'result.json';publish(result,record)
    ended=threading.Event();start=time.monotonic()
    def heartbeat():
        while not ended.wait(30): print(json.dumps(dict(event='heartbeat',seconds=time.monotonic()-start,checks=record['checks'])),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    coordinator=server=None
    try:
        original=json.loads((args.source/'result.json').read_text())
        assert original['status']=='PASS','Source HTTP qualification must be complete'
        def copy(src,dst):
            if str(src).endswith('.state'): os.link(src,dst);return dst
            return shutil.copy2(src,dst)
        shutil.copytree(args.source/'store',args.output/'store',copy_function=copy)
        profile=json.loads((args.source/'profile.json').read_text())
        fixture=json.loads((args.source/'fixture.json').read_text())
        for key in list(os.environ):
            if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
        os.environ.update(profile['execution']['environment'])
        config=PoolConfig(gpus=(0,),ram_bytes=96*1024**3,disk_bytes=128*1024**3,min_available_bytes=16*1024**3)
        trace={}
        def factory(gpu):
            model=BranchingModel(args.output/'store',StrataBackend(profile,gpu_id=gpu),profile)
            run=model.run_greedy_request
            def measured(*a,**kw):
                before=model.backend.work_counters()
                try: return run(*a,**kw)
                finally: trace.update(before=before,after=model.backend.work_counters(),
                    request=getattr(model.backend,'last_native_request',None),
                    protocol_failed=getattr(model.backend,'_protocol_failed',False))
            model.run_greedy_request=measured;return model
        coordinator=ContextCoordinator(args.output/'store',config,factory);coordinator.ready()
        record['startup_seconds']=time.monotonic()-start
        process=coordinator._workers[0]['model'].backend.process;record['native_pid']=process.pid
        class Captured(CowAdapter):
            def generate(self,*a,**kw):
                request=super().generate(*a,**kw);self.last=request;return request
        adapter=Captured(coordinator)
        root=coordinator.store.head('http-base').tip
        coordinator.fork('http-disconnect',root)
        before=coordinator.store.head('http-disconnect')
        parent=get_state(coordinator,root);parent_hash=sha(parent['payload_file'])
        objects=set((args.output/'store'/'objects').rglob('*'))
        server=make_server(adapter,api_key='local-cancel-only')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        body=dict(branch='http-disconnect',revision=0,operation_id='http-drop',
            rendered_token_ids=fixture['prefixes'][0]+fixture['suffixes'][0],max_tokens=128,settings={'temperature':0})
        def request(value): return urllib.request.Request(base+'/cow/generate',data=json.dumps(value).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer local-cancel-only'})
        received=[]
        with urllib.request.urlopen(request(body),timeout=600) as response:
            for line in response:
                event=json.loads(line)
                if event.get('event')=='token': received.append(event['token'])
                if len(received)==3:
                    response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    break
        handle=adapter.last
        try: handle.durable.result(60)
        except CancelledError: pass
        else: raise AssertionError('Disconnected request durably published')
        assert handle.state=='cancelled' and handle.snapshot.cancelled()
        assert coordinator.store.head('http-disconnect')==before
        assert set((args.output/'store'/'objects').rglob('*'))==objects
        assert not trace['protocol_failed'] and 3<=trace['request']['generated']<128
        assert trace['after']['forward_tokens']-trace['before']['forward_tokens']==len(fixture['suffixes'][0])+trace['request']['generated']
        record.update(cancel_trace=dict(trace),received_before_disconnect=received,original_head=asdict(before))
        record['checks'].append('HTTP-CANCEL-001: real disconnect drained partial GPU request; no branch/snapshot publication')
        publish(result,record)
        following=dict(body,operation_id='http-after-drop',max_tokens=32,stream=False)
        with urllib.request.urlopen(request(following),timeout=600) as response: answer=json.loads(response.read())
        tokens=answer['answer']['tokens']
        assert tokens==original['requests'][0]['final']['answer']['tokens'][:32]
        state=get_state(coordinator,answer['receipt']['snapshot'])
        width=4*state['manifest']['logits_count']
        with (args.source/'fresh-http-A.f32.decode.f32').open('rb') as stream:
            stream.seek(31*width);frontier=stream.read(width)
        assert Path(state['frontier_file']).read_bytes()==frontier
        assert answer['answer']['counter_deltas']['forward_tokens']==len(fixture['suffixes'][0])+32
        assert sha(parent['payload_file'])==parent_hash
        metrics=coordinator.metrics();assert metrics['max_active_per_gpu']=={0:1}
        record.update(following=answer,metrics=metrics,numerical_max_error=0.,parent_payload_sha256=parent_hash,
            execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'])
        record['checks'].append('HTTP-CANCEL-002: next HTTP request/reference exact; no prefix work or parent mutation')
        server.shutdown();server.server_close();thread.join(5);server=None
        coordinator.close();coordinator=None;assert process.poll()==0
        record.update(status='PASS',native_exit=process.returncode)
    except BaseException as error: record.update(status='FAIL',failure=repr(error));raise
    finally:
        if server is not None: server.shutdown();server.server_close()
        if coordinator is not None: coordinator.close()
        ended.set();record['seconds']=time.monotonic()-start;publish(result,record)


if __name__=='__main__': main()
