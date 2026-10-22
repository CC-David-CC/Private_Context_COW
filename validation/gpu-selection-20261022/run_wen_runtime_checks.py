"""Locally authored operational checks; never edits server source or prints API keys."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--gpu',required=True);p.add_argument('--name',required=True)
p.add_argument('--ram',type=int,default=100);p.add_argument('--disk',type=int,default=100)
p.add_argument('--port',type=int,default=18180);p.add_argument('--bw',action='store_true')
p.add_argument('--http-only',action='store_true')
a=p.parse_args();base=a.base;source=base/'src/context-cow';gpus=[int(n) for n in a.gpu.split(',')]
sys.path[:0]=[str(source),str(source/'experimental/branching_context/src')]
from context_time_travel.qwen35_mixed_backend import Qwen35MixedBackend,Qwen35MixedOracle
from context_time_travel.input_spans import InputHistory
from tools.strata_tokenizer import Tokenizer

folder=base/'run'/a.name;folder.mkdir(mode=0o700,exist_ok=False)
server=folder/'server';model=base/'models/Wen-BF16.gguf'
expected='40f048a00bc2a2a8c2db163999f0af90613d136376e9b3c9372457b1769b44ca'
with model.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
subprocess.run([sys.executable,str(source/'tools/cow_server_setup.py'),'--model',str(model),
    '--native',str(base/'build/mixed/libqwen35ctt.so'),'--output',str(server),
    '--gpu',a.gpu,'--ephemeral','--context','65536','--prefill','512','--threads','8',
    '--ram-gib',str(a.ram),'--disk-gib',str(a.disk),'--min-available-gib','8','--port',str(a.port)],check=True)
profile=json.loads((server/'profile.json').read_text());e=profile['execution'];os.environ.update(e['environment'])
result={'gpu':gpus,'model_sha256':expected,'native_comparisons':0,'max_logit_error':0,
        'restores':0,'prefix_forward_tokens_on_restore':0,'tokenizer_comparisons':0}
tokenizer=Tokenizer.from_gguf(model);prefix=None;blob=None;frontier=None;schedule=None
for gpu in ([] if a.http_only else gpus):
    backend=Qwen35MixedBackend(profile,gpu_id=gpu)
    oracle=Qwen35MixedOracle(profile,embeddings={},gpu_id=gpu)
    try:
        for text in ['Hello world.','12345','cafe\u0301 a\u0301','literal <|im_start|>']:
            assert tuple(tokenizer.encode(text,parse_special=False))==backend.tokenize(text,add_special=False,parse_special=False)
            result['tokenizer_comparisons']+=1
        if prefix is None:
            prefix=backend.tokenize('The quick brown fox jumps over the lazy dog.')
            with backend.lease() as lease:
                backend.consume(lease,prefix);schedule=[backend.last_execution_receipt]
                frontier=backend.logits(lease);blob=backend.export(lease)
        history=InputHistory([{'kind':'text','tokens':list(prefix)}])
        for suffix in [(1,),(2,)]:
            with backend.lease() as lease:
                before=backend.work_counters()['forward_tokens']
                backend.restore(lease,blob,history=history,frontier=frontier)
                assert backend.work_counters()['forward_tokens']==before
                result['restores']+=1
                backend.consume(lease,suffix);actual=backend.logits(lease)
                reference=oracle.forward_inputs(history+suffix,schedule+[backend.last_execution_receipt])
                assert len(actual)==len(reference)==profile['vocab_size']
                error=max(abs(x-y) for x,y in zip(actual,reference))
                result['max_logit_error']=max(result['max_logit_error'],error)
                assert error==0,('native frontier mismatch',gpu,error)
                result['native_comparisons']+=1
    finally:oracle.close();backend.close()

key=(server/'api-key.txt').read_text().strip();endpoint=f'http://127.0.0.1:{a.port}'
def http(path,body=None):
    request=urllib.request.Request(endpoint+path,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
    except urllib.error.HTTPError as error:
        detail=error.read().decode();raise RuntimeError(f'HTTP {error.code}: {detail}') from error

with (folder/'http.log').open('w') as log:
    process=subprocess.Popen([sys.executable,str(source/'tools/cow_server.py'),str(server)],stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+180
        while True:
            if process.poll() is not None:raise RuntimeError('Server exited; inspect HTTP log')
            try:info=http('/cow/info');break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Server startup')
                time.sleep(.25)
        assert info['execution_profile_id']==profile['execution_profile_id']
        root='smoke-'+uuid.uuid4().hex[:12]
        namespace=http('/cow/namespace',dict(title='Runtime smoke',operation_id='op-'+uuid.uuid4().hex))['namespace_id']
        created=http('/cow/text/prepare',dict(branch=root,namespace_id=namespace,messages=[{'role':'user','content':'Say hello in one short sentence.'}],operation_id='op-'+uuid.uuid4().hex,stream=False))
        head=http('/cow/head?branch='+root)
        for name in ('A','B'):http('/cow/fork',dict(name=root+name,snapshot=head['tip'],source_branch=root,operation_id='op-'+uuid.uuid4().hex))
        barrier=threading.Barrier(2)
        def generate(index):
            barrier.wait()
            return http('/cow/text/answer',dict(branch=root+('A','B')[index],revision=0,
                operation_id='op-'+uuid.uuid4().hex,max_tokens=8,settings={'temperature':0},stream=False))
        with ThreadPoolExecutor(2) as pool:answers=list(pool.map(generate,(0,1)))
        assert all(answer['event']=='complete' for answer in answers)
        assert answers[0]['receipt']['snapshot']!=answers[1]['receipt']['snapshot']
        assert http('/cow/head?branch='+root)==head
        metrics=http('/cow/status')
        assert metrics['worker_count']==len(gpus)
        result['workers_used']=[gpu for gpu,count in metrics['max_active_per_gpu'].items() if count]
        if len(gpus)>1:assert len(result['workers_used'])==len(gpus)
        result['http_concurrent_generations']=2;result['parent_head_preserved']=True
        result['ram_byte_limit']=metrics['ram'].get('byte_limit')
        result['disk_byte_limit']=metrics['disk'].get('byte_limit')
        if a.bw:
            bw=base/'venvs/branchwrite/bin/python'
            code="from branchwrite.client_setup import discover; import os,json; print(json.dumps(discover(os.environ['BW_ENDPOINT'],os.environ['BW_COW_API_KEY'])))"
            env={**os.environ,'BW_COW_API_KEY':key,'BW_ENDPOINT':endpoint}
            connection=json.loads(subprocess.check_output([str(bw),'-c',code],env=env,text=True))
            path=folder/'connection.json';path.write_text(json.dumps(connection));path.chmod(0o600)
            state=folder/'bw-chat'
            with (folder/'branchwrite.log').open('w') as out:
                subprocess.run([str(bw),'-m','branchwrite','chat','init','--connection',str(path),
                    '--state',str(state),'--branch','bw-'+uuid.uuid4().hex[:12]],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=60)
                subprocess.run([str(bw),'-m','branchwrite','chat','send',str(state),'--message','Say hello in one short sentence.',
                    '--max-tokens','32'],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=180)
            result['branchwrite_committed_chat']=True
    finally:
        process.terminate()
        try:process.wait(timeout=60)
        except subprocess.TimeoutExpired:process.kill();process.wait();raise
        result['server_exit_code']=process.returncode
        result['ephemeral_sessions_remaining']=len(list((server/'.ephemeral-sessions').glob('session-*')))
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['server_exit_code']==0
assert result['ephemeral_sessions_remaining']==0
print(json.dumps(result,indent=2))
