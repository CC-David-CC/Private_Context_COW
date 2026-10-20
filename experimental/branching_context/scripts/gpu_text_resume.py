"""Text HTTP continuation across new Python/native processes; greedy only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

from native_request_probe import publish,reference
from pool_campaign import get_state
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend,_digest
from context_time_travel import controller
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server,model_text_adapter


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def phase(args):
    profile=json.loads((args.output/'profile.json').read_text())
    fixture=json.loads((args.output/'fixture.json').read_text());context=fixture['contexts'][0]
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    record=dict(status='RUNNING',phase=args.action,python_pid=os.getpid(),requests=[],checks=[])
    path=args.output/(args.action+'.json');publish(path,record)
    stopped=threading.Event();start=time.monotonic()
    def heartbeat():
        while not stopped.wait(30): print(json.dumps(dict(event='heartbeat',phase=args.action,seconds=time.monotonic()-start,checks=record['checks'])),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    coordinator=server=None
    try:
        config=PoolConfig(gpus=(0,),ram_bytes=96*1024**3,disk_bytes=128*1024**3,min_available_bytes=16*1024**3)
        def factory(gpu): return BranchingModel(args.output/'store',StrataBackend(profile,gpu_id=gpu),profile)
        coordinator=ContextCoordinator(args.output/'store',config,factory);coordinator.ready()
        record['startup_seconds']=time.monotonic()-start
        process=coordinator._workers[0]['model'].backend.process;record['native_pid']=process.pid
        adapter=CowAdapter(coordinator)
        text=model_text_adapter(adapter,profile['model_path'],profile['execution']['text_template_kwargs'])
        assert list(text._render_encode(context['root_messages'],add_generation_prompt=False))==context['base_token_ids']
        server=make_server(adapter,text_adapter=text,api_key='local-resume-only')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        def call(route,body):
            begin=time.monotonic();events=[];times=[]
            req=urllib.request.Request(base+route,data=json.dumps(body).encode(),headers={
                'Content-Type':'application/json','Authorization':'Bearer local-resume-only'})
            with urllib.request.urlopen(req,timeout=600) as response:
                if body.get('stream',True):
                    for raw in response:
                        event=json.loads(raw);events.append(event)
                        if event.get('event') in ('text','turn_complete') and 'token_id' in event: times.append(time.monotonic()-begin)
                    final=events[-1];assert final['event']=='complete'
                else: final=json.loads(response.read())
            return dict(final=final,events=events,received_times=times,durable_seconds=time.monotonic()-begin)
        if args.action=='prepare':
            root=call('/cow/text/create',dict(branch='resume-base',messages=context['root_messages'],operation_id='resume-create',stream=False))
            root_id=root['final']['receipt']['snapshot'];record['root']=get_state(coordinator,root_id)
            for name in ('A','B'):
                coordinator.fork('resume-'+name,root_id)
                row=call('/cow/text/generate',dict(branch='resume-'+name,revision=0,
                    operation_id='resume-first-'+name,messages=context[name+'_messages'],max_tokens=4,settings={'temperature':0}))
                answer=row['final']['answer'];assert len(answer['tokens'])==4
                assert answer['counter_deltas']['forward_tokens']==context['branches'][name]['suffix_count']+4
                row['name']=name;row['state']=get_state(coordinator,row['final']['receipt']['snapshot'])
                assert row['state']['controller']['status']=='active'
                record['requests'].append(row);publish(path,record)
        else:
            previous=json.loads((args.output/'prepare.json').read_text());assert previous['status']=='PASS'
            parent_hashes={row['state']['payload_file']:sha(row['state']['payload_file']) for row in previous['requests']}
            parent_hashes[previous['root']['payload_file']]=sha(previous['root']['payload_file'])
            for first in previous['requests']:
                name=first['name'];revision=first['final']['receipt']['branch']['revision']
                row=call('/cow/text/resume',dict(branch='resume-'+name,revision=revision,
                    operation_id='resume-follow-'+name,max_tokens=args.generated-4))
                assert row['final']['answer']['counter_deltas']['forward_tokens']==len(row['final']['answer']['tokens'])
                assert row['final']['answer']['counter_deltas']['prefill_tokens']==0
                row['name']=name;row['state']=get_state(coordinator,row['final']['receipt']['snapshot'])
                row['all_tokens']=first['final']['answer']['tokens']+row['final']['answer']['tokens']
                row['all_text']=first['final']['answer']['text']+row['final']['answer']['text']
                streamed=''.join(e.get('text','') for e in row['events'] if e.get('event')=='text')
                assert streamed==row['final']['answer']['text']
                retry=call('/cow/text/resume',dict(branch='resume-'+name,revision=revision,
                    operation_id='resume-follow-'+name,max_tokens=args.generated-4,stream=False))
                assert retry['final']==row['final']
                record['requests'].append(row);publish(path,record)
            assert all(sha(path)==value for path,value in parent_hashes.items())
            record['parent_hashes']=parent_hashes
        record['metrics']=coordinator.metrics();assert record['metrics']['max_active_per_gpu']=={0:1}
        server.shutdown();server.server_close();thread.join(5);server=None
        coordinator.close();coordinator=None;assert process.poll()==0
        record.update(status='PASS',native_exit=process.returncode)
    except BaseException as error: record.update(status='FAIL',failure=repr(error));raise
    finally:
        if server is not None: server.shutdown();server.server_close()
        if coordinator is not None: coordinator.close()
        stopped.set();record['seconds']=time.monotonic()-start;publish(path,record)


def campaign(args):
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    fixture=json.loads(args.fixture.read_text())
    if fixture['status']!='PASS_CPU_TOKENIZER_TEMPLATE_SOURCE_RELEVANCE':
        raise ValueError('Requires executed tokenizer/template/source-relevance fixture')
    for context in fixture['contexts']:
        for branch in context['branches'].values():
            if (not branch['ordinary_service_ids_match'] or not branch['exact_base_prefix']
                    or not branch['required_markers_present_in_consumed_code']):
                raise ValueError('Incomplete exact-prefix/tokenizer parity')
    profile=json.loads(args.profile.read_text())
    assert fixture['model_weights_sha256_profile_identity']==profile['model_weights_sha256']
    profile['execution']['text_template_kwargs']=fixture['template_kwargs']
    api=Path(__file__).resolve().parents[1]
    for directory in ('src','serve','tools'):
        for path in (api/directory).rglob('*.py'): profile['native_build']['sha256'][str(path)]=sha(path)
    profile['execution_profile_id']=_digest(dict(build=profile['native_build'],execution=profile['execution'],hardware=profile['hardware']))
    publish(args.output/'profile.json',profile);publish(args.output/'fixture.json',fixture)
    result=dict(status='RUNNING',scope='Q8 process-separated text resume, greedy only; not ordinary UI/OpenAI integration',output_limit=args.generated,
        phases={},comparisons=[],profile_id=profile['execution_profile_id'],harness_sha256=sha(__file__),fixture_sha256=sha(args.fixture))
    destination=args.output/'result.json';publish(destination,result)
    try:
        for action in ('prepare','resume'):
            cmd=[sys.executable,__file__,action,'--output',str(args.output),'--generated',str(args.generated)]
            with (args.output/(action+'.log')).open('wb') as log:
                worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps(dict(event='phase-start',phase=action,pid=worker.pid)),flush=True)
                code=worker.wait()
            value=json.loads((args.output/(action+'.json')).read_text())
            result['phases'][action]=dict(python_pid=worker.pid,exit_code=code,observed_exit=True,native_pid=value.get('native_pid'),native_exit=value.get('native_exit'))
            publish(destination,result)
            if code or value['status']!='PASS': raise AssertionError('Phase failed: '+action)
        assert result['phases']['prepare']['python_pid']!=result['phases']['resume']['python_pid']
        assert result['phases']['prepare']['native_pid']!=result['phases']['resume']['native_pid']
        fixture_context=fixture['contexts'][0]
        for key in list(os.environ):
            if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
        os.environ.update(profile['execution']['environment'])
        resumed=json.loads((args.output/'resume.json').read_text())
        from tools.strata_tokenizer import Tokenizer
        tokenizer=Tokenizer.from_gguf(profile['model_path'])
        for row in resumed['requests']:
            name=row['name'];branch=fixture_context['branches'][name]
            fresh=reference(profile,args.output,'fresh-resume-'+name,branch['full_token_ids'],
                [fixture_context['base_token_count'],branch['suffix_count']],args.generated)
            assert row['all_tokens']==fresh['tokens']
            assert Path(row['state']['frontier_file']).read_bytes()==fresh['generated_frontiers'][-1]
            expected=controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),
                'resume-first-'+name,tuple(branch['full_token_ids']),len(tokenizer.tokens))
            expected=controller.consume_many(expected,((token,tokenizer.token_bytes(token),token in profile['execution']['eos_tokens']) for token in fresh['tokens']))
            assert expected==row['state']['controller']
            result['comparisons'].append(dict(branch=name,status='PASS',outputs=len(fresh['tokens']),max_absolute_error=0.,controller_exact=True))
            publish(destination,result)
        result['status']='PASS'
    except BaseException as error: result.update(status='FAIL',failure=repr(error));raise
    finally: publish(destination,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['campaign','prepare','resume']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--fixture',type=Path)
    p.add_argument('--generated',type=int,default=128)
    args=p.parse_args()
    if args.generated<=4: p.error('generated must exceed the four-token save boundary')
    campaign(args) if args.action=='campaign' else phase(args)


if __name__=='__main__': main()
