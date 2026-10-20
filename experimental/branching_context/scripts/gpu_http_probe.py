"""Real GPU HTTP admission/stream/durability with independent fresh references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

from native_request_probe import publish,reference
from pool_campaign import get_state
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend,_digest
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server,model_text_adapter
from serve.cow_workspace import CowWorkspace
from context_time_travel import controller


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--text-fixture',type=Path)
    parser.add_argument('--max-tokens',type=int,default=128)
    args=parser.parse_args()
    if args.max_tokens<=0: parser.error('--max-tokens must be positive')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    record=dict(status='RUNNING',scope='real Q8 token-history HTTP; not ordinary chat integration',
                python_pid=os.getpid(),requests=[],checks=[],harness_sha256=sha(__file__),
                requested_output_limit=args.max_tokens)
    destination=args.output/'result.json';publish(destination,record)
    stopped=threading.Event();started=time.monotonic()
    def heartbeat():
        while not stopped.wait(30):
            print(json.dumps(dict(event='heartbeat',seconds=time.monotonic()-started,
                                  checks=record['checks'])),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    coordinator=None;server=None
    try:
        profile=json.loads((args.source/'profile.json').read_text())
        text_fixture=json.loads(args.text_fixture.read_text()) if args.text_fixture else None
        if text_fixture:
            if text_fixture['status'] not in ('PASS_CPU_TOKENIZER_TEMPLATE_ONLY',
                                              'PASS_CPU_TOKENIZER_TEMPLATE_SOURCE_RELEVANCE'):
                raise ValueError('Executed CPU tokenizer/template parity must pass first')
            if text_fixture['model_weights_sha256_profile_identity']!=profile['model_weights_sha256']:
                raise ValueError('Text fixture model identity differs')
            if any(not branch['ordinary_service_ids_match'] or not branch['exact_base_prefix']
                   for context in text_fixture['contexts'] for branch in context['branches'].values()):
                raise ValueError('Incomplete exact-prefix/tokenizer parity evidence')
            profile['execution']['text_template_kwargs']=text_fixture['template_kwargs']
            record['scope']='real GPU namespace-bound text/workspace HTTP; browser interaction and original gates separate'
            record['text_fixture_sha256']=sha(args.text_fixture)
        for key in list(os.environ):
            if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
        os.environ.update(profile['execution']['environment'])
        # New control code gets a new profile, and therefore a new freshly
        # computed prefix. No old checkpoint identity is rewritten.
        api=Path(__file__).resolve().parents[1]
        for folder in (api/'src',api/'serve'):
            for path in folder.rglob('*.py'): profile['native_build']['sha256'][str(path)]=sha(path)
        if text_fixture:
            for path in (api/'tools').glob('*.py'): profile['native_build']['sha256'][str(path)]=sha(path)
        profile['execution_profile_id']=_digest(dict(build=profile['native_build'],
            execution=profile['execution'],hardware=profile['hardware']))
        publish(args.output/'profile.json',profile)
        fixture=json.loads((args.source/'fixture.json').read_text())
        if text_fixture:
            context=text_fixture['contexts'][0]
            fixture=dict(prefixes=[context['base_token_ids']],
                suffixes=[context['branches'][name]['suffix_token_ids'] for name in ('A','B')])
        publish(args.output/'fixture.json',fixture)
        config=PoolConfig(gpus=(0,),ram_bytes=96*1024**3,disk_bytes=128*1024**3,
                          min_available_bytes=16*1024**3)
        def factory(gpu): return BranchingModel(args.output/'store',StrataBackend(profile,gpu_id=gpu),profile)
        coordinator=ContextCoordinator(args.output/'store',config,factory);coordinator.ready()
        record['startup_seconds']=time.monotonic()-started
        process=coordinator._workers[0]['model'].backend.process;record['native_pid']=process.pid
        adapter=CowAdapter(coordinator)
        text=model_text_adapter(adapter,profile['model_path'],profile['execution'].get('text_template_kwargs'),
            model_contract_id=profile['model_contract_id']) if text_fixture else None
        if text_fixture:
            assert list(text._render_encode(context['root_messages'],add_generation_prompt=False))==fixture['prefixes'][0]
            for name in ('A','B'):
                assert list(text._render_encode(context[name+'_messages'],add_generation_prompt=True))==context['branches'][name]['full_token_ids']
        workspace=CowWorkspace(coordinator.store,text.tokenizer) if text is not None else None
        server=make_server(adapter,api_key='local-probe-only',text_adapter=text,workspace=workspace)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        def call(path,body):
            req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={
                'Content-Type':'application/json','Authorization':'Bearer local-probe-only'})
            times=[];events=[];begin=time.monotonic()
            with urllib.request.urlopen(req,timeout=600) as response:
                if body.get('stream',True):
                    for line in response:
                        row=json.loads(line);events.append(row)
                        if row.get('event')=='token' or (row.get('event') in ('text','turn_complete') and 'token_id' in row):
                            times.append(time.monotonic()-begin)
                    assert events and events[-1]['event']=='complete',events[-1:]
                    final=events[-1]
                else: final=json.loads(response.read())
            return dict(final=final,events=events,received_times=times,seconds=time.monotonic()-begin)
        namespace=None
        if text_fixture:
            req=urllib.request.Request(base+'/cow/namespace',
                data=json.dumps(dict(title='GPU workspace probe',operation_id='http-namespace')).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer local-probe-only'})
            with urllib.request.urlopen(req,timeout=30) as response:
                namespace=json.loads(response.read())['namespace_id']
            assert workspace.list_branches(namespace)==[]
        root_body=dict(branch='http-base',operation_id='http-create',stream=False)
        root_body.update(messages=context['root_messages']) if text_fixture else root_body.update(token_ids=fixture['prefixes'][0])
        if text_fixture: root_body['namespace_id']=namespace
        root_row=call('/cow/text/context' if text_fixture else '/cow/create',root_body)
        root=root_row['final']['receipt']['snapshot'];root_state=get_state(coordinator,root)
        payload=root_state['payload_file'];parent_hash=sha(payload)
        record['prefix_creation']=root_row
        record['checks'].append('HTTP-GPU-001: new pinned profile and fresh prefix durably created through HTTP')
        publish(destination,record)
        for index,name in enumerate(('A','B')):
            # Fork is a finite JSON endpoint, not a stream.
            fork_body=dict(name='http-'+name,snapshot=root)
            if text_fixture: fork_body.update(source_branch='http-base',operation_id='http-fork-'+name)
            req=urllib.request.Request(base+'/cow/fork',data=json.dumps(fork_body).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer local-probe-only'})
            with urllib.request.urlopen(req,timeout=30) as response: assert json.loads(response.read())['tip']==root
            body=dict(branch='http-'+name,revision=0,operation_id='http-generate-'+name,
                rendered_token_ids=fixture['prefixes'][0]+fixture['suffixes'][index],
                max_tokens=args.max_tokens,settings={'temperature':0})
            if text_fixture:
                body.pop('rendered_token_ids');body['content']=context[name+'_messages'][-1]['content']
                assert context[name+'_messages'][:-1]==context['root_messages']
                source_control=coordinator.store.get_snapshot(root)['controller']
                assert list(text.append_initial_user_suffix(fixture['prefixes'][0],source_control,body['content']))==fixture['suffixes'][index]
            route='/cow/text/ask' if text_fixture else '/cow/generate'
            row=call(route,body)
            answer=row['final']['answer'];outputs=answer['tokens']
            assert len(row['received_times'])==len(outputs)
            assert answer['counter_deltas']['forward_tokens']==len(fixture['suffixes'][index])+len(outputs)
            state=get_state(coordinator,row['final']['receipt']['snapshot'])
            row.update(name=name,state=state)
            times=row['received_times']
            row['ttft_seconds']=times[0]
            row['received_generation_tps']=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None
            row['tpot_seconds']=[b-a for a,b in zip(times,times[1:])]
            # Replay after publication: same request/receipt and no admission.
            retry=call(route,dict(body,stream=False))
            assert retry['final']['receipt']==row['final']['receipt']
            assert retry['final']['answer']==answer
            if text_fixture:
                streamed=''.join(event.get('text','') for event in row['events'] if event.get('event')=='text')
                assert streamed==answer['text']
                req=urllib.request.Request(base+'/cow/workspace/history?branch=http-'+name,
                    headers={'Authorization':'Bearer local-probe-only'})
                with urllib.request.urlopen(req,timeout=30) as response:
                    projected=json.loads(response.read())
                assert projected==workspace.history('http-'+name)
                assert projected['chat_id']==namespace
                assert projected['messages'][-1]['text']==answer['text']
                assert projected['messages'][-1]['node_id']==state['snapshot']
                row['projected_history']=projected
            assert sha(payload)==parent_hash
            record['requests'].append(row)
            record['checks'].append('HTTP-GPU-'+str(index+2)+': '+name+' stream/durable retry, no prefix work, parent unchanged')
            publish(destination,record)
        record['metrics']=coordinator.metrics()
        assert record['metrics']['max_active_per_gpu']=={0:1}
        server.shutdown();server.server_close();thread.join(5);server=None
        coordinator.close();coordinator=None;assert process.poll()==0
        record['native_exit']=process.returncode;publish(destination,record)
        # Independent native processes execute the same pinned schedule; no
        # checkpoint reads and no pool worker remains resident during them.
        for index,row in enumerate(record['requests']):
            ref=reference(profile,args.output,'fresh-http-'+row['name'],
                fixture['prefixes'][0]+fixture['suffixes'][index],
                [len(fixture['prefixes'][0]),len(fixture['suffixes'][index])],args.max_tokens)
            assert row['final']['answer']['tokens']==ref['tokens']
            assert Path(row['state']['frontier_file']).read_bytes()==ref['generated_frontiers'][-1]
            assert Path(root_state['frontier_file']).read_bytes()==ref['prompt_frontiers'][0]
            if text_fixture:
                expected=controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),
                    'http-generate-'+row['name'],
                    tuple(fixture['prefixes'][0]+fixture['suffixes'][index]),len(text.tokenizer.tokens))
                expected=controller.consume_many(expected,((token,text.tokenizer.token_bytes(token),
                    token in profile['execution']['eos_tokens']) for token in ref['tokens']))
                assert row['state']['controller']==expected
            row['reference']=dict(seconds=ref['seconds'],work=ref['work'],command=ref['command'])
            record['checks'].append('HTTP-GPU-EXACT-'+row['name']+': tokens and final full frontier exact')
            publish(destination,record)
        record.update(status='PASS',numerical_max_error=0.,execution_profile_id=profile['execution_profile_id'],
                      model_contract_id=profile['model_contract_id'],parent_payload_sha256=parent_hash)
    except BaseException as error: record.update(status='FAIL',failure=repr(error));raise
    finally:
        if server is not None: server.shutdown();server.server_close()
        if coordinator is not None: coordinator.close()
        stopped.set();record['seconds']=time.monotonic()-started;publish(destination,record)


if __name__=='__main__': main()
