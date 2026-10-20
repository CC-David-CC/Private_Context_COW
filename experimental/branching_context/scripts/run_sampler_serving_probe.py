"""Private single-GPU Min-P serving/restart audit, not a throughput benchmark.

Use the real COW service, stdlib CLI, shared pools and native backend. The chat
server stays off. Fresh raw-native references use identical operator boundaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

from run_min_p_probe import POLICY,check_delta,compare_record,frontier_bytes


def validate_completed_request(actual,expected,*,vocab,suffix):
    compare_record(actual,expected)
    zero={key:0 for key in actual['counter_deltas']}
    check_delta(zero,actual['counter_deltas'],suffix,len(actual['tokens']),vocab=vocab)


def sampling_arguments(policy,previous=None):
    return [value for field in ('temperature','min_p','top_k','top_p',
                               'typical_p','dynatemp_range','dynatemp_exponent',
                               'xtc_probability','xtc_threshold',
                               'mirostat','mirostat_tau','mirostat_eta',
                               'adaptive_target','adaptive_decay',
                               'dry_multiplier','dry_base','dry_allowed_length','dry_penalty_last_n')
            if field in policy and (previous is None or policy[field]!=previous.get(field))
            for value in ('--'+field.replace('_','-'),str(policy[field]))]


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('campaign','prepare','resume'),required=True)
    for name in ('runtime-root','profile','fixture','reference-observer','run-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prefix',type=int,default=1024)
    parser.add_argument('--suffix',type=int,default=64)
    parser.add_argument('--outputs',type=int,default=16)
    parser.add_argument('--sampling-json',type=json.loads)
    parser.add_argument('--resume-sampling-json',type=json.loads)
    args=parser.parse_args()
    if not 4<args.outputs<=128 or args.prefix<=0 or args.suffix<=0:raise ValueError('Invalid bounded fixture')
    os.umask(0o077)
    runtime=args.runtime_root.resolve(strict=True)
    sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),
                  str(runtime/'experimental/branching_context/scripts')]
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from grammar_long_reference import Heartbeat,_apply_profile_environment
    from run_grammar_first_probe import run_stage,publish
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig,PoolTransitionTrace
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend,StrataImage,_validate_execution
    from serve.cow_adapter import CowAdapter
    from serve.cow_http import make_server,model_text_adapter
    from tools import cow_cli
    from context_time_travel.min_p import validate_policy
    policy=validate_policy(POLICY if args.sampling_json is None else args.sampling_json)
    resumed_policy=None if args.resume_sampling_json is None else validate_policy(args.resume_sampling_json)
    profile=read(args.profile);_apply_profile_environment(profile);_validate_execution(profile)
    output=args.run_dir.absolute()
    if args.stage=='campaign':
        output.mkdir(mode=0o700,parents=True,exist_ok=False)
        result=dict(status='RUNNING',stages=[],script_sha256=sha(__file__),profile_sha256=sha(args.profile))
        publish(output/'campaign.json',result)
        for stage in ('reference-A','reference-B','prepare','resume'):
            command=[sys.executable,'-B',str(args.reference_observer if stage.startswith('reference') else Path(__file__).resolve()),
                '--stage',stage,'--runtime-root',str(runtime),'--profile',str(args.profile.resolve()),
                '--fixture',str(args.fixture.resolve()),'--run-dir',str(output),
                '--prefix',str(args.prefix),'--suffix',str(args.suffix),'--outputs',str(args.outputs)]
            if not stage.startswith('reference'):command+=['--reference-observer',str(args.reference_observer.resolve())]
            command+=['--sampling-json',json.dumps(policy)]
            if resumed_policy is not None:command+=['--resume-sampling-json',json.dumps(resumed_policy)]
            receipt=run_stage(command,output/(stage+'.stdout'),output/(stage+'.stderr'))
            result['stages'].append(dict(stage=stage,**receipt))
            path=output/stage/'result.json';child=read(path) if path.exists() else {}
            healthy=(receipt['exit_code']==0 and receipt['pid_absent'] and not receipt['cleanup_needed']
                and not receipt['timed_out'] and not receipt['live_group_pids'] and child.get('status')=='PASS'
                and child.get('native_exit_code')==0 and child.get('native_pid_absent') is True)
            result['status']='RUNNING' if healthy else 'FAIL';publish(output/'campaign.json',result)
            if not healthy:raise AssertionError('Failed or incomplete stage: '+stage)
        result['status']='PASS';publish(output/'campaign.json',result);return
    directory=output/args.stage;directory.mkdir(mode=0o700,exist_ok=False)
    heartbeat=Heartbeat(directory/'heartbeat.json').start()
    trace_path=directory/'native.jsonl';os.environ['CTT_NATIVE_TRACE']=str(trace_path)
    result=dict(status='RUNNING',python_pid=os.getpid(),rows=[],failures=[],script_sha256=sha(__file__),
        profile_sha256=sha(args.profile),scope=__doc__,pool_config=dict(GPU=[0],ram_bytes=64*1024**3,disk_bytes=120*1024**3),
        numerical_max_error=None,comparisons=0)
    result['pool_config']['checkpoint_idle_seconds']=1800
    publish(directory/'result.json',result)
    coordinator=server=server_thread=None
    observed={};models={}
    try:
        references={b:read(output/('reference-'+b)/'result.json') for b in ('A','B')}
        for ref in references.values():
            assert ref['status']=='PASS' and ref['native_pid_absent'] is True
            assert not Path('/proc',str(ref['native_pid'])).exists() and not Path('/proc',str(ref['python_pid'])).exists()
        fixture=read(output/'reference-A/selected-fixture.json')
        prefix=tuple(fixture['prefix_token_ids']);suffixes={b:tuple(v) for b,v in fixture['suffix_token_ids'].items()}
        assert len(prefix)==args.prefix and all(len(v)==args.suffix for v in suffixes.values())
        assert len(references['A']['tokens'])>4
        def factory(gpu):
            backend=StrataBackend(profile)
            model=BranchingModel(output/'store',backend,profile);model.gpu_id=gpu
            models[gpu]=model
            original=model.run_sampled_request
            def audited(snapshot,ids,count,**kwargs):
                operation=kwargs['stream'];resume=kwargs.get('resume',False)
                branch='B' if operation=='minp-B' else 'A'
                generated=4 if resume else 0
                samples={};source_image=None;audit_before=None
                original_feed,original_decode=backend.consume,backend.consume_generated
                def audit_source():
                    nonlocal source_image,audit_before
                    if source_image is None:
                        source_image=model.cache[snapshot].blob
                        assert isinstance(source_image,StrataImage)
                        audit_before=backend.audit(source_image)
                def sample(lease,name):
                    raw=frontier_bytes(backend.logits(lease),backend.vocab_size)
                    path=directory/(operation+'-'+name+'.f32');path.write_bytes(raw);samples[name]=sha(path)
                def feed(lease,tokens):
                    assert tuple(tokens)==suffixes[branch] and not resume
                    audit_source();original_feed(lease,tokens);sample(lease,'suffix')
                def decode(lease,tokens):
                    nonlocal generated
                    audit_source();original_decode(lease,tokens);generated+=len(tokens)
                    sample(lease,'generated-'+str(generated))
                backend.consume,backend.consume_generated=feed,decode
                try:answer=original(snapshot,ids,count,**kwargs)
                finally:backend.consume,backend.consume_generated=original_feed,original_decode
                audit_after=backend.audit(source_image)
                assert audit_after['sha256']==audit_before['sha256']
                observed[operation]=dict(tokens=answer['tokens'],controller=model.controller(answer['snapshot']),
                    samples=samples,parent_audit_before=audit_before,parent_audit_after=audit_after,
                    request_seconds=answer['request_seconds'],generation_seconds=answer['generation_seconds'])
                return answer
            model.run_sampled_request=audited
            return model
        heartbeat.set('native-start')
        started=time.monotonic()
        transitions=PoolTransitionTrace(args.stage,max_events=20000)
        coordinator=ContextCoordinator(output/'store',PoolConfig(gpus=(0,),ram_bytes=64*1024**3,
            disk_bytes=120*1024**3,min_available_bytes=0,checkpoint_idle_seconds=1800),factory,transition_trace=transitions)
        coordinator.ready(600)
        result['startup_seconds']=time.monotonic()-started
        model=models[0];backend=model.backend
        result['native_pid']=backend.process.pid
        key=secrets.token_hex(24)
        token_adapter=CowAdapter(coordinator)
        text=model_text_adapter(token_adapter,profile['model_path'],profile['execution'].get('text_template_kwargs'),
                                model_contract_id=profile['model_contract_id'])
        port=0 if args.stage=='prepare' else read(output/'prepare/result.json')['port']
        server=make_server(token_adapter,port=port,api_key=key,text_adapter=text)
        server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
        result['port']=server.server_port
        base_url='http://127.0.0.1:'+str(server.server_port)
        client=cow_cli.CowClient(base_url,key,timeout=600)
        journal=cow_cli.RequestJournal(output/'client-state')
        def disk_frontier(snapshot):
            manifest=model.validate_checkpoint(snapshot)
            path=model.checkpoint_directory/(manifest['payload_sha256']+'.state')
            raw=path.read_bytes();size=backend.vocab_size*4
            frontier=raw[-size:];frontier_bytes(__import__('struct').unpack('<'+'f'*backend.vocab_size,frontier),backend.vocab_size)
            return hashlib.sha256(frontier).hexdigest()
        def assert_request(operation,branch,count,resume,answer):
            actual=observed[operation];actual['counter_deltas']=answer['counter_deltas']
            ref=references[branch]
            offset=4 if resume else 0
            expected_tokens=ref['tokens'][offset:offset+count]
            expected_controller=ref['after_four_controller'] if count==4 and not resume else ref['controller']
            names=['generated-'+str(n) for n in range(offset+1,offset+len(expected_tokens)+1)]
            if not resume:names.insert(0,'suffix')
            expected=dict(tokens=expected_tokens,controller=expected_controller,samples={name:ref['samples'][name] for name in names})
            validate_completed_request(actual,expected,vocab=backend.vocab_size,suffix=0 if resume else len(suffixes[branch]))
            assert disk_frontier(answer['snapshot'])==ref['samples']['generated-'+str(offset+len(expected_tokens))]
            result['comparisons']+=len(names)+1
            result['rows'].append(dict(operation=operation,branch=branch,resume=resume,**actual,
                snapshot=answer['snapshot'],durable_frontier_sha256=disk_frontier(answer['snapshot'])))
        if args.stage=='prepare':
            heartbeat.set('create-prefix',backend.process.pid)
            body=dict(branch='Prefix',token_ids=list(prefix),operation_id='prefix',stream=True)
            created=client.stream('/cow/create',body,lambda value:None)
            root=created['receipt']['snapshot'];result['root']=root
            assert disk_frontier(root)==references['A']['samples']['prefix']
            result['comparisons']+=1
            for branch in ('A','B'):coordinator.fork(branch,root)
            result['parent_manifest']=model.validate_checkpoint(root)
            for branch,count in (('A',4),('B',args.outputs)):
                operation='minp-'+branch
                heartbeat.set('serve-'+operation,backend.process.pid)
                head=token_adapter.head(branch)
                body=dict(branch=branch,rendered_token_ids=list(prefix+suffixes[branch]),revision=head.revision,
                    operation_id=operation,max_tokens=count,settings=policy,stream=True)
                if branch=='A':result['A-original-body']=body
                started=time.monotonic();reply=client.stream('/cow/generate',body,lambda value:None)
                result.setdefault('http_seconds',{})[operation]=time.monotonic()-started
                answer=coordinator._requests[operation].snapshot.result(5)
                assert_request(operation,branch,count,False,answer)
                result[branch+'-receipt']=reply['receipt']
                publish(directory/'result.json',result)
            assert model.validate_checkpoint(root)==result['parent_manifest']
        else:
            prepared=read(output/'prepare/result.json')
            assert prepared['status']=='PASS' and prepared['native_pid_absent'] is True
            assert not Path('/proc',str(prepared['python_pid'])).exists() and not Path('/proc',str(prepared['native_pid'])).exists()
            root=prepared['root']
            parent_manifest=model.validate_checkpoint(root)
            heartbeat.set('cli-continue',backend.process.pid)
            command=[sys.executable,'-B',str(runtime/'tools/cow_cli.py'),'--url',base_url,
                '--state-dir',str(output/'client-state'),'continue','--branch','A','--max-tokens',str(len(references['A']['tokens'])-4),
                '--operation-id','resume-A']
            if resumed_policy is not None:
                command+=sampling_arguments(resumed_policy,policy)
            env=dict(os.environ,COW_API_KEY=key)
            with (directory/'continue.stdout').open('wb') as out,(directory/'continue.stderr').open('wb') as err:
                code=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=600).returncode
            assert code==0,'CLI continuation failed'
            answer=coordinator._requests['resume-A'].snapshot.result(5)
            assert_request('resume-A','A',len(references['A']['tokens'])-4,True,answer)
            result['restored_source_counter_deltas']=answer['counter_deltas']
            assert answer['counter_deltas']['loaded_file_bytes']>0
            current=coordinator.store.head('A')
            trace_before=sha(trace_path);before_calls=len(observed)
            heartbeat.set('api-retry-receipt',backend.process.pid)
            # The raw token API is not a CLI text-journal endpoint. Replay its
            # identical body through the shared transport rather than writing
            # an invalid journal record for /cow/generate.
            original=prepared['A-original-body']
            recovered=client.stream('/cow/generate',original,lambda value:None)
            assert sha(trace_path)==trace_before and len(observed)==before_calls
            assert coordinator.store.head('A')==current
            assert recovered['receipt']['snapshot']==prepared['A-receipt']['snapshot']
            result['ack_recovery']=dict(native_pipe_unchanged=True,branch_unchanged=True,
                scope='Raw API replay of acknowledged original operation; not simulated lost CLI acknowledgment',
                receipt_snapshot=recovered['receipt']['snapshot'])
            original=json.loads(json.dumps(original))
            original['settings']['min_p']=.1
            try:client.stream('/cow/generate',original,lambda value:None)
            except cow_cli.ClientError:pass
            else:raise AssertionError('Changed sampler reused a committed operation ID')
            assert sha(trace_path)==trace_before and coordinator.store.head('A')==current
            assert model.validate_checkpoint(root)==parent_manifest
            result['changed_policy_rejected_without_native_calls']=True
        result['metrics']=coordinator.metrics();result['pool_trace']=transitions.snapshot()
        assert not result['pool_trace']['overflowed']
        result['numerical_max_error']=0;result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['failures'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if server is not None:server.shutdown();server.server_close()
        if server_thread is not None:server_thread.join(10)
        if coordinator is not None:coordinator.close()
        if models:
            process=models[0].backend.process
            result['native_exit_code']=process.poll();result['native_pid_absent']=not Path('/proc',str(process.pid)).exists()
            if result['native_exit_code']!=0 or not result['native_pid_absent']:result['status']='FAIL'
        heartbeat.close();publish(directory/'result.json',result)
    if result['status']!='PASS':raise AssertionError('Failed native cleanup')


if __name__=='__main__':main()
