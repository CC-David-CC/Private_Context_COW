"""Image Mirostat 2 feedback through CLI/GPU, expiry and verified regeneration.

This bounded sampled audit tests durable request boundaries, not every streamed
token frontier. Tool proposals are syntax fixtures and are never executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time

from context_time_travel.input_spans import InputHistory,history_for_snapshot,history_segment
from context_time_travel.store import Store,_json_bytes
from context_time_travel.sampled_controller import PlainSamplingSession,start as start_sampled
from context_time_travel import mirostat2
from context_time_travel.sampling_proposal import SelectionProposal,MIROSTAT2_PROPOSAL_PROTOCOL
from context_time_travel.seed_controls import seed_stream,reseed_controller
from run_vision_grammar_probe import GRAMMARS,engine_for
from run_cli_seed_stop_probe import validate_reseed_transition
from run_vision_ingest_probe import Native,compare_files,fingerprint,save
from tools.cow_cli import CowClient
from vision_fixture import reference_positions

SAMPLE_ARGS=('--temperature','.8','--min-p','0','--top-k','0','--top-p','1')


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile=json.loads(args.profile.read_text());engine=engine_for(profile)
    assert 'CTT_NATIVE_TRACE' not in profile['execution']['environment'], 'Profile would overwrite the audit trace'
    sampler_args=('--mirostat','2','--mirostat-tau','4','--mirostat-eta','.05')
    if not 0<=args.context_repeats<=1024:raise ValueError('Context repetition count outside fixture bound')
    registry=out/'registry.json';save(registry,dict(schema=1,states=GRAMMARS))
    config=out/'vision-config.json';config.write_bytes(args.vision_config.read_bytes())
    image=out/'diagram.png';image.write_bytes(args.image.read_bytes())
    context='\n'.join(f'def helper_{i}(value):\n    return value + {i}\n' for i in range(args.context_repeats))
    parts=out/'parts.json';save(parts,[dict(type='image',file=image.name),
        dict(type='text',text=(context+'\n' if context else '')+'Inspect this diagram and propose one small code patch.')])
    store_path=out/'store';store=None;server=client=None;port=0;done=threading.Event();phase={'name':'startup'}
    trace=out/'native-sampling.jsonl'
    key='bounded-image-grammar-canary';env={**os.environ,'COW_API_KEY':key,'PYTHONDONTWRITEBYTECODE':'1',
        'CTT_NATIVE_TRACE':str(trace)}
    result=dict(status='RUNNING',scope=__doc__,profile=fingerprint(args.profile),config=fingerprint(config),
        registry=fingerprint(registry),image=fingerprint(image),servers=[],cli_calls=[],comparisons=[],
        controller_checks=[],references=[],removed_embeddings=[],context_repeats=args.context_repeats,sampler_fixture='mirostat2')
    started=time.monotonic()
    def heartbeat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            pid=os.getpid(),server_pid=server.pid if server is not None else None)),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def cli(*arguments):
        argv=[sys.executable,'-B','-m','tools.cow_cli','--url',client.base_url,'--timeout','300',
            '--state-dir',str(out/'client-state'),'--machine-result',*arguments]
        trace_start=trace.stat().st_size if trace.exists() else 0
        begin=time.monotonic();run=subprocess.run(argv,env=env,capture_output=True,timeout=300)
        with trace.open('rb') if trace.exists() else open(os.devnull,'rb') as stream:
            stream.seek(trace_start);raw=stream.read()
        assert not raw or raw.endswith(b'\n'), 'Incomplete native trace record'
        events=[json.loads(line) for line in raw.splitlines() if line.strip()]
        events=[row for row in events if row.get('sampling_protocol')==MIROSTAT2_PROPOSAL_PROTOCOL]
        operation=arguments[arguments.index('--operation-id')+1] if '--operation-id' in arguments else None
        index=len(result['cli_calls']);stdout=out/f'cli-{index:02d}.stdout';stderr=out/f'cli-{index:02d}.stderr'
        stdout.write_bytes(run.stdout);stderr.write_bytes(run.stderr)
        result['cli_calls'].append(dict(argv=argv,exit_code=run.returncode,seconds=time.monotonic()-begin,
            stdout=fingerprint(stdout),stderr=fingerprint(stderr),operation=operation,native_events=events))
        save(out/'result.json',result);assert run.returncode==0,run.stderr.decode(errors='replace')
        if run.stdout.strip():value=json.loads(run.stdout)
        else:
            text=run.stderr.decode();marker='durable receipt:\n';assert text.count(marker)==1,text
            value=json.loads(text.split(marker,1)[1])
        tokens=value.get('answer',{}).get('tokens',[])
        replay=arguments[0]=='retry' and operation is None
        assert [row['selected_token'] for row in events]==([] if replay else tokens),(arguments,len(events),tokens)
        save(out/'result.json',result);return value
    def start(producer):
        nonlocal server,client,port,store
        index=len(result['servers']);stdout=out/f'server-{index}.stdout';stderr=out/f'server-{index}.stderr'
        audit=out/f'server-{index}-audit.json'
        argv=[sys.executable,'-B','-u',str(Path(__file__).with_name('run_vision_serving_rebuild_probe.py')),
            '--serve-worker',str(store_path),str(audit),'--profile',str(args.profile),'--store',str(store_path),
            '--text','--port',str(port),'--api-key',key,'--ram-gib','16','--disk-gib','32',
            '--grammar-registry',str(registry),'--checkpoint-idle-seconds','1800']
        if producer:argv+=['--vision-config',str(config)]
        with stdout.open('wb') as output,stderr.open('wb') as error:
            server=subprocess.Popen(argv,env=env,stdout=output,stderr=error,start_new_session=True)
        row=dict(pid=server.pid,argv=argv,producer_configured=producer,audit_path=str(audit),
            stdout=str(stdout),stderr=str(stderr));result['servers'].append(row);begin=time.monotonic()
        while True:
            if server.poll() is not None:raise RuntimeError('Server startup failed: '+stderr.read_text(errors='replace'))
            ready=[json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
            ready=[item for item in ready if item.get('event')=='ready']
            if ready:break
            if time.monotonic()-begin>300:raise TimeoutError('Server readiness')
            time.sleep(.1)
        port=ready[-1]['address'][1];client=CowClient('http://127.0.0.1:'+str(port),key,300);store=Store(store_path)
        status=client.request('GET','/cow/status');row['ready_status']=status
        row['native_pids']=[value['pid'] for value in status['workers'].values()]
        assert len(row['native_pids'])==1 and row['native_pids'][0] is not None
        save(out/'result.json',result)
    def stop():
        nonlocal server
        if server is None:return
        row=result['servers'][-1]
        if server.poll() is None:
            if client is not None:row['last_status']=client.request('GET','/cow/status')
            server.send_signal(signal.SIGINT)
        try:code=server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=15);raise TimeoutError('Server shutdown')
        row.update(exit_code=code,process_absent=not Path(f'/proc/{server.pid}').exists(),
            native_absent=all(not Path(f'/proc/{pid}').exists() for pid in row.get('native_pids',[])))
        server=None
        if Path(row['audit_path']).exists():row['audit']=json.loads(Path(row['audit_path']).read_text())
        assert code==0 and row['process_absent'] and row['native_absent'],row
        assert row['audit']['status']=='PASS';save(out/'result.json',result)
    def checkpoint(snapshot,label):
        envelope=json.loads((store_path/'checkpoints'/(snapshot+'.json')).read_text());manifest=envelope['manifest']
        assert hashlib.sha256(_json_bytes(manifest)).hexdigest()==envelope['checksum']
        payload=store_path/'checkpoints'/(manifest['payload_sha256']+'.state')
        assert fingerprint(payload)['sha256']==manifest['payload_sha256']
        with payload.open('rb') as stream:stream.seek(manifest['state_bytes']);raw=stream.read()
        assert len(raw)==4*manifest['logits_count'];path=out/(label+'.f32');path.write_bytes(raw)
        return dict(snapshot=snapshot,path=str(path),manifest=manifest,payload=fingerprint(payload))
    def exact(a,b,label):
        value=dict(label=label,**compare_files(a,b));result['comparisons'].append(value)
        assert value['exact'],value;save(out/'result.json',result)
    def generation(operation):
        envelope=json.loads((store_path/'pool-operations'/(operation+'.json')).read_text())
        assert hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest()==envelope['checksum']
        return envelope['intent']['generation']
    def reseed_record(branch,before,after,seed):
        left=out/(branch+'-before-reseed.json');right=out/(branch+'-after-reseed.json')
        save(left,before);save(right,after)
        return dict(branch=branch,before=fingerprint(left),after=fingerprint(right),seed=seed,
            cursor_before=before['next_draw'],cursor_after=after['next_draw'])
    operations={};targets={};payloads={}
    try:
        start(True);phase['name']='CLI image prepare'
        namespace=cli('namespace','--title','Image Min-P RNG','--operation-id','grammar-namespace')['namespace_id']
        root=cli('prepare','--branch','A','--namespace-id',namespace,'--parts-file',str(parts),'--operation-id','grammar-root')
        targets['prefix']=checkpoint(root['receipt']['snapshot'],'prefix')
        prefix=history_for_snapshot(store,root['receipt']['snapshot'])
        assert root['answer']['counter_deltas']['forward_tokens']==len(prefix)
        phase['name']='CLI partial EDIT and fork'
        partial=cli('answer','--branch','A','--revision','1','--max-tokens','4','--grammar-state','EDIT',
            *SAMPLE_ARGS,*sampler_args,'--seed','42','--operation-id','grammar-partial')
        targets['partial']=checkpoint(partial['receipt']['snapshot'],'partial');operations['grammar-partial']=generation('grammar-partial')
        assert operations['grammar-partial']['schema']==12
        assert store.get_snapshot(partial['receipt']['snapshot'])['controller']['status']=='active'
        cli('fork','--source-branch','A','--name','B','--snapshot',partial['receipt']['snapshot'],'--operation-id','grammar-fork-B')
        b_head=store.head('B');b_bytes=(store.objects/(b_head.tip+'.json')).read_bytes()
        cli('fork','--source-branch','A','--name','D','--snapshot',partial['receipt']['snapshot'],'--operation-id','sampling-fork-D')
        phase['name']='CLI finish EDIT and select TEST'
        a=cli('continue','--branch','A','--revision','2','--max-tokens','32','--operation-id','grammar-finish-A')
        targets['edit']=checkpoint(a['receipt']['snapshot'],'edit');operations['grammar-finish-A']=generation('grammar-finish-A')
        testing=cli('append-user','--branch','A','--revision','3','--text','Run the unit tests now.',
            '--grammar-state','TEST','--max-tokens','32',*SAMPLE_ARGS,*sampler_args,'--seed','99','--operation-id','grammar-test-A')
        targets['test']=checkpoint(testing['receipt']['snapshot'],'test');operations['grammar-test-A']=generation('grammar-test-A')
        assert store.head('B')==b_head and (store.objects/(b_head.tip+'.json')).read_bytes()==b_bytes
        result['sibling_progress_unchanged']=True
        cli('fork','--source-branch','A','--name','C','--snapshot',a['receipt']['snapshot'],'--operation-id','grammar-fork-C')
        phase['name']='CLI new image with TEST grammar'
        image_answer=cli('append-user','--branch','C','--revision','0','--parts-file',str(parts),
            '--grammar-state','TEST','--max-tokens','32',*SAMPLE_ARGS,*sampler_args,'--seed','101','--operation-id','grammar-image-C')
        targets['image']=checkpoint(image_answer['receipt']['snapshot'],'image');operations['grammar-image-C']=generation('grammar-image-C')
        assert operations['grammar-image-C']['schema']==13
        result['answers']=dict(partial=partial,edit=a,test=testing,image=image_answer)
        phase['name']='CLI active grammar reseed'
        before_reseed=store.get_snapshot(store.head('D').tip)['controller']
        cli('reseed','--branch','D','--revision','0','--seed','123','--operation-id','sampling-reseed-D')
        after_reseed=store.get_snapshot(store.head('D').tip)['controller']
        validate_reseed_transition(before_reseed,after_reseed,123)
        targets['reseed-point']=checkpoint(store.head('D').tip,'reseed-point')
        rerolled=cli('continue','--branch','D','--revision','1','--max-tokens','32','--operation-id','sampling-finish-D')
        targets['reseed']=checkpoint(rerolled['receipt']['snapshot'],'reseed');operations['sampling-finish-D']=generation('sampling-finish-D')
        result['answers']['reseed']=rerolled
        result['reseed_checks']=[reseed_record('D',before_reseed,after_reseed,123)]
        phase['name']='CLI plain sampled policy change and reseed'
        cli('fork','--source-branch','A','--name','E','--snapshot',root['receipt']['snapshot'],'--operation-id','sampling-fork-E')
        plain=cli('answer','--branch','E','--revision','0','--max-tokens','4',*SAMPLE_ARGS,*sampler_args,'--seed','314',
            '--operation-id','sampling-plain')
        targets['plain-partial']=checkpoint(plain['receipt']['snapshot'],'plain-partial');operations['sampling-plain']=generation('sampling-plain')
        assert store.get_snapshot(plain['receipt']['snapshot'])['controller']['status']=='active','Plain fixture ended before mid-scope checks'
        warmer=cli('continue','--branch','E','--revision','1','--max-tokens','4','--temperature','.9','--min-p','0',
            '--top-k','0','--top-p','1',*sampler_args,'--mirostat-tau','3.5','--operation-id','sampling-warmer')
        targets['plain-warmer']=checkpoint(warmer['receipt']['snapshot'],'plain-warmer');operations['sampling-warmer']=generation('sampling-warmer')
        before_reseed=store.get_snapshot(store.head('E').tip)['controller']
        cli('reseed','--branch','E','--revision','2','--seed','271','--operation-id','sampling-reseed-E')
        after_reseed=store.get_snapshot(store.head('E').tip)['controller'];validate_reseed_transition(before_reseed,after_reseed,271)
        targets['plain-reseed-point']=checkpoint(store.head('E').tip,'plain-reseed-point')
        plain_tail=cli('continue','--branch','E','--revision','3','--max-tokens','4','--operation-id','sampling-plain-tail')
        targets['plain']=checkpoint(plain_tail['receipt']['snapshot'],'plain');operations['sampling-plain-tail']=generation('sampling-plain-tail')
        result['answers'].update(plain_partial=plain,plain_warmer=warmer,plain_tail=plain_tail)
        result['reseed_checks'].append(reseed_record('E',before_reseed,after_reseed,271))
        assert store.head('B')==b_head and (store.objects/(b_head.tip+'.json')).read_bytes()==b_bytes
        assert fingerprint(targets['prefix']['payload']['path'])==targets['prefix']['payload']
        phase['name']='CLI changed plain/grammar/typed-image retries'
        for case,source_operation,branch,seed in (
                ('retry-plain','sampling-plain','E',411),
                ('retry-test','grammar-test-A','A',412),
                ('retry-image','grammar-image-C','C',413)):
            request=cli('retry','--from-operation',source_operation,'--branch',branch,
                '--temperature','.8','--min-p','0','--top-k','0','--top-p','1',*sampler_args,
                '--seed',str(seed),'--operation-id',case)
            targets[case]=checkpoint(request['receipt']['snapshot'],case)
            operations[case]=generation(case);result['answers'][case]=request
            descriptor=operations[case]
            delta=InputHistory(descriptor['ids']) if descriptor['schema']==13 else tuple(descriptor['ids'])
            assert request['answer']['counter_deltas']['forward_tokens']==len(delta)+len(request['answer']['tokens'])
            assert descriptor['retry_source_operation']==source_operation
            if case=='retry-image':assert request['answer']['input_preparation']['encoder_calls']==0
        # Retain isolated oracle payloads before removing all managed derived
        # embeddings. The second server cannot read these external files.
        for path in sorted((store_path/'checkpoints').glob('*.sve')):
            target=out/('oracle-'+fingerprint(path)['sha256']+'.sve');target.write_bytes(path.read_bytes())
            payloads[fingerprint(path)['sha256']]=target
        stop()
        for path in sorted((store_path/'checkpoints').glob('*.sve')):
            assert path.resolve().parent==(store_path/'checkpoints').resolve() and not path.is_symlink()
            result['removed_embeddings'].append(fingerprint(path));path.unlink()
        phase['name']='new producer-free server, CLI grammar resume'
        start(False)
        b=cli('continue','--branch','B','--revision',str(b_head.revision),'--max-tokens','32','--operation-id','grammar-finish-B')
        assert b['answer']['tokens']==a['answer']['tokens']
        assert b['answer']['counter_deltas']['forward_tokens']==len(b['answer']['tokens'])
        assert b['answer']['counter_deltas']['loaded_file_bytes']>0
        assert store.get_snapshot(b['receipt']['snapshot'])['controller']==store.get_snapshot(a['receipt']['snapshot'])['controller']
        result['controller_checks'].append(dict(label='A-B-resumed-controller',exact=True))
        targets['cold-edit']=checkpoint(b['receipt']['snapshot'],'cold-edit')
        exact(targets['edit']['path'],targets['cold-edit']['path'],'A-B-restarted-edit')
        replay=cli('retry','--from-operation','grammar-image-C')
        assert replay['answer']['tokens']==image_answer['answer']['tokens'];result['replay']=replay
        assert not list((store_path/'checkpoints').glob('*.sve'))
        stop();result['answers']['cold_edit']=b
        phase['name']='CLI changed typed-image retry after embedding eviction'
        assert not list((store_path/'checkpoints').glob('*.sve'))
        start(True)
        regenerated=cli('retry','--from-operation','grammar-image-C','--branch','C',
            '--temperature','.8','--min-p','0','--top-k','0','--top-p','1',*sampler_args,
            '--seed','414','--operation-id','retry-image-reencoded')
        targets['retry-image-reencoded']=checkpoint(regenerated['receipt']['snapshot'],'retry-image-reencoded')
        operations['retry-image-reencoded']=generation('retry-image-reencoded')
        result['answers']['retry-image-reencoded']=regenerated
        assert operations['retry-image-reencoded']['ids']==operations['grammar-image-C']['ids']
        assert regenerated['answer']['input_preparation']['encoder_calls']==1
        assert regenerated['answer']['input_preparation']['images'][0]['source']=='reencoded'
        delta=InputHistory(operations['retry-image-reencoded']['ids'])
        assert regenerated['answer']['counter_deltas']['forward_tokens']==len(delta)+len(regenerated['answer']['tokens'])
        assert fingerprint(targets['prefix']['payload']['path'])==targets['prefix']['payload']
        stop()
        result['targets']=targets;result['operations']=operations
        # Independently execute both current paths, importing no checkpoint.
        # Check every generated choice; compare all available durable frontiers.
        for case in ('test','image','reseed','plain','retry-plain','retry-test','retry-image','retry-image-reencoded'):
            phase['name']='fresh raw reference '+case
            history=history_for_snapshot(store,targets[case]['snapshot']);schedule=targets[case]['manifest']['execution_schedule']
            chosen=({'test':['grammar-partial','grammar-finish-A','grammar-test-A'],
                'image':['grammar-partial','grammar-finish-A','grammar-image-C'],
                'reseed':['grammar-partial','sampling-finish-D'],
                'plain':['sampling-plain','sampling-warmer','sampling-plain-tail'],
                'retry-plain':['retry-plain'],
                'retry-test':['grammar-partial','grammar-finish-A','retry-test'],
                'retry-image':['grammar-partial','grammar-finish-A','retry-image'],
                'retry-image-reencoded':['grammar-partial','grammar-finish-A','retry-image-reencoded']})[case]
            scopes={}
            for operation in chosen:
                descriptor=operations[operation]
                source=history_for_snapshot(store,descriptor['source_head']['tip'])
                delta=InputHistory(descriptor['ids']) if descriptor['schema']==13 else tuple(descriptor['ids'])
                prompt=source+delta;scopes[len(prompt)]=dict(state=descriptor['grammar']['agent_state'] if descriptor['grammar'] else None,
                    stream=descriptor['stream_operation_id'],text=prompt.text_tokens,descriptor=descriptor,operation=operation)
            argv=list(profile['execution']['sut_command']);argv[0]=str(args.reference_binary)
            native=Native(argv,out/('reference-'+case));row=dict(case=case,pid=native.proc.pid,choices=[],argv=argv)
            result['references'].append(row);session=None;values=None;plain_scope=False
            mask=bytearray(b'\xff'*((engine.vocab_size+7)//8))
            if engine.vocab_size%8:mask[-1]&=(1<<(engine.vocab_size%8))-1
            full_mask=bytes(mask)
            def restore_session(state,text):
                if state.get('constraint') is not None:return engine.restore(state,text)
                return PlainSamplingSession.restore(state,text,engine.vocab_size,engine.tokenizer.token_bytes,
                    lambda token:token in engine.eos_tokens)
            try:
                for event_index,event in enumerate(schedule):
                    start_cell=event['position'];end=start_cell+event['count'];delta=history_segment(history,start_cell,end)
                    if event['operator']=='batched-prefill':
                        if any(span['kind']=='image' for span in delta.spans):
                            tokens=delta.native_tokens;lines=[f'COW_VISION_INPUT_V1 {start_cell} {len(tokens)} '+
                                str(sum(span['kind']=='image' for span in delta.spans)),' '.join(map(str,tokens))]
                            cell=start_cell
                            for span in delta.spans:
                                if span['kind']=='text':cell+=len(span['tokens']);continue
                                fields=[cell]+[span[key] for key in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
                                lines.append(' '.join(map(str,fields))+' '+json.dumps(str(payloads[span['embedding_sha256']])));cell+=span['rows']
                            manifest=out/f'{case}-{event_index}.manifest';manifest.write_text('\n'.join(lines)+'\n')
                            prefix_history=history_segment(history,0,end);images=[];cell=0
                            for span in prefix_history.spans:
                                if span['kind']=='text':cell+=len(span['tokens']);continue
                                images.append(dict(start=cell,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']));cell+=span['rows']
                            positions=out/f'{case}-{event_index}.positions.i32'
                            positions.write_bytes(b''.join(struct.pack('<3i',*p) for p in reference_positions(end,images,profile['execution']['capacity'])))
                            native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(positions)))
                        else:native.request('feed '+' '.join(map(str,delta.text_tokens)))
                    else:
                        assert event['operator']=='normal-verifier-decode' and len(delta)==1
                        if start_cell in scopes:
                            scope=scopes[start_cell];descriptor=scope['descriptor'];policy=descriptor['sampling']
                            executed=[call for call in result['cli_calls'] if call['operation']==scope['operation']]
                            assert len(executed)==1,(case,scope['operation'])
                            native_events=executed[0]['native_events'];scope_index=0
                            plain_scope=descriptor['grammar'] is None
                            if descriptor['resume']:
                                assert session is not None
                                state=session.current_controller
                                if ((case=='reseed' and start_cell==targets['partial']['manifest']['consumed'])
                                        or (case=='plain' and start_cell==targets['plain-warmer']['manifest']['consumed'])):
                                    state=reseed_controller(state,seed=123 if case=='reseed' else 271,
                                        history=scope['text'],vocab_size=engine.vocab_size)
                                session=restore_session(state,scope['text'])
                                if policy is not None:session.configure_generation(policy)
                            else:
                                seed=descriptor['controls']['seed']
                                stream=seed_stream(seed) if seed is not None else scope['stream']
                                state=(start_sampled(policy,stream,scope['text'],engine.vocab_size,engine.eos_tokens)
                                    if plain_scope else engine.start(scope['state'],stream,scope['text'],engine.vocab_size,sampling=policy))
                                session=restore_session(state,scope['text'])
                        assert session is not None and values is not None
                        token=delta.text_tokens[0]
                        expected_choice={}
                        def propose(mask,temp,p,*,draw_start,draw_candidates,source_digest,**filters):
                            assert p==0. and filters.get('dry_multiplier',0.)==0.
                            saved=mirostat2.validate_state(filters['sampler_state'])
                            choice=mirostat2.choose(values,mask,temperature=temp,draw53=draw_candidates[0],
                                mu=saved['mu'],tau=filters['mirostat_tau'],eta=filters['mirostat_eta'])
                            expected_choice.update(choice,draw_start=draw_start,draw_count=1,
                                source_digest=source_digest,mu=saved['mu'])
                            feedback=dict(saved,mu=choice['next_mu'])
                            return SelectionProposal(choice['token'],draw_start,1,feedback,source_digest)
                        selected=(session.choose(full_mask,propose) if plain_scope else session._choose_proposal(propose))
                        actual=native_events[scope_index];scope_index+=1
                        assert actual['selected_token']==selected and actual['consumed']==start_cell
                        assert actual['source_digest']==expected_choice['source_digest']
                        for key in ('draw_start','draw_count'):
                            assert actual[key]==expected_choice[key],(case,key,actual,expected_choice)
                        for key in ('draw53','draw_count','mu','next_mu','selected_log_probability',
                                'selected_probability','surprise_bits','kept_count'):
                            assert actual['native_receipt'][key]==expected_choice[key],(case,key,actual,expected_choice)
                        assert selected==token,(case,start_cell,selected,token)
                        native.request('decode '+str(token));session._consume_selected(token,engine.tokenizer.token_bytes(token),token in engine.eos_tokens)
                        row['choices'].append(dict(position=start_cell,token=token,sampled_cpu_exact=True,
                            native_receipt_exact=True,operation=scope['operation'],expected=expected_choice))
                    path,values=native.frontier('event-'+str(event_index))
                    for label,target in targets.items():
                        target_history=history_for_snapshot(store,target['snapshot'])
                        if len(target_history)==end and history_segment(history,0,end)==target_history:
                            exact(path,target['path'],case+'-'+label)
                assert session.current_controller==store.get_snapshot(targets[case]['snapshot'])['controller']
                result['controller_checks'].append(dict(label=case+'-fresh-controller',exact=True))
                row['work']=native.request('stats');assert row['work']['forward_tokens']==len(history)
            finally:row['exit']=native.close()
        assert fingerprint(targets['prefix']['payload']['path'])==targets['prefix']['payload']
        work={}
        for label,answer in result['answers'].items():
            counters=answer['answer']['counter_deltas']
            assert counters['sampling_mirostat2_calls']==len(answer['answer']['tokens']),(label,counters)
            assert counters['sampling_v4_calls']==0,(label,counters)
            for key in ('sampling_mirostat2_calls','sampling_v4_calls'):
                work[key]=work.get(key,0)+counters[key]
        assert work['sampling_mirostat2_calls']>0 and work['sampling_v4_calls']==0
        events=[event for call in result['cli_calls'] for event in call['native_events']]
        assert events and all(event['draw_count']==1 for event in events)
        assert any(event['next_mu']!=event['mu'] for event in events)
        result['feedback_work_including_cold_continuation']=work
        result['native_trace']=fingerprint(trace)
        result.update(status='PASS',max_abs_error=0.,prefix_cells=len(prefix),prefix_text_tokens=len(prefix.text_tokens),
            image_rows=len(prefix)-len(prefix.text_tokens))
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();result['seconds']=time.monotonic()-started;save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','vision-config','image','reference-binary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--context-repeats',type=int,default=0)
    campaign(parser.parse_args())
