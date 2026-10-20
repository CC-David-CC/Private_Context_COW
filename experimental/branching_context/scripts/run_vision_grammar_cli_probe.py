"""Image GBNF through CLI/HTTP/GPU, restart and fresh schedule-matched oracles.

This bounded greedy audit tests durable request boundaries, not every streamed
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
from run_vision_grammar_probe import GRAMMARS,engine_for
from run_vision_ingest_probe import Native,compare_files,fingerprint,save
from tools.cow_cli import CowClient
from vision_fixture import reference_positions


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile=json.loads(args.profile.read_text());engine=engine_for(profile)
    if not 0<=args.context_repeats<=1024:raise ValueError('Context repetition count outside fixture bound')
    registry=out/'registry.json';save(registry,dict(schema=1,states=GRAMMARS))
    config=out/'vision-config.json';config.write_bytes(args.vision_config.read_bytes())
    image=out/'diagram.png';image.write_bytes(args.image.read_bytes())
    context='\n'.join(f'def helper_{i}(value):\n    return value + {i}\n' for i in range(args.context_repeats))
    parts=out/'parts.json';save(parts,[dict(type='image',file=image.name),
        dict(type='text',text=context+'\nInspect this diagram and propose one small code patch.')])
    store_path=out/'store';store=None;server=client=None;port=0;done=threading.Event();phase={'name':'startup'}
    key='bounded-image-grammar-canary';env={**os.environ,'COW_API_KEY':key,'PYTHONDONTWRITEBYTECODE':'1'}
    result=dict(status='RUNNING',scope=__doc__,profile=fingerprint(args.profile),config=fingerprint(config),
        registry=fingerprint(registry),image=fingerprint(image),servers=[],cli_calls=[],comparisons=[],
        controller_checks=[],references=[],removed_embeddings=[],context_repeats=args.context_repeats)
    started=time.monotonic()
    def heartbeat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            pid=os.getpid(),server_pid=server.pid if server is not None else None)),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def cli(*arguments):
        argv=[sys.executable,'-B','-m','tools.cow_cli','--url',client.base_url,'--timeout','300',
            '--state-dir',str(out/'client-state'),'--machine-result',*arguments]
        begin=time.monotonic();run=subprocess.run(argv,env=env,capture_output=True,timeout=300)
        index=len(result['cli_calls']);stdout=out/f'cli-{index:02d}.stdout';stderr=out/f'cli-{index:02d}.stderr'
        stdout.write_bytes(run.stdout);stderr.write_bytes(run.stderr)
        result['cli_calls'].append(dict(argv=argv,exit_code=run.returncode,seconds=time.monotonic()-begin,
            stdout=fingerprint(stdout),stderr=fingerprint(stderr)))
        save(out/'result.json',result);assert run.returncode==0,run.stderr.decode(errors='replace')
        if run.stdout.strip():return json.loads(run.stdout)
        text=run.stderr.decode();marker='durable receipt:\n';assert text.count(marker)==1,text
        return json.loads(text.split(marker,1)[1])
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
    operations={};targets={};payloads={}
    try:
        start(True);phase['name']='CLI image prepare'
        namespace=cli('namespace','--title','Image GBNF','--operation-id','grammar-namespace')['namespace_id']
        root=cli('prepare','--branch','A','--namespace-id',namespace,'--parts-file',str(parts),'--operation-id','grammar-root')
        targets['prefix']=checkpoint(root['receipt']['snapshot'],'prefix')
        prefix=history_for_snapshot(store,root['receipt']['snapshot'])
        assert root['answer']['counter_deltas']['forward_tokens']==len(prefix)
        phase['name']='CLI partial EDIT and fork'
        partial=cli('answer','--branch','A','--revision','1','--max-tokens','4','--grammar-state','EDIT','--operation-id','grammar-partial')
        targets['partial']=checkpoint(partial['receipt']['snapshot'],'partial');operations['grammar-partial']=generation('grammar-partial')
        assert operations['grammar-partial']['schema']==10
        assert store.get_snapshot(partial['receipt']['snapshot'])['controller']['status']=='active'
        cli('fork','--source-branch','A','--name','B','--snapshot',partial['receipt']['snapshot'],'--operation-id','grammar-fork-B')
        b_head=store.head('B');b_bytes=(store.objects/(b_head.tip+'.json')).read_bytes()
        phase['name']='CLI finish EDIT and select TEST'
        a=cli('continue','--branch','A','--revision','2','--max-tokens','32','--operation-id','grammar-finish-A')
        targets['edit']=checkpoint(a['receipt']['snapshot'],'edit');operations['grammar-finish-A']=generation('grammar-finish-A')
        testing=cli('append-user','--branch','A','--revision','3','--text','Run the unit tests now.',
            '--grammar-state','TEST','--max-tokens','32','--operation-id','grammar-test-A')
        targets['test']=checkpoint(testing['receipt']['snapshot'],'test');operations['grammar-test-A']=generation('grammar-test-A')
        assert store.head('B')==b_head and (store.objects/(b_head.tip+'.json')).read_bytes()==b_bytes
        result['sibling_progress_unchanged']=True
        cli('fork','--source-branch','A','--name','C','--snapshot',a['receipt']['snapshot'],'--operation-id','grammar-fork-C')
        phase['name']='CLI new image with TEST grammar'
        image_answer=cli('append-user','--branch','C','--revision','0','--parts-file',str(parts),
            '--grammar-state','TEST','--max-tokens','32','--operation-id','grammar-image-C')
        targets['image']=checkpoint(image_answer['receipt']['snapshot'],'image');operations['grammar-image-C']=generation('grammar-image-C')
        assert operations['grammar-image-C']['schema']==11
        result['answers']=dict(partial=partial,edit=a,test=testing,image=image_answer)
        assert fingerprint(targets['prefix']['payload']['path'])==targets['prefix']['payload']
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
        result['targets']=targets;result['operations']=operations
        # Independently execute both current paths, importing no checkpoint.
        # Check every generated choice; compare all available durable frontiers.
        for case in ('test','image'):
            phase['name']='fresh raw reference '+case
            history=history_for_snapshot(store,targets[case]['snapshot']);schedule=targets[case]['manifest']['execution_schedule']
            chosen=['grammar-partial','grammar-finish-A','grammar-test-A' if case=='test' else 'grammar-image-C']
            scopes={}
            for operation in chosen:
                descriptor=operations[operation]
                if descriptor['resume']:continue
                source=history_for_snapshot(store,descriptor['source_head']['tip'])
                delta=InputHistory(descriptor['ids']) if descriptor['schema']==11 else tuple(descriptor['ids'])
                prompt=source+delta;scopes[len(prompt)]=dict(state=descriptor['grammar']['agent_state'],
                    stream=descriptor['stream_operation_id'],text=prompt.text_tokens)
            argv=list(profile['execution']['sut_command']);argv[0]=str(args.reference_binary)
            native=Native(argv,out/('reference-'+case));row=dict(case=case,pid=native.proc.pid,choices=[],argv=argv)
            result['references'].append(row);session=None;values=None
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
                            scope=scopes[start_cell];state=engine.start(scope['state'],scope['stream'],scope['text'],engine.vocab_size)
                            session=engine.restore(state,scope['text'])
                        assert session is not None and values is not None
                        token=delta.text_tokens[0];selected=session.choose(values);assert selected==token,(case,start_cell,selected,token)
                        native.request('decode '+str(token));session._consume_selected(token,engine.tokenizer.token_bytes(token),token in engine.eos_tokens)
                        row['choices'].append(dict(position=start_cell,token=token,legal_greedy_exact=True))
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
