"""Real Q4 image questions through CLI/HTTP, then producer-free server restart.

Four retained independent vectors plus four newly executed uncached vectors.
Pending client journals simulate missed local acknowledgments, not power loss.
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

from context_time_travel import controller
from context_time_travel.input_spans import InputHistory,history_for_snapshot
from context_time_travel.store import Store,_json_bytes
from run_vision_ingest_probe import Native,compare_files,fingerprint,save
from serve.cow_http import model_text_components
from tools.cow_cli import CowClient,RequestJournal
from vision_fixture import reference_positions


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();prior=json.loads((baseline/'result.json').read_text());assert prior['status']=='PASS'
    profile=json.loads(Path(prior['profile']['path']).read_text())
    build=json.loads(Path(prior['build']['path']).read_text())
    config_path=out/'vision-config.json';config_path.write_bytes(Path(prior['producer_config']['path']).read_bytes())
    parts=model_text_components(profile['model_path'],profile['execution'].get('text_template_kwargs'),
        model_contract_id=profile['model_contract_id'],mixed_inputs=True)
    prefix=InputHistory(prior['inputs']['prefix']);deltas={key:InputHistory(value) for key,value in prior['inputs']['deltas'].items()}
    separator=tuple(parts['tokenizer'].encode('\n',parse_special=False))
    later={branch:InputHistory([dict(kind='text',tokens=list(separator))])+deltas[other]
           for branch,other in (('A','B'),('B','A'))}
    result=dict(status='RUNNING',scope=__doc__,baseline=fingerprint(baseline/'result.json'),profile=prior['profile'],
        build=prior['build'],config=fingerprint(config_path),servers=[],cli_calls=[],branches={},comparisons=[],
        removed_embeddings=[],removed_client_files=[],replays=[],fresh_references=[])
    store_path=out/'store';store=None;server=None;client=None;port=0
    key='bounded-image-suffix-canary';env={**os.environ,'COW_API_KEY':key,'PYTHONDONTWRITEBYTECODE':'1'}
    done=threading.Event();phase={'name':'startup'};started=time.monotonic()
    def heartbeat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            python_pid=os.getpid(),server_pid=server.pid if server is not None else None)),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def cli(*arguments,journal=None):
        argv=[sys.executable,'-B','-m','tools.cow_cli','--url',client.base_url,'--timeout','300',
            '--state-dir',str(journal or out/'client-state'),'--machine-result',*arguments]
        index=len(result['cli_calls']);begin=time.monotonic()
        run=subprocess.run(argv,env=env,capture_output=True,timeout=300)
        stdout=out/f'cli-{index:02d}.stdout';stderr=out/f'cli-{index:02d}.stderr'
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
        argv=[sys.executable,'-B','-u','-m','serve.cow_http','--profile',prior['profile']['path'],'--store',str(store_path),
            '--text','--port',str(port),'--api-key',key,'--ram-gib','16','--disk-gib','32','--checkpoint-idle-seconds','1800']
        if producer:argv+=['--vision-config',str(config_path)]
        begin=time.monotonic()
        with stdout.open('wb') as output,stderr.open('wb') as error:
            server=subprocess.Popen(argv,env=env,stdout=output,stderr=error,start_new_session=True)
        row=dict(pid=server.pid,argv=argv,producer_configured=producer,stdout=str(stdout),stderr=str(stderr))
        result['servers'].append(row)
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited: '+stderr.read_text(errors='replace'))
            ready=[json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
            ready=[item for item in ready if item.get('event')=='ready']
            if ready:break
            if time.monotonic()-begin>300:raise TimeoutError('Server readiness timed out')
            time.sleep(.1)
        port=ready[-1]['address'][1];client=CowClient('http://127.0.0.1:'+str(port),key,300);store=Store(store_path)
        row['ready_status']=client.request('GET','/cow/status');row['startup_seconds']=time.monotonic()-begin
        row['native_pids']=[value['pid'] for value in row['ready_status']['workers'].values()]
        assert len(row['native_pids'])==1 and row['native_pids'][0] is not None
        save(out/'result.json',result)
    def stop():
        nonlocal server
        if server is None:return
        row=result['servers'][-1]
        if 'ready_status' in row:row['last_status']=client.request('GET','/cow/status')
        server.send_signal(signal.SIGINT)
        try:code=server.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGTERM);server.wait(15);raise TimeoutError('Server shutdown timed out')
        row.update(exit_code=code,process_absent=not Path(f'/proc/{server.pid}').exists(),
            native_processes_absent=all(not Path(f'/proc/{pid}').exists() for pid in row.get('native_pids',[])))
        server=None
        assert code==0 and row['process_absent'] and row['native_processes_absent'],row
        save(out/'result.json',result)
    def vector(snapshot,label):
        envelope=json.loads((store_path/'checkpoints'/(snapshot+'.json')).read_text());manifest=envelope['manifest']
        assert hashlib.sha256(_json_bytes(manifest)).hexdigest()==envelope['checksum']
        payload=store_path/'checkpoints'/(manifest['payload_sha256']+'.state')
        assert fingerprint(payload)['sha256']==manifest['payload_sha256']
        with payload.open('rb') as stream:stream.seek(manifest['state_bytes']);raw=stream.read()
        assert len(raw)==4*manifest['logits_count'];path=out/(label+'.f32');path.write_bytes(raw);return path
    def exact(reference,candidate,label):
        row=dict(label=label,reference=fingerprint(reference),candidate=fingerprint(candidate),**compare_files(reference,candidate))
        result['comparisons'].append(row);save(out/'result.json',result);assert row['exact'],row
    def generation(operation):
        envelope=json.loads((store_path/'pool-operations'/(operation+'.json')).read_text())
        assert hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest()==envelope['checksum']
        return envelope['intent']['generation']
    def check_controller(snapshot,source,delta,operation,tokens):
        history=history_for_snapshot(store,source)+delta
        expected=controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),operation,
            history.text_tokens,len(parts['tokenizer'].tokens))
        expected=controller.consume_many(expected,((token,parts['tokenizer'].token_bytes(token),
            token in profile['execution']['eos_tokens']) for token in tokens))
        assert store.get_snapshot(snapshot)['controller']==expected
    payloads={fingerprint(path)['sha256']:path for path in baseline.glob('*.sve')}
    def positions(history,label):
        images=[];count=0
        for span in history.spans:
            if span['kind']=='text':count+=len(span['tokens'])
            else:
                images.append(dict(start=count,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']))
                count+=span['rows']
        path=out/(label+'-positions.i32')
        path.write_bytes(b''.join(struct.pack('<3i',*row) for row in reference_positions(count,images,profile['execution']['capacity'])))
        return path
    def manifest(delta,count,label):
        tokens=[];images=[]
        for span in delta.spans:
            if span['kind']=='text':tokens.extend(span['tokens'])
            else:images.append((count+len(tokens),span));tokens.extend([span['pad_token_id']]*span['rows'])
        lines=[f'COW_VISION_INPUT_V1 {count} {len(tokens)} {len(images)}',' '.join(map(str,tokens))]
        for position,span in images:
            fields=[position]+[span[key] for key in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
            lines.append(' '.join(map(str,fields))+' '+json.dumps(str(payloads[span['embedding_sha256']])))
        path=out/(label+'.manifest');path.write_text('\n'.join(lines)+'\n');return path
    try:
        start(True);namespace=cli('namespace','--title','Incremental image questions','--operation-id','suffix-namespace')['namespace_id']
        originals={}
        for span in [*prefix.spans,*deltas['B'].spans]:
            if span['kind']=='image' and span['original_sha256'] not in originals:
                path=out/('image-'+span['original_sha256']+'.png')
                path.write_bytes((baseline/'store/image-originals'/(span['original_sha256']+'.input')).read_bytes());originals[span['original_sha256']]=path
        root_parts=[]
        for index,span in enumerate([s for s in prefix.spans if s['kind']=='image']):
            root_parts.extend([dict(type='text',text=' First diagram. ' if index==0 else ' Second diagram. '),
                dict(type='image',file=originals[span['original_sha256']].name)])
        root_parts.append(dict(type='text',text='Keep these diagrams as shared context.'))
        root_file=out/'root-parts.json';save(root_file,root_parts);phase['name']='CLI initial context'
        root=cli('context','--branch','A','--namespace-id',namespace,'--parts-file',str(root_file),'--operation-id','suffix-root')
        result['root']=root;assert root['answer']['counter_deltas']['forward_tokens']==1097
        for branch,index in (('A',0),('B',2)):
            exact(prior['references'][index]['paths'][0],vector(root['receipt']['snapshot'],'root-'+branch),'root-'+branch)
        parent=store_path/'checkpoints'/(root['receipt']['payload_sha256']+'.state');parent_before=fingerprint(parent)
        cli('fork','--source-branch','A','--name','B','--snapshot',root['receipt']['snapshot'],'--operation-id','suffix-fork-B')
        files={}
        for branch in ('A','B'):
            image=next(span for span in deltas[branch].spans if span['kind']=='image')
            value=[dict(type='text',text='New diagram. '),dict(type='image',file=originals[image['original_sha256']].name),
                dict(type='text',text=' What color is the left half of this new diagram? Answer with one color only.')]
            path=out/('question-'+branch+'.json');save(path,value);files[branch]=path
        for branch,revision,index in (('A',1,0),('B',0,2)):
            phase['name']='CLI first image question '+branch;operation='suffix-question-'+branch
            value=cli('ask','--branch',branch,'--revision',str(revision),'--parts-file',str(files[branch]),
                '--max-tokens','8','--operation-id',operation)
            assert value['answer']['tokens']==prior['branches'][branch]['tokens']
            assert value['answer']['counter_deltas']['forward_tokens']==86
            assert InputHistory(generation(operation)['ids'])==deltas[branch]
            exact(prior['references'][index]['paths'][3],vector(value['receipt']['snapshot'],branch+'-first'),branch+'-first')
            check_controller(value['receipt']['snapshot'],root['receipt']['snapshot'],deltas[branch],operation,value['answer']['tokens'])
            result['branches'][branch]=dict(first=value,first_controller_exact=True)
        for branch,other,revision in (('A','B',2),('B','A',1)):
            phase['name']='CLI next image question '+branch;operation='suffix-next-'+branch
            source=result['branches'][branch]['first']['receipt']['snapshot']
            value=cli('append-user','--branch',branch,'--revision',str(revision),'--parts-file',str(files[other]),
                '--max-tokens','8','--operation-id',operation)
            assert value['answer']['counter_deltas']['forward_tokens']==85+len(value['answer']['tokens'])
            assert InputHistory(generation(operation)['ids'])==later[branch]
            check_controller(value['receipt']['snapshot'],source,later[branch],operation,value['answer']['tokens'])
            result['branches'][branch].update(next=value,next_controller_exact=True,
                next_frontier=fingerprint(vector(value['receipt']['snapshot'],branch+'-next')))
        assert fingerprint(parent)==parent_before
        # Construct a separate pending journal with the exact submitted bodies:
        # a missed client-side acknowledgment fixture, not a killed network.
        pending=out/'pending-client-state';journal=RequestJournal(out/'client-state');missed=RequestJournal(pending)
        for operation in ('suffix-question-A','suffix-next-A'):
            saved=journal.get(operation);assert saved['state']=='complete'
            missed.begin(operation,saved['url'],saved['path'],saved['body_json'].encode())
        embeddings=list((store_path/'checkpoints').glob('*.sve'));assert len(embeddings)==3;stop()
        for path in embeddings:
            assert path.resolve().parent==(store_path/'checkpoints').resolve() and not path.is_symlink()
            result['removed_embeddings'].append(fingerprint(path));path.unlink()
        for path in [*originals.values(),root_file,*files.values()]:
            result['removed_client_files'].append(fingerprint(path));path.unlink()
        phase['name']='new producer-free server';start(False)
        for operation in ('suffix-question-A','suffix-next-A'):
            value=cli('retry-request','--operation-id',operation,journal=pending)
            assert value['answer']['recovered'] and value['answer']['input_preparation']['source']=='committed-replay'
            result['replays'].append(value)
        for branch,revision in (('A',3),('B',2)):
            phase['name']='cold text followup '+branch;operation='suffix-followup-'+branch
            source=result['branches'][branch]['next']['receipt']['snapshot']
            value=cli('append-user','--branch',branch,'--revision',str(revision),'--text',
                'What color is the right half of the newest diagram? Answer with one color only.',
                '--max-tokens','8','--operation-id',operation)
            suffix=tuple(generation(operation)['ids']);assert suffix==separator+tuple(prior['inputs']['followup'])
            assert value['answer']['counter_deltas']['forward_tokens']==len(suffix)+len(value['answer']['tokens'])
            assert value['answer']['counter_deltas']['loaded_file_bytes']>0
            check_controller(value['receipt']['snapshot'],source,suffix,operation,value['answer']['tokens'])
            result['branches'][branch].update(cold=value,cold_controller_exact=True,cold_suffix=list(suffix),
                cold_frontier=fingerprint(vector(value['receipt']['snapshot'],branch+'-cold')))
        assert not list((store_path/'checkpoints').glob('*.sve')) and fingerprint(parent)==parent_before
        result['graph']=client.request('GET','/cow/workspace/graph?chat='+namespace)
        assert result['graph']['shared_prefix_messages']==1;result['parent_payload_unchanged']=parent_before;stop()
        assert result['servers'][0]['pid']!=result['servers'][1]['pid']
        assert result['servers'][0]['native_pids']!=result['servers'][1]['native_pids']
        for branch in ('A','B'):
            phase['name']='independent fresh native '+branch
            argv=list(profile['execution']['sut_command']);argv[0]=build['binaries']['reference']['path']
            assert fingerprint(argv[0])['sha256']==build['binaries']['reference']['sha256']
            native=Native(argv,out/('reference-'+branch));row=dict(branch=branch,pid=native.proc.pid,argv=argv)
            result['fresh_references'].append(row)
            try:
                assert native.ready['reference']=='fresh-no-checkpoints'
                native.request('feed_image '+json.dumps(str(baseline/'prefix.manifest'))+' '+json.dumps(str(baseline/'prefix-positions.i32')))
                native.request('feed_image '+json.dumps(str(baseline/('delta-'+branch+'.manifest')))+' '+json.dumps(str(baseline/('delta-'+branch+'-positions.i32'))))
                for token in result['branches'][branch]['first']['answer']['tokens']:native.request('decode '+str(token))
                first_history=prefix+deltas[branch]+tuple(result['branches'][branch]['first']['answer']['tokens'])
                packet=manifest(later[branch],len(first_history),'next-'+branch);coords=positions(first_history+later[branch],'next-'+branch)
                native.request('feed_image '+json.dumps(str(packet))+' '+json.dumps(str(coords)))
                _,values=native.frontier('next');tokens=[]
                for index in range(8):
                    token=max(range(len(values)),key=values.__getitem__);tokens.append(token)
                    native.request('decode '+str(token));path,values=native.frontier('next-output-'+str(index+1))
                    if token in profile['execution']['eos_tokens']:break
                assert tokens==result['branches'][branch]['next']['answer']['tokens']
                exact(path,result['branches'][branch]['next_frontier']['path'],branch+'-next-fresh')
                native.request('feed '+' '.join(map(str,result['branches'][branch]['cold_suffix'])))
                _,values=native.frontier('cold');following=[]
                for index in range(8):
                    token=max(range(len(values)),key=values.__getitem__);following.append(token)
                    native.request('decode '+str(token));path,values=native.frontier('cold-output-'+str(index+1))
                    if token in profile['execution']['eos_tokens']:break
                assert following==result['branches'][branch]['cold']['answer']['tokens']
                exact(path,result['branches'][branch]['cold_frontier']['path'],branch+'-cold-fresh')
                row.update(work=native.request('stats'),next_tokens=tokens,cold_tokens=following)
                assert row['work']['forward_tokens']==len(first_history)+len(later[branch])+len(tokens)+len(result['branches'][branch]['cold_suffix'])+len(following)
            finally:row['exit']=native.close()
            assert row['exit']['exit_code']==0 and row['exit']['process_absent']
        encodings=[*root['answer']['input_preparation']['encodings']]
        for branch in result['branches'].values():
            for stage in ('first','next'):encodings+=branch[stage]['answer']['input_preparation']['encodings']
        assert len(encodings)==3 and all(row['exit_code']==0 and row['process_absent'] and not Path(f"/proc/{row['pid']}").exists() for row in encodings)
        result.update(status='PASS',seconds=time.monotonic()-started,max_abs_error=0.,encoder_runs=encodings,
            limitation='Greedy Q4/GPU0, short answers. Four retained and four new fresh vectors; not every frontier. Pending client-journal acknowledgments are simulated. No 8K/sampler/pressure/Q8/split-GPU/original-gate qualification.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),seconds=result['seconds'],cli_calls=len(result['cli_calls']))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    campaign(parser.parse_args())
