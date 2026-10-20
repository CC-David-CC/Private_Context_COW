"""Q4 CLI recovery after numerical eviction, with independent fresh references.

Cached embeddings need no producer. Missing embeddings use the pinned encoder.
Reconstruction changes the native arrangement; it does not restore evicted state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
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
from tools.cow_cli import CowClient
from vision_fixture import reference_positions


def serve_worker():
    root=Path(sys.argv[2]).resolve();audit_path=Path(sys.argv[3]);reads=[]
    allowed=[root/'checkpoints',root/'image-staging']
    def audit(event,arguments):
        if event!='open' or not isinstance(arguments[0],(str,bytes)):return
        path=Path(os.fsdecode(arguments[0])).resolve()
        if path.suffix!='.sve':return
        write_only=(arguments[2]&os.O_ACCMODE)==os.O_WRONLY
        accepted=write_only or any(path.is_relative_to(folder) for folder in allowed)
        reads.append(dict(path=str(path),flags=arguments[2],write_only=write_only,accepted=accepted,thread=threading.get_ident()))
        if not accepted:raise AssertionError('Server attempted external embedding access: '+str(path))
    sys.addaudithook(audit);sys.argv=['serve.cow_http',*sys.argv[4:]]
    try:runpy.run_module('serve.cow_http',run_name='__main__')
    finally:save(audit_path,dict(status='PASS' if all(row['accepted'] for row in reads) else 'FAIL',opens=reads))


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();prior=json.loads((baseline/'result.json').read_text());assert prior['status']=='PASS'
    profile=json.loads(Path(prior['profile']['path']).read_text());build=json.loads(Path(prior['build']['path']).read_text())
    config=out/'vision-config.json';config.write_bytes(Path(prior['producer_config']['path']).read_bytes())
    parts=model_text_components(profile['model_path'],profile['execution'].get('text_template_kwargs'),
        model_contract_id=profile['model_contract_id'],mixed_inputs=True)
    first=next(span for span in prior['inputs']['prefix'] if span['kind']=='image')
    image=out/'diagram.png';image.write_bytes((baseline/'store/image-originals'/(first['original_sha256']+'.input')).read_bytes())
    result=dict(status='RUNNING',scope=__doc__,baseline=fingerprint(baseline/'result.json'),profile=prior['profile'],
        build=prior['build'],config=fingerprint(config),servers=[],cli_calls=[],evictions=[],cases={},comparisons=[],
        source_digest_checks=[],references=[],encoder_runs=[])
    root_path=out/'store';store=None;server=None;client=None;port=0;done=threading.Event();phase={'name':'startup'};begin=time.monotonic()
    environment={**os.environ,'COW_API_KEY':'bounded-reconstruction-canary','PYTHONDONTWRITEBYTECODE':'1'}
    def heartbeat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-begin,
            server_pid=server.pid if server else None)),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def cli(*arguments):
        argv=[sys.executable,'-B','-m','tools.cow_cli','--url',client.base_url,'--timeout','300','--state-dir',str(out/'client-state'),
            '--machine-result',*arguments]
        start=time.monotonic();run=subprocess.run(argv,env=environment,capture_output=True,timeout=300);index=len(result['cli_calls'])
        stdout=out/f'cli-{index}.stdout';stderr=out/f'cli-{index}.stderr';stdout.write_bytes(run.stdout);stderr.write_bytes(run.stderr)
        result['cli_calls'].append(dict(argv=argv,exit_code=run.returncode,seconds=time.monotonic()-start,stdout=fingerprint(stdout),stderr=fingerprint(stderr)))
        save(out/'result.json',result);assert run.returncode==0,run.stderr.decode(errors='replace')
        if run.stdout.strip():return json.loads(run.stdout)
        text=run.stderr.decode();marker='durable receipt:\n';assert text.count(marker)==1,text;return json.loads(text.split(marker,1)[1])
    def start(producer):
        nonlocal server,client,port,store
        index=len(result['servers']);stdout=out/f'server-{index}.stdout';stderr=out/f'server-{index}.stderr';audit=out/f'server-{index}-audit.json'
        argv=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--serve-worker',str(root_path),str(audit),
            '--profile',prior['profile']['path'],'--store',str(root_path),'--text','--port',str(port),
            '--api-key',environment['COW_API_KEY'],'--ram-gib','16','--disk-gib','32','--checkpoint-idle-seconds','1800']
        if producer:argv+=['--vision-config',str(config)]
        started=time.monotonic()
        with stdout.open('wb') as output,stderr.open('wb') as error:server=subprocess.Popen(argv,env=environment,stdout=output,stderr=error,start_new_session=True)
        row=dict(pid=server.pid,argv=argv,producer=producer,stdout=str(stdout),stderr=str(stderr),audit=str(audit));result['servers'].append(row)
        while True:
            if server.poll() is not None:raise RuntimeError('Server startup failed: '+stderr.read_text(errors='replace'))
            ready=[json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
            ready=[row for row in ready if row.get('event')=='ready']
            if ready:break
            if time.monotonic()-started>300:raise TimeoutError('Server readiness timeout')
            time.sleep(.1)
        port=ready[-1]['address'][1];client=CowClient('http://127.0.0.1:'+str(port),environment['COW_API_KEY'],300);store=Store(root_path)
        row.update(startup_seconds=time.monotonic()-started,ready_status=client.request('GET','/cow/status'))
        row['native_pids']=[value['pid'] for value in row['ready_status']['workers'].values()];assert len(row['native_pids'])==1
    def stop():
        nonlocal server
        if server is None:return
        row=result['servers'][-1]
        if 'ready_status' in row:row['last_status']=client.request('GET','/cow/status')
        server.send_signal(signal.SIGINT)
        try:code=server.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGTERM);server.wait(15);raise TimeoutError('Server shutdown timeout')
        row.update(exit_code=code,process_absent=not Path(f'/proc/{server.pid}').exists(),
            native_absent=all(not Path(f'/proc/{pid}').exists() for pid in row.get('native_pids',[])))
        server=None;assert code==0 and row['process_absent'] and row['native_absent'],row
        row['audit_result']=json.loads(Path(row['audit']).read_text());assert row['audit_result']['status']=='PASS'
        save(out/'result.json',result)
    def vector(snapshot,label):
        envelope=json.loads((root_path/'checkpoints'/(snapshot+'.json')).read_text());manifest=envelope['manifest']
        assert hashlib.sha256(_json_bytes(manifest)).hexdigest()==envelope['checksum']
        payload=root_path/'checkpoints'/(manifest['payload_sha256']+'.state');assert fingerprint(payload)['sha256']==manifest['payload_sha256']
        with payload.open('rb') as stream:stream.seek(manifest['state_bytes']);raw=stream.read()
        assert len(raw)==4*manifest['logits_count'];path=out/(label+'.f32');path.write_bytes(raw);return path
    def evict(keep_embedding):
        assert server is None and result['servers'][-1]['native_absent']
        removed=[]
        for path in sorted((root_path/'checkpoints').iterdir()):
            if path.is_file() and path.suffix in ('.json','.state','.sve'):
                if keep_embedding and path.suffix=='.sve':continue
                assert not path.is_symlink() and path.resolve().parent==(root_path/'checkpoints').resolve()
                removed.append(fingerprint(path));path.unlink()
        assert not list((root_path/'checkpoints').glob('*.state')) and not list((root_path/'checkpoints').glob('*.json'))
        result['evictions'].append(dict(keep_embedding=keep_embedding,removed=removed,
            retained_embeddings=[fingerprint(path) for path in (root_path/'checkpoints').glob('*.sve')]))
    def exact(reference,candidate,label):
        row=dict(label=label,reference=fingerprint(reference),candidate=fingerprint(candidate),**compare_files(reference,candidate))
        result['comparisons'].append(row);save(out/'result.json',result);assert row['exact'],row
    def run_case(name,branch,revision,command,question,encoder_calls):
        phase['name']='CLI '+name;source=store.head(branch).tip;history=history_for_snapshot(store,source);record=store.get_snapshot(source)
        operation='rebuild-'+name;answer=cli(command,'--branch',branch,'--revision',str(revision),'--text',question,
            '--max-tokens','8','--operation-id',operation)
        work=answer['answer']['source_recomputation'];assert work['prefix_forward_tokens']==len(history)
        assert work['prior_computation_preserved'] is False and work['durable'] is False
        assert work['input_preparation']['encoder_calls']==encoder_calls and work['reencode_callback_calls']==encoder_calls
        assert store.get_snapshot(source)==record
        envelope=json.loads((root_path/'pool-operations'/(operation+'.json')).read_text());generation=envelope['intent']['generation']
        assert hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest()==envelope['checksum'];suffix=tuple(generation['ids'])
        assert answer['answer']['counter_deltas']['forward_tokens']==len(suffix)+len(answer['answer']['tokens'])
        assert 'Notice: saved prefix is unavailable' in Path(result['cli_calls'][-1]['stderr']['path']).read_text()
        expected=controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),operation,
            (history+suffix).text_tokens,len(parts['tokenizer'].tokens))
        expected=controller.consume_many(expected,((token,parts['tokenizer'].token_bytes(token),token in profile['execution']['eos_tokens'])
            for token in answer['answer']['tokens']))
        assert store.get_snapshot(answer['receipt']['snapshot'])['controller']==expected
        result['cases'][name]=dict(result=answer,source=source,source_history=history.spans,suffix=list(suffix),
            source_controller_sha256=hashlib.sha256(_json_bytes(record['controller'])).hexdigest(),controller_exact=True,
            frontier=fingerprint(vector(answer['receipt']['snapshot'],name+'-final')))
        assert work['controller_sha256']==result['cases'][name]['source_controller_sha256']
        result['encoder_runs']+=work['input_preparation']['encodings'];save(out/'result.json',result)
    def packet(history,label):
        tokens=[];images=[]
        for span in history.spans:
            if span['kind']=='text':tokens.extend(span['tokens'])
            else:images.append((len(tokens),span));tokens.extend([span['pad_token_id']]*span['rows'])
        lines=[f'COW_VISION_INPUT_V1 0 {len(tokens)} {len(images)}',' '.join(map(str,tokens))]
        for position,span in images:
            fields=[position]+[span[key] for key in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
            lines.append(' '.join(map(str,fields))+' '+json.dumps(str(out/'oracle-only.sve')))
        manifest=out/(label+'.manifest');manifest.write_text('\n'.join(lines)+'\n')
        coords=reference_positions(len(tokens),[dict(start=pos,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']) for pos,span in images],profile['execution']['capacity'])
        positions=out/(label+'-positions.i32');positions.write_bytes(b''.join(struct.pack('<3i',*row) for row in coords));return manifest,positions
    try:
        start(True);namespace=cli('namespace','--title','Explicit vision reconstruction','--operation-id','rebuild-namespace')['namespace_id']
        phase['name']='tiny image context'
        root=cli('context','--branch','A','--namespace-id',namespace,'--image',str(image),'--text',
            'Reference memory. '*32+'Keep this diagram as shared context.','--operation-id','rebuild-root')
        result['root']=root;prefix=history_for_snapshot(store,root['receipt']['snapshot']);assert 128<=len(prefix)<=512
        result['prefix_cells']=len(prefix);result['prefix_text_tokens']=len(prefix.text_tokens)
        prefix_vector=vector(root['receipt']['snapshot'],'initial-prefix');result['prefix_frontier']=fingerprint(prefix_vector)
        embedding=next((root_path/'checkpoints').glob('*.sve'));(out/'oracle-only.sve').write_bytes(embedding.read_bytes())
        result['oracle_embedding']=fingerprint(out/'oracle-only.sve');result['encoder_runs']+=root['answer']['input_preparation']['encodings']
        cli('fork','--source-branch','A','--name','B','--operation-id','rebuild-fork-B')
        image.unlink();stop();evict(True)
        start(False);run_case('cached','A',1,'ask','What color is the left half of the diagram? Answer with one color only.',0)
        stop();evict(False)
        start(True);run_case('reencoded','B',0,'ask','What color is the right half of the diagram? Answer with one color only.',1)
        stop();evict(False)
        start(True);run_case('generated','A',2,'append-user','What color is the right half of the newest diagram? Answer with one color only.',1)
        stop()
        assert len({row['pid'] for row in result['servers']})==4
        assert len({row['native_pids'][0] for row in result['servers']})==4
        for name,case in result['cases'].items():
            phase['name']='fresh reference '+name;history=InputHistory(case['source_history']);manifest,positions=packet(history,name)
            argv=list(profile['execution']['sut_command']);argv[0]=build['binaries']['reference']['path']
            assert fingerprint(argv[0])['sha256']==build['binaries']['reference']['sha256']
            native=Native(argv,out/('reference-'+name));row=dict(case=name,pid=native.proc.pid,argv=argv);result['references'].append(row)
            try:
                assert native.ready['reference']=='fresh-no-checkpoints'
                native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(positions)))
                source_frontier,_=native.frontier('reconstructed')
                expected_digest=case['result']['answer']['source_recomputation']['frontier_sha256']
                check=dict(case=name,reported_sha256=expected_digest,reference=fingerprint(source_frontier),exact=expected_digest==fingerprint(source_frontier)['sha256'])
                result['source_digest_checks'].append(check);assert check['exact'],check
                if name in ('cached','reencoded'):exact(source_frontier,prefix_vector,name+'-original-prefix')
                else:result['accepted_arrangement_difference']=compare_files(source_frontier,result['cases']['cached']['frontier']['path'])
                native.request('feed '+' '.join(map(str,case['suffix'])));_,values=native.frontier('question');tokens=[]
                for index in range(8):
                    token=max(range(len(values)),key=values.__getitem__);tokens.append(token)
                    native.request('decode '+str(token));path,values=native.frontier('output-'+str(index+1))
                    if token in profile['execution']['eos_tokens']:break
                assert tokens==case['result']['answer']['tokens'];exact(path,case['frontier']['path'],name+'-final')
                row.update(tokens=tokens,work=native.request('stats'))
                assert row['work']['forward_tokens']==len(history)+len(case['suffix'])+len(tokens)
            finally:row['exit']=native.close()
            assert row['exit']['exit_code']==0 and row['exit']['process_absent']
        assert len(result['encoder_runs'])==3
        assert all(row['exit_code']==0 and row['process_absent'] and not Path(f"/proc/{row['pid']}").exists() for row in result['encoder_runs'])
        result.update(status='PASS',seconds=time.monotonic()-begin,max_abs_error=0.,
            limitation='Tiny greedy Q4/GPU0; five numerical vector comparisons and three reconstructed-frontier digest checks, not every-token or original-gate qualification. Old generated arrangement comparison is informational. Process exit/explicit eviction, not crash or power loss.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],prefix_cells=result['prefix_cells'],comparisons=len(result['comparisons']),
        digest_checks=len(result['source_digest_checks']),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--serve-worker':serve_worker()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
        campaign(parser.parse_args())
