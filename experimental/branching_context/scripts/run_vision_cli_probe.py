"""Q4 CLI -> authenticated HTTP -> GPU, then a new restore-only server process.

Uses retained independent native references for identical inputs/boundaries.
This bounded greedy experiment does not qualify samplers, image suffixes or speed.
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

from run_vision_ingest_probe import compare_files, fingerprint, save
from context_time_travel.store import _json_bytes
from tools.cow_cli import CowClient


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();prior=json.loads((baseline/'result.json').read_text());assert prior['status']=='PASS'
    profile=json.loads(Path(prior['profile']['path']).read_text())
    encoder_build=json.loads(Path(prior['encoder_build']['path']).read_text())
    metadata=json.loads((args.artifact_base/'image-baseline-v2/result.json').read_text())
    executable=encoder_build['binary']['path'];projector=metadata['encodings'][0]['argv'][2];model=profile['model_path']
    config=dict(schema=1,executable=executable,projector=projector,embedding_width=2560,threads=32,
        max_rows=1024,original_bytes_limit=256*1024**2,
        pinned_sha256={executable:encoder_build['binary']['sha256'],projector:metadata['files'][projector]['sha256'],
                       model:metadata['files'][model]['sha256']})
    config_path=out/'vision-config.json';save(config_path,config)
    result=dict(status='RUNNING',scope=__doc__,baseline=fingerprint(baseline/'result.json'),
        profile=fingerprint(prior['profile']['path']),encoder_build=prior['encoder_build'],
        config=fingerprint(config_path),servers=[],cli_calls=[],branches={},comparisons=[],removed_embeddings=[])
    store=out/'store';server=None;client=None;phase={'name':'startup'};done=threading.Event();started=time.monotonic()
    key='bounded-vision-cli-canary';environment={**os.environ,'COW_API_KEY':key,'PYTHONDONTWRITEBYTECODE':'1'}
    def heartbeat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            server_pid=server.pid if server is not None else None)),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def cli(*arguments):
        command=[sys.executable,'-B','-m','tools.cow_cli','--url',client.base_url,'--timeout','300',
                 '--state-dir',str(out/'client-state'),'--machine-result',*arguments]
        index=len(result['cli_calls']);begin=time.monotonic()
        completed=subprocess.run(command,env=environment,capture_output=True,timeout=300)
        stdout=out/f'cli-{index:02d}.stdout';stderr=out/f'cli-{index:02d}.stderr'
        stdout.write_bytes(completed.stdout);stderr.write_bytes(completed.stderr)
        result['cli_calls'].append(dict(argv=command,exit_code=completed.returncode,seconds=time.monotonic()-begin,
            stdout=fingerprint(stdout),stderr=fingerprint(stderr)))
        save(out/'result.json',result)
        assert completed.returncode==0,completed.stderr.decode(errors='replace')
        if completed.stdout.strip():return json.loads(completed.stdout)
        # Existing non-streaming CLI mutations print their receipt on stderr;
        # --machine-result promises stdout JSON only for streamed mutations.
        marker='durable receipt:\n'
        text=completed.stderr.decode('utf-8')
        assert text.count(marker)==1,text
        return json.loads(text.split(marker,1)[1])
    def start(with_encoder):
        nonlocal server,client
        index=len(result['servers']);stdout=out/f'server-{index}.stdout';stderr=out/f'server-{index}.stderr'
        command=[sys.executable,'-B','-u','-m','serve.cow_http','--profile',prior['profile']['path'],
            '--store',str(store),'--text','--port','0','--api-key',key,'--ram-gib','16','--disk-gib','32',
            '--checkpoint-idle-seconds','1800']
        if with_encoder:command+=['--vision-config',str(config_path)]
        begin=time.monotonic()
        with stdout.open('wb') as output,stderr.open('wb') as error:
            server=subprocess.Popen(command,env=environment,stdout=output,stderr=error,start_new_session=True)
        row=dict(pid=server.pid,argv=command,producer_configured=with_encoder,stdout=str(stdout),stderr=str(stderr))
        result['servers'].append(row)
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited before readiness: '+stderr.read_text(errors='replace'))
            ready=[json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
            ready=[value for value in ready if value.get('event')=='ready']
            if ready:break
            if time.monotonic()-begin>300:raise TimeoutError('Server readiness timed out')
            time.sleep(.1)
        client=CowClient('http://127.0.0.1:'+str(ready[-1]['address'][1]),key,300)
        status=client.request('GET','/cow/status');row['ready_status']=status;row['startup_seconds']=time.monotonic()-begin
        row['native_pids']=[value['pid'] for value in status['workers'].values()]
        assert len(row['native_pids'])==1 and row['native_pids'][0] is not None
        save(out/'result.json',result)
    def stop():
        nonlocal server
        if server is None:return
        row=result['servers'][-1]
        if 'ready_status' in row:row['last_status']=client.request('GET','/cow/status')
        server.send_signal(signal.SIGINT)
        try:code=server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=15)
            raise TimeoutError('Server failed to shut down gracefully')
        row.update(exit_code=code,process_absent=not Path(f'/proc/{server.pid}').exists(),
            native_processes_absent=all(not Path(f'/proc/{pid}').exists() for pid in row.get('native_pids',[])))
        server=None
        assert code==0 and row['process_absent'] and row['native_processes_absent'],row
        save(out/'result.json',result)
    def compare(snapshot,reference,label):
        envelope=json.loads((store/'checkpoints'/(snapshot+'.json')).read_text());manifest=envelope['manifest']
        assert hashlib.sha256(_json_bytes(manifest)).hexdigest()==envelope['checksum']
        payload=store/'checkpoints'/(manifest['payload_sha256']+'.state')
        assert fingerprint(payload)['sha256']==manifest['payload_sha256']
        with payload.open('rb') as stream:
            stream.seek(manifest['state_bytes']);raw=stream.read()
        assert len(raw)==4*manifest['logits_count']
        target=out/(label+'.f32');target.write_bytes(raw)
        comparison=dict(label=label,reference=fingerprint(reference),candidate=fingerprint(target),**compare_files(reference,target))
        result['comparisons'].append(comparison);assert comparison['exact'],comparison
    def answer(branch,side,operation,reference_branch,revision):
        value=cli('ask','--branch',branch,'--revision',str(revision),'--text',
            'What color is the '+side+' half of the first diagram? Answer briefly.',
            '--max-tokens','8','--operation-id',operation)
        expected=prior['branches'][reference_branch]
        assert value['answer']['tokens']==expected['answer']['tokens']
        assert value['answer']['text']==('Red' if reference_branch=='A' else 'Blue')
        assert value['answer']['counter_deltas']['forward_tokens']==28
        reference=next(row['expected'] for row in prior['comparisons'] if row['label']==reference_branch+'-final')
        compare(value['receipt']['snapshot'],reference,branch+'-final')
        expected_controller=json.loads((baseline/'store/objects'/(expected['durable']['snapshot']+'.json')).read_text())['controller']
        expected_controller['stream_key']=hashlib.sha256(operation.encode()).hexdigest()
        actual=json.loads((store/'objects'/(value['receipt']['snapshot']+'.json')).read_text())['controller']
        assert actual==expected_controller
        result['branches'][branch]=dict(result=value,controller_exact=True)
        save(out/'result.json',result)
    try:
        start(True);namespace=cli('namespace','--title','Vision CLI canary','--operation-id','cli-namespace')['namespace_id']
        parts=[]
        for index,upload in enumerate(prior['uploads']):
            raw=(baseline/'store/image-originals'/(upload['image_id']+'.input')).read_bytes()
            path=out/f'diagram-{index}.png';path.write_bytes(raw)
            parts.extend([dict(type='text',text=' First diagram. ' if index==0 else ' Second diagram. '),
                          dict(type='image',file=path.name)])
        parts.append(dict(type='text',text='Keep these diagrams as shared context.'))
        parts_path=out/'parts.json';save(parts_path,parts);phase['name']='CLI image context'
        root=cli('context','--branch','A','--namespace-id',namespace,'--parts-file',str(parts_path),'--operation-id','cli-prepare-A')
        result['root']=root;assert root['answer']['counter_deltas']['forward_tokens']==1097
        compare(root['receipt']['snapshot'],prior['comparisons'][0]['expected'],'prefix')
        root_manifest=json.loads((store/'checkpoints'/(root['receipt']['snapshot']+'.json')).read_text())['manifest']
        parent_payload=store/'checkpoints'/(root_manifest['payload_sha256']+'.state');parent_before=fingerprint(parent_payload)
        cli('fork','--source-branch','A','--name','B','--snapshot',root['receipt']['snapshot'],'--operation-id','cli-fork-B')
        phase['name']='CLI questions A/B';answer('A','left','cli-question-A','A',1);answer('B','right','cli-question-B','B',0)
        result['graph_before_restart']=client.request('GET','/cow/workspace/graph?chat='+namespace)
        assert result['graph_before_restart']['shared_prefix_messages']==1
        assert fingerprint(parent_payload)==parent_before
        embeddings=list((store/'checkpoints').glob('*.sve'));assert len(embeddings)==2
        stop()
        for path in embeddings:
            assert path.resolve().parent==(store/'checkpoints').resolve() and not path.is_symlink()
            result['removed_embeddings'].append(fingerprint(path));path.unlink()
        phase['name']='new restore-only server';start(False)
        replay=cli('retry','--from-operation','cli-question-A')
        assert replay['answer']['text']=='Red';result['replay']=replay
        cli('fork','--source-branch','A','--name','C','--snapshot',root['receipt']['snapshot'],'--operation-id','cli-fork-C')
        phase['name']='cold CLI question C';answer('C','left','cli-question-C','A',0)
        result['cold_work']=result['branches']['C']['result']['answer']['counter_deltas']
        assert result['cold_work']['loaded_file_bytes']>0
        assert not list((store/'checkpoints').glob('*.sve'))
        assert fingerprint(parent_payload)==parent_before
        result['parent_payload_unchanged']=parent_before
        stop();assert result['servers'][0]['pid']!=result['servers'][1]['pid']
        assert result['servers'][0]['native_pids']!=result['servers'][1]['native_pids']
        encodings=root['answer']['input_preparation']['encodings']
        assert len(encodings)==2 and all(row['exit_code']==0 and row['process_absent'] for row in encodings)
        assert all(not Path(f"/proc/{row['pid']}").exists() for row in encodings)
        result.update(status='PASS',seconds=time.monotonic()-started,max_abs_error=0.,
            limitation='Four root/final vectors against retained uncached references; greedy, two-token answers, no new image suffixes or 8K/performance/controller qualification.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),seconds=result['seconds'],
        answers={name:value['result']['answer']['text'] for name,value in result['branches'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-base',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);campaign(parser.parse_args())
