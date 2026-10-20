"""Bounded canonical mixed API versus fresh raw native references on a real GPU.

Imports the already measured SVE fixture, with provenance recorded. No new
encoder execution, cold checkpoint restore or client capability is claimed.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import threading
import time

from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import DiskCheckpointPool, SharedCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError
from context_time_travel.strata_backend import StrataBackend, _digest
from context_time_travel.vision_artifacts import ImageArtifacts
from context_time_travel.vision_execution import RESUME_CONTRACT, VISION_PROTOCOL, POSITION_CONVENTION
from run_vision_ingest_probe import Native, compare_files, fingerprint, save


def campaign(args):
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    base=json.loads(args.profile.read_text()); build=json.loads(args.build.read_text())
    fixture=json.loads(args.fixture.read_text())
    assert fixture['status']=='PASS' and fixture['fixture']=='natural'
    manifest=args.fixture.parent/'input.manifest'
    position_file=args.fixture.parent/'reference-positions.i32'
    fields=shlex.split(manifest.read_text().splitlines()[2])
    start,rows,nx,ny,width,pad=map(int,fields[:6])
    payload=Path(fields[7]); assert fingerprint(payload)==fixture['embedding']
    ids=fixture['input_tokens']; suffixes=fixture['suffixes']
    profile=copy.deepcopy(base)
    profile['native_build']={'sha256':{item['path']:item['sha256'] for item in build['binaries'].values()},
                             'build_receipt':fingerprint(args.build)}
    execution=profile['execution']
    execution['sut_command'][0]=build['binaries']['cow']['path']
    execution['sut_command']+=['--vision','--gpu','0']
    execution.update(resume_contract=RESUME_CONTRACT,vision_protocol=VISION_PROTOCOL,
        embedding_width=width,GPU=[0],checkpoint_cache=dict(byte_limit=16*1024**3,entry_limit=256))
    execution['environment']['CUDA_VISIBLE_DEVICES']='0'
    profile['model_contract'].update(input_kind='canonical-mixed-spans-v1',positions=POSITION_CONVENTION)
    profile['model_contract_id']=_digest(profile['model_contract'])
    profile['execution_profile_id']=_digest(dict(build=profile['native_build'],execution=execution,
                                                hardware=profile['hardware']))
    save(out/'profile.json',profile)
    result=dict(status='RUNNING',scope='canonical mixed API and hot branching, greedy only',
        fixture=fingerprint(args.fixture),build=fingerprint(args.build),profile=fingerprint(out/'profile.json'),
        source_parent='ad6653c57f88a21e08ee6ec84a94d01ce6181c7b',comparisons=[],runs=[],checks=[],
        disk_restart='UNIMPLEMENTED',cli='UNIMPLEMENTED',encoder_calls=0)
    done=threading.Event(); phase={'name':'startup'}
    def heartbeat():
        while not done.wait(30):
            print(json.dumps(dict(phase=phase['name'],python_pid=os.getpid(),time_ns=time.time_ns())),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def exact(a,b,label):
        value=dict(label=label,**compare_files(a,b)); result['comparisons'].append(value)
        save(out/'result.json',result); assert value['exact'],value
    def frontier(values,label):
        path=out/(label+'.f32'); path.write_bytes(struct.pack('<'+'f'*len(values),*values)); return path
    references={}; saved={}; backend=None; ram=None
    try:
        for branch in ('A','B'):
            for repeat in range(2):
                phase['name']=f'reference-{branch}-{repeat}'
                argv=list(execution['sut_command']); argv[0]=build['binaries']['reference']['path']
                native=Native(argv,out/f'reference-{branch}-{repeat}')
                row=dict(name=phase['name'],argv=argv,pid=native.proc.pid,startup_seconds=native.startup_seconds)
                result['runs'].append(row)
                try:
                    native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(position_file)))
                    paths=[native.frontier('prefix')[0]]
                    native.request('feed '+' '.join(map(str,suffixes[branch])))
                    path,values=native.frontier('suffix'); paths.append(path); generated=[]
                    for index in range(8):
                        token=max(range(len(values)),key=values.__getitem__); generated.append(token)
                        native.request('decode '+str(token))
                        path,values=native.frontier('output-'+str(index+1)); paths.append(path)
                        if token in execution['eos_tokens']:break
                    if repeat:
                        assert generated==references[branch]['tokens']
                        for index,(a,b) in enumerate(zip(references[branch]['paths'],paths)):
                            exact(a,b,f'{branch}-fresh-repeat-{index}')
                    else: references[branch]=dict(paths=paths,tokens=generated)
                    row['work']=native.request('stats')
                finally:row['exit']=native.close()
                save(out/'result.json',result)
        phase['name']='canonical-api'
        os.environ['CTT_NATIVE_TRACE']=str(out/'api-native-trace.jsonl')
        started=time.monotonic(); backend=StrataBackend(profile,gpu_id=0)
        result['runs'].append(dict(name='canonical-api',pid=backend.process.pid,startup_seconds=time.monotonic()-started))
        api=BranchingModel(out/'store',backend,profile)
        ram=SharedCheckpointPool(16*1024**2,checkpoint_idle_seconds=1800)
        disk=DiskCheckpointPool(api.store,32*1024**2,checkpoint_idle_seconds=1800)
        producer=dict(kind='retained-executed-SVE-fixture',baseline=fingerprint(args.image.parent/'result.json'),
                      fixture=fingerprint(args.fixture),image=fingerprint(args.image),embedding=fingerprint(payload))
        save(out/'imported-encoder-fixture.json',producer)
        artifacts=ImageArtifacts(api.store,ram,disk,encoder_profile_id=_digest(producer),
                                  expected_width=width,original_bytes_limit=16*1024**2)
        with artifacts.prepare(original=args.image.read_bytes(),normalized=args.image.read_bytes(),
                               encode=lambda _:payload.read_bytes(),pad_token_id=pad) as image:
            spans=[dict(kind='text',tokens=ids[:start]),image.span,
                   dict(kind='text',tokens=ids[start+rows:])]
            prefix=api.create_inputs(spans,artifacts=artifacts)
        assert api.history(prefix).native_tokens==tuple(ids)
        parent=api.cache[prefix]
        before=backend.work_counters()['forward_tokens']
        assert api.create_inputs(spans,artifacts=artifacts)==prefix
        assert backend.work_counters()['forward_tokens']==before
        result['checks'].append(dict(name='exact-root-reuse',forward_delta=0))
        result['parent_before']=backend.audit(parent.blob)
        parent_path=frontier(api.next_logits(prefix),'api-prefix')
        for branch in ('A','B'):
            phase['name']='canonical-'+branch
            ref=api.store.fork(branch,prefix)
            before=backend.work_counters()['forward_tokens']
            with api.materialize(prefix):pass
            assert backend.work_counters()['forward_tokens']==before
            tip=api.feed(prefix,suffixes[branch]); paths=[parent_path,frontier(api.next_logits(tip),f'api-{branch}-suffix')]
            active=api.start_generation(tip,dict(temperature=0,eos_tokens=execution['eos_tokens']),'same-greedy-stream')
            generated=[]; boundaries=[]
            for index in range(8):
                batch=api.generate(active,1)
                if not batch.accepted_tokens:break
                active=batch.snapshot_id; generated.extend(batch.accepted_tokens); boundaries.append(active)
                paths.append(frontier(api.next_logits(active),f'api-{branch}-output-{index+1}'))
                if generated[-1] in execution['eos_tokens']:break
            assert generated==references[branch]['tokens']
            assert len(paths)==len(references[branch]['paths'])
            for index,(a,b) in enumerate(zip(references[branch]['paths'],paths)):
                exact(a,b,f'{branch}-canonical-{index}')
            saved[branch]=dict(snapshot=active,boundaries=boundaries,tokens=generated,paths=paths)
            assert api.history(active)==InputHistory(spans)+tuple(suffixes[branch])+tuple(generated)
            result['checks'].append(dict(name='branch-'+branch,snapshot=active,initial_branch=ref.name,
                tokens=generated,controller=api.controller(active),
                schedule=api.cache[active].execution_schedule))
            assert backend.audit(parent.blob)==result['parent_before']
            save(out/'result.json',result)
        phase['name']='binding-rejection-and-retry'
        wrong=copy.deepcopy(spans); wrong[1]['embedding_sha256']='0'*64
        with backend.lease() as lease:
            before=backend.work_counters()
            try:backend.restore(lease,parent.blob,history=InputHistory(wrong),frontier=parent.frontier)
            except IntegrityError:pass
            else:raise AssertionError('wrong image reached restore')
            after=backend.work_counters(); assert before==after
        altered=copy.deepcopy(parent.execution_schedule); altered[0]['position_sha256']='0'*64
        api.cache[prefix]=replace(parent,execution_schedule=altered)
        before=backend.work_counters()['restores'] if 'restores' in backend.stats else backend.work_counters()['forward_tokens']
        try:
            with api.materialize(prefix):pass
        except IntegrityError:pass
        else:raise AssertionError('altered canonical receipt was accepted')
        api.cache[prefix]=parent
        result['checks'].append(dict(name='wrong-image-and-position-receipt-rejected'))
        if len(saved['A']['tokens'])>=4:
            four=saved['A']['boundaries'][3]
            replay=api.generate(four,len(saved['A']['tokens'])-4)
            assert list(replay.accepted_tokens)==saved['A']['tokens'][4:]
            assert api.controller(replay.snapshot_id)==api.controller(saved['A']['snapshot'])
            exact(saved['A']['paths'][-1],frontier(api.next_logits(replay.snapshot_id),'api-retry-final'),'A-retry-controller-and-frontier')
        phase['name']='native-request-api'
        request=api.run_greedy_request(prefix,suffixes['B'],len(saved['B']['tokens']),stream='same-greedy-stream')
        assert request['tokens']==saved['B']['tokens']
        assert api.controller(request['snapshot'])==api.controller(saved['B']['snapshot'])
        exact(saved['B']['paths'][-1],frontier(api.next_logits(request['snapshot']),'api-resident-request'),'B-resident-request')
        result['work']=backend.work_counters(); result['parent_after']=backend.audit(parent.blob)
        assert result['parent_before']==result['parent_after']
        result['snapshots']=dict(prefix=prefix,A=saved['A']['snapshot'],B=saved['B']['snapshot'])
        result['status']='PASS'
    except BaseException as error:
        result['status']='FAIL'; result['error']=repr(error); raise
    finally:
        if backend is not None:
            pid=backend.process.pid; backend.close()
            result['runs'][-1]['exit']=dict(pid=pid,exit_code=backend.process.returncode,
                process_absent=not Path(f'/proc/{pid}').exists())
        if ram is not None:ram.close()
        done.set(); save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('profile','build','fixture','image','output'):parser.add_argument('--'+key,type=Path,required=True)
    campaign(parser.parse_args())
