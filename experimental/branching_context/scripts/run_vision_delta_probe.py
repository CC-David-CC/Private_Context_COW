"""Real Q4 incremental image inputs versus fresh, checkpoint-free native runs.

Core API only. Every generated frontier is checked for this bounded fixture;
client/coordinator image suffix admission remains a subsequent integration task.
"""
import argparse
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import struct
import threading
import time

from PIL import Image, ImageOps
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import SharedCheckpointPool, DiskCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store
from context_time_travel.strata_backend import StrataBackend
from serve.cow_http import model_text_components, vision_preparer_from_config
from run_vision_ingest_probe import Native, compare_files, fingerprint, save
from vision_fixture import reference_positions


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=args.artifact_base.resolve();baseline=args.baseline.resolve()
    prior=json.loads((baseline/'result.json').read_text());assert prior['status']=='PASS'
    profile=json.loads(Path(prior['profile']['path']).read_text())
    build_path=base/'vc05-native-build-v1/build.json';build=json.loads(build_path.read_text())
    parts=model_text_components(profile['model_path'],profile['execution'].get('text_template_kwargs'),
        model_contract_id=profile['model_contract_id'],mixed_inputs=True)
    renderer=parts['mixed_renderer'];store=Store(out/'store')
    preparer=vision_preparer_from_config(store,renderer,profile,args.producer_config)
    ram=SharedCheckpointPool(16*1024**3,checkpoint_idle_seconds=1800)
    disk=DiskCheckpointPool(store,32*1024**3,checkpoint_idle_seconds=1800)
    artifacts=preparer.artifacts(ram,disk);pad,=parts['tokenizer'].encode('<|image_pad|>',parse_special=True)
    result=dict(status='RUNNING',scope=__doc__,profile=fingerprint(prior['profile']['path']),
        baseline=fingerprint(baseline/'result.json'),build=fingerprint(build_path),
        producer_config=fingerprint(args.producer_config),encoder_runs=[],references=[],workers=[],
        branches={},comparisons=[],removed_embeddings=[])
    backend=None;done=threading.Event();phase={'name':'prepare artifacts'};started=time.monotonic()
    def beat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            python_pid=os.getpid(),native_pid=backend.process.pid if backend is not None else None)),flush=True)
    threading.Thread(target=beat,daemon=True).start()
    payloads={}
    def image(raw,label):
        normalized=preparer.normalizer.normalize(raw)
        def encode(value):
            encoded=preparer.encoder.encode(value,store=store,pool=ram,disk=disk,gpu_id=0)
            result['encoder_runs'].append(encoded.receipt);return encoded.embedding.payload
        with artifacts.prepare(original=raw,normalized=normalized.payload,encode=encode,pad_token_id=pad) as held:
            path=out/(label+'.sve');path.write_bytes(held.embedding.payload)
            payloads[held.span['embedding_sha256']]=path
            return held.span
    def raw_positions(history,label):
        images=[];cursor=0
        for span in history.spans:
            if span['kind']=='text':cursor+=len(span['tokens'])
            else:
                images.append(dict(start=cursor,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']))
                cursor+=span['rows']
        path=out/(label+'-positions.i32')
        path.write_bytes(b''.join(struct.pack('<3i',*row) for row in reference_positions(cursor,images,profile['execution']['capacity'])))
        return path
    def manifest(delta,prior_count,label):
        # Independent transport projection; reference coordinates never call
        # the production layout/position binding functions.
        tokens=[];images=[]
        for span in delta.spans:
            if span['kind']=='text':tokens.extend(span['tokens'])
            else:
                images.append((prior_count+len(tokens),span));tokens.extend([span['pad_token_id']]*span['rows'])
        lines=[f'COW_VISION_INPUT_V1 {prior_count} {len(tokens)} {len(images)}',' '.join(map(str,tokens))]
        for position,span in images:
            fields=[position]+[span[key] for key in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
            lines.append(' '.join(map(str,fields))+' '+json.dumps(str(payloads[span['embedding_sha256']])))
        path=out/(label+'.manifest');path.write_text('\n'.join(lines)+'\n');return path
    def vector(api,snapshot,label):
        path=out/(label+'.f32');values=api.next_logits(snapshot)
        path.write_bytes(struct.pack('<'+'f'*len(values),*values));return path
    def exact(reference,candidate,label):
        row=dict(label=label,reference=fingerprint(reference),candidate=fingerprint(candidate),**compare_files(reference,candidate))
        result['comparisons'].append(row);save(out/'result.json',result);assert row['exact'],row
    def start():
        nonlocal backend
        backend=StrataBackend(profile,gpu_id=0);result['workers'].append(dict(pid=backend.process.pid))
        return BranchingModel(out/'store',backend,profile)
    def stop():
        nonlocal backend
        if backend is None:return
        row=result['workers'][-1];row['work']=backend.work_counters();row['vision']=backend._request('vision_info')
        backend.close();row.update(exit_code=backend.process.returncode,process_absent=not Path(f"/proc/{row['pid']}").exists())
        backend=None;assert row['exit_code']==0 and row['process_absent']
    references={};cold_expected={}
    try:
        originals=[(baseline/'store/image-originals'/(item['image_id']+'.input')).read_bytes() for item in prior['uploads']]
        root_images=[image(raw,'root-'+str(index)) for index,raw in enumerate(originals)]
        with Image.open(io.BytesIO(originals[0])) as source:
            output=io.BytesIO();ImageOps.mirror(source).save(output,format='PNG');mirrored=output.getvalue()
        branch_images={'A':root_images[0],'B':image(mirrored,'mirrored')}
        root_parts=[]
        for index,span in enumerate(root_images):
            root_parts.extend([dict(type='text',text=' First diagram. ' if index==0 else ' Second diagram. '),
                               dict(type='image',image_id=span['original_sha256'])])
        root_parts.append(dict(type='text',text='Keep these diagrams as shared context.'))
        prefix=renderer.encode([dict(role='user',content=root_parts)],root_images,add_generation_prompt=False)
        assert len(prefix)==1097
        deltas={}
        for branch,span in branch_images.items():
            content=[dict(type='text',text='New diagram. '),dict(type='image',image_id=span['original_sha256']),
                dict(type='text',text=' What color is the left half of this new diagram? Answer with one color only.')]
            deltas[branch]=renderer.encode([dict(role='user',content=content)],[span],add_generation_prompt=True)
        assert deltas['A'].native_tokens==deltas['B'].native_tokens and deltas['A'].digest!=deltas['B'].digest
        first_manifest=manifest(prefix,0,'prefix');first_positions=raw_positions(prefix,'prefix')
        followup=tuple(renderer.encoder._user_frame('What color is the right half of the newest diagram? Answer with one color only.'))
        result['inputs']=dict(prefix=prefix.spans,deltas={name:value.spans for name,value in deltas.items()},
            followup=list(followup),prefix_cells=len(prefix),delta_cells={name:len(value) for name,value in deltas.items()},
            identical_delta_pad_projections=True,different_delta_identities=True)
        for branch,delta in deltas.items():
            second_manifest=manifest(delta,len(prefix),'delta-'+branch)
            second_positions=raw_positions(prefix+delta,'delta-'+branch)
            for repeat in range(2):
                phase['name']=f'fresh-{branch}-{repeat}'
                argv=list(profile['execution']['sut_command']);argv[0]=build['binaries']['reference']['path']
                assert fingerprint(argv[0])['sha256']==build['binaries']['reference']['sha256']
                native=Native(argv,out/f'reference-{branch}-{repeat}')
                row=dict(branch=branch,repeat=repeat,pid=native.proc.pid,argv=argv);result['references'].append(row)
                try:
                    native.request('feed_image '+json.dumps(str(first_manifest))+' '+json.dumps(str(first_positions)))
                    paths=[native.frontier('prefix')[0]]
                    native.request('feed_image '+json.dumps(str(second_manifest))+' '+json.dumps(str(second_positions)))
                    path,values=native.frontier('delta');paths.append(path);tokens=[]
                    for index in range(args.outputs):
                        token=max(range(len(values)),key=values.__getitem__);tokens.append(token)
                        native.request('decode '+str(token));path,values=native.frontier('output-'+str(index+1));paths.append(path)
                        if token in profile['execution']['eos_tokens']:break
                    native.request('feed '+' '.join(map(str,followup)));path,values=native.frontier('followup');paths.append(path)
                    following=[]
                    for index in range(args.outputs):
                        token=max(range(len(values)),key=values.__getitem__);following.append(token)
                        native.request('decode '+str(token));path,values=native.frontier('followup-output-'+str(index+1));paths.append(path)
                        if token in profile['execution']['eos_tokens']:break
                    row.update(tokens=tokens,following=following,paths=list(map(str,paths)),work=native.request('stats'))
                    assert row['work']['forward_tokens']==len(prefix)+len(delta)+len(tokens)+len(followup)+len(following)
                    if repeat:
                        assert tokens==references[branch]['tokens'] and following==references[branch]['following']
                        for index,(expected,actual) in enumerate(zip(references[branch]['paths'],paths)):
                            exact(expected,actual,f'{branch}-fresh-repeat-{index}')
                        assert len(paths)==len(references[branch]['paths'])
                    else:references[branch]=row
                finally:row['exit']=native.close()
                assert row['exit']['exit_code']==0 and row['exit']['process_absent']
        phase['name']='canonical image deltas';api=start();root=api.create_inputs(prefix.spans,artifacts=artifacts)
        parent=backend.audit(api.cache[root].blob);result['parent_before']=parent
        root_path=vector(api,root,'api-prefix');initial={}
        for branch,delta in deltas.items():
            reference=references[branch];before=backend.work_counters();tip=api.feed_inputs(root,delta.spans,artifacts=artifacts)
            after=backend.work_counters();work={key:after[key]-value for key,value in before.items()
                if type(value) is int and type(after.get(key)) is int}
            assert work['forward_tokens']==len(delta)
            paths=[root_path,vector(api,tip,branch+'-delta')]
            active=api.start_generation(tip,dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),'delta-'+branch)
            tokens=[]
            for index in range(len(reference['tokens'])):
                generated=api.generate(active,1);active=generated.snapshot_id;tokens.extend(generated.accepted_tokens)
                paths.append(vector(api,active,branch+'-output-'+str(index+1)))
            assert tokens==reference['tokens'];initial[branch]=active
            # Also exercise the single owned native request with the same new
            # image delta, independent of the per-token audit loop above.
            before=backend.work_counters()['forward_tokens']
            request=api.run_greedy_request(root,delta,args.outputs,stream='delta-'+branch,artifacts=artifacts)
            assert request['tokens']==tokens and backend.work_counters()['forward_tokens']-before==len(delta)+len(tokens)
            assert api.controller(request['snapshot'])==api.controller(active)
            exact(paths[-1],vector(api,request['snapshot'],branch+'-whole-request'),branch+'-whole-request')
            tip=api.feed(active,followup)
            paths.append(vector(api,tip,branch+'-followup'))
            active=api.start_generation(tip,dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),'followup-'+branch)
            following=[]
            for index in range(len(reference['following'])):
                generated=api.generate(active,1);active=generated.snapshot_id;following.extend(generated.accepted_tokens)
                paths.append(vector(api,active,branch+'-followup-output-'+str(index+1)))
            assert following==reference['following'] and len(paths)==len(reference['paths'])
            for index,(expected,actual) in enumerate(zip(reference['paths'],paths)):exact(expected,actual,f'{branch}-API-{index}')
            assert backend.audit(api.cache[root].blob)==parent
            result['branches'][branch]=dict(initial_snapshot=initial[branch],tokens=tokens,following=following,
                paths=list(map(str,paths)),suffix_work=work,whole_request=request,controller_exact=True)
            cold_expected[branch]=dict(snapshot=active,controller=api.controller(active),path=paths[-1])
            api.checkpoint(initial[branch]);save(out/'result.json',result)
        result['parent_after']=backend.audit(api.cache[root].blob);assert result['parent_after']==parent
        # Generated colors may differ; only the supplied P||delta projections
        # are deliberately identical. Never compare generated history as input.
        input_end=len(prefix)+len(deltas['A'])
        assert api.history(initial['A']).prefix(input_end).native_tokens==api.history(initial['B']).prefix(input_end).native_tokens
        assert api.history(initial['A'])!=api.history(initial['B'])
        stop()
        for path in list(disk.directory.glob('*.sve')):
            assert path.parent.resolve()==disk.directory.resolve() and not path.is_symlink()
            result['removed_embeddings'].append(fingerprint(path));path.unlink()
        assert len(result['removed_embeddings'])==3
        encodes=len(result['encoder_runs']);phase['name']='cold delta checkpoint';api=start();cold=[]
        for branch in ('A','B'):
            before=backend.work_counters()['forward_tokens'];receipt=api.restore_checkpoint(initial[branch])
            path=vector(api,initial[branch],'cold-'+branch+'-initial')
            assert backend.work_counters()['forward_tokens']==before and receipt['forward_tokens']==0
            exact(references[branch]['paths'][1+len(references[branch]['tokens'])],path,'cold-'+branch+'-restore')
            request=api.run_greedy_request(initial[branch],followup,args.outputs,stream='followup-'+branch)
            assert request['tokens']==references[branch]['following']
            assert backend.work_counters()['forward_tokens']-before==len(followup)+len(request['tokens'])
            assert api.controller(request['snapshot'])==cold_expected[branch]['controller']
            exact(cold_expected[branch]['path'],vector(api,request['snapshot'],'cold-'+branch+'-final'),'cold-'+branch+'-final')
            cold.append(dict(branch=branch,receipt=receipt,continuation=request,controller_exact=True))
        assert len(result['encoder_runs'])==encodes and backend._request('vision_info')['validated_embedding_bytes']==0
        result['cold']=cold;stop()
        assert all(row['exit_code']==0 and row['process_absent'] for row in result['encoder_runs'])
        result.update(status='PASS',seconds=time.monotonic()-started,max_abs_error=0.,
            limitation='Core typed-delta API only; no coordinator/CLI image-suffix admission, new Python restart, 8K/performance, sampler or original-gate qualification.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:ram.close();done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),seconds=result['seconds'],
        delta_cells=result['inputs']['delta_cells'],outputs={name:len(row['tokens']) for name,row in result['branches'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-base',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--producer-config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--outputs',type=int,default=8);campaign(parser.parse_args())
