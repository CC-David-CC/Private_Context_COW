"""Real Q4 image upload/preparer/coordinator canary with fresh native references.

Calls production Python boundaries directly; HTTP/CLI are not exercised here.
Python remains alive while the native worker exits, references run and receipts
are recovered by a new worker. No checkpoint is imported by either reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import time

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import history_for_snapshot
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store
from context_time_travel.strata_backend import StrataBackend
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_renderer import renderer_identity
from serve.cow_vision import CowImageNormalizer, CowVisionPromptEncoder
from serve.cow_vision_encoder import CowNativeVisionEncoder
from serve.cow_vision_preparer import CowVisionPreparer
from serve.frontend import ChatTemplate
from tools.gguf_reader import GGUFFile
from tools.strata_tokenizer import Tokenizer
from run_vision_ingest_probe import Native, compare_files, fingerprint, save
from vision_fixture import reference_positions


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=args.artifact_base.resolve();profile=json.loads(args.profile.read_text())
    build=json.loads(args.encoder_build.read_text())
    baseline=json.loads((base/'image-baseline-v2/result.json').read_text())
    projector=baseline['encodings'][0]['argv'][2];model=profile['model_path']
    executable=build['binary']['path'];pins={executable:build['binary']['sha256']}
    for path in (projector,model):pins[path]=baseline['files'][path]['sha256']
    encoder=CowNativeVisionEncoder(executable=executable,projector=projector,model=model,
        pinned_sha256=pins,width=2560,threads=32)
    tokenizer=Tokenizer.from_gguf(model);template_source=GGUFFile(Path(model)).metadata['tokenizer.chat_template']
    with tempfile.TemporaryDirectory(prefix='cow-production-template-') as folder:
        path=Path(folder)/'template.jinja';path.write_text(template_source);template=ChatTemplate(path)
    prompt=CowPromptEncoder(tokenizer,template,{'enable_thinking':False})
    identity=renderer_identity(tokenizer,template_source,prompt.template_kwargs,model_contract_id=profile['model_contract_id'])
    renderer=CowVisionPromptEncoder(prompt,renderer_id=identity);store=Store(out/'store')
    preparer=CowVisionPreparer(store,CowImageNormalizer(),renderer,encoder,original_bytes_limit=256*1024**2)
    namespace=store.create_namespace('Production image input canary','vision-namespace').namespace_id
    result=dict(status='RUNNING',scope=__doc__,python_pid=os.getpid(),profile=fingerprint(args.profile),
        encoder_build=fingerprint(args.encoder_build),preparer_id=preparer.identity,workers=[],uploads=[],
        roots=[],branches={},references=[],comparisons=[],replays=[])
    coordinator=None;models=[];done=threading.Event();phase={'name':'startup'};started=time.monotonic()
    def beat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            python_pid=os.getpid(),native_pids=[row['pid'] for row in result['workers']])),flush=True)
    threading.Thread(target=beat,daemon=True).start()
    def factory(gpu):
        backend=StrataBackend(profile,gpu_id=gpu)
        try:api=BranchingModel(out/'store',backend,profile)
        except BaseException:backend.close();raise
        models.append(api);row=dict(pid=backend.process.pid,owner_thread=threading.get_ident(),commands={});result['workers'].append(row)
        request=backend._request;close=backend.close
        def owned_request(command):
            assert threading.get_ident()==row['owner_thread'],'Native request escaped owner'
            name=command.split(' ',1)[0];row['commands'][name]=row['commands'].get(name,0)+1
            return request(command)
        def owned_close():
            assert threading.get_ident()==row['owner_thread'],'Native close escaped owner'
            if not backend._closed:row['final_work']=backend.work_counters()
            return close()
        backend._request=owned_request;backend.close=owned_close
        return api
    def start():
        manager=ContextCoordinator(out/'store',PoolConfig(gpus=(0,),ram_bytes=16*1024**3,
            disk_bytes=32*1024**3,checkpoint_idle_seconds=1800),factory,input_preparer=preparer)
        try:manager.ready(300)
        except BaseException:manager.close();raise
        return manager
    def stop():
        nonlocal coordinator
        if coordinator is None:return
        coordinator.close();coordinator=None;backend=models[-1].backend;row=result['workers'][-1]
        row.update(exit_code=backend.process.returncode,process_absent=not Path(f"/proc/{row['pid']}").exists())
        assert row['exit_code']==0 and row['process_absent']
    def vector(snapshot,label):
        key=(profile['model_contract_id'],profile['execution_profile_id'],snapshot)
        with coordinator.pool.pin(key) as checkpoint:values=checkpoint.frontier
        path=out/(label+'.f32');path.write_bytes(struct.pack('<'+'f'*len(values),*values));return str(path)
    def root(branch):
        operation='prepare-'+branch
        return coordinator.prepare_root(branch,body,operation,presentation=preparer.presentation(body,operation))
    try:
        coordinator=start();parts=[]
        for index,path in enumerate(args.image):
            upload=preparer.upload(namespace,path.read_bytes(),pool=coordinator.pool,disk=coordinator.disk)
            result['uploads'].append(upload)
            parts.extend([dict(type='text',text=' First diagram. ' if index==0 else ' Second diagram. '),
                          dict(type='image',image_id=upload['image_id'])])
        parts.append(dict(type='text',text='Keep these diagrams as shared context.'))
        body=dict(schema=1,namespace_id=namespace,messages=[dict(role='user',content=parts)],add_generation_prompt=False)
        for index,branch in enumerate(('A','B')):
            phase['name']='prepare-'+branch;handle=root(branch);durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            history=history_for_snapshot(store,durable['snapshot'])
            assert answer['counter_deltas']['forward_tokens']==(len(history) if index==0 else 0)
            assert len(answer['input_preparation']['encodings'])==(len(args.image) if index==0 else 0)
            result['roots'].append(dict(branch=branch,durable=durable,answer=answer,frontier=vector(durable['snapshot'],branch+'-root')))
            save(out/'result.json',result)
        prefix=history_for_snapshot(store,result['roots'][0]['durable']['snapshot'])
        (out/'canonical-inputs.json').write_bytes(prefix._encoded)
        images=[];position=0;lines=[f'COW_VISION_INPUT_V1 0 {len(prefix)} {len(args.image)}', ' '.join(map(str,prefix.native_tokens))]
        for span in prefix.spans:
            if span['kind']=='text':position+=len(span['tokens']);continue
            payload=coordinator.disk.directory/(span['embedding_sha256']+'.sve')
            lines.append(f'{position} {span["rows"]} {span["grid_width"]} {span["grid_height"]} {span["width"]} {span["pad_token_id"]} {span["embedding_sha256"]} '+json.dumps(str(payload)))
            images.append(dict(start=position,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']))
            position+=span['rows']
        manifest=out/'reference-input.manifest';manifest.write_text('\n'.join(lines)+'\n')
        positions=out/'reference-positions.i32'
        positions.write_bytes(b''.join(struct.pack('<3i',*row) for row in reference_positions(len(prefix),images,profile['execution']['capacity'])))
        payload=coordinator.disk.directory/(result['roots'][0]['durable']['payload_sha256']+'.state');parent_bytes=fingerprint(payload)
        questions={branch:'What color is the '+side+' half of the first diagram? Answer briefly.'
                   for branch,side in (('A','left'),('B','right'))}
        suffixes={branch:tuple(prompt._user_frame(content)) for branch,content in questions.items()}
        for branch in ('A','B'):
            phase['name']='question-'+branch;operation='question-'+branch
            presentation=dict(schema=1,namespace_id=namespace,kind='initial_turn',renderer_id=renderer.renderer_id,
                messages=[dict(id=hashlib.sha256(operation.encode()).hexdigest()[:32],role='user',content=questions[branch])],
                output_message_id=hashlib.sha256((operation+'-answer').encode()).hexdigest()[:32])
            handle=coordinator.submit(branch,suffixes[branch],args.outputs,operation,expected_revision=1,presentation=presentation)
            durable=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert answer['counter_deltas']['forward_tokens']==len(suffixes[branch])+len(answer['tokens'])
            assert fingerprint(payload)==parent_bytes
            result['branches'][branch]=dict(durable=durable,answer=answer,question=questions[branch],
                frontier=vector(durable['snapshot'],branch+'-final'))
            save(out/'result.json',result)
        stop()
        for branch in ('A','B'):
            phase['name']='fresh-reference-'+branch
            argv=list(profile['execution']['sut_command']);argv[0]=str(base/'vc05-native-build-v1/strata-vision-reference')
            native=Native(argv,out/('reference-'+branch));row=dict(branch=branch,pid=native.proc.pid,argv=argv)
            result['references'].append(row)
            try:
                native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(positions)))
                root_path,_=native.frontier('prefix');native.request('feed '+' '.join(map(str,suffixes[branch])))
                _,values=native.frontier('suffix');tokens=[]
                for index in range(args.outputs):
                    token=max(range(len(values)),key=values.__getitem__);tokens.append(token);native.request('decode '+str(token))
                    final_path,values=native.frontier('output-'+str(index+1))
                    if token in profile['execution']['eos_tokens']:break
                assert tokens==result['branches'][branch]['answer']['tokens']
                for label,expected,actual in ((branch+'-root',root_path,result['roots'][0]['frontier']),
                    (branch+'-final',final_path,result['branches'][branch]['frontier'])):
                    check=dict(label=label,**compare_files(expected,actual));result['comparisons'].append(check);assert check['exact'],check
                row.update(tokens=tokens,work=native.request('stats'))
                assert row['work']['forward_tokens']==len(prefix)+len(suffixes[branch])+len(tokens)
            finally:row['exit']=native.close()
            assert row['exit']['exit_code']==0 and row['exit']['process_absent'];save(out/'result.json',result)
        phase['name']='reopen-root-receipts';coordinator=start()
        for previous in result['roots']:
            handle=root(previous['branch']);durable=handle.durable.result(300)
            assert durable['branch']==previous['durable']['branch'] and handle.snapshot.result(5).get('recovered')
            result['replays'].append(dict(branch=previous['branch'],exact=True))
        stop();assert result['workers'][-1]['final_work']['forward_tokens']==0
        result.update(status='PASS',seconds=time.monotonic()-started,max_abs_error=0.,prefix_cells=len(prefix),
            parent_payload=parent_bytes,limitation='Direct production boundaries, no HTTP/CLI; root/final oracle vectors only, greedy controller; Python driver survives worker restart.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],prefix_cells=result['prefix_cells'],comparisons=len(result['comparisons']),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-base',type=Path,required=True);parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--encoder-build',type=Path,required=True);parser.add_argument('--image',type=Path,action='append',required=True)
    parser.add_argument('--outputs',type=int,default=8);parser.add_argument('--output',type=Path,required=True)
    campaign(parser.parse_args())
