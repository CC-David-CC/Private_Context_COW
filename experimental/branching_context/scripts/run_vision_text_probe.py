"""Real Q4 text-adapter/workspace flow and encoder-free cold continuation.

Uses retained independent native references for the identical prefix/questions.
No HTTP or CLI transport is exercised. Python survives native-worker restart.
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
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store
from context_time_travel.strata_backend import StrataBackend
from serve.cow_adapter import CowAdapter
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_renderer import renderer_identity
from serve.cow_text_adapter import CowTextAdapter
from serve.cow_vision import CowImageNormalizer, CowVisionPromptEncoder
from serve.cow_vision_encoder import CowNativeVisionEncoder
from serve.cow_vision_preparer import CowVisionPreparer
from serve.cow_workspace import CowWorkspace
from serve.frontend import ChatTemplate
from tools.gguf_reader import GGUFFile
from tools.strata_tokenizer import Tokenizer
from run_vision_ingest_probe import compare_files, fingerprint, save


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();prior=json.loads((baseline/'result.json').read_text());assert prior['status']=='PASS'
    profile=json.loads(Path(prior['profile']['path']).read_text())
    build=json.loads(Path(prior['encoder_build']['path']).read_text())
    old_encoder=prior['roots'][0]['answer']['input_preparation']['encodings'][0]
    # The original encoder baseline pins executable/projector/metadata bytes.
    metadata=json.loads((args.artifact_base/'image-baseline-v2/result.json').read_text())
    projector=metadata['encodings'][0]['argv'][2];model=profile['model_path'];exe=build['binary']['path']
    native_encoder=CowNativeVisionEncoder(executable=exe,projector=projector,model=model,width=2560,threads=32,
        pinned_sha256={exe:build['binary']['sha256'],projector:metadata['files'][projector]['sha256'],model:metadata['files'][model]['sha256']})
    assert native_encoder.identity==old_encoder['encoder_id']
    tokenizer=Tokenizer.from_gguf(model);source=GGUFFile(Path(model)).metadata['tokenizer.chat_template']
    with tempfile.TemporaryDirectory(prefix='cow-text-vision-template-') as folder:
        path=Path(folder)/'template.jinja';path.write_text(source);template=ChatTemplate(path)
    prompt=CowPromptEncoder(tokenizer,template,{'enable_thinking':False})
    renderer_id=renderer_identity(tokenizer,source,prompt.template_kwargs,model_contract_id=profile['model_contract_id'])
    mixed=CowVisionPromptEncoder(prompt,renderer_id=renderer_id);store=Store(out/'store')
    preparer=CowVisionPreparer(store,CowImageNormalizer(),mixed,native_encoder,original_bytes_limit=256*1024**2)
    namespace=store.create_namespace('Text image canary','vision-namespace').namespace_id
    result=dict(status='RUNNING',scope=__doc__,baseline=fingerprint(baseline/'result.json'),
        profile=fingerprint(prior['profile']['path']),workers=[],branches={},comparisons=[],removed_embeddings=[])
    manager=None;models=[];done=threading.Event();phase={'name':'startup'};started=time.monotonic()
    def beat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            native_pids=[row['pid'] for row in result['workers']])),flush=True)
    threading.Thread(target=beat,daemon=True).start()
    def factory(gpu):
        backend=StrataBackend(profile,gpu_id=gpu)
        try:api=BranchingModel(out/'store',backend,profile)
        except BaseException:backend.close();raise
        models.append(api);row=dict(pid=backend.process.pid,owner_thread=threading.get_ident(),commands={});result['workers'].append(row)
        request=backend._request;close=backend.close
        def owned_request(command):
            assert threading.get_ident()==row['owner_thread'],'Native IPC escaped owner'
            key=command.split(' ',1)[0];row['commands'][key]=row['commands'].get(key,0)+1
            return request(command)
        def owned_close():
            assert threading.get_ident()==row['owner_thread'],'Native close escaped owner'
            if not backend._closed:
                row['final_work']=backend.work_counters();row['final_vision']=backend._request('vision_info')
            return close()
        backend._request=owned_request;backend.close=owned_close
        return api
    def start(with_encoder):
        coordinator=ContextCoordinator(out/'store',PoolConfig(gpus=(0,),ram_bytes=16*1024**3,
            disk_bytes=32*1024**3,checkpoint_idle_seconds=1800),factory,input_preparer=preparer if with_encoder else None)
        try:coordinator.ready(300)
        except BaseException:coordinator.close();raise
        adapter=CowAdapter(coordinator)
        text=CowTextAdapter(adapter,tokenizer,prompt.render,prompt.encode_rendered,
            append_user_suffix=prompt.append_user_suffix,append_initial_user_suffix=prompt.append_initial_user_suffix,
            initial_root_mode=prompt.initial_root_mode,renderer_id=renderer_id,mixed_renderer=mixed,
            vision_preparer=preparer if with_encoder else None)
        return coordinator,adapter,text,CowWorkspace(store,tokenizer,root_mode_resolver=text.root_mode_for_projection)
    def stop():
        nonlocal manager
        if manager is None:return
        manager.close();manager=None;backend=models[-1].backend;row=result['workers'][-1]
        row.update(exit_code=backend.process.returncode,process_absent=not Path(f"/proc/{row['pid']}").exists())
        assert row['exit_code']==0 and row['process_absent']
    def compare(snapshot,reference,label):
        key=(profile['model_contract_id'],profile['execution_profile_id'],snapshot)
        with manager.pool.pin(key) as checkpoint:values=checkpoint.frontier
        path=out/(label+'.f32');path.write_bytes(struct.pack('<'+'f'*len(values),*values))
        check=dict(label=label,reference=fingerprint(reference),candidate=fingerprint(path),**compare_files(reference,path))
        result['comparisons'].append(check);assert check['exact'],check
    def answer(branch,question,revision,operation,reference_branch):
        handle=text.ask(branch,question,revision,operation,8);receipt=handle.durable.result(300);snapshot=handle.snapshot.result(5)
        reference=prior['branches'][reference_branch]
        assert snapshot['tokens']==reference['answer']['tokens']
        assert snapshot['counter_deltas']['forward_tokens']==len(prompt._user_frame(question))+len(snapshot['tokens'])
        vector=next(row['expected'] for row in prior['comparisons'] if row['label']==reference_branch+'-final')
        compare(receipt['snapshot'],vector,branch+'-final')
        expected=json.loads((baseline/'store/objects'/(reference['durable']['snapshot']+'.json')).read_text())['controller']
        expected['stream_key']=hashlib.sha256(operation.encode()).hexdigest()
        actual=store.get_snapshot(receipt['snapshot'])['controller'];assert actual==expected
        content=handle.text_result();assert content==('Red' if reference_branch=='A' else 'Blue')
        view=workspace.history(branch);assert view['messages'][-1]['content']==content and len(view['image_inputs'])==2
        result['branches'][branch]=dict(receipt=receipt,snapshot=snapshot,text=content,history=view,controller_exact=True)
        save(out/'result.json',result)
    try:
        manager,adapter,text,workspace=start(True);parts=[]
        for index,old in enumerate(prior['uploads']):
            raw=(baseline/'store/image-originals'/(old['image_id']+'.input')).read_bytes()
            upload=preparer.upload(namespace,raw,pool=manager.pool,disk=manager.disk)
            parts.extend([dict(type='text',text=' First diagram. ' if index==0 else ' Second diagram. '),
                dict(type='image',image_id=upload['image_id'])])
        parts.append(dict(type='text',text='Keep these diagrams as shared context.'))
        messages=[dict(role='user',content=parts)];phase['name']='prepare'
        root_handle=text.create('A',messages,'prepare-A',namespace_id=namespace)
        root=root_handle.durable.result(300)
        result['root_preparation']=root_handle.snapshot.result(5)
        assert root['snapshot']==prior['roots'][0]['durable']['snapshot']
        assert workspace.history('A')['initial_root_mode']=='context_only'
        compare(root['snapshot'],prior['comparisons'][0]['expected'],'prefix')
        adapter.fork('B',root['snapshot'],source_branch='A',operation_id='fork-B')
        for branch,side,revision in (('A','left',1),('B','right',0)):
            phase['name']='ask-'+branch
            answer(branch,'What color is the '+side+' half of the first diagram? Answer briefly.',revision,'question-'+branch,branch)
        graph=workspace.graph(namespace);assert len(graph['branches'])==2 and graph['shared_prefix_messages']==1
        save(out/'graph-before-restart.json',graph)
        paths=list(manager.disk.directory.glob('*.sve'));assert len(paths)==2
        stop()
        for path in paths:
            assert path.resolve().parent==(out/'store/checkpoints').resolve() and path.is_file() and not path.is_symlink()
            result['removed_embeddings'].append(fingerprint(path));path.unlink()
        phase['name']='encoder-free-disk-continuation';manager,adapter,text,workspace=start(False)
        replay=text.retry('question-A','A',2,'retry-A');assert replay['replayed'] and replay['answer']['text']=='Red'
        result['replay']=replay
        adapter.fork('C',root['snapshot'],source_branch='A',operation_id='fork-C')
        answer('C','What color is the left half of the first diagram? Answer briefly.',0,'question-C','A')
        result['cold_work']=result['branches']['C']['snapshot']['counter_deltas']
        assert result['cold_work']['forward_tokens']==28 and result['cold_work']['loaded_file_bytes']>0
        assert not list((out/'store/checkpoints').glob('*.sve'))
        save(out/'graph-after-restart.json',workspace.graph(namespace));stop()
        assert result['workers'][-1]['final_work']['forward_tokens']==28
        assert result['workers'][-1]['final_vision']['validated_embedding_bytes']==0
        result.update(status='PASS',seconds=time.monotonic()-started,max_abs_error=0.,
            limitation='No HTTP/CLI, no new Python-process restart; four prefix/final comparisons against retained independent native references, greedy only.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),
        answers={key:value['text'] for key,value in result['branches'].items()},seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-base',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);campaign(parser.parse_args())
