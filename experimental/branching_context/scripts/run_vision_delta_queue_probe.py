"""Real Q4 coordinator requests with leased image deltas.

Uses retained independently executed native references for identical inputs.
Optional worker suffix preparation executes a real encoder cache miss.
HTTP/CLI suffix delivery is not exercised by this probe.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel import controller
from context_time_travel.input_spans import InputHistory, history_for_snapshot
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store
from context_time_travel.strata_backend import StrataBackend
from serve.cow_adapter import CowAdapter
from serve.cow_http import model_text_components, vision_preparer_from_config
from serve.cow_text_adapter import CowTextAdapter
from serve.cow_workspace import CowWorkspace
from run_vision_ingest_probe import Native, compare_files, fingerprint, save


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();prior=json.loads((baseline/'result.json').read_text());assert prior['status']=='PASS'
    profile=json.loads(Path(prior['profile']['path']).read_text());store=Store(out/'store')
    parts=model_text_components(profile['model_path'],profile['execution'].get('text_template_kwargs'),
        model_contract_id=profile['model_contract_id'],mixed_inputs=True)
    preparer=vision_preparer_from_config(store,parts['mixed_renderer'],profile,prior['producer_config']['path'])
    result=dict(status='RUNNING',scope=__doc__,baseline=fingerprint(baseline/'result.json'),profile=prior['profile'],
        workers=[],imports=[],branches={},cold={},comparisons=[],replays=[],removed_embeddings=[],encoder_calls=0,
        fresh_references=[],prepared_suffixes={},worker_suffix_preparation=args.prepare_suffixes,encoder_runs=[])
    prefix=InputHistory(prior['inputs']['prefix']);deltas={name:InputHistory(value) for name,value in prior['inputs']['deltas'].items()}
    models=[];manager=None;artifacts=None;done=threading.Event();phase={'name':'startup'};started=time.monotonic()
    def beat():
        while not done.wait(30):print(json.dumps(dict(phase=phase['name'],seconds=time.monotonic()-started,
            native_pids=[row['pid'] for row in result['workers']])),flush=True)
    threading.Thread(target=beat,daemon=True).start()
    real_encode=preparer.encoder.encode
    def checked_encoding(*values,**options):
        result['encoder_calls']+=1
        if not args.prepare_suffixes:raise AssertionError('This retained-artifact canary must not encode images')
        assert options['gpu_id']==0 and threading.get_ident()==result['workers'][-1]['owner_thread']
        encoded=real_encode(*values,**options);result['encoder_runs'].append(encoded.receipt);return encoded
    preparer.encoder.encode=checked_encoding
    def factory(gpu):
        backend=StrataBackend(profile,gpu_id=gpu)
        try:api=BranchingModel(out/'store',backend,profile)
        except BaseException:backend.close();raise
        models.append(api);row=dict(pid=backend.process.pid,owner_thread=threading.get_ident(),
            initial_work=backend.work_counters(),commands={});result['workers'].append(row)
        request=backend._request;close=backend.close
        def owned_request(command):
            assert threading.get_ident()==row['owner_thread'],'Native IPC escaped worker'
            name=command.split(' ',1)[0];row['commands'][name]=row['commands'].get(name,0)+1
            return request(command)
        def owned_close():
            assert threading.get_ident()==row['owner_thread'],'Native close escaped worker'
            if not backend._closed:row.update(final_work=backend.work_counters(),vision=backend._request('vision_info'))
            return close()
        backend._request=owned_request;backend.close=owned_close
        return api
    def start(with_producer):
        nonlocal manager
        manager=ContextCoordinator(out/'store',PoolConfig(gpus=(0,),ram_bytes=16*1024**3,disk_bytes=32*1024**3,
            checkpoint_idle_seconds=1800),factory,input_preparer=preparer if with_producer else None)
        try:manager.ready(300)
        except BaseException:manager.close();manager=None;raise
        adapter=CowAdapter(manager)
        text=CowTextAdapter(adapter,**parts,vision_preparer=preparer if with_producer else None)
        return adapter,text,CowWorkspace(store,parts['tokenizer'],root_mode_resolver=text.root_mode_for_projection)
    def stop():
        nonlocal manager
        if manager is None:return
        result['workers'][-1]['last_status']=manager.metrics();manager.close();manager=None
        backend=models[-1].backend;row=result['workers'][-1]
        row.update(exit_code=backend.process.returncode,process_absent=not Path(f"/proc/{row['pid']}").exists())
        assert row['exit_code']==0 and row['process_absent']
    def vector(snapshot,label,reference=None):
        with manager.pool.pin((profile['model_contract_id'],profile['execution_profile_id'],snapshot)) as checkpoint:
            values=checkpoint.frontier
        path=out/(label+'.f32');path.write_bytes(struct.pack('<'+'f'*len(values),*values))
        if reference is None:return path
        check=dict(label=label,reference=fingerprint(reference),candidate=fingerprint(path),**compare_files(reference,path))
        result['comparisons'].append(check);assert check['exact'],check
    def controller_check(snapshot,expected_snapshot,operation):
        expected=json.loads((baseline/'store/objects'/(expected_snapshot+'.json')).read_text())['controller']
        expected['stream_key']=hashlib.sha256(operation.encode()).hexdigest()
        assert store.get_snapshot(snapshot)['controller']==expected
    try:
        adapter,text,workspace=start(True)
        namespace=store.create_namespace('Resolved image questions','queue-namespace').namespace_id
        artifacts=preparer.artifacts(manager.pool,manager.disk)
        payloads={fingerprint(path)['sha256']:path for path in baseline.glob('*.sve')}
        # Import previously executed encoder outputs with their exact original,
        # normalization and encoder identities. This is not a new encoder test.
        imported=set()
        for span in [*prefix.spans,*deltas['B'].spans]:
            if span['kind']!='image' or span['original_sha256'] in imported:continue
            imported.add(span['original_sha256'])
            original=baseline/'store/image-originals'/(span['original_sha256']+'.input')
            raw=original.read_bytes();normalized=preparer.normalizer.normalize(raw)
            preparer.upload(namespace,raw,pool=manager.pool,disk=manager.disk)
            path=payloads[span['embedding_sha256']]
            # Leave B's new image as an original-only upload in preparation mode:
            # its suffix must run the pinned encoder on the selected worker.
            if not args.prepare_suffixes or any(s==span for s in prefix.spans):
                with artifacts.prepare(original=raw,normalized=normalized.payload,encode=lambda _,p=path:p.read_bytes(),
                    pad_token_id=span['pad_token_id']) as held:assert held.span==span
            result['imports'].append(dict(original=fingerprint(original),embedding=fingerprint(path),span=span))
        root_parts=[]
        for index,span in enumerate([s for s in prefix.spans if s['kind']=='image']):
            root_parts.extend([dict(type='text',text=' First diagram. ' if index==0 else ' Second diagram. '),
                               dict(type='image',image_id=span['original_sha256'])])
        root_parts.append(dict(type='text',text='Keep these diagrams as shared context.'))
        phase['name']='root';handle=text.create('A',[dict(role='user',content=root_parts)],'queue-root',namespace_id=namespace)
        root=handle.durable.result(300);root_answer=handle.snapshot.result(5);result['root']=dict(receipt=root,answer=root_answer)
        assert root_answer['counter_deltas']['forward_tokens']==len(prefix)
        assert not root_answer['input_preparation']['encodings']
        for branch in ('A','B'):vector(root['snapshot'],'root-'+branch,prior['references'][0 if branch=='A' else 2]['paths'][0])
        parent=manager.disk.directory/(root['payload_sha256']+'.state');before=fingerprint(parent)
        adapter.fork('B',root['snapshot'],source_branch='A',operation_id='queue-fork-B')
        presentations={}
        for branch,revision in (('A',1),('B',0)):
            phase['name']='delta-'+branch;operation='queue-question-'+branch
            image=next(s for s in deltas[branch].spans if s['kind']=='image')
            if args.prepare_suffixes:
                content=[dict(type='text',text='New diagram. '),dict(type='image',image_id=image['original_sha256']),
                    dict(type='text',text=' What color is the left half of this new diagram? Answer with one color only.')]
                request=dict(schema=2,namespace_id=namespace,kind='initial_turn',messages=[dict(role='user',content=content)],
                    add_generation_prompt=True)
                presentation=preparer.presentation(request,operation);presentations[branch]=presentation
                with manager.prepare_suffix(branch,revision,request,operation,8,presentation=presentation) as prepared:
                    assert prepared.history==deltas[branch] and prepared.receipt['forward_tokens']==0
                    result['prepared_suffixes'][branch]=prepared.receipt
                    handle=manager.submit(branch,prepared.history,8,operation,expected_revision=revision,
                        presentation=presentation,artifacts=prepared.artifacts)
            else:
                content='New diagram. [image sha256:'+image['original_sha256']+'] What color is the left half of this new diagram? Answer with one color only.'
                presentation=text._presentation(namespace,'initial_turn',[dict(role='user',content=content)],operation,
                    text._message_id(operation,'assistant'),canonical=prefix);presentations[branch]=presentation
                handle=manager.submit(branch,deltas[branch],8,operation,expected_revision=revision,
                    presentation=presentation,artifacts=artifacts)
            receipt=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert answer['tokens']==prior['branches'][branch]['tokens'] and answer['counter_deltas']['forward_tokens']==86
            vector(receipt['snapshot'],branch+'-final',prior['references'][0 if branch=='A' else 2]['paths'][3])
            controller_check(receipt['snapshot'],prior['branches'][branch]['whole_request']['snapshot'],operation)
            descriptor=manager.committed_generation(operation)
            assert descriptor['generation']['schema']==9 and descriptor['generation']['ids']==deltas[branch].spans
            assert descriptor['receipt']['presentation']['input_span']['count']==84
            view=workspace.history(branch)
            assert len(view['image_inputs'])==3 and view['messages'][-1]['content']==('Red' if branch=='A' else 'Blue')
            result['branches'][branch]=dict(receipt=receipt,answer=answer,history=view,controller_exact=True,
                descriptor=descriptor['generation'])
            assert fingerprint(parent)==before;save(out/'result.json',result)
        assert workspace.graph(namespace)['shared_prefix_messages']==1
        embeddings=list(manager.disk.directory.glob('*.sve'));assert len(embeddings)==3
        stop()
        assert result['workers'][0]['final_work']['forward_tokens']==1269
        for path in embeddings:
            assert path.resolve().parent==(out/'store/checkpoints').resolve() and not path.is_symlink()
            result['removed_embeddings'].append(fingerprint(path));path.unlink()
        phase['name']='reopen without producer';adapter,text,workspace=start(False)
        for branch,revision in (('A',1),('B',0)):
            operation='queue-question-'+branch
            handle=manager.submit(branch,deltas[branch],8,operation,expected_revision=revision,presentation=presentations[branch])
            receipt=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert answer['recovered'] and answer['tokens']==prior['branches'][branch]['tokens']
            replay=text.retry(operation,branch,revision+1,'replay-'+branch)
            assert replay['replayed'] and replay['answer']['text']==('Red' if branch=='A' else 'Blue')
            result['replays'].append(dict(branch=branch,receipt=receipt,answer=answer,text_replay=replay))
        for branch,revision in (('A',2),('B',1)):
            phase['name']='cold-followup-'+branch;operation='queue-followup-'+branch
            content='What color is the right half of the newest diagram? Answer with one color only.'
            source=manager.store.head(branch)
            source_history=history_for_snapshot(store,source.tip);source_controller=store.get_snapshot(source.tip)['controller']
            suffix=tuple(parts['append_user_suffix'](text._text_tail(source_history),source_controller,content))
            assert suffix==tuple(parts['tokenizer'].encode('\n',parse_special=False))+tuple(prior['inputs']['followup'])
            handle=text.append_user(branch,content,revision,operation,8)
            receipt=handle.durable.result(300);answer=handle.snapshot.result(5)
            assert answer['tokens']==prior['branches'][branch]['following']
            assert answer['counter_deltas']['forward_tokens']==len(suffix)+len(answer['tokens']) and answer['counter_deltas']['loaded_file_bytes']>0
            candidate=vector(receipt['snapshot'],'cold-'+branch+'-final')
            expected=controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),operation,
                (source_history+suffix).text_tokens,len(parts['tokenizer'].tokens))
            expected=controller.consume_many(expected,((token,parts['tokenizer'].token_bytes(token),
                token in profile['execution']['eos_tokens']) for token in answer['tokens']))
            assert store.get_snapshot(receipt['snapshot'])['controller']==expected
            value=handle.text_result();assert value==('Blue' if branch=='A' else 'Red')
            result['cold'][branch]=dict(receipt=receipt,answer=answer,text=value,controller_exact=True,
                controller_reference='canonical greedy start/consume equation',suffix=list(suffix),frontier=fingerprint(candidate))
        assert fingerprint(parent)==before and not list((out/'store/checkpoints').glob('*.sve'))
        result['parent_payload_unchanged']=before;stop()
        assert result['workers'][1]['final_work']['forward_tokens']==sum(len(row['suffix'])+len(row['answer']['tokens']) for row in result['cold'].values())
        assert result['workers'][1]['vision']['validated_embedding_bytes']==0
        assert result['encoder_calls']==(1 if args.prepare_suffixes else 0)
        # Serving deliberately adds an inter-turn newline; the earlier raw core
        # fixture did not. Execute new uncached references for this exact input.
        build=json.loads(Path(prior['build']['path']).read_text())
        for branch in ('A','B'):
            phase['name']='fresh-serving-followup-'+branch
            argv=list(profile['execution']['sut_command']);argv[0]=build['binaries']['reference']['path']
            assert fingerprint(argv[0])['sha256']==build['binaries']['reference']['sha256']
            native=Native(argv,out/('reference-'+branch));row=dict(branch=branch,pid=native.proc.pid,argv=argv)
            result['fresh_references'].append(row)
            try:
                assert native.ready['reference']=='fresh-no-checkpoints'
                native.request('feed_image '+json.dumps(str(baseline/'prefix.manifest'))+' '+json.dumps(str(baseline/'prefix-positions.i32')))
                native.request('feed_image '+json.dumps(str(baseline/('delta-'+branch+'.manifest')))+' '+json.dumps(str(baseline/('delta-'+branch+'-positions.i32'))))
                _,values=native.frontier('initial')
                for index,token in enumerate(prior['branches'][branch]['tokens']):
                    assert max(range(len(values)),key=values.__getitem__)==token
                    native.request('decode '+str(token));_,values=native.frontier('initial-output-'+str(index+1))
                native.request('feed '+' '.join(map(str,result['cold'][branch]['suffix'])))
                _,values=native.frontier('followup');tokens=[]
                for index in range(8):
                    token=max(range(len(values)),key=values.__getitem__);tokens.append(token)
                    native.request('decode '+str(token));path,values=native.frontier('output-'+str(index+1))
                    if token in profile['execution']['eos_tokens']:break
                assert tokens==result['cold'][branch]['answer']['tokens']
                row.update(tokens=tokens,work=native.request('stats'))
                assert row['work']['forward_tokens']==len(prefix)+len(deltas[branch])+len(prior['branches'][branch]['tokens'])+len(result['cold'][branch]['suffix'])+len(tokens)
                candidate=Path(result['cold'][branch]['frontier']['path'])
                check=dict(label='cold-'+branch+'-fresh-final',reference=fingerprint(path),candidate=fingerprint(candidate),**compare_files(path,candidate))
                result['comparisons'].append(check);assert check['exact'],check
            finally:row['exit']=native.close()
            assert row['exit']['exit_code']==0 and row['exit']['process_absent'];save(out/'result.json',result)
        result.update(status='PASS',seconds=time.monotonic()-started,max_abs_error=0.,
            limitation='Coordinator deltas only; worker suffix prep when requested, no HTTP/CLI suffix flow; four retained-reference and two new fresh-reference vectors, greedy, native-worker restart in one Python driver.')
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        try:stop()
        finally:done.set();save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],comparisons=len(result['comparisons']),seconds=result['seconds'],
        cold_work={name:row['answer']['counter_deltas']['forward_tokens'] for name,row in result['cold'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-suffixes',action='store_true')
    campaign(parser.parse_args())
