"""Bounded Q4 image GBNF fork/restart audit, with independent raw references.

Uses retained, real encoder fixtures; no new encoding is claimed. Reference
processes import no checkpoint and compute positions with the fixture's CPU
reference. This is an audit, not a serving-throughput benchmark.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import DiskCheckpointPool, SharedCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend, _digest
from context_time_travel.vision_artifacts import ImageArtifacts
from run_vision_ingest_probe import Native, compare_files, fingerprint, save
from run_vision_restart_probe import audit_unchanged, controller_file, frontier
from tools.strata_tokenizer import Tokenizer
from vision_fixture import fixture_parts


GRAMMARS={
    'EDIT':dict(format='gbnf',source='''root ::= "{\\"tool\\":\\"propose_patch\\",\\"path\\":" path ",\\"patch\\":" patch "}"
path ::= "\\"src/main.py\\"" | "\\"tests/test_main.py\\""
patch ::= "\\"return 1\\"" | "\\"return 2\\""'''),
    'TEST':dict(format='gbnf',source='root ::= "{\\"tool\\":\\"run_tests\\",\\"suite\\":\\"unit\\"}"'),
}


def engine_for(profile):
    tokenizer=Tokenizer.from_gguf(profile['model_path'])
    return GrammarEngine(tokenizer,GRAMMARS,profile['execution']['eos_tokens'])


def raw_scope(native,engine,history,state,limit,stream,label):
    decoder=engine.start(state,stream,history,engine.vocab_size)
    session=engine.restore(decoder,history);paths=[];tokens=[]
    _,values=native.frontier(label+'-before')
    for index in range(limit):
        # Independent CPU legal-token argmax over the raw fresh model logits;
        # the SUT instead uses its checked native finite-mask selector.
        token=session.choose(values);tokens.append(token)
        native.request('decode '+str(token))
        session._consume_selected(token,engine.tokenizer.token_bytes(token),token in engine.eos_tokens)
        path,values=native.frontier(label+'-'+str(index+1));paths.append(str(path))
        if session.is_stopped():break
    assert session.is_stopped() and session.is_complete(),'Fixture exhausted its bound before grammar completion'
    return dict(tokens=tokens,frontiers=paths,controller=session.current_controller)


def worker(config,stage,out):
    profile=json.loads(Path(config['profile']).read_text());engine=engine_for(profile)
    result=dict(status='RUNNING',python_pid=os.getpid(),stage=stage,encoder_calls=0,
        engine_id=engine.engine_id,tokenizer_sha256=engine.tokenizer_sha256,
        spec_sha256=engine.spec_sha256)
    backend=ram=None;started=time.monotonic()
    try:
        if stage.startswith('reference-'):
            branch=stage[-1];argv=list(profile['execution']['sut_command']);argv[0]=config['reference_binary']
            native=Native(argv,out/'native');result['native_pid']=native.proc.pid
            try:
                manifest=Path(config['fixture']).parent/'input.manifest'
                native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(manifest.parent/'reference-positions.i32')))
                prefix=str(native.frontier('prefix')[0]);suffix=config['suffixes'][branch]
                native.request('feed '+' '.join(map(str,suffix)));suffix_path=str(native.frontier('suffix')[0])
                source=InputHistory(config['spans']);text_history=source.text_tokens+tuple(suffix)
                value=raw_scope(native,engine,text_history,'EDIT',config['limit'],'vision-edit','edit')
                result.update(prefix=prefix,suffix=suffix_path,answer=value)
                if branch=='A':
                    result['test_scope']=raw_scope(native,engine,text_history+tuple(value['tokens']),
                        'TEST',config['limit'],'vision-test','test')
                result['work']=native.request('stats')
            finally:result['native_exit']=native.close()
        else:
            os.environ['CTT_NATIVE_TRACE']=str(out/'native-trace.jsonl')
            backend=StrataBackend(profile,gpu_id=0);result['native_pid']=backend.process.pid
            api=BranchingModel(config['store'],backend,profile)
            def scope(snapshot,state,label,*,resume=False,limit=None):
                tokens=[];paths=[];snapshots=[]
                for index in range(config['limit'] if limit is None else limit):
                    answer=api.run_state_request(snapshot,(),1,engine=engine,state=state,
                        stream='vision-edit' if state=='EDIT' else 'vision-test',resume=resume or index>0)
                    snapshot=answer['snapshot'];tokens.extend(answer['tokens']);snapshots.append(snapshot)
                    paths.append(frontier(api,snapshot,out/(label+'-'+str(index+1)+'.f32')))
                    if answer['stopped']:break
                return dict(snapshot=snapshot,tokens=tokens,frontiers=paths,snapshots=snapshots,
                    controller=api.controller(snapshot),stopped=answer['stopped'],grammar_complete=answer['grammar_complete'])
            if stage=='prepare':
                fixture=json.loads(Path(config['fixture']).read_text())
                parts=fixture_parts(config['fixture'],config['images'])
                ram=SharedCheckpointPool(1024**3);disk=DiskCheckpointPool(api.store,4*1024**3)
                producer=dict(kind='retained-executed-encoder-fixture',fixture=fingerprint(config['fixture']),parts=parts)
                artifacts=ImageArtifacts(api.store,ram,disk,encoder_profile_id=_digest(producer),
                    expected_width=parts[0]['width'],original_bytes_limit=256*1024**2)
                ids=fixture['input_tokens'];spans=[];cursor=0
                with ExitStack() as held:
                    for part in parts:
                        original=Path(part['original']['path']).read_bytes();path=Path(part['embedding']['path'])
                        item=held.enter_context(artifacts.prepare(original=original,normalized=original,
                            encode=lambda _,p=path:p.read_bytes(),pad_token_id=part['pad']))
                        spans.extend([dict(kind='text',tokens=ids[cursor:part['start']]),item.span]);cursor=part['start']+part['rows']
                    spans.append(dict(kind='text',tokens=ids[cursor:]));parent=api.create_inputs(spans,artifacts=artifacts)
                result.update(prefix_snapshot=parent,spans=spans,prefix=frontier(api,parent,out/'prefix.f32'))
                api.store.fork('prefix',parent);api.save_branch('prefix',0,parent,'save-vision-grammar-prefix')
                result['parent_audit']=backend.audit(api.cache[parent].blob);result['parent_audits_after_branches']=[]
                starts={};answers={}
                for name in ('A','B'):
                    before=backend.work_counters()['forward_tokens'];tip=api.feed(parent,config['suffixes'][name])
                    assert backend.work_counters()['forward_tokens']-before==len(config['suffixes'][name])
                    starts[name]=dict(snapshot=tip,suffix=frontier(api,tip,out/(name+'-suffix.f32')))
                # Hold B's partial EDIT while A finishes EDIT and enters TEST.
                b=scope(starts['B']['snapshot'],'EDIT','B-partial',limit=1)
                b_bytes=api.snapshot_bytes(b['snapshot']);b_controller=api.controller(b['snapshot'])
                for name in ('A','B'):
                    first=scope(starts['A']['snapshot'],'EDIT','A-first',limit=4) if name=='A' else b
                    if name=='A':
                        assert len(first['tokens'])==4 and not first['stopped']
                        result['generated_four']=first['snapshot'];api.checkpoint(first['snapshot'])
                    tail=scope(first['snapshot'],'EDIT',name+'-tail',resume=True)
                    assert tail['stopped'] and tail['grammar_complete']
                    answers[name]=dict(snapshot=tail['snapshot'],tokens=first['tokens']+tail['tokens'],
                        frontiers=first['frontiers']+tail['frontiers'],suffix=starts[name]['suffix'],controller=tail['controller'])
                    if name=='A':
                        result['test_scope']=scope(tail['snapshot'],'TEST','A-test')
                        assert result['test_scope']['stopped'] and result['test_scope']['grammar_complete']
                        assert api.snapshot_bytes(b['snapshot'])==b_bytes and api.controller(b['snapshot'])==b_controller
                        result['sibling_progress_unchanged']=True
                    audit_unchanged(backend,api.cache[parent].blob,result['parent_audit'],result['parent_audits_after_branches'])
                result['answers']=answers
                result['saved_files']=[fingerprint(p) for p in sorted(api.checkpoint_directory.iterdir()) if p.suffix in ('.json','.state')]
            else:
                prepared=json.loads(Path(config['prepare_result']).read_text())
                forbidden={str(Path(config['fixture']).resolve()),*(str(Path(p).resolve()) for p in config['images'])}
                def no_image_reads_or_processes(event,args):
                    if event=='open' and isinstance(args[0],(str,bytes)):
                        path=os.fsdecode(args[0]);mode=args[1] if len(args)>1 else None
                        # Restoration may use native state transport; no image
                        # encoder or embedding access is needed for this scope.
                        if path.endswith('.sve') or str(Path(path).resolve()) in forbidden:
                            raise AssertionError('Image read during strict grammar restore: '+path)
                    if event=='subprocess.Popen':raise AssertionError('Extra subprocess during strict grammar restore')
                sys.addaudithook(no_image_reads_or_processes)
                assert not list(api.checkpoint_directory.glob('*.sve'))
                snapshot=prepared['generated_four'] if stage=='resume-generated' else prepared['prefix_snapshot']
                result['restore']=api.restore_checkpoint(snapshot);assert result['restore']['forward_tokens']==0
                result['restored_frontier']=frontier(api,snapshot,out/'restored.f32')
                assert backend._request('vision_info')['validated_embedding_bytes']==0
                if stage=='resume-generated':
                    result['answer']=scope(snapshot,'EDIT','generated-restored',resume=True)
                else:
                    result['answers']={}
                    for name in ('A','B'):
                        before=backend.work_counters()['forward_tokens'];tip=api.feed(snapshot,config['suffixes'][name])
                        assert backend.work_counters()['forward_tokens']-before==len(config['suffixes'][name])
                        suffix=frontier(api,tip,out/(name+'-suffix.f32'))
                        answer=scope(tip,'EDIT',name+'-restored');answer['suffix']=suffix;result['answers'][name]=answer
                assert backend._request('vision_info')['validated_embedding_bytes']==0
            result['work']=backend.work_counters()
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        if backend is not None:
            pid=backend.process.pid;backend.close()
            result['native_exit']=dict(pid=pid,exit_code=backend.process.returncode,process_absent=not Path(f'/proc/{pid}').exists())
        if ram is not None:ram.close()
        result['seconds']=time.monotonic()-started;save(out/'result.json',result)


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    original=json.loads(args.fixture_config.read_text());fixture=json.loads(Path(original['fixture']).read_text())
    config=dict(profile=original['profile'],fixture=original['fixture'],images=original.get('images',[original['image']]),
        suffixes=fixture['suffixes'],store=str(out/'store'),prepare_result=str(out/'prepare/result.json'),
        reference_binary=original['reference_binary'],limit=args.limit)
    save(out/'config.json',config)
    result=dict(status='RUNNING',stages=[],comparisons=[],controller_comparisons=[],encoder_calls=0,
        scope='Q4 image greedy GBNF; every consumed frontier; no complete VC07-VC09 qualification',grammars=GRAMMARS)
    def run(stage,name):
        directory=out/name;directory.mkdir();argv=[sys.executable,'-B',__file__,'--worker',stage,
            '--config',str(out/'config.json'),'--output',str(directory)]
        with (directory/'stdout.log').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:
            child=subprocess.Popen(argv,stdout=stdout,stderr=stderr);started=time.monotonic()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:print(json.dumps(dict(stage=name,pid=child.pid,elapsed_seconds=time.monotonic()-started)),flush=True)
        row=dict(stage=name,argv=argv,python_pid=child.pid,exit_code=code,
            process_absent=not Path(f'/proc/{child.pid}').exists(),seconds=time.monotonic()-started)
        result['stages'].append(row);save(out/'result.json',result);assert code==0,row
        value=json.loads((directory/'result.json').read_text())
        assert value['status']=='PASS' and value['native_exit']['exit_code']==0 and value['native_exit']['process_absent']
        return value
    def exact(a,b,label):
        entry=dict(label=label,**compare_files(a,b));result['comparisons'].append(entry);save(out/'result.json',result)
        assert entry['exact'],entry
    def controller_exact(a,b,label):
        assert a==b,label
        result['controller_comparisons'].append(dict(label=label,sha256=_digest(a)))
    try:
        prepared=run('prepare','prepare');config['spans']=prepared['spans'];save(out/'config.json',config)
        removed=[]
        for path in (out/'store/checkpoints').glob('*.sve'):removed.append(fingerprint(path));path.unlink()
        assert removed;result['removed_embeddings']=removed
        for name in ('A','B'):
            references=[run('reference-'+name,'reference-'+name+'-'+str(index)) for index in range(2)]
            expected=prepared['answers'][name]
            for index,ref in enumerate(references):
                answer=ref['answer'];assert answer['tokens']==expected['tokens']
                exact(ref['prefix'],prepared['prefix'],name+'-prefix-'+str(index))
                exact(ref['suffix'],expected['suffix'],name+'-suffix-'+str(index))
                assert len(answer['frontiers'])==len(expected['frontiers'])
                for i,(a,b) in enumerate(zip(answer['frontiers'],expected['frontiers'])):exact(a,b,f'{name}-reference-{index}-output-{i+1}')
                controller_exact(answer['controller'],expected['controller'],name+'-reference-controller-'+str(index))
                if name=='A':
                    actual,prior=ref['test_scope'],prepared['test_scope'];assert actual['tokens']==prior['tokens']
                    assert len(actual['frontiers'])==len(prior['frontiers'])
                    for i,(a,b) in enumerate(zip(actual['frontiers'],prior['frontiers'])):exact(a,b,f'TEST-reference-{index}-output-{i+1}')
                    controller_exact(actual['controller'],prior['controller'],'TEST-reference-controller-'+str(index))
        resumed=run('resume','resume');exact(prepared['prefix'],resumed['restored_frontier'],'restored-prefix')
        for name in ('A','B'):
            expected,actual=prepared['answers'][name],resumed['answers'][name]
            assert actual['tokens']==expected['tokens'];exact(expected['suffix'],actual['suffix'],name+'-restored-suffix')
            assert len(actual['frontiers'])==len(expected['frontiers'])
            for i,(a,b) in enumerate(zip(expected['frontiers'],actual['frontiers'])):exact(a,b,f'{name}-restored-output-{i+1}')
            controller_exact(expected['controller'],actual['controller'],name+'-restored-controller')
        restored=run('resume-generated','resume-generated');expected=prepared['answers']['A'];actual=restored['answer']
        assert actual['tokens']==expected['tokens'][4:]
        exact(expected['frontiers'][3],restored['restored_frontier'],'restored-generated-four')
        assert len(actual['frontiers'])==len(expected['frontiers'][4:])
        for i,(a,b) in enumerate(zip(expected['frontiers'][4:],actual['frontiers'])):exact(a,b,'generated-restored-output-'+str(i+5))
        controller_exact(expected['controller'],actual['controller'],'generated-restored-controller')
        for entry in prepared['saved_files']:assert fingerprint(entry['path'])==entry,'Saved checkpoint bytes changed'
        result.update(status='PASS',max_abs_error=0.0,sibling_progress_unchanged=prepared['sibling_progress_unchanged'])
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker');parser.add_argument('--config',type=Path)
    parser.add_argument('--fixture-config',type=Path);parser.add_argument('--limit',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.worker:worker(json.loads(args.config.read_text()),args.worker,args.output)
    else:campaign(args)
