"""One resident multi-token GBNF request per image branch; real GPU audit.

Companion to run_vision_grammar_probe. Reuses its immutable disk prefix and
independently validated answers; does not re-encode or prefill the saved prefix.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend
from run_vision_grammar_probe import engine_for
from run_vision_ingest_probe import compare_files, fingerprint, save
from run_vision_restart_probe import audit_unchanged, frontier


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());prepared=json.loads(Path(config['prepare_result']).read_text())
    profile=json.loads(Path(config['profile']).read_text());engine=engine_for(profile)
    result=dict(status='RUNNING',python_pid=os.getpid(),comparisons=[],branches={},encoder_calls=0,
        source_campaign=fingerprint(args.config.parent/'result.json'),profile=fingerprint(config['profile']))
    backend=None;started=time.monotonic()
    try:
        os.environ['CTT_NATIVE_TRACE']=str(out/'native-trace.jsonl')
        backend=StrataBackend(profile,gpu_id=0);result['native_pid']=backend.process.pid
        api=BranchingModel(config['store'],backend,profile)
        forbidden={str(Path(config['fixture']).resolve()),*(str(Path(path).resolve()) for path in config['images'])}
        def audit(event,args):
            if event=='open' and isinstance(args[0],(str,bytes)):
                path=os.fsdecode(args[0])
                if path.endswith('.sve') or str(Path(path).resolve()) in forbidden:
                    raise AssertionError('Resident restore read an image/embedding: '+path)
            if event=='subprocess.Popen':raise AssertionError('Resident request spawned another process')
        sys.addaudithook(audit)
        parent=prepared['prefix_snapshot'];result['restore']=api.restore_checkpoint(parent)
        assert result['restore']['forward_tokens']==0
        result['native_vision']=backend._request('vision_info')
        assert result['native_vision']['validated_embedding_bytes']==0
        def exact(left,right,label):
            entry=dict(label=label,**compare_files(left,right));result['comparisons'].append(entry)
            assert entry['exact'],entry
        exact(prepared['prefix'],frontier(api,parent,out/'prefix.f32'),'restored-prefix')
        parent_audit=backend.audit(api.cache[parent].blob);result['parent_audits']=[]
        for name in ('A','B'):
            source_objects=set(api.store.objects.iterdir());progress=[]
            def provisional(token):
                assert set(api.store.objects.iterdir())==source_objects,'Streamed progress published a snapshot too soon'
                progress.append(token)
            before=backend.work_counters()
            answer=api.run_state_request(parent,config['suffixes'][name],config['limit'],
                engine=engine,state='EDIT',stream='vision-edit',on_token=provisional)
            after=backend.work_counters();deltas={key:after[key]-before[key] for key in after
                if type(after[key]) is int and type(before.get(key)) is int}
            expected=prepared['answers'][name]
            assert answer['tokens']==expected['tokens']==progress
            assert answer['stopped'] and answer['grammar_complete']
            assert api.controller(answer['snapshot'])==expected['controller']
            assert deltas['forward_tokens']==len(config['suffixes'][name])+len(answer['tokens'])
            assert deltas['captures']==1 and deltas['restores']==1
            exact(expected['frontiers'][-1],frontier(api,answer['snapshot'],out/(name+'-final.f32')),name+'-resident-final')
            audit_unchanged(backend,api.cache[parent].blob,parent_audit,result['parent_audits'])
            result['branches'][name]=dict(tokens=answer['tokens'],snapshot=answer['snapshot'],controller_exact=True,
                deltas=deltas,generation_seconds=answer['generation_seconds'],request_seconds=answer['request_seconds'])
            save(out/'result.json',result)
        for item in prepared['saved_files']:assert fingerprint(item['path'])==item
        result.update(status='PASS',work=backend.work_counters(),max_abs_error=0.)
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:
        if backend is not None:
            pid=backend.process.pid;backend.close()
            result['native_exit']=dict(pid=pid,exit_code=backend.process.returncode,process_absent=not Path(f'/proc/{pid}').exists())
        result['seconds']=time.monotonic()-started;save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    campaign(parser.parse_args())
