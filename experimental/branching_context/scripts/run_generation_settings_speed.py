"""Resident generation smoke: reuse an audited question, up to 512 outputs.

Only the already-audited leading frontier/tokens are independently compared.
Remaining outputs are a performance smoke, not full-vector qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('runtime-root','profile','audit-dir','run-dir'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--outputs',type=int,default=512)
    args=parser.parse_args()
    if not 1<=args.outputs<=512:raise ValueError('Output limit must be in [1,512]')
    runtime=args.runtime_root.resolve(strict=True)
    sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),
                  str(runtime/'experimental/branching_context/scripts')]
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend
    from grammar_long_reference import Heartbeat,_apply_profile_environment
    from run_grammar_first_probe import publish,finish_backend
    from run_min_p_probe import check_delta,frontier_bytes
    from run_generation_settings_probe import DEFAULT_POLICIES,STREAM
    os.umask(0o077)
    output=args.run_dir.absolute();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    audit=args.audit_dir.resolve(strict=True)
    campaign=read(audit/'campaign.json')
    assert campaign['status']=='PASS' and campaign['profile_sha256']==sha(args.profile)
    profile=read(args.profile);_apply_profile_environment(profile)
    trace=output/'native-trace.jsonl';os.environ['CTT_NATIVE_TRACE']=str(trace)
    heartbeat=Heartbeat(output/'heartbeat.json').start()
    result=dict(status='RUNNING',scope=__doc__,script_sha256=sha(__file__),
        profile_sha256=sha(args.profile),audit_campaign_sha256=sha(audit/'campaign.json'),
        python_pid=os.getpid(),rows=[],failures=[])
    backend=None
    publish(output/'result.json',result)
    try:
        heartbeat.set('native-start')
        started=time.monotonic();backend=StrataBackend(profile)
        result.update(native_pid=backend.process.pid,startup_seconds=time.monotonic()-started)
        model=BranchingModel(audit/'prepare-shared/store',backend,profile)
        question=read(audit/'prepare-shared/result.json')['question']
        result['restore_receipt']=model.restore_checkpoint(question)
        assert result['restore_receipt']['forward_tokens']==0
        parent=model.snapshot_bytes(question)
        parent_image=model.cache[question].blob
        parent_hash=backend.audit(parent_image)['sha256']
        manifest=model.validate_checkpoint(question)
        for policy in DEFAULT_POLICIES:
            if policy['name'] not in ('temperature-zero','temperature-low','temperature-high'):continue
            name=policy['name'];reference=read(audit/('reference-'+name)/'result.json')
            heartbeat.set(name,backend.process.pid)
            consumed=[];samples={};generated=0
            original=backend.consume_generated
            ncheck=min(args.outputs,len(reference['tokens']))
            def consume(lease,ids):
                nonlocal generated
                original(lease,ids);generated+=len(ids)
                if generated==ncheck:
                    raw=frontier_bytes(backend.logits(lease),backend.vocab_size)
                    samples['generated-'+str(generated)]=hashlib.sha256(raw).hexdigest()
            def captures():
                return sum(read_line.get('command')=='capture' for read_line in
                           (json.loads(line) for line in trace.read_text().splitlines()))
            before=backend.work_counters();captures_before=captures()
            backend.consume_generated=consume
            started=time.monotonic()
            try:
                answer=model.run_sampled_request(question,(),args.outputs,sampling=policy['sampling'],
                    stream=STREAM,resume=True,on_token=lambda token:consumed.append(time.monotonic()))
            finally:backend.consume_generated=original
            elapsed=time.monotonic()-started
            after=backend.work_counters();count=len(answer['tokens'])
            check_delta(before,after,0,count,vocab=backend.vocab_size)
            assert after['loaded_file_bytes']==before['loaded_file_bytes']
            assert captures()-captures_before==1,'Expected one capture for the complete generation'
            assert answer['tokens'][:ncheck]==reference['tokens'][:ncheck]
            expected=reference['samples']['generated-'+str(ncheck)]
            assert samples['generated-'+str(ncheck)]==(expected['sha256'] if isinstance(expected,dict) else expected)
            state=model.controller(answer['snapshot'])
            assert state['next_draw']==(0 if policy['sampling']['temperature']==0 else count)
            final=frontier_bytes(model.cache[answer['snapshot']].frontier,backend.vocab_size)
            (output/(name+'-final.f32')).write_bytes(final)
            intervals=[b-a for a,b in zip(consumed,consumed[1:])]
            row=dict(name=name,policy=policy['sampling'],output_limit=args.outputs,output_count=count,
                stopped=state['status']=='stopped',generation_seconds=answer['generation_seconds'],
                generation_tps=count/answer['generation_seconds'],request_seconds=elapsed,
                first_consumed_seconds=consumed[0]-started,median_inter_consumed_seconds=statistics.median(intervals) if intervals else None,
                captures=1,counters_before=before,counters_after=after,
                verified_leading_token_count=ncheck,verified_leading_frontier_sha256=samples['generated-'+str(ncheck)],
                remaining_outputs_scope='Performance smoke; no fresh full-frontier oracle after the checked leading frontier',
                final_frontier_sha256=hashlib.sha256(final).hexdigest())
            result['rows'].append(row);publish(output/'result.json',result)
        assert model.snapshot_bytes(question)==parent
        assert backend.audit(parent_image)['sha256']==parent_hash
        assert model.validate_checkpoint(question)==manifest
        memory={}
        for line in Path('/proc',str(backend.process.pid),'status').read_text().splitlines():
            if line.split(':',1)[0] in ('VmRSS','VmHWM'):memory[line.split(':',1)[0]]=line.split(':',1)[1].strip()
        result['native_process_memory']=memory
        result['end_gpu_memory_sample']=subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        result.update(parent_unchanged=True,status='PASS')
    except BaseException as exc:
        result['status']='FAIL';result['failures'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if backend is not None:finish_backend(backend,result)
        heartbeat.close();publish(output/'result.json',result)


if __name__=='__main__':main()
