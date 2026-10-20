"""Real lifecycle cycles, retained counters and direct process memory samples.

Bounded CLI runs are development evidence. Registered long soak additionally
requires the fixed 24-hour/64-cycle protocol; no reduced gate-count option.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import struct
import sys
import time

from lifecycle_evidence import summarize

LONG_SECONDS=86400
LONG_CYCLES=64
WARMUP_CYCLES=8


def parse_memory(status, gpu_rows, pid):
    match=re.search(r'^VmRSS:\s+(\d+) kB$',status,re.MULTILINE)
    if match is None:raise ValueError('Native RSS instrumentation absent')
    memory=[]
    for line in gpu_rows.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==2 and fields[0]==str(pid):
            if not fields[1].isdigit():raise ValueError('GPU process memory instrumentation unavailable')
            memory.append(int(fields[1])*1024*1024)
    if not memory:raise ValueError('Native GPU PID absent from process memory measurement')
    return dict(rss_bytes=int(match.group(1))*1024,gpu_bytes=sum(memory))


def process_memory(pid):
    status=Path('/proc')/str(pid)/'status'
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                  '--format=csv,noheader,nounits'],text=True)
    return parse_memory(status.read_text(),rows,pid)


def qualifies_long_soak(report):
    """Duration/coverage policy only; run_soak supplies actual execution evidence."""
    return (report.get('status')=='MEASURED' and type(report.get('cycles')) is int
        and report['cycles']>=LONG_CYCLES and report.get('seconds',0)>=LONG_SECONDS
        and report.get('warmup_cycles')==WARMUP_CYCLES
        and report.get('numerical_comparisons',0)>=report['cycles']*4
        and math.isfinite(report.get('max_absolute_error',float('nan')))
        and report.get('summary',{}).get('status')=='MEASURED'
        and type(report.get('summary',{}).get('frozen_host_heap_bound')) is int)


def run_soak(profile, directory, *, minimum_cycles=10, minimum_seconds=0):
    from context_time_travel.real import BranchingModel,FreshOracle
    from context_time_travel.numerics import assert_logits_close
    if profile.get('calibration_status')!='FROZEN':raise ValueError('Frozen real profile required')
    if type(minimum_cycles) is not int or minimum_cycles<=WARMUP_CYCLES:
        raise ValueError('Need cycles beyond fixed eight-cycle warmup')
    if type(minimum_seconds) not in (int,float) or not math.isfinite(minimum_seconds) or minimum_seconds<0:
        raise ValueError('Nonnegative duration required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    report=dict(schema=1,status='RUNNING',cycles=0,seconds=0,warmup_cycles=WARMUP_CYCLES,
        requested_minimum_cycles=minimum_cycles,requested_minimum_seconds=minimum_seconds,
        hardware=profile['hardware'],model_contract_id=profile['model_contract_id'],
        execution_profile_id=profile['execution_profile_id'],native_build=profile['native_build'],
        numerical_comparisons=0,max_absolute_error=0,rows=[],checks=[])
    backend=oracle=sut=None;started=time.monotonic()
    def save():
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def check(snapshot,history):
        expected=oracle.forward(history)
        assert oracle.last['computed_tokens']==len(history)
        actual=sut.next_logits(snapshot)
        try:result=assert_logits_close(actual,expected,atol=profile['atol'],rtol=profile['rtol'])
        except BaseException:
            for name,values in [('actual',actual),('expected',expected)]:
                (directory/(name+'.f32')).write_bytes(struct.pack('<'+str(len(values))+'f',*values))
            raise
        assert sut.history(snapshot)==history and sut.consumed(snapshot)==len(history)
        report['max_absolute_error']=max(report['max_absolute_error'],result.max_absolute_error)
        report['numerical_comparisons']+=1
        report['checks'].append(dict(snapshot=snapshot,history=history,max_absolute_error=result.max_absolute_error))
    save()
    try:
        backend=BranchingModel.load_backend(profile);oracle=FreshOracle(profile)
        report['native_pid']=backend.process.pid
        prefix=tuple(profile['fixtures']['prefix'][:9])
        tokens=[profile['fixtures'][name][0] for name in ('suffix_a','suffix_b')]
        assert tokens[0]!=tokens[1]
        baseline=dict(backend._refresh_stats())
        while report['cycles']<minimum_cycles or time.monotonic()-started<minimum_seconds:
            cycle=report['cycles'];cycle_started=time.monotonic()
            sut=BranchingModel(directory/('cycle-'+str(cycle).zfill(6)),backend,profile)
            before=dict(backend._refresh_stats())
            root=sut.create(prefix);original=sut.snapshot_bytes(root);check(root,prefix)
            for index,token in enumerate(tokens):
                scope=sut.start_generation(root,dict(allowed=[token]),'soak-'+str(cycle)+'-'+str(index))
                generated=sut.generate(scope,1)
                assert generated.accepted_tokens==(token,)
                controller=sut.controller(generated.snapshot_id)
                assert controller['next_draw']==1 and controller['accepted_in_scope']==1
                check(generated.snapshot_id,prefix+(token,))
            assert sut.snapshot_bytes(root)==original
            sut.evict_all();gc.collect();backend.release_checkpoints();backend._request('reset')
            replay_before=backend._refresh_stats()['forward_tokens']
            check(generated.snapshot_id,prefix+(tokens[-1],))
            assert backend._refresh_stats()['forward_tokens']-replay_before==len(prefix)+1
            assert sut.controller(generated.snapshot_id)['next_draw']==1
            sut.evict_all();gc.collect();backend.release_checkpoints();backend._request('reset')
            after=dict(backend._refresh_stats())
            assert after['consumed']==0
            assert after['forward_tokens']-before['forward_tokens']==2*len(prefix)+3
            # Stable live ownership must match the original empty backend, not a moving baseline.
            from lifecycle_evidence import LIVE
            assert all(after[k]==baseline[k] for k in LIVE)
            row=dict(before=before,after=after,seconds=time.monotonic()-cycle_started,
                     **process_memory(backend.process.pid))
            report['rows'].append(row);report['cycles']+=1
            report['seconds']=time.monotonic()-started
            if report['cycles']>WARMUP_CYCLES:
                report['summary']=summarize(report['rows'],warmup_cycles=WARMUP_CYCLES)
            save();print('cycle',cycle,'seconds',round(row['seconds'],3),flush=True)
        report.update(status='MEASURED',oracle_computed_tokens=oracle.tokens_computed)
        report['long_soak_qualified']=qualifies_long_soak(report)
        return report
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        active_error=sys.exc_info()[0] is not None
        cleanup_errors=[]
        for action in ([sut.evict_all] if sut is not None else [])+([oracle.close] if oracle is not None else [])+([backend.close] if backend is not None else []):
            try:action()
            except BaseException as error:cleanup_errors.append(repr(error))
        if cleanup_errors:report.update(status='FAIL',cleanup_errors=cleanup_errors,long_soak_qualified=False)
        report['seconds']=time.monotonic()-started;save()
        if cleanup_errors and not active_error:raise RuntimeError('Lifecycle teardown failed: '+repr(cleanup_errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cycles',type=int,default=10,help='Bounded development cycles; not release qualification')
    args=parser.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    report=run_soak(json.loads(args.profile.read_bytes()),args.output,minimum_cycles=args.cycles)
    print(json.dumps({k:report[k] for k in ['status','cycles','seconds','numerical_comparisons','max_absolute_error','long_soak_qualified']},indent=2))


if __name__=='__main__':main()
