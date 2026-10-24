"""Bounded real-model branch/restore probe against an independent fresh oracle.

This is an initial adapter probe, not certification of every G1-G5 requirement.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parents[1])]
from context_time_travel.real import BranchingModel
from context_time_travel.oracle import FreshOracle
from context_time_travel.store import IntegrityError
from context_time_travel.numerics import assert_logits_close

def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();profile=json.loads(a.profile.read_text());a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    started=time.monotonic();backend=None;oracle=None;checks=[];reports=[]
    # Freeze tolerances before observing any candidate output. Do not widen on failure.
    atol,rtol=.01,.001
    cpu=profile['hardware'].get('backend')=='CPU';admission=None
    if not cpu:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        selected=profile['execution']['GPU'];assert len(selected)==1
        admission=next(line for line in raw.splitlines() if int(line.split(',')[0])==selected[0])
        if int(admission.split(',')[2])<24576:raise RuntimeError('Probe requires 24 GiB free including 8 GiB reserve')
    save(a.output/'intent.json',dict(schema=1,purpose='Independent dense MiniCPM branch, checkpoint and eviction correctness',
        profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),model_sha256=profile['model_weights_sha256'],
        atol=atol,rtol=rtol,admission=admission,cpu_only=cpu,weights_updated=False,promoted=False))
    os.environ.update(profile['execution']['environment'])
    try:
        oracle=FreshOracle(profile)
        prefix=tuple(oracle.tokenize('A saved context has two independent futures. The next answer is'))
        # Use genuine tokenizer IDs; the forced alternatives cannot accidentally be identical.
        aa=tuple(oracle.tokenize(' red'));bb=tuple(oracle.tokenize(' blue'))
        assert aa!=bb and prefix and len(prefix)+len(aa)+len(bb)<profile['execution']['capacity']
        from context_time_travel.execution_receipt import tokens_digest
        def fresh_split(ids,sizes):
            cursor=0;chunks=[]
            for size in sizes:chunks.append([cursor,size]);cursor+=size
            return oracle.forward_schedule(ids,[dict(operator='batched-prefill',position=0,count=len(ids),
                tokens_sha256=tokens_digest(ids),chunks=chunks)])
        combined={'p':oracle.forward(prefix),'a':oracle.forward(prefix+aa),'b':oracle.forward(prefix+bb)}
        expected={'p':combined['p'],'a':fresh_split(prefix+aa,[len(prefix),len(aa)]),
                  'b':fresh_split(prefix+bb,[len(prefix),len(bb)])}
        batching_drift={name:max(abs(x-y) for x,y in zip(combined[name],expected[name])) for name in ('a','b')}
        again=oracle.forward(prefix);assert again==expected['p'];checks.append('fresh_oracle_repeated_full_work')
        backend=BranchingModel.load_backend(profile);sut=BranchingModel(a.output/'store',backend,profile)
        root=sut.create(prefix);original=sut.snapshot_bytes(root)
        def check(label,snapshot,history,reference):
            assert sut.history(snapshot)==history and sut.consumed(snapshot)==len(history)
            assert sut.position_max(snapshot)==len(history)-1
            result=assert_logits_close(sut.next_logits(snapshot),reference,atol=atol,rtol=rtol)
            reports.append(dict(label=label,maximum_error=max(abs(x-y) for x,y in zip(sut.next_logits(snapshot),reference))))
            checks.append(label)
        check('root_frontier',root,prefix,expected['p'])
        left=sut.feed(root,aa);right=sut.feed(root,bb)
        check('independent_suffix_a',left,prefix+aa,expected['a'])
        check('independent_suffix_b',right,prefix+bb,expected['b'])
        assert sut.snapshot_bytes(root)==original;checks.append('parent_immutable')
        wrong=oracle.forward(prefix+aa+bb)
        try:assert_logits_close(sut.next_logits(right),wrong,atol=atol,rtol=rtol)
        except AssertionError:checks.append('wrong_sibling_history_detected')
        else:raise AssertionError('Fixture failed to distinguish contaminated sibling history')
        sut.checkpoint(right);sut.cache.clear();sut.restore_checkpoint(right)
        check('disk_checkpoint_restore',right,prefix+bb,expected['b'])
        saved=sut.cache[right];bad=replace(saved,execution_profile_id='wrong-execution-profile')
        sut.cache[right]=bad
        try:
            try:sut.next_logits(right)
            except IntegrityError:checks.append('mismatched_execution_profile_rejected')
            else:raise AssertionError('Mismatched execution profile was accepted')
        finally:sut.cache[right]=saved
        sut.evict_all()
        try:sut.next_logits(right)
        except IntegrityError:checks.append('evicted_exact_computation_not_silently_replayed')
        else:raise AssertionError('Evicted exact computation was silently replayed')
        reconstruction=sut.rebuild_materialization(right)
        assert reconstruction['prior_computation_preserved'] is False
        check('explicit_reconstruction_after_full_cache_eviction',right,prefix+bb,combined['b'])
        sut.rebuild_materialization(root)
        check('rewind_to_original_root',root,prefix,expected['p'])
        assert sut.snapshot_bytes(root)==original
        result=dict(schema=1,passed=True,checks=checks,logit_comparisons=reports,
            oracle_requests=oracle.requests,oracle_tokens_computed=oracle.tokens_computed,
            backend_counters=backend.work_counters(),capabilities=backend.capabilities().to_json(),
            batching_drift_fresh_contexts=batching_drift,reconstruction=reconstruction,
            seconds=time.monotonic()-started,weights_updated=False,promoted=False,
            certification='Bounded initial adapter probe; full G1-G5 matrix not claimed')
        save(a.output/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException as e:
        save(a.output/'failure.json',dict(type=type(e).__name__,message=str(e),completed_checks=checks,
            seconds=time.monotonic()-started,weights_updated=False));raise
    finally:
        if backend is not None:backend.close()
        if oracle is not None:oracle.close()

if __name__=='__main__':main()
