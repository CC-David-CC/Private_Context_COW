"""Measure independent Prefill and native COW without the canonical store loop.

Diagnostic layers only: this does not replace complete API correctness tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time


def frontier_hash(values):
    return hashlib.sha256(struct.pack('<'+'f'*len(values),*values)).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',choices=['direct','native-cow'],required=True)
    a = p.parse_args()
    sys.path.insert(0,str(a.api/'src'))
    sys.path.insert(0,str(a.api/'scripts'))
    from context_time_travel.real import BranchingModel
    from durable_resume import BatchedOracle
    read = lambda n:json.loads((a.campaign/n).read_text())
    if read('result.json')['status'] != 'PASS' or read('compare.json')['status'] != 'PASS':
        raise RuntimeError('Require executed matching-schedule validation')
    profile,fixture,reference,prepared = [read(n+'.json') for n in ('profile','fixture','resume','prepare')]
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    a.output.mkdir(parents=True,exist_ok=False)
    result = dict(status='RUNNING',layer=a.layer,branches={},execution_profile_id=profile['execution_profile_id'],
        controller_scope='No controller continuity qualification at these diagnostic layers')
    backend = None
    try:
        if a.layer == 'direct':
            oracle = BatchedOracle(profile,a.output,'direct-speed')
            for name in ('A','B'):
                count = len(reference['branches'][name]['generated'])
                tokens,frontiers = oracle.continuation(fixture['prefix']+fixture[name],
                    [len(fixture['prefix']),len(fixture[name])],count)
                if tokens != reference['branches'][name]['generated']: raise AssertionError('Direct tokens differ')
                if frontier_hash(frontiers[-1]) != reference['branches'][name]['states'][-1]['frontier_sha256']:
                    raise AssertionError('Direct frontier differs')
                work = oracle.last['work']
                result['branches'][name] = dict(native_work=work,process_seconds=oracle.last['seconds'],
                    output_tokens=len(tokens),tokens_per_second=len(tokens)/work['decode_seconds'],
                    exact_validation='PASS',scope='Native compute-loop wall time includes frontier audit output')
            oracle.close()
        else:
            started = time.monotonic()
            backend = BranchingModel.load_backend(profile)
            result['startup_seconds'] = time.monotonic()-started
            with backend.lease() as lease:
                started = time.monotonic()
                backend.consume(lease,tuple(fixture['prefix']))
                result['prefix_seconds'] = time.monotonic()-started
                root = backend.export(lease)
                frontier = backend.logits(lease)
                if frontier_hash(frontier) != prepared['root']['frontier_sha256']:
                    raise AssertionError('Native COW prefix differs')
                original = hashlib.sha256(backend.serialize(lease,root)).hexdigest()
            for name in ('A','B'):
                before = backend.work_counters()
                with backend.lease() as lease:
                    backend.restore(lease,root,history=fixture['prefix'],frontier=frontier)
                    restored = backend.work_counters()
                    if restored['consumed_tokens'] != before['consumed_tokens']:
                        raise AssertionError('Resident native COW recomputed prefix')
                    started = time.monotonic()
                    backend.consume(lease,tuple(fixture[name]))
                    suffix_seconds = time.monotonic()-started
                    image = backend.export(lease)
                    times = dict(consume_seconds=0.,capture_seconds=0.,frontier_seconds=0.)
                    tokens = reference['branches'][name]['generated']
                    started_all = time.monotonic()
                    for token in tokens:
                        start = time.monotonic(); backend.consume_generated(lease,(token,))
                        times['consume_seconds'] += time.monotonic()-start
                        start = time.monotonic(); image = backend.export(lease)
                        times['capture_seconds'] += time.monotonic()-start
                        start = time.monotonic(); final = backend.logits(lease)
                        times['frontier_seconds'] += time.monotonic()-start
                    elapsed = time.monotonic()-started_all
                    if frontier_hash(final) != reference['branches'][name]['states'][-1]['frontier_sha256']:
                        raise AssertionError('Forced native COW continuation differs')
                after = backend.work_counters()
                count = len(fixture[name])+len(tokens)
                if after['consumed_tokens']-restored['consumed_tokens'] != count:
                    raise AssertionError('Unexpected native COW work')
                result['branches'][name] = dict(**times,suffix_seconds=suffix_seconds,
                    generation_seconds=elapsed,output_tokens=len(tokens),tokens_per_second=len(tokens)/elapsed,
                    counter_deltas={k:v-before.get(k,0) for k,v in after.items() if type(v) is int},exact_validation='PASS')
            with backend.lease() as lease:
                backend.restore(lease,root,history=fixture['prefix'],frontier=frontier)
                if hashlib.sha256(backend.serialize(lease,root)).hexdigest() != original:
                    raise AssertionError('Saved parent mutated')
            result['parent_byte_audit'] = 'PASS'
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL',error=str(error)); raise
    finally:
        if backend:
            backend.close()
            result['native_exit_observed'] = backend.process.poll() is not None
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
