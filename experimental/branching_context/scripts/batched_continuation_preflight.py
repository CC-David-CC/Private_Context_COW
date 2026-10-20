"""Real fresh-process batch continuation comparison, without checkpoint imports."""
import argparse
import json
import os
from pathlib import Path
import time

from durable_resume import BatchedOracle, numerical_record, publish, vector
from context_time_travel.numerics import compare_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    profile = json.loads(args.profile.read_text())
    if profile['execution']['generated_operator'] != 'batched-prefill':
        raise ValueError('Batches-only pinned profile required')
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    oracle = BatchedOracle(profile,args.output,'batch-preflight')
    history = [100+i%17 for i in range(128)]+[117,118,119,120]
    report = dict(status='RUNNING',scope='uncached batches-only first-output primitive; not restart qualification',
                  history=history,observations=[],started=time.time())
    publish(args.output/'result.json',report)
    try:
        oracle.forward(history)
        first = oracle.last.copy()
        token = first['work']['generated'][0]
        actual = vector(args.output/(first['frontier']+'.decode.f32'))
        report['observations'].append(first)
        publish(args.output/'result.json',report)
        references = []
        for _ in range(2):
            references.append(oracle.forward(history+[token]))
            report['observations'].append(oracle.last.copy())
            publish(args.output/'result.json',report)
        baseline = compare_logits(*references,atol=0,rtol=0)
        bound = 2*baseline.max_absolute_error
        comparison = compare_logits(actual,references[0],atol=bound,rtol=0)
        report.update(token=token,atol=bound,rtol=0,
            fresh_repeat=numerical_record(baseline),batch_continuation_vs_full_history=numerical_record(comparison),
            status='PASS' if comparison.pass_elementwise else 'FAIL')
        if not comparison.pass_elementwise: raise AssertionError('Batched continuation differs from fresh batch')
    except BaseException as error:
        report.update(status='FAIL',error=str(error)); raise
    finally:
        publish(args.output/'result.json',report)


if __name__ == '__main__': main()
