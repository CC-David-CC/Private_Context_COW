"""Diagnose retained first-generated-token frontiers without running inference."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    values = array.array('f')
    values.frombytes(path.read_bytes())
    if sys.byteorder != 'little': values.byteswap()
    if len(values)!=248320 or not all(map(math.isfinite,values)):
        raise ValueError('Incomplete/nonfinite frontier')
    return values


def compare(a,b):
    maximum = max(abs(x-y) for x,y in zip(a,b))
    index = max(range(len(a)),key=lambda i:abs(a[i]-b[i]))
    return dict(max_absolute_error=maximum,worst_logit=index,actual=a[index],reference=b[index],
                argmax_actual=max(range(len(a)),key=a.__getitem__),
                argmax_reference=max(range(len(b)),key=b.__getitem__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    run = args.run
    resumed = json.loads((run/'resume.json').read_text())
    measured = json.loads((run/'compare.json').read_text())
    branch = resumed['branches']['A']
    candidate = next(s for s in branch['states'] if s['frontier']=='resumed-A-generated-1.f32')
    suffix = next(r for r in measured['comparisons'] if r['frontier']=='resumed-A-suffix.f32')
    normal = suffix['oracle_work']
    if normal['work']['generated'] != branch['generated'][:1]:
        raise ValueError('Normal reference did not consume the same generated token')
    candidate_path = run/candidate['frontier']
    normal_path = run/(normal['frontier']+'.decode.f32')
    full_path = run/('compare-oracle-'+str(len(measured['comparisons']))+'.f32')
    token_path = full_path.with_suffix('.tokens')
    if [int(t) for t in token_path.read_text().split(',')] != candidate['history']:
        raise ValueError('Fresh batch did not read the same complete history')
    a,b,c = (read(p) for p in (candidate_path,normal_path,full_path))
    result = dict(status='BLOCKED_EXECUTION_EQUIVALENCE',scope='first output, suffix A; retained actual GPU data',
        inputs=candidate['history'],profile_sha256=hashlib.sha256((run/'profile.json').read_bytes()).hexdigest(),
        restored_vs_uncached_normal_decode=compare(a,b),restored_vs_uncached_full_batched=compare(a,c),
        uncached_normal_vs_uncached_full_batched=compare(b,c),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (candidate_path,normal_path,full_path,token_path)})
    (run/'frontier-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
