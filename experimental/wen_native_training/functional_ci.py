"""CI acceptance from complete BW task verdicts, never internal score parity."""
import argparse
import json
from pathlib import Path


def index_results(rows):
    result = {}
    for row in rows:
        key = (row['id'],row['guided'],row['distance'])
        if (type(row['guided']) is not bool or type(row['distance']) is not int
                or type(row['verdict']['passed']) is not bool or key in result):
            raise ValueError('Unique complete functional verdicts required')
        result[key] = row['verdict']['passed']
    if not result:
        raise ValueError('Empty functional suite cannot pass')
    return result


def compare(baseline,candidate):
    before, after = index_results(baseline), index_results(candidate)
    if set(before) != set(after):
        raise ValueError('Baseline and candidate must cover the same complete suite')
    def keys(predicate):
        return [dict(id=key[0],guided=key[1],distance=key[2]) for key in sorted(before) if predicate(key)]
    return dict(status='PASS' if all(after.values()) else 'FAIL',total=len(after),
                baseline_passed=sum(before.values()),candidate_passed=sum(after.values()),
                repaired=keys(lambda key:not before[key] and after[key]),
                regressed=keys(lambda key:before[key] and not after[key]),
                still_failing=keys(lambda key:not before[key] and not after[key]),
                acceptance='complete native BW task verdicts and actual tool/file effects',
                internal_math_used_for_acceptance=False,base_distribution_preservation=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    receipt = compare(json.loads(args.baseline.read_text(encoding='utf-8')),
                      json.loads(args.candidate.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('repaired','regressed','still_failing')}))
    raise SystemExit(0 if receipt['status']=='PASS' else 1)


if __name__=='__main__': main()
