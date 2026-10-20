"""Read incremental campaign records without printing full controllers/logits."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    args = parser.parse_args()
    report = dict(case=str(args.case),qualification='PENDING unless terminal result is PASS',phases={})
    for name in ('result','prepare','resume'):
        path = args.case/(name+'.json')
        if not path.exists(): continue
        raw = json.loads(path.read_text())
        row = {key:raw[key] for key in ('status','error','python_pid','native_pid','native_exit_observed') if key in raw}
        row['branches'] = {}
        for branch,value in raw.get('branches',{}).items():
            measured = value['result'];count = len(measured['tokens'])
            row['branches'][branch] = dict(outputs=count,
                generation_seconds=measured['generation_seconds'],request_seconds=measured['request_seconds'],
                generation_tps=count/measured['generation_seconds'],request_tps=count/measured['request_seconds'])
        row['checked_boundaries'] = {key:list(values) for key,values in raw.get('boundary_checks',{}).items()}
        report['phases'][name] = row
    report['completed_fresh_native_runs'] = {}
    for path in sorted(args.case.glob('fresh-*.stdout')):
        if not path.stat().st_size: continue
        try: raw = json.loads(path.read_text())
        except json.JSONDecodeError: continue
        report['completed_fresh_native_runs'][path.stem] = {
            key:value for key,value in raw.items() if isinstance(value,(str,int,float,bool))}
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__': main()
