"""Compact executed evidence; raw histories/controllers remain in the private archive."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def state_summary(state):
    controller = state['controller']
    return dict(snapshot=state['snapshot'],consumed=state['consumed'],history_sha256=digest(state['history']),
        controller_sha256=digest(controller),controller_status=controller['status'],
        rng_draw_cursor=controller.get('next_draw'),accepted_in_scope=controller.get('accepted_in_scope'),
        frontier_sha256=state['frontier_sha256'],schedule_sha256=state['schedule_sha256'])


def summarize(root):
    result_path = root/'result.json'
    result = json.loads(result_path.read_text())
    if result['status'] != 'PASS': raise ValueError('Summarize completed PASS cases only; preserve failures separately')
    if not all(p['exit_code'] == 0 and p['native_exit_observed'] for p in result['phases']):
        raise ValueError('Missing process-exit evidence')
    report = dict(result=result,source=str(root),raw_sha256={'result.json':sha(result_path)},phases={})
    for phase in ('prepare','resume'):
        path = root/(phase+'.json')
        raw = json.loads(path.read_text())
        if raw['status'] != 'PASS' or not raw['native_exit_observed']:
            raise ValueError('Incomplete native stage')
        report['raw_sha256'][path.name] = sha(path)
        row = dict(python_pid=raw['python_pid'],native_pid=raw['native_pid'],
            startup_seconds=raw['startup_seconds'],counters=raw['counters'],branches={})
        for name,branch in raw['branches'].items():
            measured = branch['result']
            count = len(measured['tokens'])
            row['branches'][name] = dict(outputs=count,tokens_sha256=digest(measured['tokens']),
                generation_seconds=measured['generation_seconds'],request_seconds=measured['request_seconds'],
                generation_tps=count/measured['generation_seconds'],request_tps=count/measured['request_seconds'],
                state=state_summary(branch['state']))
            if 'save_and_audit_seconds' in branch:
                row['branches'][name]['save_and_audit_seconds'] = branch['save_and_audit_seconds']
        row['checked_generated_boundaries'] = {name:list(map(int,checks)) for name,checks in raw['boundary_checks'].items()}
        if phase == 'prepare':
            row.update(prefix_seconds=raw['prefix_seconds'],root=state_summary(raw['root']),
                       midpoint=state_summary(raw['midpoint']),parent_payload_sha256=raw['parent_payload_sha256'])
        else:
            for name in ('restore','mid_restore'):
                row[name] = raw[name]
            row['generated_resume'] = dict(outputs=len(raw['generated_resume']['result']['tokens']),
                state=state_summary(raw['generated_resume']['state']))
        report['phases'][phase] = row
    report['timing_scope'] = 'one measured request per branch; request excludes startup and later disk-save/audit acknowledgment'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = summarize(args.case)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='SUMMARIZED',case=str(args.case),
        outputs={k:v['outputs'] for k,v in report['phases']['resume']['branches'].items()},
        generation_tps={k:v['generation_tps'] for k,v in report['phases']['resume']['branches'].items()},
        request_tps={k:v['request_tps'] for k,v in report['phases']['resume']['branches'].items()},
        restore_seconds=report['phases']['resume']['restore']['seconds'],
        restored_native_bytes=report['phases']['resume']['restore']['counter_deltas']['loaded_file_bytes'])))


if __name__ == '__main__': main()
