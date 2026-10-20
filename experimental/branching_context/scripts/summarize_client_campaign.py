"""Summarize completed client measurements; never certify a partial campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(directory):
    client_path, worker_path = directory/'client.json', directory/'worker.json'
    client = json.loads(client_path.read_bytes())
    worker = json.loads(worker_path.read_bytes())
    digest = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    native_exit_codes = worker.get('native_exit_codes')
    native_exit_success = worker.get('native_exit_success') is True
    numeric_native_exit = (isinstance(native_exit_codes, dict) and bool(native_exit_codes)
                           and all(type(code) is int and code == 0
                                   for code in native_exit_codes.values()))
    if (client['status'] != 'PASS' or worker['status'] != 'PASS'
            or not worker.get('native_exit_observed') or not native_exit_success
            or not numeric_native_exit or client['worker_sha256'] != digest):
        raise ValueError('Require completed matching client/worker records and successful numeric native exit')
    rows = {row['request']: row for row in worker['requests']}
    if len(rows) != len(worker['requests']) or set(rows) != set(client['requests']):
        raise ValueError('Client/worker request identities differ')
    modes = {row['mode'] for row in rows.values()}
    if len(modes) != 1: raise ValueError('Mixed measurement modes')
    mode = modes.pop()
    work = []
    for label, timing in client['requests'].items():
        row = rows[label]
        if (not timing['durable'] or timing['outputs'] != row['outputs']
                or timing['warmup'] != row['warmup'] or timing['mode'] != mode):
            raise ValueError('Incomplete or inconsistent request: '+label)
        delta = row['counter_deltas']
        if delta['loaded_file_bytes'] != 0:
            raise ValueError('Unexpected disk work inside native request')
        restore = row['disk_restore']
        if mode == 'disk' and (restore is None or restore['forward_tokens'] != 0):
            raise ValueError('Missing zero-forward disk restore receipt')
        if mode == 'resident' and restore is not None:
            raise ValueError('Disk restore in resident measurement')
        work.append(dict(request=label, warmup=row['warmup'], outputs=row['outputs'],
                         forward_tokens=delta['forward_tokens'],
                         loaded_file_bytes=delta['loaded_file_bytes'],
                         disk_restore=restore, durable_save_seconds=row['durable_save_seconds']))
    fields = ('time_to_first_token_seconds', 'steady_received_tps',
              'median_tpot_seconds', 'p95_tpot_seconds', 'request_tps', 'durable_request_tps')
    branches = {}
    for branch in ('A', 'B'):
        measured = [r for label,r in client['requests'].items()
                    if label.startswith(branch+'-') and not r['warmup']]
        if len(measured) != 3: raise ValueError('Require three measured requests per branch')
        branches[branch] = dict(measured_requests=len(measured),
            medians={field:statistics.median(r[field] for r in measured) for field in fields})
    return dict(status='PASS', scope=client['scope'], mode=mode,
        profile_id=worker['profile_id'], startup_seconds=worker['startup_seconds'],
        native_exit_observed=worker['native_exit_observed'],
        native_exit_codes=native_exit_codes, native_exit_success=native_exit_success,
        branches=branches, work=work,
        client_sha256=hashlib.sha256(client_path.read_bytes()).hexdigest(), worker_sha256=digest,
        limitations=['Same-profile greedy reference; not cumulative certification',
                     'Repeated deterministic completions may reuse existing durable payloads',
                     'Warm filesystem cache; no cold-storage or HTTP/network qualification'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
