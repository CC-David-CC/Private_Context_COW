"""Read completed restart evidence into a compact pause receipt; no inference."""
import hashlib
import json
from pathlib import Path

base = Path('/home/dflanag3/branching-g3')
case = base/'digest-reuse-v1-20261014/q8-real-64k-restart'
report = dict(source=str(case),result=json.loads((case/'result.json').read_text()),phases={},raw_sha256={})
for phase in ('prepare','resume'):
    path = case/(phase+'.json');r = json.loads(path.read_text())
    row = {key:r[key] for key in ('status','python_pid','native_pid','native_exit_observed','startup_seconds')}
    row['branches'] = {name:dict(outputs=len(branch['result']['tokens']),
        generation_seconds=branch['result']['generation_seconds'],request_seconds=branch['result']['request_seconds'],
        frontier_sha256=branch['state']['frontier_sha256']) for name,branch in r['branches'].items()}
    for key in ('restore','mid_restore','parent_payload_sha256'):
        if key in r: row[key] = r[key]
    report['phases'][phase] = row
    report['raw_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
events = base/'digest-client-v1-20261014/q8-client-resident-64k/events.jsonl'
counts = {}
if events.exists():
    for line in events.read_text().splitlines():
        event = json.loads(line)['event'];key = event['event'];counts[key] = counts.get(key,0)+1
report['interrupted_client_events'] = counts
report['interrupted_client_worker'] = json.loads((base/'digest-client-v1-20261014/worker.json').read_text())
path = base/'native-request-probe-20261014/pause-restart-summary.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['result']['status'],comparisons=report['result']['comparisons'],
    restore=report['phases']['resume']['restore'],client_events=counts)))
