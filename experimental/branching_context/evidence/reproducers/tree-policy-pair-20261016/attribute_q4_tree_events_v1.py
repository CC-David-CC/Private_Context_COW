"""Decompose already captured same-host monotonic events without rerunning."""
import base64
import hashlib
import json
import statistics
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
pair=json.loads((root/'policy-pair-v2.json').read_bytes());arms={}
for arm,data in pair['arms'].items():
    rows=[]
    for generation in data['hot_requests']:
        path=Path(generation['trace_file']);trace=json.loads(path.read_bytes())
        assert len(trace['requests'])==1
        request=trace['requests'][0];events=[]
        assert request['summary']['validation']['status']=='valid'
        for entry in request['events']:
            event=json.loads(base64.b64decode(entry['raw_line_b64'],validate=True))
            events.append((event,entry['arrival_monotonic_ns']))
        started=[entry for entry in events if entry[0].get('event')=='request_started']
        snapshots=[entry for entry in events if entry[0].get('event')=='snapshot']
        durable=[entry for entry in events if entry[0].get('event')=='durable']
        tokens=[entry for entry in events if type(entry[0].get('token_id')) is int]
        assert len(started)==len(snapshots)==len(durable)==1 and len(tokens)==128
        worker_start=round(started[0][0]['time']*1e9)
        begin=request['request_started_monotonic_ns'];first=tokens[0][1];last=tokens[-1][1]
        assert begin<=worker_start<=first<=last<=snapshots[0][1]<=durable[0][1]
        row=dict(tag=generation['tag'],context=generation['context_index'],question=generation['question'],
            request_to_worker_seconds=(worker_start-begin)/1e9,
            worker_to_first_arrival_seconds=(first-worker_start)/1e9,
            request_started_event_transport_seconds=(started[0][1]-worker_start)/1e9,
            last_token_to_snapshot_seconds=(snapshots[0][1]-last)/1e9,
            snapshot_to_durable_seconds=(durable[0][1]-snapshots[0][1])/1e9,
            capture_seconds=snapshots[0][0]['capture_seconds'],
            native_request_seconds=snapshots[0][0]['request_seconds'],
            generation_seconds=snapshots[0][0]['generation_seconds'])
        rows.append(row)
    keys=[key for key in rows[0] if key.endswith('_seconds')]
    arms[arm]=dict(rows=rows,medians={key:statistics.median(row[key] for row in rows) for key in keys})
receipt=dict(schema='cow-existing-same-host-event-attribution-v1',status='PASS',arms=arms,
    pair_sha256=hashlib.sha256((root/'policy-pair-v2.json').read_bytes()).hexdigest(),
    source_producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations=['Same llm-60 CLOCK_MONOTONIC across client and server processes; server float seconds converted to ns',
        'Worker-to-first-arrival includes server/protocol/transport; not pure GPU time',
        'Capture is nested within last-token-to-snapshot; do not sum nested durations',
        'Read-only analysis of executed unmodified profiles; no new inference or runtime instrumentation'])
with (root/'event-attribution-v1.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
for arm,data in arms.items():print(json.dumps(dict(arm=arm,**data)))
