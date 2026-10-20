"""Read existing trace clocks; no runtime instrumentation or new inference."""
import hashlib
import json
import statistics
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
record=json.loads((root/'policy-pair-v2.json').read_bytes());arms={}
for arm,summary in record['arms'].items():
    path=Path(summary['result_path']);campaign=json.loads(path.read_bytes())
    result=[]
    for service in campaign['services']:
        ordinal=service['ordinal'];trace=path.parent/f'native-{ordinal}.jsonl'
        records=[json.loads(line) for line in trace.read_text().splitlines() if line]
        assert all(left['time']<=right['time'] for left,right in zip(records,records[1:]))
        labels={call['label'] for call in campaign['calls'] if call['service']==ordinal}
        generations=[row for row in campaign['generations'] if row['call_label'] in labels]
        feeds=[index for index,row in enumerate(records) if row.get('command','').startswith('feed ')]
        assert len(feeds)==len(generations),(arm,ordinal,len(feeds),len(generations))
        for generation,index in zip(generations,feeds):
            feed=records[index];previous=records[index-1]
            assert previous.get('command')=='stats'
            assert len(feed['command'].split())-1==generation['suffix_tokens']
            restore_index=next(i for i in range(index-1,-1,-1) if records[i].get('command','').startswith(('restore ','load ')))
            restore=records[restore_index]
            row=dict(tag=generation['tag'],context=generation['context_index'],question=generation['question'],
                cli_seconds=generation['process_seconds'],
                ttft_seconds=generation.get('arrival_summary',{}).get('metrics_ns',{}).get('request_to_first_token',0)/1e9,
                feed_response_gap_seconds=feed['time']-previous['time'],
                source_restore_response_gap_seconds=restore['time']-records[restore_index-1]['time'],
                source_restore_kind=restore['command'].split()[0])
            result.append(row)
    hot=[row for row in result if row['tag'].startswith('round')]
    assert len(hot)==18
    arms[arm]=dict(rows=hot,
        median_feed_response_gap_seconds=statistics.median(row['feed_response_gap_seconds'] for row in hot),
        median_source_restore_response_gap_seconds=statistics.median(row['source_restore_response_gap_seconds'] for row in hot))
receipt=dict(schema='cow-existing-native-protocol-attribution-v1',status='PASS',arms=arms,
    pair_sha256=hashlib.sha256((root/'policy-pair-v2.json').read_bytes()).hexdigest(),
    source_producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations=['Gaps use time.time trace emission clocks; asserted monotonic over observed data, not device events',
        'Adjacent response gaps include intervening Python/protocol work; upper bounds, not pure kernel timings',
        'Read-only analysis of completed original runs; no new model execution or instrumentation overhead'])
with (root/'protocol-attribution-v1.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
for arm,data in arms.items():
    print(json.dumps(dict(arm=arm,median_feed_response_gap_seconds=data['median_feed_response_gap_seconds'],
        median_source_restore_response_gap_seconds=data['median_source_restore_response_gap_seconds'],rows=data['rows'])))
