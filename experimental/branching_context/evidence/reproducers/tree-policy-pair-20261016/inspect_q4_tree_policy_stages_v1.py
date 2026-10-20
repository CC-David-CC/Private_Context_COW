"""Print aggregate measured native stages and observed request components."""
import json
import sys
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_bytes())
for name,arm in record['arms'].items():
    totals=arm['native_counter_totals']
    times={key:round(value/1e9,6) for key,value in totals.items() if key.endswith('_ns')}
    counts={key:value for key,value in totals.items() if not key.endswith('_ns') and any(word in key for word in ('hash','copy','copied','captur','restor','loaded','cache'))}
    print(json.dumps(dict(arm=name,native_stage_seconds=times,counters=counts)))
    for row in arm['hot_requests']:
        metrics=row['arrival_summary']['metrics_ns']
        print(json.dumps(dict(arm=name,tag=row['tag'],context=row['context_index'],question=row['question'],
            cli_seconds=row['process_seconds'],ttft_seconds=metrics['request_to_first_token']/1e9,
            last_token_to_durable_seconds=metrics['last_token_to_durable']/1e9,
            tpot_seconds=metrics['token_interval_mean']/1e9)))
