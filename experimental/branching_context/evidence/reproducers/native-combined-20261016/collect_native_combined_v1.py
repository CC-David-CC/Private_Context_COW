"""Produce a bounded primary summary from terminal, executed evidence."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1')
reference = Path('/dev/shm/context-cow-native-combined-reference-20261016-v1')
def read(path):
    return json.loads(path.read_bytes())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def run_receipt(name, expected=0):
    path = root / name
    receipts = {field: int((path / field).read_text()) for field in ('command.exit', 'wrapper.exit')}
    assert all(code == expected for code in receipts.values()), (name, receipts)
    return dict(path=str(path), **receipts)
cpu = read(root / 'cpu-v1.json')
baseline = read(root / 'baseline-conflict-v1.json')
campaign = read(root / 'gpu-campaign-v1/result.json')
audit = read(reference / 'result.json')
smoke = read(root / 'gpu-smoke-reference-v1/result.json')
assert cpu['status'] == 'PASS' and cpu['tests'] == 200 and cpu['source_unchanged']
assert baseline['status'] == 'FAIL' and baseline['tests'] == 7 and len(baseline['failures']) == 1
assert campaign['status'] == 'PASS_PIPELINE_NUMERICS_PENDING' and not campaign['failures']
assert audit['status'] == smoke['status'] == 'PASS' and audit['numerical_max_error'] == smoke['numerical_max_error'] == 0
receipts = {name: run_receipt(name) for name in ('cpu-command-v1', 'g0-command-v1',
    'gpu-smoke-command-v1', 'gpu-smoke-reference-command-v1', 'gpu-campaign-command-v1',
    'gpu-campaign-reference-command-v1', 'root-query-measurement-command-v2', 'profile-command-v2')}
receipts['baseline-conflict-command-v1'] = run_receipt('baseline-conflict-command-v1', 1)
profile = read(root / 'gpu-candidate-v2/profile.json')
hot = [row for row in campaign['generations'] if row['tag'].startswith('round')]
assert len(hot) == 18
assert sum(row['output_count'] for row in hot) == 2304
assert len(campaign['generations']) == 32
assert all(row['unchanged'] for row in campaign['roots'].values())
assert campaign['final_status']['max_active_per_gpu'] == {'0': 1}
for row in campaign['generations']:
    assert row['counter_deltas']['prefill_tokens'] == row['suffix_tokens']
times = [row['process_seconds'] for row in hot]
metrics = [row['arrival_summary']['metrics_ns'] for row in hot]
query = read(root / 'root-query-measurement-v2.json')
query_rows = []
for row in query['contexts']:
    measured = row['observations'][1:]
    query_rows.append(dict(arm=row['arm'], context=row['context'],
        median_catalog_seconds=row['medians']['catalog'], median_root_seconds=row['medians']['root'],
        median_catalog_snapshot_read_bytes=statistics.median(item['values']['catalog']['counter_deltas']['snapshot_read_bytes'] for item in measured),
        median_root_snapshot_read_bytes=statistics.median(item['values']['root']['counter_deltas']['snapshot_read_bytes'] for item in measured)))
data = dict(schema='cow-native-combined-primary-v1', status='FOCUSED-PASS',
    source_commit='97bab1a24ee53df036a3ac5bd083e18ec7f5cc71',
    profile_sha256=sha(root / 'gpu-candidate-v2/profile.json'), execution_profile_id=profile['execution_profile_id'],
    model_contract_id=profile['model_contract_id'], backend_sha256=campaign['native_sha256'],
    oracle_sha256=audit['oracle_sha256'], controller_scope=audit['controller_scope'],
    checked_boundaries=audit['checked_boundaries'],
    cpu=dict(tests=cpu['tests'], failures=cpu['failures'], errors=cpu['errors'], skips=cpu['skips'],
        source_unchanged=cpu['source_unchanged'], source_manifest_sha256=sha(root / 'cpu-v1.json')),
    separate_baseline_conflict=dict(tests=7, passed=6, failed=1, failures=baseline['failures'],
        explanation='Unchanged missing-checkpoint rejection expectation conflicts with canonical-only metadata actions and explicit reconstruction; not hidden or reclassified as PASS'),
    gpu=dict(generations=32, actual_output_tokens=sum(row['output_count'] for row in campaign['generations']),
        comparisons=audit['comparisons'], maximum_logit_error=audit['numerical_max_error'],
        independent_reference_processes=len(audit['references']), all_reference_pids_absent=audit['all_reference_pids_absent'],
        smoke_comparisons=smoke['comparisons'], parent_checkpoint_bytes_unchanged=True,
        max_active_per_gpu=campaign['final_status']['max_active_per_gpu'], services=campaign['services']),
    hot_requests=dict(count=len(hot), individual_seconds=times, median_seconds=statistics.median(times),
        sum_seconds=sum(times), total_output_tokens=2304, whole_cli_tokens_per_second=2304/sum(times),
        median_ttft_seconds=statistics.median(row['request_to_first_token'] for row in metrics)/1e9,
        median_tpot_seconds=statistics.median(row['token_interval_mean'] for row in metrics)/1e9,
        median_last_token_to_durable_seconds=statistics.median(row['last_token_to_durable'] for row in metrics)/1e9,
        loaded_native_file_bytes=sum(row['counter_deltas']['loaded_file_bytes'] for row in hot),
        individual_counter_deltas=[row['counter_deltas'] for row in hot]),
    metadata_query=query_rows, receipts=receipts,
    limitations=['CPU ownership fault injection is not real-GPU injected-fault qualification',
        'Original G1-G5 cumulative gates remain unqualified; cross-arrangement equivalence is not claimed',
        'Greedy, Q4, FP16 KV, one GPU request at a time; Q8 deferred',
        'CLI time includes checkpoint durability, CLI startup and observer serialization/fsync',
        'Historical control stores have different prior histories; throughput is not an isolated causal speedup estimate',
        'Fresh oracle arrays stored on RAM filesystem; actual durable checkpoint store remains on SSD',
        'No physical disk-read byte or peak-memory claim; native loaded bytes can stage from RAM'])
with (root / 'primary-v1.json').open('x') as stream:
    json.dump(data, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps({key: data[key] for key in ('status', 'profile_sha256', 'execution_profile_id', 'metadata_query')}))
print(json.dumps({key: data['hot_requests'][key] for key in ('median_seconds', 'whole_cli_tokens_per_second',
    'median_ttft_seconds', 'median_tpot_seconds', 'median_last_token_to_durable_seconds', 'loaded_native_file_bytes')}))
