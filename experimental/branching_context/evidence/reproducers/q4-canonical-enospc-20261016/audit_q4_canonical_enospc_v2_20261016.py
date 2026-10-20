"""Primary read-only evidence audit; never starts inference or imports the harness."""
import hashlib
import json
from pathlib import Path
import sys

root = Path('/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1')
case = root / 'gpu-q4-v2/campaign'
wrapper = root / 'launch-v4/run-command-v1'
export = root / 'readonly-audit-v2/canonical-export'
cache = root / 'checkpoint-volume-v1'

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()

r = read(case / 'result.json')
assert r['status'] == 'PASS'
assert all((wrapper / name).read_text().strip() == '0' for name in ('command.exit', 'wrapper.exit'))
cleanup = read(root / 'mount-run-v3/terminal-receipt.json')
assert cleanup['probe_exit'] == cleanup['cleanup_exit'] == 0
assert cleanup['cache_unmounted'] and cleanup['canonical_unmounted'] and cleanup['loop_detached']
ro_cleanup = read(root / 'readonly-audit-v2/cleanup.json')
assert ro_cleanup['export_exit'] == ro_cleanup['cleanup_exit'] == 0
assert ro_cleanup['mount_removed'] and ro_cleanup['loop_detached']
assert ro_cleanup['image_sha256_before'] == ro_cleanup['image_sha256_after'] == cleanup['image_sha256_after']
assert sha(root / 'canonical-ext4-64m.img') == cleanup['image_sha256_after']
export_receipt = read(root / 'readonly-audit-v2/export.json')
assert export_receipt['readonly_source'] is True
for relative, record in export_receipt['files'].items():
    path = export / relative
    assert not path.is_symlink() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']

pins = r['pins']
sys.path.insert(0, pins['api_src'])
from context_time_travel.store import Store
store = Store(export)
source = r['source_before']['head']['tip']
target = r['retry']['response']['receipt']['snapshot']
branch = r['target_branch']
operation = r['operation_id']
receipt = r['retry']['response']['receipt']

events = r['fault']['events']
errors = [row for row in events if row.get('event') == 'error']
assert len(errors) == 1 and errors[0]['cancelled'] is False
assert '[Errno 28] No space left on device' in errors[0]['error']
assert not any(row.get('event') in ('durable', 'complete') or row.get('durable') is True or 'receipt' in row for row in events)
writer_rows = [json.loads(line) for line in (case / 'control/writer-events.jsonl').read_bytes().splitlines()]
gates = [row for row in writer_rows if row['kind'] == 'branch-reference-gate']
failures = [row for row in writer_rows if row['kind'] == 'branch-reference-write-error']
assert len(gates) == len(failures) == 1
writer = r['fault']['writer']
assert writer == failures[0] and writer['errno'] == 28
assert writer['operation_id'] == operation and writer['branch'] == branch and writer['branch_write'] is True
assert writer['old_reference_sha256'] == writer['reference_sha256_after'] == r['fault_request']['old_reference_sha256']
assert writer['target_snapshot'] == target and writer['stage_files_after'] == []
assert writer['serialized_reference_bytes'] > 0
assert gates[0]['new_head']['tip'] == target
assert r['fault_request']['parent_head']['name'] == branch
assert gates[0]['old_head'] == {key: r['fault_request']['parent_head'][key] for key in ('revision', 'root', 'tip')}
fill = read(case / 'filler-receipt.json')
assert fill['errno'] == 28 and fill['available_bytes_after'] == 0
assert fill['operation_id'] == operation and fill['target_snapshot'] == target
assert 0 < fill['filled_bytes'] <= 64 * 1024**2 and 0 < fill['allocated_bytes'] <= 64 * 1024**2
assert fill['canonical_st_dev'] != pins['canonical_mount']['checkpoint']['st_dev']
assert read(case / 'control/release.json')['operation_id'] == operation
assert not (case / 'control/abort.json').exists()
assert (case / 'control/gate-consumed.json').is_file()

snapshots = [row for row in events if row.get('event') == 'snapshot']
assert len(snapshots) == 1 and snapshots[0]['snapshot'] == target
tokens = snapshots[0]['tokens']
assert len(tokens) == 16
assert [row['token_id'] for row in events if row.get('event') in ('text', 'turn_complete') and 'token_id' in row] == tokens
assert store.lineage(target)[1] == store.lineage(source)[1] + tokens
follow_events = r['following_request']['events']
follow_snapshots = [row for row in follow_events if row.get('event') == 'snapshot']
assert len(follow_snapshots) == 1 and len(follow_snapshots[0]['tokens']) == 8
follow = follow_snapshots[0]['snapshot']
assert store.lineage(follow)[1] == store.lineage(target)[1] + follow_snapshots[0]['tokens']
for parent, child, count in ((source, target, 16), (target, follow, 8)):
    p, c = (store.get_snapshot(identity)['controller'] for identity in (parent, child))
    assert c['accepted_in_scope'] == p['accepted_in_scope'] + count
    assert c['policy'] == p['policy'] and c['next_draw'] == p['next_draw'] == 0
    assert c['grammar_state'] is p['grammar_state'] is None
assert store.head(branch).tip == follow and store.head(branch).revision == r['fault_request']['parent_head']['revision'] + 2
moves = [row for row in store._read_branch(branch)['events'] if row['type'] == 'move']
assert len(moves) == 2 and sum(row['operation_id'] == operation for row in moves) == 1
assert store.checkpoint_receipt(operation)['checkpoint_receipt']['payload_sha256'] == receipt['payload_sha256']
assert r['fault']['operation_status']['durable'] is False
assert r['fault']['operation_status']['publication_retryable'] is True
assert r['retry']['operation_status']['receipt'] == receipt and r['retry']['operation_status']['durable'] is True
assert r['retry']['native_trace_unchanged'] is True
raw = (case / 'native-trace.jsonl').read_bytes().splitlines(keepends=True)
trace = [json.loads(line) for line in raw]
digest = hashlib.sha256()
matches = []
for number, line in enumerate(raw, 1):
    digest.update(line)
    if digest.hexdigest() == r['retry']['native_trace_sha256']:
        matches.append(number)
assert len(matches) == 1
assert max(row.get('response', {}).get('forward_tokens', 0) for row in trace[:matches[0]]) == 16
closed = [row for row in trace if row.get('closed') is True]
assert len(closed) == 1
counters = closed[0]['counters']
assert counters['forward_tokens'] == 24 and counters['prefill_tokens'] == 0
assert counters['integrity_mismatches'] == 0
assert counters['model_bytes_hashed_this_process'] == 0 and counters['model_file_hash_receipts_reused'] == 15
assert follow_snapshots[0]['counter_deltas']['forward_tokens'] == 8 and follow_snapshots[0]['counter_deltas']['prefill_tokens'] == 0
assert r['service_stop']['exit_code'] == r['native_shutdown']['native_exit_code'] == 0
assert r['service_stop']['graceful'] and r['native_shutdown']['close_succeeded']
pids = [r['server_pid'], r['native_process_before']['pid']]
assert all(not Path('/proc', str(pid)).exists() for pid in pids)
assert all(row['pid'] == pids[1] for row in trace)
assert sha(Path(pins['native_binary'])) == pins['native_sha256']
assert sha(Path(pins['profile_path'])) == pins['profile_sha256']
files = {}
for snapshot in (source, target, follow):
    manifest_path = cache / (snapshot + '.json')
    manifest = read(manifest_path)['manifest']
    payload = cache / (manifest['payload_sha256'] + '.state')
    assert not payload.is_symlink() and sha(payload) == manifest['payload_sha256']
    assert payload.stat().st_size == manifest['state_bytes'] + 4 * manifest['logits_count']
    assert sha(export / 'objects' / (snapshot + '.json')) == snapshot
    files[snapshot] = dict(payload_sha256=manifest['payload_sha256'], payload_bytes=payload.stat().st_size,
                           manifest_sha256=sha(manifest_path), canonical_sha256=snapshot)
target_record = files[target]
assert target_record['payload_sha256'] == r['fault']['target_checkpoint']['payload_sha256'] == receipt['payload_sha256']
assert target_record['manifest_sha256'] == r['fault']['target_checkpoint']['manifest_sha256']
original = Path(pins['original_store'])
for relative, record in pins['original_store_metadata']['files'].items():
    assert sha(original / relative) == record['sha256'] and (original / relative).stat().st_size == record['bytes']
for name, record in r['source_before']['checkpoint_files'].items():
    assert sha(cache / name) == sha(original / 'checkpoints' / name) == record['sha256']
    assert (cache / name).stat().st_size == record['bytes']
assert store.head('CliA').tip == source and store.head('CliA').revision == 4
assert not list(export.rglob('.stage-*')) and not list(export.glob('.q4-canonical-enospc-*.filler'))
assert not list(cache.glob('.stage-*'))
report = dict(schema='q4-canonical-enospc-primary-v1', status='FOCUSED-PASS',
    scope='Real Q4 canonical branch-reference ENOSPC while checkpoint storage remains available',
    command_exit=0, wrapper_exit=0, saved_prefix_tokens=len(store.lineage(source)[1]),
    fault_outputs=16, following_outputs=8, original_errno=28, error_events=1,
    filler_bytes=fill['filled_bytes'], filler_allocated_bytes=fill['allocated_bytes'],
    canonical_available_at_fault=0, serialized_reference_bytes=writer['serialized_reference_bytes'],
    old_reference_hash_preserved_at_failure=True, prepared_checkpoint_preserved=True,
    source_canonical_and_payload_bytes_unchanged=True, single_retry_move=True,
    retry_native_work_unchanged=True, retry_trace_prefix_lines=matches[0],
    canonical_tokens_and_greedy_controller_progress_verified=True,
    native_forward_tokens=counters['forward_tokens'], native_prefill_tokens=counters['prefill_tokens'],
    native_loaded_file_bytes=counters['loaded_file_bytes'], native_copied_bytes=counters['copied_bytes'],
    native_integrity_hashed_bytes=counters['integrity_device_hashed_bytes'],
    native_integrity_mismatches=counters['integrity_mismatches'],
    process_ids=pids, all_pids_absent=True, http_exit=0, native_exit=0,
    profile_sha256=pins['profile_sha256'], native_sha256=pins['native_sha256'],
    execution_profile_id=receipt['execution_profile_id'], model_contract_id=receipt['model_contract_id'],
    numerical_max_error=None, files=files, metadata_export_sha256=sha(root / 'readonly-audit-v2/export.json'),
    raw_result_sha256=sha(case / 'result.json'), raw_trace_sha256=sha(case / 'native-trace.jsonl'),
    limitations=['Branch-reference writer only; not every canonical metadata write or crash stage',
                 'No fresh numerical oracle, power-loss test, sampler expansion or performance qualification',
                 'Cumulative G1-G5 remains unqualified'])
output = root / 'primary-audit-gpu-q4-v2.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, sort_keys=True, indent=2)
    stream.write('\n')
output.chmod(0o600)
print(json.dumps(report, sort_keys=True))
