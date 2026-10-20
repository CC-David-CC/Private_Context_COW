"""Audit the aborted filler experiment without inference or disk mutation."""
import hashlib
import json
from pathlib import Path
import sys

root = Path('/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1')
case = root / 'gpu-q4-v1/campaign'
export = root / 'readonly-audit-v1/canonical-export'
cache = root / 'checkpoint-volume-v1'
def read(path): return json.loads(path.read_bytes())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2): h.update(block)
    return h.hexdigest()
r = read(case / 'result.json')
assert r['status'] == 'FAIL'
assert r['error'] == 'CanonicalEnospcError: filesystem still has 417792 available bytes after ENOSPC'
for name in ('command.exit', 'wrapper.exit'):
    assert (root / 'launch-v3/run-command-v1' / name).read_text().strip() == '1'
    assert (root / 'readonly-export-command-v1' / name).read_text().strip() == '0'
cleanup = read(root / 'mount-run-v2/terminal-receipt.json')
assert cleanup['probe_exit'] == 1 and cleanup['cleanup_exit'] == 0
assert cleanup['cache_unmounted'] and cleanup['canonical_unmounted'] and cleanup['loop_detached']
readonly = read(root / 'readonly-audit-v1/cleanup.json')
assert readonly['export_exit'] == readonly['cleanup_exit'] == 0
assert readonly['image_sha256_before'] == readonly['image_sha256_after'] == cleanup['image_sha256_after']
assert sha(root / 'canonical-ext4-64m.img') == cleanup['image_sha256_after']
metadata = read(root / 'readonly-audit-v1/export.json')
for relative, row in metadata['files'].items():
    assert sha(export / relative) == row['sha256'] and (export / relative).stat().st_size == row['bytes']
assert not list(export.glob('.q4-canonical-enospc-*.filler')) and not list(export.rglob('.stage-*'))
sys.path.insert(0, r['pins']['api_src'])
from context_time_travel.store import Store
store = Store(export)
source = r['source_before']['head']['tip']
assert store.head('CliA').tip == source and store.head('CliA').revision == 4
branch = r['target_branch']
assert store.head(branch).tip == source and store.head(branch).revision == 0
assert not [row for row in store._read_branch(branch)['events'] if row['type'] == 'move']
rows = [json.loads(line) for line in (case / 'control/writer-events.jsonl').read_bytes().splitlines()]
assert len(rows) == 1 and rows[0]['kind'] == 'branch-reference-gate'
assert rows[0]['operation_id'] == r['operation_id']
assert not (case / 'control/release.json').exists()
assert not (case / 'control/gate-consumed.json').exists()
assert (case / 'control/abort.json').is_file()
target = rows[0]['target_snapshot']
assert len(store.lineage(target)[1]) == len(store.lineage(source)[1]) + 16
assert r['service_stop_after_gate_abort']['exit_code'] == 0
assert r['service_stop_after_gate_abort']['graceful']
trace = [json.loads(line) for line in (case / 'native-trace.jsonl').read_bytes().splitlines()]
closed = [row for row in trace if row.get('closed') is True]
assert len(closed) == 1
counters = closed[0]['counters']
assert counters['forward_tokens'] == 16 and counters['prefill_tokens'] == 0 and counters['integrity_mismatches'] == 0
pids = [r['server_pid'], r['native_process_before']['pid']]
assert all(not Path('/proc', str(pid)).exists() for pid in pids)
sys.path.insert(0, str(Path(r['pins']['runtime_root']) / 'experimental/branching_context/scripts'))
from run_cow_client_recovery import _trace_worker
native = _trace_worker(case / 'pool-trace.json', pids[0])
assert native['native_pid'] == pids[1] and native['native_exit_code'] == 0 and native['close_succeeded']
files = {}
for snapshot in (source, target):
    manifest_path = cache / (snapshot + '.json')
    manifest = read(manifest_path)['manifest']
    payload = cache / (manifest['payload_sha256'] + '.state')
    assert sha(payload) == manifest['payload_sha256']
    assert payload.stat().st_size == manifest['state_bytes'] + 4 * manifest['logits_count']
    assert sha(export / 'objects' / (snapshot + '.json')) == snapshot
    files[snapshot] = dict(payload_sha256=manifest['payload_sha256'], payload_bytes=payload.stat().st_size,
                           manifest_sha256=sha(manifest_path), canonical_sha256=snapshot)
original = Path(r['pins']['original_store'])
for name, row in r['source_before']['checkpoint_files'].items():
    assert sha(cache / name) == sha(original / 'checkpoints' / name) == row['sha256']
for relative, row in r['pins']['original_store_metadata']['files'].items():
    assert sha(original / relative) == row['sha256']
report = dict(schema='q4-canonical-enospc-attempt1-primary-v1', status='FAIL-HARNESS-PRE-RELEASE',
    observed_error=r['error'], command_exit=1, wrapper_exit=1, cleanup_exit=0,
    actual_branch_writer_released=False, branch_writer_enospc_executed=False,
    source_bytes_unchanged=True, old_branch_unchanged=True, filler_and_stage_files_absent=True,
    saved_prefix_tokens=len(store.lineage(source)[1]), generated_tokens_before_abort=16,
    native_forward_tokens=counters['forward_tokens'], native_prefill_tokens=counters['prefill_tokens'],
    native_integrity_mismatches=counters['integrity_mismatches'], http_exit=0, native_exit=0,
    process_ids=pids, all_pids_absent=True, files=files, numerical_max_error=None,
    raw_result_sha256=sha(case / 'result.json'), readonly_export_sha256=sha(root / 'readonly-audit-v1/export.json'),
    image_sha256_after=cleanup['image_sha256_after'],
    limitation='No target writer errno28 test occurred; filler zero-available guard aborted first')
output = root / 'primary-audit-gpu-q4-v1-abort.json'
with output.open('x') as stream: json.dump(report, stream, sort_keys=True, indent=2); stream.write('\n')
output.chmod(0o600)
print(json.dumps(report, sort_keys=True))
