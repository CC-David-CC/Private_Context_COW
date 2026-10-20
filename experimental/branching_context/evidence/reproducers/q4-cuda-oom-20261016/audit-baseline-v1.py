"""Read-only independent audit of the actual Q4 CUDA OOM baseline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/home/dflanag3/spatial-artifacts/q4-cuda-oom-20261016-v1')
run = root / 'gpu-q4-v2/campaign'
store = root / 'input-store-v1'
def read(path): return json.loads(path.read_bytes())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(8 * 1024**2): h.update(b)
    return h.hexdigest()
def require(condition, message):
    if not condition: raise AssertionError(message)
result_path = run / 'result.json'
require(sha(result_path) == 'b20eadc9d7aa9c5f33db3283d3835d20be0c12dfdd7a61ef7133e9cdc602868b', 'raw result pin')
r = read(result_path)
require(r['status'] == 'FAIL-RECOVERY', 'must retain failed automatic recovery status')
for name in ('command.exit', 'wrapper.exit'):
    require((root / 'gpu-command-v2' / name).read_text().strip() == '1', name)
require(read(run / 'progress.json')['status'] == r['status'], 'progress final status')
require(not r.get('cleanup_failures'), 'cleanup errors')
require(r['profile_sha256'] == 'f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c', 'profile identity')
require(r['native_sha256'] == '4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451', 'native identity')
require(sha(Path(r['profile_path'])) == r['profile_sha256'], 'profile bytes')
require(sha(Path(r['native_binary'])) == r['native_sha256'], 'native bytes')
for side in ('first_service', 'restart_service'):
    for row in r[side]['imports']['modules'].values():
        require(r['expected_imports'].get(row['path']) == row['sha256'] == sha(Path(row['path'])), 'import witness')
source = r['expected_source_snapshot']
require(sha(store / 'objects' / (source + '.json')) == r['source_before']['snapshot_record_sha256'] == source, 'source record')
for name, row in r['source_before']['checkpoint_files'].items():
    path = store / 'checkpoints' / name
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'source checkpoint bytes')
prep = read(root / 'preparation.json')
old_objects = [row for row in prep['metadata_hashes'] if row['relative_path'].startswith('objects/')]
for row in old_objects:
    require(sha(store / row['relative_path']) == row['sha256'], 'old canonical object mutated')
refs = read(store / 'refs.json')
directory = store / refs['directory']
source_head = read(directory / 'CliA.json')['head']
require(source_head == {k:v for k,v in r['source_before']['head'].items() if k != 'name'}, 'source head after full campaign')
require(r['canonical_after_fault']['target_head'] == r['target_before']['head'], 'fault moved branch')
require(r['canonical_after_fault']['source_head'] == r['source_before']['head'], 'fault moved source')
require(r['canonical_after_fault']['target_controller'] == 'active', 'controller after fault')
for path in directory.glob('*.json'):
    branch = read(path)
    require(r['fault_operation'] not in branch['receipts'], 'failed fault op durable receipt')
fault_events = r['fault_response']['events']
require(sum(e.get('event') == 'error' for e in fault_events) == 1, 'single fault error')
require(not any(e.get('event') in ('token','text','snapshot','durable','complete') for e in fault_events), 'fault published work')
require(r['fault_response']['terminal']['error'] == 'Invalid consumed native token event', 'baseline adapter error loss')
require(r['same_service_followup']['event'] == 'error' and r['same_service_followup']['error'] == 'Native request protocol failed; restart the backend', 'baseline same-process recovery failure')
before, after = r['native_oom']['stats_before'], r['native_oom']['stats_after']
require(after['vmm_driver_failures'] - before['vmm_driver_failures'] == 1, 'actual SUT driver failure')
require(after['vmm_out_of_memory_failures'] - before['vmm_out_of_memory_failures'] == 1, 'actual SUT OOM')
require(after['vmm_last_driver_error'] == 2 and after['vmm_last_failed_operation'] == 'cuMemCreate', 'actual SUT CUDA OOM operation')
for name in ('forward_tokens','prefill_tokens','consumed_tokens','captures'):
    require(after[name] == before[name], 'fault changed ' + name)
require(before['forward_tokens'] == before['prefill_tokens'] == 0 and after['consumed'] == 7999, 'fault work/position')
require(r['native_oom']['response']['status'] == 'error' and 'cannot prepare private tail' in r['native_oom']['response']['message'], 'raw native error')
helper = r['pressure_helper']
require(helper['fill']['held_bytes'] == helper['release']['released_bytes'] == 11228151808, 'helper bytes')
require(helper['fill']['physical_allocations'] == 5354 and helper['fill']['last_error'] == 2, 'helper real allocations')
require(helper['terminal']['exit_code'] == 0 and helper['terminal']['pid_absent_after_wait'], 'helper release exit')
require(r['first_service']['native_close_observation']['native_worker_closed'][0]['returncode'] == -9, 'poisoned child kill recorded')
require(r['first_service']['native_close_observation']['native_worker_closed'][0]['close_succeeded'] is False, 'no false clean close')
require(r['restart_service']['native_close_trace']['native_exit_code'] == 0 and r['restart_service']['native_close_trace']['close_succeeded'], 'restart native close')
for side in ('first_service', 'restart_service'):
    require(r[side]['stop']['exit_code'] == 0, 'service process exit')
pids = [r['first_service']['pid'], r['first_service']['native_process']['pid'], helper['identity']['pid'], r['restart_service']['pid'], r['restart_service']['native_close_trace']['native_pid']]
require(len(set(pids)) == 5 and all(not Path('/proc', str(pid)).exists() for pid in pids), 'process separation and termination')
follow = r['restart_followup']
require(follow['event'] == 'complete' and len(follow['tokens']) == 8, 'restart continuation')
d = follow['answer']['counter_deltas']
require(d['forward_tokens'] == d['consumed_tokens'] == 8 and d['prefill_tokens'] == 0 and d['loaded_file_bytes'] > 0, 'restart zero-prefix work')
require(d['integrity_mismatches'] == 0, 'restart integrity')
durable = [e['receipt'] for e in follow['events'] if e['event'] == 'durable']
require(len(durable) == 1, 'single restart durable receipt')
receipt = durable[0]
branch = read(directory / (r['target_branch'] + '.json'))
require(branch['head'] == {k:v for k,v in receipt['branch'].items() if k != 'name'} and branch['head']['revision'] == 1, 'exact restart branch revision')
require(r['restart_operation'] in branch['receipts'], 'restart durable receipt persisted')
def lineage(tip):
    seen, chunks = set(), []
    while tip is not None:
        require(tip not in seen, 'ancestry cycle')
        seen.add(tip)
        path = store / 'objects' / (tip + '.json')
        require(sha(path) == tip, 'canonical ancestry object identity')
        row = read(path)
        chunks.append(row['inputs'])
        tip = row['parent']
    return [token for chunk in reversed(chunks) for token in chunk]
source_ids, target_ids = lineage(source), lineage(receipt['snapshot'])
require(len(source_ids) == 7999 and target_ids == source_ids + follow['tokens'], 'restart exact canonical history')
require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'GPU idle after cleanup')
report = dict(schema='q4-cuda-oom-primary-audit-v1', audit_status='PASS', experiment_status='FAIL-RECOVERY',
    host='llm-60', actual_sut_cuda_oom=True, sut_oom_count=1, cuda_error=2, failed_operation='cuMemCreate',
    source_tokens=7999, fault_forward_tokens=0, fault_capture_delta=0, fault_branch_revision_delta=0,
    fault_durable_receipt=False, same_service_recovery=False, explicit_service_restart=True,
    restarted_outputs=8, restarted_forward_tokens=8, restarted_prefix_prefill_tokens=0,
    restarted_loaded_file_bytes=d['loaded_file_bytes'], restart_copied_bytes=d['copied_bytes'],
    restart_device_hashed_bytes=d['integrity_device_hashed_bytes'], restart_integrity_mismatches=0,
    helper_held_released_bytes=helper['fill']['held_bytes'], helper_allocations=5354,
    native_exit_codes=[-9,0], service_exit_codes=[0,0], helper_exit=0,
    all_owned_pids_absent=True, gpu_empty=True, original_objects_verified=len(old_objects),
    source_checkpoint_bytes=327985908, source_checkpoint_unchanged=True,
    command_exit=1, wrapper_exit=1, numerical_max_error=None, numerical_oracle_executed=False,
    post_error_live_gpu_parent_audit=False,
    result_sha256=sha(result_path), profile_sha256=r['profile_sha256'], native_sha256=r['native_sha256'],
    driver_sha256=r['runner_sha256'], audit_script_sha256=sha(Path(__file__)),
    limitations=['No fresh numerical oracle','No live GPU parent audit after poisoning','No automatic same-service worker recovery','Baseline client hides original native error','One bounded greedy allocation-failure case, not cumulative G2/G3 qualification'])
output = root / 'primary-audit-v2.json'
with output.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2, sort_keys=True); f.write('\n')
print(json.dumps(report, sort_keys=True))