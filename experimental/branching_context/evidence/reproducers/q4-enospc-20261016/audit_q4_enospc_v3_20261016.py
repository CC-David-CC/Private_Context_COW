"""Independent terminal evidence and read-only payload audit, no inference."""
import hashlib
import json
from pathlib import Path
import sys

root = Path('/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1')
case = root / 'gpu-q4-enospc-v3/probe-result'
wrapper = Path('/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v5/gpu-command-v3')
mount = Path(sys.argv[1]).resolve(strict=True)
def read(path): return json.loads(path.read_bytes())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024**2): h.update(block)
    return h.hexdigest()

r=read(case/'result.json')
assert r['status']=='PASS'
assert all((wrapper/name).read_text().strip()=='0' for name in ('command.exit','wrapper.exit'))
cleanup=read(root/'private-mount-run-v5/terminal-receipt.json')
assert cleanup['probe_exit']==cleanup['cleanup_exit']==0
assert cleanup['mount_removed'] and cleanup['loop_detached']
sys.path.insert(0,r['api_src'])
from context_time_travel.store import Store
store=Store(r['store_path'])
source=r['source_before']['head']['tip']
target=r['retry']['response']['receipt']['snapshot']
branch=r['target_branch']
operation=r['fault_operation_id']
receipt=r['retry']['response']['receipt']
events=r['fault']['stream']
errors=[row for row in events if row['event']=='error']
assert len(errors)==1 and errors[0]['cancelled'] is False
assert '[Errno 28] No space left on device' in errors[0]['error']
assert not any(row.get('event') in ('durable','complete') or row.get('durable') is True or 'receipt' in row for row in events)
writer=r['fault']['writer']
assert writer==read(case/'writer-errors.jsonl')
assert writer['errno']==28 and writer['operation_id']==operation and writer['branch']==branch
assert not writer['published_payload_exists'] and not writer['published_manifest_exists']
assert writer['staging_files_after']==[]
assert writer['payload_name']==receipt['payload_sha256']+'.state'
assert writer['requested_payload_bytes']==331328756
fill=read(case/'filler-result.json')
assert fill==r['fault']['filler'] and fill['errno']==28
assert fill['available_after'] < writer['requested_payload_bytes']
assert r['fault']['filler_cleanup']['unlinked'] is True
assert (case/'writer-release').read_bytes()==b'RELEASE\n'
snapshots=[row for row in events if row['event']=='snapshot']
assert len(snapshots)==1 and snapshots[0]['snapshot']==target
tokens=snapshots[0]['tokens']
assert len(tokens)==128
assert [row['token_id'] for row in events if row['event'] in ('text','turn_complete') and 'token_id' in row]==tokens
assert tuple(store.lineage(target)[1])==tuple(store.lineage(source)[1])+tuple(tokens)
follow_events=r['following_request']['events']
follow_snapshots=[row for row in follow_events if row['event']=='snapshot']
assert len(follow_snapshots)==1 and len(follow_snapshots[0]['tokens'])==8
follow=follow_snapshots[0]['snapshot']
assert tuple(store.lineage(follow)[1])==tuple(store.lineage(target)[1])+tuple(follow_snapshots[0]['tokens'])
for parent,child,count in ((source,target,128),(target,follow,8)):
    p,c=(store.get_snapshot(s)['controller'] for s in (parent,child))
    assert c['accepted_in_scope']==p['accepted_in_scope']+count
    assert c['policy']==p['policy']
    assert c['next_draw']==p['next_draw']==0
    assert c['grammar_state'] is p['grammar_state'] is None
assert store.head(branch).tip==follow and store.head(branch).revision==2
moves=[row for row in store._read_branch(branch)['events'] if row['type']=='move']
assert len(moves)==2 and sum(row['operation_id']==operation for row in moves)==1
assert store.checkpoint_receipt(operation)['checkpoint_receipt']['payload_sha256']==receipt['payload_sha256']
assert r['fault']['branch_head_unchanged'] and r['fault']['source_checkpoint_unchanged']
assert r['fault']['operation_status']['durable'] is False
assert r['fault']['operation_status']['publication_retryable'] is True
assert r['retry']['http_status']==200 and r['retry']['operation_status']['durable'] is True
assert r['retry']['response']['receipt']==r['retry']['operation_status']['receipt']
assert r['retry']['native_trace_sha256_before']==r['retry']['native_trace_sha256_after']
raw=(case/'native-trace.jsonl').read_bytes().splitlines(keepends=True)
trace=[json.loads(line) for line in raw]
digest=hashlib.sha256()
matches=[]
for n,line in enumerate(raw,1):
    digest.update(line)
    if digest.hexdigest()==r['retry']['native_trace_sha256_before']: matches.append(n)
assert len(matches)==1
at_retry=trace[:matches[0]]
assert max(row.get('response',{}).get('forward_tokens',0) for row in at_retry)==128
closed=[row for row in trace if row.get('closed') is True]
assert len(closed)==1
counters=closed[0]['counters']
assert counters['forward_tokens']==136 and counters['prefill_tokens']==0
assert counters['integrity_mismatches']==0
assert counters['model_bytes_hashed_this_process']==0 and counters['model_file_hash_receipts_reused']==15
assert follow_snapshots[0]['counter_deltas']['forward_tokens']==8
assert follow_snapshots[0]['counter_deltas']['prefill_tokens']==0
assert r['service_stop']['exit_code']==r['native_shutdown']['native_exit_code']==0
assert r['service_stop']['graceful'] and r['native_shutdown']['close_succeeded']
pids=[r['server_pid'],r['native_identity']['pid']]
assert all(not Path('/proc',str(pid)).exists() for pid in pids)
assert all(row['pid']==pids[1] for row in trace)
assert sha(Path(r['native_binary']))==r['native_sha256']
assert sha(Path(r['profile_path']))==r['profile_sha256']
files={}
for snapshot in (source,target,follow):
    manifest_path=mount/(snapshot+'.json')
    manifest=read(manifest_path)['manifest']
    payload=mount/(manifest['payload_sha256']+'.state')
    assert not payload.is_symlink() and sha(payload)==manifest['payload_sha256']
    assert payload.stat().st_size==manifest['state_bytes']+4*manifest['logits_count']
    canonical=store.root/'objects'/(snapshot+'.json')
    assert sha(canonical)==snapshot
    files[snapshot]=dict(payload_sha256=manifest['payload_sha256'],payload_bytes=payload.stat().st_size,
                         manifest_sha256=sha(manifest_path),canonical_sha256=snapshot)
assert files[target]['payload_bytes']==writer['requested_payload_bytes']
for name,record in r['source_before']['checkpoint_files'].items():
    original=Path('/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/input-store-v1/checkpoints')/name
    assert sha(mount/name)==sha(original)==record['sha256']
    assert (mount/name).stat().st_size==original.stat().st_size==record['bytes']
assert store.head('CliA').tip==source and store.head('CliA').revision==4
expected={snapshot+'.json' for snapshot in files}|{row['payload_sha256']+'.state' for row in files.values()}|{'lost+found'}
assert {p.name for p in mount.iterdir()}==expected
assert not (mount/'.q4-enospc-filler').exists() and not list(mount.glob('.stage-*'))
report=dict(schema='q4-enospc-primary-v1',status='FOCUSED-PASS',
            scope='Actual Q4 HTTP checkpoint-volume ENOSPC and same-operation retry, not cumulative G2',
            command_exit=0,wrapper_exit=0,error_events=1,original_errno=28,
            rejected_payload_bytes=writer['requested_payload_bytes'],saved_prefix_tokens=len(store.lineage(source)[1]),
            fault_outputs=128,following_outputs=8,source_full_bytes_unchanged=True,
            prepared_payload_identity_preserved=True,single_retry_move=True,
            retry_trace_prefix_lines=matches[0],retry_native_work_unchanged=True,
            canonical_tokens_and_greedy_controller_progress_verified=True,
            native_forward_tokens=counters['forward_tokens'],native_prefill_tokens=counters['prefill_tokens'],
            native_loaded_file_bytes=counters['loaded_file_bytes'],
            native_copied_bytes=counters['copied_bytes'],native_integrity_hashed_bytes=counters['integrity_device_hashed_bytes'],
            native_integrity_mismatches=counters['integrity_mismatches'],
            http_exit=0,native_exit=0,process_ids=pids,all_pids_absent=True,
            profile_sha256=r['profile_sha256'],native_sha256=r['native_sha256'],
            execution_profile_id=receipt['execution_profile_id'],model_contract_id=receipt['model_contract_id'],
            numerical_max_error=None,files=files,raw_result_sha256=sha(case/'result.json'),
            raw_trace_sha256=sha(case/'native-trace.jsonl'),
            limits=['Only checkpoint volume is full; canonical metadata volume is not full',
                    'No fresh numerical oracle, sampler expansion, peak-memory or throughput qualification',
                    'No power-loss campaign; original cumulative G1-G5 remains unqualified'])
output=root/'primary-audit-gpu-q4-enospc-v3.json'
output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
