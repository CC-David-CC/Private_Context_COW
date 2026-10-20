"""Export bounded primary evidence only after checking terminal receipts."""
import hashlib,json
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1')
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
candidate=read(root/'smoke-v2/result.json')
audit=read(root/'smoke-reference-v2/result.json')
assert candidate['status']=='PASS_PIPELINE_NUMERICS_PENDING' and audit['status']=='PASS'
assert audit['campaign_sha256']==sha(root/'smoke-v2/result.json')
assert audit['numerical_max_error']==0 and audit['comparisons']==12
for directory in ('smoke-command-v2','smoke-reference-command-v2'):
    for name in ('command.exit','wrapper.exit'):
        assert (root/directory/name).read_text().strip()=='0'
owned=[]
for row in candidate['services']:
    assert row['stop']['exit_code']==row['worker']['native_exit_code']==0
    owned.extend((row['http_pid'],row['worker']['native_pid']))
owned.extend(row['pid'] for row in audit['references'])
assert len(set(owned))==4 and all(not Path('/proc',str(pid)).exists() for pid in owned)
generation=candidate['generations'][0]
assert generation['source_tokens']==7231 and generation['suffix_tokens']==512 and generation['output_count']==8
assert generation['counter_deltas']['forward_tokens']==520 and generation['counter_deltas']['prefill_tokens']==512
assert len(candidate['calls'])==3 and candidate['calls'][-1]['receipt']==candidate['calls'][1]['receipt']
report=dict(schema='cow-q4-cli-smoke-primary-v1',status='FOCUSED-PASS',host='llm-60',gpu='RTX PRO 6000 Blackwell GPU0',
    scope='Actual terminal CLI -> COW HTTP -> native execution; saved-root reuse, new question and local completed-operation retry',
    prefix_tokens=7231,suffix_tokens=512,output_tokens=8,numerical_max_error=0,full_vector_comparisons=12,
    candidate_frontiers_checked=['saved-prefix','final-consumed-output'],fresh_repeat_frontiers=10,
    generated_tokens_exact=True,full_greedy_controller_exact=True,canonical_history_exact=True,
    original_prefix_bytes_unchanged=True,idempotent_retry_receipt_exact=True,
    operation_retry_scope='Completed operation is replayed from the unchanged CLI journal; this smoke does not re-POST a lost acknowledgment',
    native_forward_tokens=520,native_prefill_tokens=512,loaded_native_bytes=generation['counter_deltas']['loaded_file_bytes'],
    cli_process_seconds=generation['process_seconds'],startup_seconds=candidate['services'][0]['startup_seconds'],
    decode_compute_ns=generation['counter_deltas']['decode_compute_ns'],decode_prepare_ns=generation['counter_deltas']['decode_prepare_ns'],
    decode_finish_ns=generation['counter_deltas']['decode_finish_ns'],captures=generation['counter_deltas']['captures'],restores=generation['counter_deltas']['restores'],
    hashed_bytes=generation['counter_deltas']['integrity_device_hashed_bytes'],integrity_mismatches=generation['counter_deltas']['integrity_mismatches'],
    native_sha256=candidate['native_sha256'],cli_sha256=candidate['cli_sha256'],profile_sha256=candidate['profile_sha256'],
    execution_profile_id=candidate['execution_profile_id'],model_contract_id=candidate['model_contract_id'],
    candidate_result_sha256=sha(root/'smoke-v2/result.json'),audit_result_sha256=sha(root/'smoke-reference-v2/result.json'),
    runner_sha256=candidate['runner_sha256'],audit_script_sha256=audit['script_sha256'],source_scope=candidate['source_scope'],
    owned_pids=owned,all_owned_pids_absent=True,command_and_wrapper_exits=[0,0],
    preserved_failures=['Smoke v1 rejected root revision 1 due to an incorrect revision-zero assumption; root was valid',
                        'Reference audit v1 stopped after 10 successful repeat vector checks: omitted required snapshot argument to Store.read_lease',
                        'Reference audit v2 reused the two completed fresh reference runs after checking raw-vector hashes, input IDs, native schedule, work and process receipts'],
    limitations=['Single eight-output request, not a throughput benchmark','TTFT/TPOT observer and multi-tree campaign unrun',
                 'Matching profile/operator sequence only; no cross-arrangement equivalence','Greedy controller only',
                 'No new cumulative G1-G5 qualification','Q8 deferred; chat server off; private only'])
target=root/'smoke-primary-v1.json'
with target.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(report,sort_keys=True))
