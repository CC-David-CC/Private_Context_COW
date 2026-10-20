"""Reconcile complete candidate evidence without turning a baseline failure green."""
import hashlib,json,statistics
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
candidate=read(root/'gpu-campaign-v1/result.json')
audit=read(root/'gpu-campaign-reference-v1/result.json')
smoke=read(root/'gpu-smoke-reference-v1/result.json')
cpu=read(root/'cpu-v4.json');focus=read(root/'focus-v6.json')
profile=read(root/'gpu-candidate-v1/profile.json')
assert candidate['status']=='PASS_PIPELINE_NUMERICS_PENDING'
assert audit['status']==smoke['status']=='PASS' and audit['numerical_max_error']==smoke['numerical_max_error']==0
assert cpu['status']=='FAIL' and cpu['tests']==101 and len(cpu['failures'])==1 and not cpu['errors']
assert focus['status']=='PASS' and focus['tests']==17
commands=('gpu-smoke-command-v1','gpu-smoke-reference-command-v1','gpu-campaign-command-v1',
          'gpu-campaign-reference-command-v1','focus-command-v6','g0-command-v7','prepare-command-v1')
for name in commands:
    assert all((root/name/f).read_text().strip()=='0' for f in ('command.exit','wrapper.exit')),name
assert all((root/'baseline-branch-command-v1'/f).read_text().strip()=='1' for f in ('command.exit','wrapper.exit'))
assert 'InvalidOperation not raised' in (root/'baseline-branch-command-v1/stderr.log').read_text()
owned=[]
for source in (candidate,read(root/'gpu-smoke-v1/result.json')):
    for service in source['services']:
        assert service['stop']['exit_code']==service['worker']['native_exit_code']==0
        owned.extend((service['http_pid'],service['worker']['native_pid']))
for reference in audit['references']+smoke['references']:
    assert reference['exit_code']==0;owned.append(reference['pid'])
assert all(not Path('/proc',str(pid)).exists() for pid in owned)
assert candidate['final_status']['max_active_per_gpu']=={'0':1}
rows=candidate['generations'];assert len(rows)==16 and sum(r['output_count'] for r in rows)==2048
hot=[row for row in rows if row['tag']=='round0']
assert len(hot)==6 and all(row['counter_deltas']['loaded_file_bytes']==0 for row in hot)
for row in rows:
    assert row['counter_deltas']['forward_tokens']==row['suffix_tokens']+row['output_count']
    assert row['counter_deltas']['prefill_tokens']==row['suffix_tokens']
for row in candidate['roots'].values():assert row['unchanged']
record=dict(schema='cow-bounded-tree-retention-primary-v1',status='FOCUSED-GPU-PASS; NOT-INTEGRATED',
    expanded_cpu_status='FAIL: one unchanged baseline missing-checkpoint expectation conflict',
    expanded_cpu_tests=101,expanded_cpu_failures=cpu['failures'],focused_cpu_tests=17,G0_tests=51,
    source_commit='eff02e29af5556305a8acca75d41e01d3ca276c9',
    profile_sha256=sha(root/'gpu-candidate-v1/profile.json'),execution_profile_id=profile['execution_profile_id'],
    model_contract_id=profile['model_contract_id'],model_weights_sha256=profile['model_weights_sha256'],
    native_sha256=sha(Path(profile['execution']['sut_command'][0])),oracle_sha256=sha(Path(profile['execution']['oracle_command'][0])),
    changed_production_sha256='7042b31f1f4bb23580cd21213052ad27e8ab861b011dca98a95ede9db2602f62',
    cache_policy=profile['execution']['checkpoint_cache'],generations=17,actual_outputs=2056,
    campaign_generations=16,campaign_outputs=2048,full_vector_comparisons=audit['comparisons']+smoke['comparisons'],
    numerical_max_error=0,all_tokens_controllers_histories_exact=True,parent_bytes_unchanged=True,
    all_owned_pids_absent=True,owned_pids=owned,new_oracle_processes=len(audit['references'])+len(smoke['references']),
    active_requests_per_gpu=1,zero_saved_prefix_forward_work=True,
    campaign_result_sha256=sha(root/'gpu-campaign-v1/result.json'),audit_result_sha256=sha(root/'gpu-campaign-reference-v1/result.json'),
    roots=[dict(prefix_tokens=case['source_tokens'],suffix_tokens=case['suffix_tokens']) for case in hot],
    hot_round=dict(requests=6,loaded_native_file_bytes=0,median_cli_request_seconds=statistics.median(row['process_seconds'] for row in hot),
        total_request_seconds=sum(row['process_seconds'] for row in hot),actual_outputs=sum(row['output_count'] for row in hot),
        requests_detail=[dict(context=row['context_index'],question=row['question'],seconds=row['process_seconds'],
            outputs=row['output_count'],counter_deltas=row['counter_deltas'],arrival_summary=row.get('arrival_summary')) for row in hot]),
    restart=dict(requests=3,loaded_native_file_bytes=[row['counter_deltas']['loaded_file_bytes'] for row in rows if row['tag']=='disk-restart'],
                 seconds=[row['process_seconds'] for row in rows if row['tag']=='disk-restart']),
    limitations=['Q4 greedy matching-native-operator scope only; Q8 deferred',
        'Candidate saved-prefix/final checks; fresh-repeat checks at every dumped frontier',
        'One campaign, six hot measured requests; same-budget old-policy causal speed control UNRUN',
        'Entry limit increased3 to8, byte limit unchanged; do not attribute latency differences solely to code',
        'No sustained pressure or real-GPU OOM/corruption campaign on new retained-image profile',
        'Existing CPU expectation conflict remains failed; no tests/tolerances/equations weakened',
        'No cumulative G1-G5 qualification, chat off, private Git only'])
record['hot_round']['aggregate_cli_tps']=record['hot_round']['actual_outputs']/record['hot_round']['total_request_seconds']
with (root/'primary-v1.json').open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:record[key] for key in ('status','generations','actual_outputs','full_vector_comparisons','numerical_max_error','new_oracle_processes')}))
