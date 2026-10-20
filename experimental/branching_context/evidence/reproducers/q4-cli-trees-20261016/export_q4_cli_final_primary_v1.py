"""Final read-only consistency audit and compact qualification receipt."""
import hashlib,json
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1')
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
pipeline=read(root/'campaign-v1/result.json');audit=read(root/'campaign-reference-v1/result.json')
qualified=read(root/'qualified-primary-v1.json');backup=read(root/'compact-backup-v1/readback.json')
cpu=read(root/'integrated-harness-v1/result.json')
profile=read(Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json'))
assert qualified['status']=='FOCUSED-PASS' and audit['status']==backup['status']==cpu['status']=='PASS'
assert audit['comparisons']==818 and audit['numerical_max_error']==0
assert len(pipeline['generations'])==32 and qualified['all_outputs']==4096
assert all(row['output_count']==128 for row in pipeline['generations'])
assert qualified['campaign_result_sha256']==audit['campaign_sha256']==sha(root/'campaign-v1/result.json')
assert qualified['audit_result_sha256']==sha(root/'campaign-reference-v1/result.json')
assert len(audit['references'])==12 and len(audit['generations'])==32
assert cpu['tests']==22 and not any(cpu[key] for key in ('failures','errors','skips','xfails')) and cpu['source_unchanged']
commands=('campaign-command-v1','campaign-reference-command-v1','integrated-harness-command-v1','g0-command-v2',
          'qualified-summary-command-v1','chart-command-v3','backup-command-v2','backup-readback-command-v1')
for name in commands:
    assert all((root/name/file).read_text().strip()=='0' for file in ('command.exit','wrapper.exit')),name
assert 'G0: PASS: 51 tests' in (root/'g0-command-v2/stdout.log').read_text()
pids=[]
for service in pipeline['services']:
    assert service['stop']['exit_code']==service['worker']['native_exit_code']==0
    pids.extend((service['http_pid'],service['worker']['native_pid']))
for reference in audit['references']:
    assert reference['exit_code']==0;pids.append(reference['pid'])
assert len(set(pids))==16 and all(not Path('/proc',str(pid)).exists() for pid in pids)
assert pipeline['final_status']['max_active_per_gpu']=={'0':1} and pipeline['final_status']['queued']==0
assert not any(pipeline['final_status']['active_per_gpu'].values())
observations=[status for call in pipeline['calls'] for status in (call['status_before'],call['status_after'])]
report=dict(schema='cow-q4-cli-three-tree-primary-v1',status='FOCUSED-PASS',host='llm-60',
    scope='Actual CLI/HTTP/native three saved source-tree workload, greedy matching profile, one active GPU request',
    prefix_tokens=[7231,7845,8087],suffix_tokens=[512,488,461,448,478,464],output_limit=128,
    generation_requests=32,actual_output_tokens=4096,failures=0,skips=0,xfails=0,
    numerical_max_error=0,full_vector_comparisons=818,
    comparison_breakdown=dict(fresh_repeat=780,candidate_saved_prefix=6,candidate_final=32),
    every_candidate_intermediate_logit_checked=False,all_tokens_exact=True,full_greedy_controllers_exact=True,
    native_forward_work_excludes_saved_prefix=True,parent_checkpoint_bytes_unchanged=True,
    logical_retry_previous_future_retained=True,local_completed_operation_retry_scope='Earlier eight-output smoke; local journal, not server lost-ack replay',
    source_tier_counts=qualified['source_tier_counts'],source_tier_proof='Bounded nonoverflowing pool trace bound by operation/branch/snapshot and request monotonic interval',
    measured_ram_pool_peak_accounted_bytes=max(status['ram']['peak_accounted_bytes'] for status in observations),
    measured_disk_pool_max_file_bytes=max(status['disk']['file_bytes'] for status in observations),
    memory_scope='Internal checkpoint pool accounting; not peak system RAM or VRAM',
    ram_pool_limit_gib=96,disk_pool_limit_gib=128,context_capacity=98304,prefill_chunk=8192,kv_layout='FP16',
    native_sha256=qualified['native_sha256'],oracle_sha256=audit['oracle_sha256'],model_contract_id=qualified['model_contract_id'],
    model_weights_sha256=profile['model_weights_sha256'],execution_profile_id=qualified['execution_profile_id'],
    profile_sha256=qualified['profile_sha256'],cli_sha256=qualified['cli_sha256'],observer_sha256=qualified['observer_sha256'],
    model_verification=audit['model_verification'],combined_cpu_tests=22,G0_tests=51,
    command_wrapper_exits={name:[0,0] for name in commands},owned_pids=pids,all_owned_pids_absent=True,
    measurements_sha256=sha(root/'qualified-primary-v1.json'),audit_sha256=sha(root/'campaign-reference-v1/result.json'),
    png_sha256=sha(root/'latency-breakdown-v2.png'),svg_sha256=sha(root/'latency-breakdown-v2.svg'),
    backup=backup,script_sha256=sha(Path(__file__)),
    numerical_scope='Independent fresh repeat checks every dumped frontier; candidate prefix/final checks only. No arbitrary arrangement equivalence.',
    remaining=['Longer histories/questions/outputs on this combined client version','Pressure/crash/controller expansions','Clean native-without-COW baseline and LAN measurements',
               'Q8 deferred','Cumulative G1-G5 unqualified'],
    preserved_harness_failures=['Initial revision-zero assumption in smoke harness','Read-lease snapshot argument omitted in first audit',
                               'Verifier absent from reduced runtime package; full frozen source G0 succeeds',
                               'Summary expected an unexported preloads counter; corrected to direct timed pool traces',
                               'Plot environment lacked matplotlib; isolated plotting environment used without changing inference environment',
                               'First chart labels overlapped; revised chart visually checked',
                               'Backup v1 had a syntax error before execution; v2 transfers and independently verifies every member'])
with (root/'final-primary-v1.json').open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:report[key] for key in ('status','generation_requests','actual_output_tokens','full_vector_comparisons','numerical_max_error','all_owned_pids_absent','measured_ram_pool_peak_accounted_bytes','measured_disk_pool_max_file_bytes')}))
