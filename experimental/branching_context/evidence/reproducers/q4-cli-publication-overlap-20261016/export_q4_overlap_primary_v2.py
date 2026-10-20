"""Audit terminal receipts and summarize the two distinct overlap experiments."""
import base64,hashlib,json,statistics
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-overlap-20261016-v1')
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
control=read(root/'controlled-v1/result.json');bench=read(root/'benchmark-v1/result.json');cpu=read(root/'integrated-cpu-v1/result.json')
assert control['status']==bench['status']==cpu['status']=='PASS'
assert control['numerical_max_error']==bench['numerical_max_error']==0
assert control['comparisons']==4 and bench['comparisons']==28
assert control['controlled_witness']['later_gpu_token_before_first_publication_release']
assert cpu['tests']==9 and not any(cpu[key] for key in ('failures','errors','skips','xfails')) and cpu['source_unchanged']
commands=('controlled-command-v1','benchmark-command-v1','guard-green-command-v1','integrated-cpu-command-v1','g0-command-v2')
for name in commands:
    assert all((root/name/file).read_text().strip()=='0' for file in ('command.exit','wrapper.exit')),name
assert 'G0: PASS: 51 tests' in (root/'g0-command-v2/stdout.log').read_text()
profile=read(Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json'))
owned=[]
for case in (control,bench):
    assert case['service_stop']['exit_code']==case['worker']['native_exit_code']==0
    owned.extend((case['http_pid'],case['worker']['native_pid']))
    assert all(row['unchanged'] for row in case['roots'].values())
    assert case['final_status']['max_active_per_gpu']=={'0':1} and case['final_status']['queued']==0
    assert not any(case['final_status']['active_per_gpu'].values())
assert len(set(owned))==4 and all(not Path('/proc',str(pid)).exists() for pid in owned)
groups=bench['groups'];pairs=[]
for repeat in range(3):
    serial=next(row for row in groups if row['label']=='serial'+str(repeat))
    queued=next(row for row in groups if row['label']=='queued'+str(repeat))
    assert serial['outputs']==queued['outputs']==512 and not serial['deliberate_hold'] and not queued['deliberate_hold']
    pairs.append(dict(repetition=repeat,serial_seconds=serial['seconds'],queued_seconds=queued['seconds'],
                      elapsed_reduction_percent=100*(1-queued['seconds']/serial['seconds'])))
serial_median=statistics.median(row['serial_seconds'] for row in pairs)
queued_median=statistics.median(row['queued_seconds'] for row in pairs)
pool=read(root/'benchmark-v1/pool.json');assert not pool['overflowed']
sources={row['operation_id']:row for row in pool['events'] if row['event']=='request_source_acquired'}
source_overlap=[]
for group in groups:
    if not group['queued']:continue
    for earlier,later in zip(group['admission_order'],group['admission_order'][1:]):
        trace=read(group['operation_traces'][earlier])
        durable=[json.loads(base64.b64decode(row['raw_line_b64'])) for row in trace['requests'][0]['events']]
        durable=next(event for event in durable if event['event']=='durable')
        source_overlap.append(dict(earlier_operation=earlier,later_operation=later,
            later_source_acquired_before_prior_server_durable=sources[later]['monotonic_ns']<durable['time']*1e9))
total_outputs=sum(row['outputs'] for case in (control,bench) for row in case['generations'])
assert total_outputs==3616
report=dict(schema='cow-q4-cli-publication-overlap-primary-v1',status='FOCUSED-PASS',
    qualification='Correct consumed state, FIFO scheduling and controlled publication overlap; natural token overlap not demonstrated',
    host='llm-60',gpu='RTX PRO 6000 Blackwell GPU0',active_requests_per_gpu=1,
    controlled=dict(generations=4,outputs_each=8,deliberate_publication_hold=True,speed_measurement=False,witness=control['controlled_witness'],
                    all_heads_unchanged_until_release=True,no_durable_ack_until_release=True),
    benchmark=dict(generations=28,outputs_each=128,warmup_generations=4,measured_generations=24,pairs=pairs,
        serial_median_observed_client_group_seconds=serial_median,queued_median_observed_client_group_seconds=queued_median,
        ratio_of_median_elapsed_reduction_percent=100*(1-queued_median/serial_median),
        median_paired_elapsed_reduction_percent=statistics.median(row['elapsed_reduction_percent'] for row in pairs),
        serial_median_observed_client_tps=512/serial_median,queued_median_observed_client_tps=512/queued_median,
        natural_later_token_before_prior_durable=sum(row['later_first_token_before_earlier_durable'] for row in bench['natural_overlap_windows']),
        natural_windows=len(bench['natural_overlap_windows']),natural_windows_detail=bench['natural_overlap_windows'],
        source_acquisition_windows=source_overlap,
        timing_scope='Actual instrumented CLI subprocess starts through all subprocess exits, including observer trace serialization/fsync; excludes startup, branch creation and parent numerical/controller audits'),
    generations=32,actual_output_tokens=total_outputs,full_vector_comparisons=32,numerical_max_error=0,all_tokens_exact=True,
    full_greedy_controllers_exact=True,canonical_histories_exact=True,parent_bytes_unchanged=True,zero_saved_prefix_work=True,
    reference_scope=bench['reference_scope'],reference_result_sha256=bench['reference_result_sha256'],new_oracle_processes=0,
    candidate_frontiers_checked='Final consumed frontier for each request only; saved prefixes inherit prior byte-identical qualification',
    numerical_tolerances='Exact bytes; no widened tolerance',CPU_controls=9,G0_tests=51,failures=0,skips=0,xfails=0,
    profile_sha256=bench['profile_sha256'],execution_profile_id=bench['execution_profile_id'],model_contract_id=bench['model_contract_id'],
    model_weights_sha256=profile['model_weights_sha256'],native_sha256=sha(Path(profile['execution']['sut_command'][0])),
    oracle_sha256=sha(Path(profile['execution']['oracle_command'][0])),cli_sha256=bench['cli_sha256'],observer_sha256=bench['observer_sha256'],
    runner_sha256=bench['runner_sha256'],controlled_result_sha256=sha(root/'controlled-v1/result.json'),benchmark_result_sha256=sha(root/'benchmark-v1/result.json'),
    owned_pids=owned,all_owned_pids_absent=True,command_wrapper_exits={name:[0,0] for name in commands},
    preserved_failures=['Root overlap guard RED: module absent','Luna first CPU test expected submit-time conflict; corrected to actual queued stale-revision behavior',
                        'First G0 source package omitted locked-references.json/spec; complete separately frozen package passes',
                        'Primary summary v1 passed a string path to a Path-only read helper; v2 accepts either without rerunning inference'],
    limitations=['Only Q4, greedy, identical execution profile/operator arrangement','No new independently initialized oracle processes; exact prior references reused',
                 'Three source bundles from Strata, not unrelated repositories','No clean uninstrumented/native-only speed comparison',
                 'Natural run had no later generated token before previous durable receipt','No whole-validation-cost 8% qualification',
                 'No LAN timing, physical peak-memory, multi-GPU inference, power-loss or concurrent real-GPU save-failure qualification',
                 'Q8 deferred; original cumulative G1-G5 unqualified; chat stays off; private Git only'])
with (root/'primary-v1.json').open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(dict(status=report['status'],actual_output_tokens=total_outputs,comparisons=32,numerical_max_error=0,
    serial_seconds=serial_median,queued_seconds=queued_median,ratio_of_medians_reduction_percent=report['benchmark']['ratio_of_median_elapsed_reduction_percent'],
    median_paired_reduction_percent=report['benchmark']['median_paired_elapsed_reduction_percent'],
    natural_token_overlap=report['benchmark']['natural_later_token_before_prior_durable'],windows=report['benchmark']['natural_windows'])))
