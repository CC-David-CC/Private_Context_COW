"""Reconcile executed equal-budget policy arms; preserve all timing outliers."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--control-run',default='campaign-v1')
parser.add_argument('--control-audit',default='reference-v1')
parser.add_argument('--control-command',default='campaign-command-v1')
parser.add_argument('--control-audit-command',default='reference-command-v1')
parser.add_argument('--candidate-run',default='gpu-repeated-v2')
parser.add_argument('--candidate-audit',default='gpu-repeated-reference-v2')
parser.add_argument('--candidate-command',default='gpu-repeated-command-v2')
parser.add_argument('--candidate-audit-command',default='gpu-repeated-reference-command-v2')
parser.add_argument('--output',required=True)
parser.add_argument('--timing-scope',required=True)
args=parser.parse_args()
parent=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
control=Path('/home/dflanag3/spatial-artifacts/q4-tree-policy-control-20261016-v1')

def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')

paths={'control':control/'profile.json','candidate':parent/'gpu-candidate-v1/profile.json'}
profiles={name:read(path) for name,path in paths.items()}
assert profiles['control']['execution']==profiles['candidate']['execution']
assert profiles['control']['hardware']==profiles['candidate']['hardware']
for key in ('model_contract_id','model_weights_sha256'):
    assert profiles['control'][key]==profiles['candidate'][key]
assert profiles['control']['execution']['checkpoint_cache']==dict(byte_limit=8*1024**3,entry_limit=8)
closures={name:p['native_build']['selected_runtime_source_closure']['files_sha256'] for name,p in profiles.items()}
assert closures['control'].keys()==closures['candidate'].keys()
changed=[name for name in closures['control'] if closures['control'][name]!=closures['candidate'][name]]
assert len(changed)==1 and changed[0].replace('\\','/').endswith('context_time_travel/coordinator.py'),changed
arms={};all_pids=[]
for name,root,run,audit_name,command,audit_command in (
    ('control',control,args.control_run,args.control_audit,args.control_command,args.control_audit_command),
    ('candidate',parent,args.candidate_run,args.candidate_audit,args.candidate_command,args.candidate_audit_command)):
    result_path=root/run/'result.json';audit_path=root/audit_name/'result.json'
    campaign=read(result_path);audit=read(audit_path)
    for directory in (command,audit_command):
        for receipt in ('command.exit','wrapper.exit'):
            assert (root/directory/receipt).read_text().strip()=='0',(name,directory,receipt)
    assert campaign['status']=='PASS_PIPELINE_NUMERICS_PENDING' and not campaign['failures']
    assert audit['status']=='PASS' and not audit['failures'] and audit['numerical_max_error']==0
    assert audit['campaign_sha256']==sha(result_path)
    assert campaign['profile_sha256']==audit['profile_sha256']==sha(paths[name])
    assert len(campaign['generations'])==len(audit['generations'])==32
    assert sum(row['output_count'] for row in campaign['generations'])==4096
    assert audit['comparisons']==818 and len(audit['references'])==12
    assert campaign['final_status']['max_active_per_gpu']=={'0':1}
    for row in campaign['roots'].values():assert row['unchanged']
    owned=[]
    for service in campaign['services']:
        assert service['stop']['exit_code']==service['worker']['native_exit_code']==0
        owned.extend((service['http_pid'],service['worker']['native_pid']))
    for reference in audit['references']:
        assert reference['exit_code']==0;owned.append(reference['pid'])
    assert all(not Path('/proc',str(pid)).exists() for pid in owned)
    all_pids.extend(owned)
    for modules in campaign['source_imports'].values():
        for row in modules.values():
            path=Path(row['path']);assert sha(path)==row['sha256']==profiles[name]['native_build']['sha256'][str(path)]
    hot=[]
    for row in campaign['generations']:
        counters=row['counter_deltas']
        assert counters['forward_tokens']==row['suffix_tokens']+row['output_count']
        assert counters['prefill_tokens']==row['suffix_tokens']
        assert sha(Path(row['frontier_file']))==row['frontier_sha256']
        if row['tag'].startswith('round'):
            hot.append(row)
    assert len(hot)==18
    times=[row['process_seconds'] for row in hot]
    plain=[row['process_seconds'] for row in campaign['generations'] if row['tag'].startswith('plain')]
    observed=[row['process_seconds'] for row in campaign['generations'] if row['tag'].startswith('observed')]
    metrics=[row['arrival_summary']['metrics_ns'] for row in hot]
    native_keys=sorted(set.intersection(*(set(row['counter_deltas']) for row in hot)))
    arms[name]=dict(profile_sha256=sha(paths[name]),execution_profile_id=profiles[name]['execution_profile_id'],
        result_path=str(result_path),result_sha256=sha(result_path),audit_path=str(audit_path),audit_sha256=sha(audit_path),
        generations=32,outputs=4096,full_vector_comparisons=818,numerical_max_error=0,
        independent_references=len(audit['references']),new_oracle_processes=audit.get('new_native_executions',len(audit['references'])),
        model_verification=audit['model_verification'],source_closure=closures[name],owned_pids=owned,
        all_owned_pids_absent=True,hot_requests=hot,median_cli_seconds=statistics.median(times),
        total_cli_seconds=sum(times),aggregate_cli_tps=2304/sum(times),
        loaded_native_file_bytes=sum(row['counter_deltas']['loaded_file_bytes'] for row in hot),
        hot_requests_without_native_load=sum(row['counter_deltas']['loaded_file_bytes']==0 for row in hot),
        median_ttft_seconds=statistics.median(row['request_to_first_token'] for row in metrics)/1e9,
        median_tpot_seconds=statistics.median(row['token_interval_mean'] for row in metrics)/1e9,
        median_last_token_to_durable_seconds=statistics.median(row['last_token_to_durable'] for row in metrics)/1e9,
        native_counter_totals={key:sum(row['counter_deltas'][key] for row in hot) for key in native_keys},
        instrumentation_controls=dict(plain_cli_seconds=plain,observed_cli_seconds=observed,
            median_plain_seconds=statistics.median(plain),median_observed_seconds=statistics.median(observed)),
        disk_restart=[row for row in campaign['generations'] if row['tag']=='disk-restart'])

by_arm={name:{(row['tag'],row['context_index'],row['question']):row for row in arm['hot_requests']} for name,arm in arms.items()}
assert by_arm['control'].keys()==by_arm['candidate'].keys()
pairs=[]
for key,left in by_arm['control'].items():
    right=by_arm['candidate'][key]
    assert left['output_tokens']==right['output_tokens'] and left['output_count']==right['output_count']==128
    assert left['source_tokens']==right['source_tokens'] and left['suffix_tokens']==right['suffix_tokens']
    assert left['frontier_sha256']==right['frontier_sha256']
    assert Path(left['frontier_file']).read_bytes()==Path(right['frontier_file']).read_bytes()
    pairs.append(dict(tag=key[0],context_index=key[1],question=key[2],control_seconds=left['process_seconds'],
        candidate_seconds=right['process_seconds'],reduction_percent=100*(1-right['process_seconds']/left['process_seconds']),
        outputs=128,final_frontier_exact=True,control_loaded_bytes=left['counter_deltas']['loaded_file_bytes'],
        candidate_loaded_bytes=right['counter_deltas']['loaded_file_bytes']))
assert arms['candidate']['loaded_native_file_bytes']==0
record=dict(schema='cow-equal-budget-tree-policy-pair-v1',status='CORRECTNESS-PASS; TIMING-REPORTED',
    source_producer_sha256=sha(Path(__file__)),timing_scope=args.timing_scope,
    configuration=dict(execution=profiles['candidate']['execution'],hardware=profiles['candidate']['hardware'],
        model_contract_id=profiles['candidate']['model_contract_id'],model_weights_sha256=profiles['candidate']['model_weights_sha256']),
    native_sha256=sha(Path(profiles['candidate']['execution']['sut_command'][0])),
    oracle_sha256=sha(Path(profiles['candidate']['execution']['oracle_command'][0])),
    changed_production_files=changed,arms=arms,paired_observations=pairs,
    paired_final_vector_comparisons=18,paired_numerical_max_error=0,
    ratio_of_medians_reduction_percent=100*(1-arms['candidate']['median_cli_seconds']/arms['control']['median_cli_seconds']),
    median_paired_reduction_percent=statistics.median(row['reduction_percent'] for row in pairs),
    total_time_reduction_percent=100*(1-arms['candidate']['total_cli_seconds']/arms['control']['total_cli_seconds']),
    owned_pids=sorted(set(all_pids)),all_owned_pids_absent=True,
    limitations=['Three request repetitions within one campaign per arm; not three independent campaigns',
        'Whole CLI timing includes observer serialization/fsync, capture and durable publication; startup and audit separate',
        'Warm/uncontrolled filesystem page cache; native file bytes may stage from RAM and are not physical disk reads',
        'No physical peak-memory measurement or clean native-only baseline; full validation overhead8% target unqualified',
        'Matching native operators, Q4 greedy only; Q8 deferred; no cumulative G1-G5 promotion',
        'Earlier expanded CPU101 remains100pass/1 baseline expectation failure',
        'Publication-handoff fix is a separate unintegrated source and is not covered by these profiles'])
destination=Path(args.output)
assert destination.is_relative_to(parent) or destination.is_relative_to(control)
write(destination,record)
print(json.dumps(dict(status=record['status'],comparison_vectors=18,
    ratio_of_medians_reduction_percent=record['ratio_of_medians_reduction_percent'],
    median_paired_reduction_percent=record['median_paired_reduction_percent'],
    total_time_reduction_percent=record['total_time_reduction_percent'],
    arms={name:{key:arm[key] for key in ('median_cli_seconds','total_cli_seconds','aggregate_cli_tps',
        'loaded_native_file_bytes','hot_requests_without_native_load','median_ttft_seconds',
        'median_tpot_seconds','median_last_token_to_durable_seconds')} for name,arm in arms.items()})))
