"""Attribute actual CLI arrival, publication and source-tier measurements."""
import argparse,base64,hashlib,json,statistics
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--require-audit',action='store_true')
args=parser.parse_args()
root=args.root;campaign=root/'campaign-v1'
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)
data=read(campaign/'result.json')
require(data['status']=='PASS_PIPELINE_NUMERICS_PENDING','Candidate pipeline did not pass')
for directory in ('campaign-command-v1','integrated-harness-command-v1','g0-command-v2'):
    require(all((root/directory/file).read_text().strip()=='0' for file in ('command.exit','wrapper.exit')),'Nonzero wrapper '+directory)
calls={row['label']:row for row in data['calls']}
pool_events=[]
for index in range(1,len(data['services'])+1):
    trace=read(campaign/f'pool-{index}.json')
    require(not trace['overflowed'],'Pool trace overflowed')
    pool_events.extend(dict(service=index,**event) for event in trace['events'])
rows=[]
for generation in data['generations']:
    row=calls[generation['call_label']]
    matches=[event for event in pool_events if event['event']=='request_source_acquired' and event.get('operation_id')==generation['operation']]
    require(len(matches)==1,'Need direct source acquisition trace')
    source=matches[0]
    require(source['source_snapshot']==generation['source_snapshot'] and source['branch']==generation['branch'],'Pool source binding differs')
    if source['source_kind']=='model_cache':
        tier='native snapshot'
    else:
        require(source['source_kind']=='shared_pool' and generation['observed'],'Shared-pool classification requires request clock binding')
        request_start=row['observation']['requests'][0]['request_started_monotonic_ns']
        starts=[event for event in pool_events if event['event']=='disk_preload_started' and event['service']==source['service']
                and event['key']==source['key'] and request_start<=event['monotonic_ns']<=source['monotonic_ns']]
        require(len(starts)<=1,'Ambiguous disk preload during one serial request')
        tier='disk to RAM' if starts else 'RAM pool'
        if starts:
            start=starts[0]
            succeeded=[event for event in pool_events if event['event']=='disk_preload_succeeded' and event.get('load_id')==start['load_id'] and event['service']==source['service']]
            require(len(succeeded)==1 and succeeded[0]['monotonic_ns']<=source['monotonic_ns'],'Disk preload did not complete before source pin')
    counters=generation['counter_deltas']
    require(counters['forward_tokens']==generation['suffix_tokens']+generation['output_count'] and counters['prefill_tokens']==generation['suffix_tokens'],'Incorrect prefix work')
    require(counters.get('integrity_mismatches',0)==0,'Integrity mismatch')
    small=dict(context=generation['context_index'],question=generation['question'],tag=generation['tag'],branch=generation['branch'],
               outputs=generation['output_count'],source_tier=tier,source_trace_sequence=source['sequence'],service=source['service'],
               cli_process_seconds=generation['process_seconds'],native_loaded_file_bytes=counters.get('loaded_file_bytes'),
               source_tokens=generation['source_tokens'],suffix_tokens=generation['suffix_tokens'],forward_tokens=counters['forward_tokens'],
               prefill_tokens=counters['prefill_tokens'],captures=counters.get('captures'),restores=counters.get('restores'),
               hashed_bytes=counters.get('integrity_device_hashed_bytes'),copied_bytes=counters.get('copied_bytes'),
               native_generation_stages_ms={name:counters['decode_'+name+'_ns']/1e6 for name in ('prepare','compute','finish')})
    stages=small['native_generation_stages_ms']
    small['native_prepare_share_percent']=100*stages['prepare']/sum(stages.values())
    if generation['observed']:
        observation=row['observation']
        require(observation['cli_exit_code']==0 and observation['cli_exception'] is None,'CLI observer failed')
        request=observation['requests'][0];summary=request['summary']
        require(summary['validation']==dict(status='valid',errors=[]) and summary['token_ids']==generation['output_tokens'],'Invalid observation')
        metrics=summary['metrics_ns']
        small.update(ttft_seconds=metrics['request_to_first_token']/1e9,
            token_gap_median_ms=metrics['token_interval_median']/1e6,
            token_gap_p95_ms=metrics['token_interval_p95']/1e6,
            token_gap_max_ms=metrics['token_interval_max']/1e6,
            steady_arrival_tps=summary['generation_tokens_per_second'],
            last_token_to_durable_seconds=metrics['last_token_to_durable']/1e9,
            request_to_durable_seconds=metrics['request_to_durable']/1e9)
        complete=[json.loads(base64.b64decode(event['raw_line_b64'])) for event in request['events']]
        answer=next(event['answer'] for event in complete if event['event']=='complete')
        small['engine_generation_seconds']=answer['generation_seconds']
        small['engine_request_seconds']=answer['request_seconds']
        small['engine_generation_tps']=small['outputs']/answer['generation_seconds']
    rows.append(small)
def aggregate(selected):
    require(selected,'Empty measurement group')
    keys=('cli_process_seconds','ttft_seconds','token_gap_median_ms','token_gap_p95_ms','token_gap_max_ms',
          'steady_arrival_tps','last_token_to_durable_seconds','request_to_durable_seconds','engine_generation_tps','native_prepare_share_percent')
    return dict(requests=len(selected),outputs=sum(row['outputs'] for row in selected),
                medians={key:statistics.median(row[key] for row in selected if key in row) for key in keys if any(key in row for row in selected)})
main=[row for row in rows if row['tag'].startswith('round')]
resident=[row for row in rows if row['tag'].startswith('observed')]
restart=[row for row in rows if row['tag']=='disk-restart']
pairs=[]
for repeat in range(data['repetitions']):
    plain=next(row for row in rows if row['tag']=='plain'+str(repeat))
    observed=next(row for row in rows if row['tag']=='observed'+str(repeat))
    require(plain['source_tier']==observed['source_tier']=='native snapshot','Paired controls are not both native reuse')
    pairs.append(dict(repetition=repeat,plain_seconds=plain['cli_process_seconds'],observed_seconds=observed['cli_process_seconds'],
                      observed_wall_change_percent=100*(observed['cli_process_seconds']/plain['cli_process_seconds']-1)))
pair_median=statistics.median(row['observed_wall_change_percent'] for row in pairs)
report=dict(schema='cow-q4-cli-three-tree-measurements-v1',status='MEASUREMENTS_COMPLETE_NUMERICS_PENDING',
    summarizer_sha256=sha(Path(__file__)),
    host='llm-60',gpu='RTX PRO 6000 Blackwell GPU0',profile_sha256=data['profile_sha256'],execution_profile_id=data['execution_profile_id'],
    model_contract_id=data['model_contract_id'],native_sha256=data['native_sha256'],cli_sha256=data['cli_sha256'],observer_sha256=data['observer_sha256'],
    campaign_result_sha256=sha(campaign/'result.json'),source_scope=data['source_scope'],rows=rows,
    alternating_trees=aggregate(main),native_snapshot_reuse=aggregate(resident),process_restart_disk=aggregate(restart),
    paired_observer_controls=pairs,paired_observer_median_wall_change_percent=pair_median,
    observer_overhead_exceeds_8_percent=pair_median>8,
    validation_cost_scope='Observer overhead only. Native preparation share includes synchronization/COW/hash work; neither is total validation overhead.',
    source_tier_counts={tier:sum(row['source_tier']==tier for row in rows) for tier in sorted(set(row['source_tier'] for row in rows))},
    branch_actions=data['branch_actions'],services=data['services'],source_roots_unchanged=all(row['unchanged'] for row in data['roots'].values()),
    all_outputs=sum(row['outputs'] for row in rows),generation_requests=len(rows),
    final_pool_status=data['final_status'],filesystem_cache=data['filesystem_cache'],
    limitations=['No controlled cold filesystem cache','Three Strata source bundles, not unrelated repositories',
                 'Client runs on inference host; no LAN transport measurement','Candidate saved-prefix/final-frontier sampling only',
                 'Greedy matching-profile scope; no arbitrary arrangement equivalence','No peak RSS/VRAM claim',
                 'Q8 deferred; no cumulative G1-G5 promotion'])
if args.require_audit:
    audit=read(root/'campaign-reference-v1/result.json')
    require(audit['status']=='PASS' and audit['numerical_max_error']==0,'Independent numerical audit did not pass')
    require(audit['campaign_sha256']==report['campaign_result_sha256'],'Audit binds another campaign')
    require(len(audit['generations'])==len(rows),'Audit omits generations')
    require(all((root/'campaign-reference-command-v1'/name).read_text().strip()=='0' for name in ('command.exit','wrapper.exit')),'Reference wrapper failed')
    report.update(status='FOCUSED-PASS',numerical_max_error=0,full_vector_comparisons=audit['comparisons'],
                  audit_result_sha256=sha(root/'campaign-reference-v1/result.json'),fresh_oracle_processes=len(audit['references']),
                  tokens_and_full_greedy_controllers_exact=True)
with args.output.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:report[key] for key in ('status','generation_requests','all_outputs','source_tier_counts','alternating_trees','native_snapshot_reuse','process_restart_disk','paired_observer_median_wall_change_percent','observer_overhead_exceeds_8_percent')}))
