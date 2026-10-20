"""Independent read-only audit: real OOM lifecycle and resumed final numerics."""
import hashlib,json,math,struct,sys
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1')
fixture,label=sys.argv[1:]
assert fixture in ('tiny','target') and label.startswith('v') and label[1:].isdigit()
campaign=root/(fixture+'-recovery-'+label)/'campaign'
output=root/(fixture+'-recovery-primary-'+label+'.json')
assert not output.exists()
def read(path): return json.loads(path.read_bytes())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024**2): h.update(b)
    return h.hexdigest()
def need(value,message):
    if not value: raise AssertionError(message)
report=dict(schema='q4-cuda-recovery-independent-v1',status='RUNNING',fixture=fixture,
            scope='Actual allocation-failure worker recovery and separate-process disk restore, deterministic greedy only',
            script_sha256=sha(Path(__file__)),failures=[],skips=0,numerical_max_error=None)
try:
    r=read(campaign/'result.json');producer=read(root/(fixture+'-prepared-v1')/'result.json')
    reference=root/(fixture+'-reference-v2');expected=read(reference/'result.json')
    need(r['status']=='PASS','Lifecycle harness did not pass')
    need(producer['status']==expected['status']=='PASS','Missing passing independent fixtures')
    for directory in (fixture+'-recovery-run-command-'+label,):
        for name in ('command.exit','wrapper.exit'):
            need((root/directory/name).read_text().strip()=='0',directory+'/'+name)
    source=producer['source_snapshot'];store=Path(r['store']);profile=read(Path(r['profile_path']))
    need(r['profile_sha256']==producer['profile_sha256']==expected['profile_sha256']==sha(Path(r['profile_path'])), 'Profile binding')
    need(profile['execution_profile_id']==producer['execution_profile_id'],'Execution identity')
    need(r['native_sha256']==sha(Path(r['native_binary'])),'Native binary hash')
    for row in r['imports']['modules'].values():need(sha(Path(row['path']))==row['sha256'],'Runtime import '+row['path'])
    source_record=store/'objects'/(source+'.json')
    need(sha(source_record)==source,'Canonical source content identity')
    need(source_record.read_bytes()==(root/(fixture+'-prepared-v1')/'store/objects'/(source+'.json')).read_bytes(),'Original canonical source changed')
    manifest_doc=read(store/'checkpoints'/(source+'.json'));manifest=manifest_doc['manifest']
    need((store/'checkpoints'/(source+'.json')).read_bytes()==(root/(fixture+'-prepared-v1')/'store/checkpoints'/(source+'.json')).read_bytes(),'Original source manifest changed')
    source_payload=store/'checkpoints'/(manifest['payload_sha256']+'.state')
    need(sha(source_payload)==manifest['payload_sha256']==producer['checkpoint_payload_sha256'],'Source payload mutation')
    need(manifest['consumed']==producer['consumed'],'Source position')
    source_ref=read(store/'references/CliA.json')
    need(dict(name=source_ref['name'],**source_ref['head'])==r['source_before']['head'],'Canonical source reference')
    oom=r['native_oom'];need(oom['response']['status']=='error','Missing native error')
    need(oom['stats_after']['vmm_last_failed_operation']=='cuMemCreate' and oom['stats_after']['vmm_last_driver_error']==2,'Not real CUDA OOM 2')
    need(oom['oom_failure_delta']>0,'No actual SUT allocation failure')
    need(oom['response']['message']==oom['original_error']==r['fault_http']['evidence']['error']['error'],'Original cause lost')
    need(not any(e.get('event') in ('durable','complete') for e in r['fault_http']['events']),'Fault acknowledged as durable')
    helper=r['helper'];need(helper['released_before_gate_abort'] and helper['exit_code']==0 and helper['pid_absent'],'Pressure-helper cleanup')
    need(helper['pressure']['held_bytes']==helper['release']['released_bytes']>0,'Helper allocation/release mismatch')
    trace=r['first_service']['trace_document'];events=trace['events']
    old=r['first_service']['old_native']['pid'];new=r['same_service']['replacement']['new_native_process']['pid']
    old_close=[e for e in events if e.get('event')=='native_worker_closed' and e.get('pid')==old]
    replacement=[e for e in events if e.get('event')=='native_worker_replaced']
    new_close=[e for e in events if e.get('event')=='native_worker_closed' and e.get('pid')==new]
    need(not trace['overflowed'] and trace['event_count']==len(events),'Truncated lifecycle trace')
    need(len(old_close)==len(replacement)==len(new_close)==1,'Lifecycle event count')
    need(old_close[0]['sequence']<replacement[0]['sequence']<new_close[0]['sequence'],'Replacement lifecycle ordering')
    need(replacement[0]['old_pid']==old and replacement[0]['new_pid']==new and replacement[0]['old_returncode']==old_close[0]['returncode'],'Replacement identity')
    need(new_close[0]['returncode']==0 and new_close[0]['close_succeeded'],'Replacement native did not close cleanly')
    need(r['same_service']['replacement']['worker_status']==dict(health='healthy',restarts=1,pid=new),'Replacement health')
    fresh_http=r['clean_restart_service']['http_pid'];fresh_native=r['clean_restart_service']['native_process']['pid']
    need(fresh_http!=r['first_service']['http_pid'] and len({old,new,fresh_native})==3,'Processes not independent')
    fresh_closes=[e for e in r['clean_restart_service']['trace_document']['events'] if e.get('event')=='native_worker_closed' and e.get('pid')==fresh_native]
    need(len(fresh_closes)==1 and fresh_closes[0]['returncode']==0 and fresh_closes[0]['close_succeeded'],'Restart native exit')
    for s in (r['first_service'],r['clean_restart_service']):need(s['stop']['exit_code']==0 and s['stop']['pid_absent_after_stop'],'HTTP exit')
    pids=[old,new,fresh_native,r['first_service']['http_pid'],fresh_http,helper['identity']['pid']]
    need(all(not Path('/proc',str(pid)).exists() for pid in pids),'Owned process remains live')
    expected_tokens=expected['expected_followup_tokens']
    need(len(expected_tokens)==8,'Fixture expected output length')
    expected_frontier=(reference/'expected-final.f32').read_bytes()
    expected_controller=read(reference/'expected-final-controller.json')
    width=4*248320;need(len(expected_frontier)==width,'Expected frontier incomplete')
    need(all(math.isfinite(x) for x in struct.unpack('<248320f',expected_frontier)),'Nonfinite reference')
    reference_pairs=0
    def exact(left,right,context):
        global reference_pairs
        need(len(left)==len(right)==width,context+' incomplete frontier')
        lv=struct.unpack('<248320f',left);rv=struct.unpack('<248320f',right)
        need(all(math.isfinite(x) for x in lv+rv),context+' nonfinite')
        maximum=max(abs(a-b) for a,b in zip(lv,rv))
        report['numerical_max_error']=max(report['numerical_max_error'] or 0,maximum)
        need(left==right,context+' differs, max='+str(maximum))
        reference_pairs+=1
    need(expected['comparisons']==16 and expected['numerical_max_error']==0 and expected['prefix_controller_exact'],'Fresh reference prerequisite')
    reference_frames=[];reference_pids=[]
    base_count=producer['base_tokens']
    for row in expected['references']:
        name=row['name'];directory=reference/name
        work=read(directory/(name+'.stdout'));process=read(directory/(name+'.process.json'))
        need(process['status']=='EXITED' and process['exit_code']==0 and process['pid']==row['pid'],'Fresh process exit')
        need(work['prefill_tokens']==work['prompt_consumed']==base_count,'Fresh prefix work')
        need(work['normal_forward_tokens']==len(work['generated'])==12 and work['consumed']==base_count+12,'Fresh output work')
        need(work['generated']==producer['generated_prefix']+expected_tokens,'Fresh exact token IDs')
        command=row['command']
        need(command[0]==profile['execution']['oracle_command'][0] and command[0]!=r['native_binary'],'Independent oracle executable')
        for flag in ('--prompt-cache','--conversation-cache-mib'):
            need(command[command.index(flag)+1]=='0','Oracle cache was enabled')
        reference_pids.append(row['pid'])
        raw=(directory/(name+'.f32.decode.f32')).read_bytes()
        need(len(raw)==12*width,'Fresh vector count')
        exact((root/(fixture+'-prepared-v1')/'base.f32').read_bytes(),(directory/(name+'.f32.requests.f32')).read_bytes(),name+' base')
        exact((root/(fixture+'-prepared-v1')/'prefix.f32').read_bytes(),raw[3*width:4*width],name+' generated4')
        reference_frames.append(raw)
    need(len(reference_frames)==2,'Two fresh reference processes required')
    for i in range(12):exact(reference_frames[0][i*width:(i+1)*width],reference_frames[1][i*width:(i+1)*width],'Fresh repeat '+str(i+1))
    exact(expected_frontier,reference_frames[0][11*width:12*width],'Expected final frontier')
    with source_payload.open('rb') as f:
        f.seek(manifest['state_bytes']);exact(f.read(),(root/(fixture+'-prepared-v1')/'prefix.f32').read_bytes(),'Preserved prefix frontier')
    need(all(not Path('/proc',str(pid)).exists() for pid in [producer['native_pid'],*reference_pids]),'Producer or reference process remains live')
    comparisons=[]
    def lineage(sid):
        chunks=[];seen=set();records=[]
        while sid is not None:
            need(sid not in seen,'Canonical cycle');seen.add(sid)
            path=store/'objects'/(sid+'.json');need(sha(path)==sid,'Canonical object hash')
            record=read(path);chunks.append(record['inputs']);records.append(sid);sid=record['parent']
        return [t for chunk in reversed(chunks) for t in chunk],records
    expected_history=read(root/'inputs'/(fixture+'.json'))['base_tokens']+producer['generated_prefix']+expected_tokens
    for name,key,branch_key in (('same_service','same_service','fault_branch'),('clean_restart','clean_restart','restart_branch')):
        follow=r[key]['followup'];answer=follow['answer'];sid=answer['snapshot'];ev=follow['events']
        need(answer['tokens']==expected_tokens,name+' output token mismatch')
        work=follow['counter_deltas']
        need(work['forward_tokens']==8 and work['consumed_tokens']==8 and work['prefill_tokens']==0,name+' replayed prefix or wrong work')
        durable=[(i,e) for i,e in enumerate(ev) if e.get('event')=='durable'];complete=[(i,e) for i,e in enumerate(ev) if e.get('event')=='complete']
        need(len(durable)==len(complete)==1 and durable[0][0]<complete[0][0],name+' durable event ordering')
        receipt=complete[0][1]['receipt'];need(durable[0][1]['receipt']==receipt,name+' conflicting receipts')
        need(receipt['snapshot']==receipt['branch']['tip']==sid,name+' receipt snapshot')
        need(receipt['branch']['name']==r[branch_key] and receipt['branch']['revision']==follow['branch_before']['revision']+1,name+' revision')
        need(receipt['execution_profile_id']==profile['execution_profile_id'] and receipt['model_contract_id']==profile['model_contract_id'],name+' receipt identities')
        reference_record=read(store/'references'/(r[branch_key]+'.json'))
        need(dict(name=reference_record['name'],**reference_record['head'])==follow['branch_after'],name+' stored reference head')
        record=read(store/'objects'/(sid+'.json'))
        need(record['controller']==expected_controller,name+' full controller mismatch')
        history,ancestors=lineage(sid)
        need(history==expected_history and source in ancestors,name+' wrong canonical history')
        m=read(store/'checkpoints'/(sid+'.json'))['manifest'];payload=store/'checkpoints'/(m['payload_sha256']+'.state')
        need(sha(payload)==m['payload_sha256']==receipt['payload_sha256'],name+' checkpoint hash')
        need(m['consumed']==len(expected_history) and m['positions']=='absolute-zero-based' and m['execution_profile_id']==profile['execution_profile_id'],name+' checkpoint metadata')
        with payload.open('rb') as f:f.seek(m['state_bytes']);actual=f.read()
        need(len(actual)==width,name+' incomplete saved frontier')
        values=struct.unpack('<248320f',actual);need(all(math.isfinite(x) for x in values),name+' nonfinite frontier')
        maximum=max(abs(a-b) for a,b in zip(values,struct.unpack('<248320f',expected_frontier)))
        need(actual==expected_frontier,name+' final frontier mismatch, max='+str(maximum))
        comparisons.append(dict(branch=name,snapshot=sid,full_vector_max_error=maximum,controller_exact=True,canonical_tokens=len(history),forward_tokens=8,prefill_tokens=0,payload_bytes=payload.stat().st_size,counter_deltas=work))
    report.update(status='PASS',numerical_max_error=0,full_vector_comparisons=len(comparisons),independent_reference_comparisons=reference_pairs,total_full_vector_comparisons=len(comparisons)+reference_pairs,comparisons=comparisons,
        source_snapshot=source,source_consumed=producer['consumed'],source_payload_bytes=source_payload.stat().st_size,source_payload_sha256=sha(source_payload),
        helper_allocated_bytes=helper['pressure']['held_bytes'],native_oom_failures=oom['oom_failure_delta'],native_error=oom['original_error'],
        original_native_exit=old_close[0]['returncode'],replacement_native_exit=0,restart_native_exit=0,all_owned_pids_absent=True,owned_pids=pids,
        profile_sha256=r['profile_sha256'],execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],native_sha256=r['native_sha256'],
        candidate_coordinator_sha256=r['imports']['modules']['context_time_travel.coordinator']['sha256'],candidate_backend_sha256=r['imports']['modules']['context_time_travel.strata_backend']['sha256'],
        lifecycle_result_sha256=sha(campaign/'result.json'),reference_result_sha256=sha(reference/'result.json'),
        measurement_scope='Fault-injection lifecycle experiment, not throughput benchmark',
        numerical_sampling='Both final consumed frontiers and complete output token/controller/history records; no per-token restored-logit assertion',
        limitations=['Deterministic greedy only','No arbitrary execution-arrangement equivalence','Not cumulative G1-G5 qualification'])
except BaseException as exc:
    report['status']='FAIL';report['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
finally:
    with output.open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},sort_keys=True))
if report['status']!='PASS':raise SystemExit(1)
