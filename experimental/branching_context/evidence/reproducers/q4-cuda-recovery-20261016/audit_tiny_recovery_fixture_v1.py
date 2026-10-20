"""Read-only numerical/source audit of the tiny candidate checkpoint fixture."""
import hashlib,json,math,struct
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1')
producer=root/'tiny-prepared-v1';reference=root/'tiny-reference-v2'
def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
p=read(producer/'result.json');r=read(reference/'result.json');fixture=read(root/'inputs/tiny.json')
assert p['status']==r['status']=='PASS' and p['native_exit_success'] and p['native_pid_absent']
assert r['numerical_max_error']==0 and r['comparisons']==16 and r['prefix_controller_exact']
profile_path=Path(p['profile']);profile=read(profile_path)
assert sha(profile_path)==p['profile_sha256']==r['profile_sha256']=='a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755'
assert profile['execution_profile_id']=='fed7e7d5e4b2405fc75e6d300deef7576fe4a0a8ddec88841331144568ada450'
base=read(Path('/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/profiles-v2/q4/profile.json'))
for key in ('execution','hardware','model_contract','model_contract_id','model_weights_sha256'):
    assert profile[key]==base[key],key
for row in p['imports'].values():assert sha(Path(row['path']))==row['sha256']
assert p['imports']['context_time_travel.coordinator']['sha256']=='bb6a1f8d9582c35dfaec69391658a033320e25d717b61ebfb9b0bec1f6c6c32b'
assert p['imports']['context_time_travel.strata_backend']['sha256']=='d692e00310984dcea2332d86862fde0010ff06f50b7f62b35a97d25110443de1'
width=4*248320
comparisons=0
def exact(a,b):
    global comparisons
    assert len(a)==len(b)==width
    assert all(math.isfinite(x) for x in struct.unpack('<248320f',a))
    assert a==b
    comparisons+=1
frames=[]
for row in r['references']:
    directory=reference/row['name'];name=row['name']
    work=read(directory/(name+'.stdout'))
    proc=read(directory/(name+'.process.json'))
    assert proc['status']=='EXITED' and proc['pid']==row['pid'] and proc['exit_code']==0
    assert work['prefill_tokens']==work['prompt_consumed']==128
    assert work['normal_forward_tokens']==len(work['generated'])==12 and work['consumed']==140
    assert row['command'][0]==profile['execution']['oracle_command'][0]
    assert row['command'][row['command'].index('--prompt-cache')+1]=='0'
    assert row['command'][row['command'].index('--conversation-cache-mib')+1]=='0'
    assert Path(row['command'][0])!=Path(profile['execution']['sut_command'][0])
    assert work['generated'][:4]==p['generated_prefix']
    assert work['generated'][4:]==r['expected_followup_tokens']
    raw=(directory/(name+'.f32.decode.f32')).read_bytes()
    prompt=(directory/(name+'.f32.requests.f32')).read_bytes()
    assert len(raw)==12*width and len(prompt)==width
    exact((producer/'base.f32').read_bytes(),prompt)
    exact((producer/'prefix.f32').read_bytes(),raw[3*width:4*width])
    frames.append(raw)
for i in range(12):exact(frames[0][i*width:(i+1)*width],frames[1][i*width:(i+1)*width])
exact((reference/'expected-final.f32').read_bytes(),frames[0][11*width:12*width])
store=producer/'store';source=p['source_snapshot']
manifest=read(store/'checkpoints'/(source+'.json'))['manifest']
payload=store/'checkpoints'/(manifest['payload_sha256']+'.state')
assert sha(payload)==manifest['payload_sha256']==p['checkpoint_payload_sha256']
assert payload.stat().st_size==122511368==manifest['state_bytes']+width
with payload.open('rb') as f:f.seek(manifest['state_bytes']);exact(f.read(),(producer/'prefix.f32').read_bytes())
assert manifest['consumed']==132 and manifest['execution_profile_id']==profile['execution_profile_id']
assert manifest['positions']=='absolute-zero-based' and manifest['schema']==3
seen=set();chunks=[];cursor=source
while cursor is not None:
    assert cursor not in seen;seen.add(cursor)
    path=store/'objects'/(cursor+'.json');assert sha(path)==cursor
    value=read(path);chunks.append(value['inputs']);cursor=value['parent']
assert [t for chunk in reversed(chunks) for t in chunk]==fixture['base_tokens']+p['generated_prefix']
assert sha(producer/'prefix-controller.json')==hashlib.sha256((json.dumps(value if False else read(producer/'prefix-controller.json'),sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
assert len({p['native_pid'],*[x['pid'] for x in r['references']]})==3
assert all(not Path('/proc',str(pid)).exists() for pid in [p['native_pid'],*[x['pid'] for x in r['references']]])
for directory in ('tiny-prepare-command-v1','tiny-reference-command-v2'):
    assert all((root/directory/name).read_text().strip()=='0' for name in ('command.exit','wrapper.exit'))
report=dict(schema='q4-candidate-tiny-fixture-primary-v1',status='PASS',scope='Fresh candidate checkpoint and independent matching-operator numerical reference; recovery UNRUN',
    input_tokens=128,preserved_consumed_tokens=132,prefix_generated=4,reference_generated=12,expected_recovery_tokens=8,
    numerical_max_error=0,independent_full_vector_comparisons=comparisons,fresh_repeat_frontiers=12,
    checkpoint_payload_bytes=payload.stat().st_size,checkpoint_payload_sha256=sha(payload),
    source_snapshot=source,execution_profile_id=profile['execution_profile_id'],profile_sha256=sha(profile_path),
    producer_native_pid=p['native_pid'],reference_native_pids=[x['pid'] for x in r['references']],all_native_exits_zero=True,all_pids_absent=True,
    normal_operator='normal-verifier-decode',fast_prefill_chunk_setting=8192,actual_initial_prefill_tokens=128,
    controller_scope='Deterministic greedy only',prefix_controller_match_reported=True,
    reference_checkpoint_imports=0,producer_result_sha256=sha(producer/'result.json'),reference_result_sha256=sha(reference/'result.json'),
    script_sha256=sha(Path(__file__)),limitations=['No allocation-failure recovery executed yet','No arbitrary batch-arrangement equivalence claim','Not cumulative G1/G2 qualification'])
with (root/'tiny-fixture-primary-v1.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(report,sort_keys=True))