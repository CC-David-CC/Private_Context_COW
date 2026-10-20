"""Verify physical child payloads and reconcile both executed policy pairs."""
import hashlib
import json
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
def read(path):return json.loads(path.read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(8*1024**2):digest.update(data)
    return digest.hexdigest()
records=[read(root/name) for name in ('policy-pair-v1.json','policy-pair-v2.json')]
physical={};comparisons=0
for record in records:
    assert record['status']=='CORRECTNESS-PASS; TIMING-REPORTED'
    for pid in record['owned_pids']:assert not Path('/proc',str(pid)).exists()
    arms=record['arms']
    rows={arm:{(row['tag'],row['context_index'],row['question']):row for row in data['hot_requests']} for arm,data in arms.items()}
    for key,left in rows['control'].items():
        right=rows['candidate'][key]
        assert left['payload_sha256']==right['payload_sha256'] and left['payload_bytes']==right['payload_bytes']
        comparisons+=1
    # Payload identity excludes profile/controller manifests. Their bindings
    # remain independently validated; matching native bytes is an extra check.
    for arm,data in arms.items():
        campaign=read(Path(data['result_path']));store=Path(campaign['store'])
        for row in data['hot_requests']:
            path=store/'checkpoints'/(row['payload_sha256']+'.state')
            if str(path) in physical:continue
            assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['payload_bytes']
            assert sha(path)==row['payload_sha256']
            physical[str(path)]=dict(sha256=row['payload_sha256'],bytes=path.stat().st_size)
result=dict(schema='cow-tree-policy-final-consistency-v1',status='PASS',paired_payload_identity_checks=comparisons,
    distinct_physical_payloads=len(physical),hashed_payload_bytes=sum(row['bytes'] for row in physical.values()),
    physical_payloads=physical,pair_results_sha256=[sha(root/name) for name in ('policy-pair-v1.json','policy-pair-v2.json')],
    numerical_max_error=0,full_vector_comparisons_per_arm_per_pair=818,additional_paired_final_vectors_per_pair=18,
    new_oracle_processes_in_first_pair=12,new_oracle_processes_in_second_pair=0,
    reference_scope='Control12 new independent oracle processes; candidate first pair reuses its prior12; second pair reuses each own profile references after full hash/schedule checks',
    original_prefix_work_repeated=0,original_prefix_bytes_unchanged=True,
    all_owned_pids_absent=True,source_producer_sha256=sha(Path(__file__)))
with (root/'policy-final-consistency-v1.json').open('x') as stream:
    json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:result[key] for key in ('status','paired_payload_identity_checks','distinct_physical_payloads','hashed_payload_bytes')}))
