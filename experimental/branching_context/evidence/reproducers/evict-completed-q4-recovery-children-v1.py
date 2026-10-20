"""Evict completed derived numerical test children after verified compact backup."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1').resolve()
backup=json.loads((root/'backup-recovery-readback-v1.json').read_bytes())
assert backup['status']=='PASS' and backup['archive_sha256']=='9ee31f352a332522b7510e07356d676be1032a5161777d9d487d967e0ccb7fd3'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
def canonical(store):
    return {str(p.relative_to(store)):sha(p) for p in sorted(store.rglob('*'))
            if p.is_file() and 'checkpoints' not in p.relative_to(store).parts}
prepared=[]
for fixture,label in (('tiny','v2'),('target','v1')):
    audit=read(root/(fixture+'-recovery-primary-'+label+'.json'));assert audit['status']=='PASS'
    assert all(not Path('/proc',str(pid)).exists() for pid in audit['owned_pids'])
    store=(root/(fixture+'-input-store-v1')).resolve(strict=True);assert store.is_relative_to(root)
    source=audit['source_snapshot'];children={r['snapshot'] for r in audit['comparisons']}
    assert len(children)==1 and source not in children
    child=children.pop();manifest=(store/'checkpoints'/(child+'.json')).resolve(strict=True)
    assert manifest.parent==store/'checkpoints' and not manifest.is_symlink()
    m=read(manifest)['manifest'];payload=(manifest.parent/(m['payload_sha256']+'.state')).resolve(strict=True)
    assert payload.parent==manifest.parent and not payload.is_symlink()
    assert sha(payload)==m['payload_sha256'] and payload.stat().st_nlink==1
    for other in manifest.parent.glob('*.json'):
        if other!=manifest:assert read(other)['manifest']['payload_sha256']!=m['payload_sha256']
    source_manifest=store/'checkpoints'/(source+'.json');sm=read(source_manifest)['manifest']
    source_payload=store/'checkpoints'/(sm['payload_sha256']+'.state')
    assert source_payload!=payload and sha(source_payload)==audit['source_payload_sha256']
    prepared.append((fixture,store,manifest,payload,source_manifest,source_payload,canonical(store),sha(source_manifest),sha(source_payload)))
rows=[];freed=0
for fixture,store,manifest,payload,sm,sp,before,sm_hash,sp_hash in prepared:
    row=dict(fixture=fixture,store=str(store),child_manifest=str(manifest),child_manifest_sha256=sha(manifest),
             child_payload=str(payload),child_payload_sha256=sha(payload),child_payload_bytes=payload.stat().st_size,
             removed_allocated_bytes=(manifest.stat().st_blocks+payload.stat().st_blocks)*512,
             retained_source_manifest=str(sm),retained_source_manifest_sha256=sm_hash,
             retained_source_payload=str(sp),retained_source_payload_sha256=sp_hash,
             canonical_files=len(before))
    manifest.unlink();payload.unlink()
    fd=os.open(str(manifest.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    assert canonical(store)==before and sha(sm)==sm_hash and sha(sp)==sp_hash
    row['canonical_and_source_unchanged']=True;rows.append(row);freed+=row['removed_allocated_bytes']
report=dict(schema='completed-q4-recovery-child-eviction-v1',status='PASS',rows=rows,
    freed_allocated_bytes=freed,backup_archive_sha256=backup['archive_sha256'],script_sha256=sha(Path(__file__)),
    limitation='Completed child numerical payloads are intentionally absent after eviction; audits record executed pre-eviction state and raw final vectors. Canonical histories and original prefix payloads remain.')
with (root/'completed-child-eviction-v1.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(report,sort_keys=True))
