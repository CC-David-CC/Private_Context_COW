"""Evict only the completed old-profile OOM test's derived numerical child."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cuda-oom-20261016-v1').resolve(strict=True)
store=(root/'input-store-v1').resolve(strict=True)
assert str(store)==str(root/'input-store-v1') and not (root/'input-store-v1').is_symlink()
r=json.loads((root/'gpu-q4-v2/campaign/result.json').read_bytes())
a=json.loads((root/'primary-audit-v2.json').read_bytes())
b=json.loads((root/'backup-readback-primary-v1.json').read_bytes())
assert r['status']=='FAIL-RECOVERY' and a['audit_status']=='PASS' and b['status']=='PASS'
assert b['archive_sha256']=='936220614c44346d4a9e5d168d206b4670e0dfce81913fe9d9aa20b52e8d31b1'
for pid in (2823816,2823822,2823995,2824006,2824059):assert not Path('/proc',str(pid)).exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
canonical={str(p.relative_to(store)):sha(p) for directory in ('objects','roots') for p in (store/directory).rglob('*') if p.is_file()}
refs=json.loads((store/'refs.json').read_bytes())
for p in (store/refs['directory']).glob('*.json'):canonical[str(p.relative_to(store))]=sha(p)
source=r['expected_source_snapshot']
source_manifest=store/'checkpoints'/(source+'.json')
source_payload=store/'checkpoints'/'423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810.state'
source_hashes={str(p):sha(p) for p in (source_manifest,source_payload)}
snapshot=r['restart_followup']['answer']['snapshot']
assert snapshot=='d0096065cfd11b013441cd9833cddabc6118c28c38d22a2ec71a85cf71fcce0c'
manifest=store/'checkpoints'/(snapshot+'.json')
envelope=json.loads(manifest.read_bytes());m=envelope['manifest']
assert m['snapshot']==snapshot and m['payload_sha256']=='d6d5c666254d997b32bc09e52d45c026dcb424e42829ae831f56b27710029ea0'
payload=store/'checkpoints'/(m['payload_sha256']+'.state')
assert sha(payload)==m['payload_sha256'] and payload.stat().st_size==m['state_bytes']+4*m['logits_count']
rows=[]
for p in (manifest,payload):
    assert p.resolve(strict=True).parent==store/'checkpoints' and not p.is_symlink()
    st=p.stat();rows.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=sha(p),links=st.st_nlink))
assert all(row['links']==1 for row in rows)
for p in (manifest,payload):p.unlink()
fd=os.open(store/'checkpoints',os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert all(not Path(row['path']).exists() for row in rows)
assert all(sha(store/path)==digest for path,digest in canonical.items())
assert all(sha(Path(path))==digest for path,digest in source_hashes.items())
report=dict(schema='completed-oom-derived-cache-eviction-v1',status='PASS',removed_files=rows,
    freed_allocated_bytes=sum(row['allocated_bytes'] for row in rows),canonical_files_verified=len(canonical),canonical_unchanged=True,
    source_checkpoint_unchanged=True,source_payload_bytes=source_payload.stat().st_size,
    semantics='Only optional derived child checkpoint evicted; canonical history and original prefix checkpoint retained. Compact backup excludes full numerical payloads.',
    backup_archive_sha256=b['archive_sha256'],script_sha256=sha(Path(__file__)))
with (root/'derived-child-evicted-v1.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(report,sort_keys=True))