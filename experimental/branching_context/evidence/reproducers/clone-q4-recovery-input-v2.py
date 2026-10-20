"""Independent input-store copy for the candidate recovery fault campaign."""
import hashlib,json,os,shutil,sys
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1')
fixture=sys.argv[1]
assert fixture in ('tiny','target')
prepared=root/(fixture+'-prepared-v1')
r=json.loads((prepared/'result.json').read_bytes())
assert r['status']=='PASS' and r['native_exit_success'] and r['native_pid_absent']
assert not Path('/proc',str(r['native_pid'])).exists()
assert r['profile_sha256']=='a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755'
assert r['consumed']==r['base_tokens']+4
source=prepared/'store';target=root/(fixture+'-input-store-v1')
target.mkdir(mode=0o700)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
rows=[]
for path in sorted(source.rglob('*')):
    assert not path.is_symlink()
    rel=path.relative_to(source);out=target/rel
    if path.is_dir():out.mkdir(mode=0o700,exist_ok=True)
    elif path.is_file():
        if path.suffix=='.state':continue
        shutil.copyfile(path,out);os.chmod(out,0o600)
        assert sha(path)==sha(out) and path.stat().st_ino!=out.stat().st_ino
        rows.append(dict(path=str(rel),sha256=sha(out),bytes=out.stat().st_size))
manifest=source/'checkpoints'/(r['source_snapshot']+'.json')
m=json.loads(manifest.read_bytes())['manifest']
assert m['execution_profile_id']==r['execution_profile_id'] and m['consumed']==r['consumed']
payload=source/'checkpoints'/(m['payload_sha256']+'.state')
assert sha(payload)==m['payload_sha256']==r['checkpoint_payload_sha256']
out=target/'checkpoints'/payload.name
os.link(payload,out)
assert out.stat().st_ino==payload.stat().st_ino and sha(out)==m['payload_sha256']
report=dict(status='PREPARED_NOT_RUN',schema='candidate-oom-input-copy-v1',profile_sha256=r['profile_sha256'],
    source_snapshot=r['source_snapshot'],source_tokens=r['consumed'],metadata=rows,metadata_files=len(rows),
    payload_sha256=m['payload_sha256'],payload_bytes=payload.stat().st_size,payload_hardlinked=True,
    source_store=str(source),input_store=str(target),script_sha256=sha(Path(__file__)))
with (root/(fixture+'-input-preparation-v1.json')).open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='metadata'},sort_keys=True))
if fixture=='target':
    with (root/'target-source-snapshot.txt').open('x') as f:f.write(r['source_snapshot']+'\n')
