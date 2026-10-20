"""Evict backed-up completed derived children; retain both arms' original roots."""
import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
parent=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1').resolve(strict=True)
control=Path('/home/dflanag3/spatial-artifacts/q4-tree-policy-control-20261016-v1').resolve(strict=True)
lock=open('/home/dflanag3/branching-g3/native-client-speed.gpu0.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(path):return json.loads(path.read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(8*1024**2):digest.update(data)
    return digest.hexdigest()
def canonical(store):
    return {str(path.relative_to(store)):sha(path) for path in sorted(store.rglob('*'))
        if path.is_file() and 'checkpoints' not in path.relative_to(store).parts}
def file(path):
    assert not path.is_symlink() and path.is_file()
    info=path.stat();assert stat.S_ISREG(info.st_mode) and info.st_nlink==1
    return dict(path=str(path),sha256=sha(path),bytes=info.st_size,allocated_bytes=info.st_blocks*512,inode=info.st_ino)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
pair=read(parent/'policy-pair-v2.json');assert pair['all_owned_pids_absent']
assert all(not Path('/proc',str(pid)).exists() for pid in pair['owned_pids'])
backup=read(parent/'pair-backup-v2/readback.json')
assert backup['status']=='PASS' and backup['archive_sha256']=='eb197dec541536ec745a5f7057f9e8f656f6daf1469ab7a1a3747d999bdcbaac'
assert subprocess.check_output(['ssh','box-git','sha256sum '+backup['archive']],text=True).split()[0]==backup['archive_sha256']
for command in ('pair-backup-command-v2','pair-readback-command-v2'):
    assert all((parent/command/name).read_text().strip()=='0' for name in ('command.exit','wrapper.exit'))
selected=[];protected={};canonical_before={};stores={}
for arm in ('control','candidate'):
    campaign=read(Path(pair['arms'][arm]['result_path']));store=Path(campaign['store']).resolve(strict=True)
    assert store.is_relative_to(parent) or store.is_relative_to(control)
    stores[arm]=store;cache=store/'checkpoints'
    roots={row['head']['tip'] for row in campaign['roots'].values()}
    children={row['snapshot'] for row in campaign['generations']}
    assert roots.isdisjoint(children)
    manifests={path.stem:read(path)['manifest'] for path in cache.glob('*.json')}
    assert children.issubset(manifests)
    for sid in sorted(roots):
        manifest=cache/(sid+'.json');payload=cache/(manifests[sid]['payload_sha256']+'.state')
        assert sha(payload)==manifests[sid]['payload_sha256']
        protected[str(manifest)]=sha(manifest);protected[str(payload)]=sha(payload)
    selected.extend(file(cache/(sid+'.json')) for sid in sorted(children))
    for digest in sorted({manifests[sid]['payload_sha256'] for sid in children}):
        assert len(digest)==64 and set(digest)<=set('0123456789abcdef')
        users={sid for sid,manifest in manifests.items() if manifest['payload_sha256']==digest}
        if not users.issubset(children):continue
        row=file(cache/(digest+'.state'));assert row['sha256']==digest;selected.append(row)
    canonical_before[arm]=canonical(store)
plan=dict(schema='cow-policy-pair-child-eviction-v1',status='PLAN',selected=sorted(selected,key=lambda row:row['path']),
    selected_allocated_bytes=sum(row['allocated_bytes'] for row in selected),protected_original_roots=protected,
    canonical_files={name:len(values) for name,values in canonical_before.items()},
    canonical_sha256={name:hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest() for name,values in canonical_before.items()},
    backup_archive=backup['archive'],backup_archive_sha256=backup['archive_sha256'],
    available_bytes_before=os.statvfs(parent).f_bavail*os.statvfs(parent).f_frsize)
destination=parent/('pair-child-eviction-v2.json' if args.apply else 'pair-child-eviction-plan-v2.json')
if args.apply:
    previous=read(parent/'pair-child-eviction-plan-v2.json')
    for key,value in plan.items():
        if key!='available_bytes_before':assert previous[key]==value,key
    for row in selected:
        path=Path(row['path'])
        assert path.resolve().is_relative_to(stores['control']/'checkpoints') or path.resolve().is_relative_to(stores['candidate']/'checkpoints')
        assert str(path) not in protected and file(path)==row
    for row in selected:Path(row['path']).unlink()
    for store in stores.values():
        descriptor=os.open(store/'checkpoints',os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    assert {name:canonical(store) for name,store in stores.items()}==canonical_before
    assert all(sha(Path(path))==digest for path,digest in protected.items())
    plan.update(status='PASS',canonical_bytes_unchanged=True,original_prefix_bytes_unchanged=True,
        available_bytes_after=os.statvfs(parent).f_bavail*os.statvfs(parent).f_frsize)
with destination.open('x') as stream:
    json.dump(plan,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({key:value for key,value in plan.items() if key in ('status','selected_allocated_bytes',
    'canonical_files','available_bytes_before','available_bytes_after')}))
