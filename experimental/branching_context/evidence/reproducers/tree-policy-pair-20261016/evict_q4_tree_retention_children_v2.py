"""Evict only audited completed child caches and a verified transfer archive."""
import argparse,hashlib,json,os,stat,subprocess
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1').resolve(strict=True)
store=root/'workspace-store-v1';cache=store/'checkpoints'
def read(path):return json.loads(path.read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while raw:=stream.read(8*1024**2):digest.update(raw)
    return digest.hexdigest()
def file(path):
    assert not path.is_symlink() and path.is_file()
    info=path.stat();assert stat.S_ISREG(info.st_mode) and info.st_nlink==1
    return dict(path=str(path),sha256=sha(path),bytes=info.st_size,allocated_bytes=info.st_blocks*512,inode=info.st_ino)
def canonical():
    return {str(path.relative_to(store)):sha(path) for path in sorted(store.rglob('*'))
            if path.is_file() and 'checkpoints' not in path.relative_to(store).parts}
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
primary=read(root/'primary-v1.json');assert primary['status']=='FOCUSED-GPU-PASS; NOT-INTEGRATED'
assert all(not Path('/proc',str(pid)).exists() for pid in primary['owned_pids'])
backup=read(root/'compact-backup-v1/readback.json')
assert backup['status']=='PASS' and backup['archive_sha256']=='b944c6f5fb671c9268ee3d38bc0cad7ff26d3608e84c8ce81b51536239fad803'
assert subprocess.check_output(['ssh','box-git','sha256sum '+backup['archive']],text=True).split()[0]==backup['archive_sha256']
for directory in ('gpu-smoke-command-v1','gpu-campaign-command-v1','gpu-campaign-reference-command-v1','backup-readback-command-v1'):
    assert all((root/directory/name).read_text().strip()=='0' for name in ('command.exit','wrapper.exit'))
campaign=read(root/'gpu-campaign-v1/result.json');smoke=read(root/'gpu-smoke-v1/result.json')
protected={row['head']['tip'] for row in campaign['roots'].values()}
children={row['snapshot'] for data in (campaign,smoke) for row in data['generations']}
assert children.isdisjoint(protected)
manifests={path.stem:read(path)['manifest'] for path in cache.glob('*.json')}
assert children.issubset(manifests)
root_files={}
for sid in protected:
    manifest=cache/(sid+'.json');payload=cache/(manifests[sid]['payload_sha256']+'.state')
    assert sha(payload)==manifests[sid]['payload_sha256']
    root_files[str(manifest)]=sha(manifest);root_files[str(payload)]=sha(payload)
selected=[file(cache/(sid+'.json')) for sid in sorted(children)]
for digest in sorted({manifests[sid]['payload_sha256'] for sid in children}):
    assert len(digest)==64 and set(digest)<=set('0123456789abcdef')
    users={sid for sid,manifest in manifests.items() if manifest['payload_sha256']==digest}
    if not users.issubset(children):continue
    path=cache/(digest+'.state');row=file(path);assert row['sha256']==digest
    selected.append(row)
# This full transfer archive is reproducible from private Git commit7c97c1d7.
# Selected staged sources and failed-attempt records are in the verified backup.
archive=root/'red-stage/source.tar';archive_row=file(archive)
assert archive_row['sha256']=='cd280367bf59106d2de1316058d9f8caa3a877482903c4584e180ee9ea9782aa'
selected.append(archive_row)
before=canonical();available=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
plan=dict(schema='cow-audited-child-cache-eviction-v1',status='PLAN',selected=selected,
    selected_allocated_bytes=sum(row['allocated_bytes'] for row in selected),protected_root_files=root_files,
    canonical_files=len(before),canonical_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
    available_bytes_before=available,backup_archive_sha256=backup['archive_sha256'],children=len(children),
    scope='Only completed child numerical copies and reproducible transfer tar; preserve all canonical files and original prefixes')
if not args.apply:
    with (root/'child-eviction-plan-v1.json').open('x') as stream:json.dump(plan,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({key:plan[key] for key in ('status','children','selected_allocated_bytes','canonical_files','available_bytes_before')}))
else:
    expected=read(root/'child-eviction-plan-v1.json')
    # Free space can change as the plan/launcher logs are published, and set
    # iteration order is not an ownership invariant. Compare every selected
    # inode/hash/extent and all canonical/root guards exactly.
    for key,value in plan.items():
        if key=='available_bytes_before':continue
        if key=='selected':assert sorted(expected[key],key=lambda row:row['path'])==sorted(value,key=lambda row:row['path'])
        else:assert expected[key]==value,key
    for row in selected:
        path=Path(row['path']);assert path.resolve().is_relative_to(root) and str(path) not in root_files
        assert file(path)==row
    for row in selected:Path(row['path']).unlink()
    for directory in (cache,root/'red-stage'):
        descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    assert canonical()==before
    assert all(sha(Path(path))==digest for path,digest in root_files.items())
    plan.update(status='PASS',canonical_bytes_unchanged=True,original_prefix_bytes_unchanged=True,
                available_bytes_after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
    with (root/'child-eviction-v1.json').open('x') as stream:json.dump(plan,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({key:plan[key] for key in ('status','children','selected_allocated_bytes','canonical_files','available_bytes_before','available_bytes_after')}))
