"""Evict completed derived child copies after verified private evidence backup."""
import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1').resolve(strict=True)
lock = open('/home/dflanag3/branching-g3/native-client-speed.gpu0.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
def read(path):
    return json.loads(path.read_bytes())
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024**2):
            digest.update(data)
    return digest.hexdigest()
def file(path):
    assert path.is_file() and not path.is_symlink()
    info = path.stat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    return dict(path=str(path), sha256=sha(path), bytes=info.st_size,
        allocated_bytes=info.st_blocks*512, inode=info.st_ino)
def canonical(store):
    return {str(path.relative_to(store)): sha(path) for path in sorted(store.rglob('*'))
            if path.is_file() and 'checkpoints' not in path.relative_to(store).parts}
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
backup = read(root / 'stream-backup-v1/readback.json')
assert backup['status'] == 'PASS' and backup['archive_sha256'] == 'd70e93c1ca3fc665886dc3ee41f53bb82feff2364c6d7b0d60787f6ff83a8331'
assert subprocess.check_output(['ssh', 'box-git', 'sha256sum ' + backup['archive']], text=True).split()[0] == backup['archive_sha256']
for run in ('stream-backup-command-v1', 'stream-readback-command-v1'):
    assert all((root / run / name).read_text().strip() == '0' for name in ('command.exit', 'wrapper.exit'))
campaigns = [read(root / (name + '/result.json')) for name in ('gpu-smoke-v1', 'gpu-campaign-v1')]
store = (root / 'workspace-store-v1').resolve(strict=True)
assert all(Path(row['store']).resolve() == store for row in campaigns)
pids = []
for campaign in campaigns:
    for service in campaign['services']:
        assert service['stop']['exit_code'] == service['worker']['native_exit_code'] == 0
        pids.extend([service['http_pid'], service['worker']['native_pid']])
assert all(not Path('/proc', str(pid)).exists() for pid in pids)
cache = store / 'checkpoints'
roots = {row['head']['tip'] for campaign in campaigns for row in campaign['roots'].values()}
children = {row['snapshot'] for campaign in campaigns for row in campaign['generations']}
assert len(roots) == 3 and roots.isdisjoint(children)
manifests = {path.stem: read(path)['manifest'] for path in cache.glob('*.json')}
assert children.issubset(manifests)
protected = {}
for sid in roots:
    manifest = cache / (sid + '.json')
    payload = cache / (manifests[sid]['payload_sha256'] + '.state')
    assert sha(payload) == manifests[sid]['payload_sha256']
    protected[str(manifest)] = sha(manifest)
    protected[str(payload)] = sha(payload)
selected = [file(cache / (sid + '.json')) for sid in sorted(children)]
for digest in sorted({manifests[sid]['payload_sha256'] for sid in children}):
    users = {sid for sid, manifest in manifests.items() if manifest['payload_sha256'] == digest}
    if not users.issubset(children):
        continue
    row = file(cache / (digest + '.state'))
    assert row['sha256'] == digest
    selected.append(row)
before = canonical(store)
plan = dict(schema='cow-native-combined-child-eviction-v1', status='PLAN', selected=selected,
    selected_allocated_bytes=sum(row['allocated_bytes'] for row in selected),
    protected_original_prefixes=protected, canonical_files=len(before),
    canonical_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    backup_archive=backup['archive'], backup_archive_sha256=backup['archive_sha256'],
    all_owned_service_pids_absent=True, available_bytes_before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
if args.apply:
    previous = read(root / 'child-eviction-plan-v1.json')
    assert {key: value for key, value in plan.items() if key != 'available_bytes_before'} == {
        key: value for key, value in previous.items() if key != 'available_bytes_before'}
    for row in selected:
        path = Path(row['path'])
        assert path.resolve().parent == cache and str(path) not in protected and file(path) == row
    for row in selected:
        Path(row['path']).unlink()
    descriptor = os.open(cache, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert canonical(store) == before
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    plan.update(status='PASS', canonical_bytes_unchanged=True, original_prefix_bytes_unchanged=True,
        available_bytes_after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
destination = root / ('child-eviction-v1.json' if args.apply else 'child-eviction-plan-v1.json')
with destination.open('x') as stream:
    json.dump(plan, stream, sort_keys=True, indent=2)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps({key: value for key, value in plan.items() if key in
    ('status', 'selected_allocated_bytes', 'canonical_files', 'available_bytes_before', 'available_bytes_after')}))
