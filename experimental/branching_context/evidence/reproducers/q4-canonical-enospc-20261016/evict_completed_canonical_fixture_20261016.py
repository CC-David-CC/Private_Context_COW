"""Evict exact audited disposable numerical copies after verified compact backup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path('/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1').resolve(strict=True)
def read(path): return json.loads(path.read_bytes())
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2): digest.update(block)
    return digest.hexdigest()
assert os.geteuid() == 1000
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
primary = read(root / 'primary-audit-gpu-q4-v2.json')
assert primary['status'] == 'FOCUSED-PASS' and primary['all_pids_absent']
assert all(not Path('/proc', str(pid)).exists() for pid in primary['process_ids'])
assert sha(root / 'gpu-q4-v2/campaign/result.json') == primary['raw_result_sha256']
for name, expected in [('backup-primary-attempt1-v1.json', 'e1b5839a13e2367faf1ba989e97a2593d1fed0bda2dbe57e40cc4ce7d0a46cd2'),
                       ('backup-primary-recovery-v1.json', 'bb1f3939f4fbf39358ab7adcfa28ea56e5b1673a24758e3c14d71a918bf0d224')]:
    record = read(root / name)
    assert record['status'] == 'PASS' and record['archive_sha256'] == expected and record['verified_regular_files'] == 217
cleanup = read(root / 'mount-run-v3/terminal-receipt.json')
assert cleanup['probe_exit'] == cleanup['cleanup_exit'] == 0
assert cleanup['loop_detached'] and cleanup['canonical_unmounted'] and cleanup['cache_unmounted']
image = root / 'canonical-ext4-64m.img'
assert not subprocess.check_output(['sudo', '-n', 'losetup', '-j', str(image)], text=True).strip()
assert not (root / 'store-mount').is_mount()
assert not image.is_symlink() and image.stat().st_ino == 5254728 and image.stat().st_size == 67108864
assert sha(image) == cleanup['image_sha256_after'] == 'ef39e14a5372e3cf06cf7973d37dac00a04a8e88970379bccef4021d2b8573a4'
export = root / 'readonly-audit-v2/canonical-export'
metadata = read(root / 'readonly-audit-v2/export.json')
for relative, row in metadata['files'].items():
    assert sha(export / relative) == row['sha256'] and (export / relative).stat().st_size == row['bytes']
source = '808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85'
cache = root / 'checkpoint-volume-v1'
selected = []
for snapshot, row in primary['files'].items():
    if snapshot == source: continue
    for path, expected in [(cache / (snapshot + '.json'), row['manifest_sha256']),
                           (cache / (row['payload_sha256'] + '.state'), row['payload_sha256'])]:
        assert path.parent.resolve(strict=True) == cache and path.is_file() and not path.is_symlink()
        assert sha(path) == expected and path.stat().st_nlink == 1
        selected.append((path, expected))
selected.append((image, cleanup['image_sha256_after']))
original = Path('/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/input-store-v1/checkpoints')
source_row = primary['files'][source]
source_payload = cache / (source_row['payload_sha256'] + '.state')
assert sha(source_payload) == sha(original / source_payload.name) == source_row['payload_sha256']
before_available = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
removed = []
for path, expected in selected:
    before = path.stat()
    assert sha(path) == expected
    path.unlink()
    removed.append(dict(path=str(path), inode=before.st_ino, logical_bytes=before.st_size,
                        allocated_bytes=before.st_blocks * 512, sha256=expected))
for directory in (cache, root):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
for relative, row in metadata['files'].items(): assert sha(export / relative) == row['sha256']
assert sha(source_payload) == sha(original / source_payload.name) == source_row['payload_sha256']
record = dict(schema='completed-canonical-fixture-eviction-v1', status='PASS',
              removed=removed, allocated_bytes_released=sum(row['allocated_bytes'] for row in removed),
              available_bytes_before=before_available, available_bytes_after=os.statvfs(root).f_bavail * os.statvfs(root).f_frsize,
              retained_canonical_files=len(metadata['files']), retained_canonical_bytes_verified=True,
              retained_source_payload_sha256=source_row['payload_sha256'], retained_source_payload_bytes=source_payload.stat().st_size,
              limitation='Child histories remain; their derived numerical checkpoints and fault image were intentionally evicted. Compact archives are evidence, not complete numerical-store backups.')
output = root / 'completed-fixture-eviction-v1.json'
with output.open('x') as stream: json.dump(record, stream, sort_keys=True, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
output.chmod(0o600)
print(json.dumps(record, sort_keys=True))
