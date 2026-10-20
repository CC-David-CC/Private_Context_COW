"""Evict only the completed disposable numerical fixture after verified backup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path('/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1')
image=root/'mount-preflight-v1/bounded-ext4.img'
assert image.resolve(strict=True)==image and image.is_relative_to(root)
assert image.is_file() and not image.is_symlink()
stat=image.stat()
assert stat.st_ino==5252779 and stat.st_nlink==1 and stat.st_size==1610612736
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(image)=='e437fe3e17fe3246d6016b4cf9928870943c82ddbdd679ca125e4d2db29f99d8'
assert subprocess.check_output(['losetup','-j',str(image)])==b''
assert subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()==b''
for pid in (2772280,2772286): assert not Path('/proc',str(pid)).exists()
audit=json.loads((root/'primary-audit-gpu-q4-enospc-v3.json').read_bytes())
assert audit['status']=='FOCUSED-PASS' and audit['source_full_bytes_unchanged']
for name,count in (('q4-enospc-pass-backup-primary-20261016.json',139),('q4-enospc-manifest-backup-primary-20261016.json',6)):
    receipt=json.loads((root.parent/name).read_bytes())
    assert receipt['status']=='PASS' and receipt['verified_regular_files']==count
for snapshot,record in audit['files'].items():
    manifest=root/'readonly-audit-v3/exported-manifests'/(snapshot+'.json')
    assert sha(manifest)==record['manifest_sha256']
source=Path('/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/input-store-v1/checkpoints/423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810.state')
assert source.is_file() and sha(source)==source.stem
store=root/'input-store-v1'
metadata={str(p.relative_to(store)):sha(p) for p in store.rglob('*') if p.is_file()}
assert not any(p.is_symlink() for p in store.rglob('*'))
free_before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
image.unlink()
directory=os.open(image.parent,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(directory)
finally: os.close(directory)
assert not image.exists()
assert metadata=={str(p.relative_to(store)):sha(p) for p in store.rglob('*') if p.is_file()}
assert sha(source)==source.stem
free_after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
report=dict(status='PASS',scope='Eviction of completed disposable numerical test image; not deletion of canonical history',
            removed_path=str(image),inode=stat.st_ino,logical_bytes=stat.st_size,allocated_bytes=stat.st_blocks*512,
            image_sha256_before='e437fe3e17fe3246d6016b4cf9928870943c82ddbdd679ca125e4d2db29f99d8',
            free_bytes_before=free_before,free_bytes_after=free_after,metadata_files_preserved=len(metadata),
            source_checkpoint_preserved=True,source_checkpoint_bytes=source.stat().st_size,
            full_manifests_and_hash_receipts_preserved=True,numerical_fixture_no_longer_restorable_from_image=True,
            canonical_metadata_sha256=metadata)
output=root/'completed-image-eviction-20261016.json'
with output.open('x') as stream:
    json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
output.chmod(0o600)
print(json.dumps({k:v for k,v in report.items() if k!='canonical_metadata_sha256'},sort_keys=True))
