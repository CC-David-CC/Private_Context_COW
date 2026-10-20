"""Export verified small manifests from the unmounted image, read-only."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path('/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1')
image=root/'mount-preflight-v1/bounded-ext4.img'
audit=json.loads((root/'primary-audit-gpu-q4-enospc-v3.json').read_bytes())
assert image.stat().st_ino==5252779 and not image.is_symlink()
assert subprocess.check_output(['losetup','-j',str(image)])==b''
with image.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='e437fe3e17fe3246d6016b4cf9928870943c82ddbdd679ca125e4d2db29f99d8'
output=root/'readonly-audit-v3/exported-manifests'
output.mkdir(mode=0o700,exist_ok=False)
rows=[]
for snapshot,record in audit['files'].items():
    assert len(snapshot)==64 and all(c in '0123456789abcdef' for c in snapshot)
    result=subprocess.run(['debugfs','-R','cat /'+snapshot+'.json',str(image)],capture_output=True,check=True)
    assert hashlib.sha256(result.stdout).hexdigest()==record['manifest_sha256']
    assert json.loads(result.stdout)['manifest']['payload_sha256']==record['payload_sha256']
    path=output/(snapshot+'.json')
    with path.open('xb') as stream: stream.write(result.stdout)
    path.chmod(0o600)
    rows.append(dict(path=str(path),bytes=len(result.stdout),sha256=record['manifest_sha256'],debugfs_stderr=result.stderr.decode()))
report=dict(status='PASS',scope='Read-only manifest export; numerical payloads remain in original image',image_unchanged=True,manifests=rows)
(output.parent/'manifest-export-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
