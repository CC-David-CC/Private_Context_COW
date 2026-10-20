"""Exercise only the real ext4 filler, with no model, server or native process."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

driver, mount, output = map(Path, sys.argv[1:])
assert sys.platform.startswith('linux') and os.geteuid() == 1000
assert mount.is_mount() and mount.stat().st_dev != mount.parent.stat().st_dev
assert (mount / 'checkpoints').stat().st_dev != mount.stat().st_dev
assert output.parent.stat().st_dev != mount.stat().st_dev and not output.exists()
output.mkdir(mode=0o700)
spec = importlib.util.spec_from_file_location('canonical_filler_candidate', driver)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
def fingerprint():
    files = {}
    for path in sorted(mount.rglob('*')):
        relative = path.relative_to(mount)
        if relative.parts[0] in ('lost+found', 'checkpoints'): continue
        assert not path.is_symlink()
        if path.is_file(): files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files
before = fingerprint()
result = dict(schema='canonical-ext4-filler-only-v1', status='RUNNING', driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
              no_model_or_inference=True, canonical_files_before=before)
filler = None
started = time.monotonic()
try:
    filler = candidate._fill_to_enospc(mount, 64 * 1024**2, uuid.uuid4().hex)
    filler['canonical_st_dev'] = mount.stat().st_dev
    candidate.validate_filler_receipt(filler, canonical_mount=str(mount), max_fill_bytes=64 * 1024**2,
                                      canonical_st_dev=mount.stat().st_dev)
    candidate._atomic_json(output / 'filler-receipt.json', filler)
    assert os.statvfs(mount).f_bavail == 0
    result['filler'] = filler
    candidate._remove_owned_filler(filler, mount)
    filler = None
    assert fingerprint() == before
    result.update(status='PASS', canonical_files_unchanged=True, capacity_after_release=os.statvfs(mount).f_bavail * os.statvfs(mount).f_frsize)
except BaseException as exc:
    result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
finally:
    if filler is not None:
        try: candidate._remove_owned_filler(filler, mount)
        except BaseException as exc: result['cleanup_error'] = f'{type(exc).__name__}: {exc}'
    result['elapsed_seconds'] = time.monotonic() - started
    candidate._atomic_json(output / 'result.json', result)
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
