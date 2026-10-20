"""Stage fresh launch paths and account for already allocated image bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

artifacts = Path('/home/dflanag3/spatial-artifacts')
previous = artifacts / 'q4-enospc-full-launch-v4'
current = artifacts / 'q4-enospc-full-launch-v5'
root = artifacts / 'q4-enospc-harness-20261016-v1'
driver = Path(sys.argv[1])
expected_driver_sha = sys.argv[2]
assert driver.resolve(strict=True) == driver and driver.is_relative_to(root)
assert hashlib.sha256(driver.read_bytes()).hexdigest() == expected_driver_sha
parent = root / 'gpu-q4-enospc-v3'
assert not current.exists() and not parent.exists()
pins = {
    'launch-q4-enospc.sh': '52004399477d76afcd8076cee6bdc6d3b59b484bf5a2f0023cd2cde00e446d3a',
    'private-mount-run.sh': 'e1a6e5e02adf93f1ac50d77452178034b9b2f83689b816834feca4b74b3b7637',
}
current.mkdir(mode=0o700)
parent.mkdir(mode=0o700)
hashes = {}
for name, pin in pins.items():
    data = (previous / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin
    text = data.decode().replace('q4-enospc-full-launch-v4', 'q4-enospc-full-launch-v5')
    if name == 'private-mount-run.sh':
        old_driver = root / 'preflight-green-v6/source/experimental/branching_context/scripts/run_q4_enospc.py'
        replacements = (
            (str(old_driver).replace(str(root), '$ROOT'), str(driver)),
            (str(old_driver.parent).replace(str(root), '$ROOT'), str(driver.parent)),
            ('gpu-q4-enospc-v2/probe-result', 'gpu-q4-enospc-v3/probe-result'),
            ('private-mount-preflight-v4', 'private-mount-preflight-v5'),
            ('private-mount-run-v4', 'private-mount-run-v5'),
        )
        for old, new in replacements:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
    else:
        old = 'readonly MIN_FREE_BYTES=$((EXPECTED_IMAGE_BYTES + 16 * 1024 * 1024 * 1024))'
        assert text.count(old) == 1
        text = text.replace(old, '''# Only unallocated image growth can consume additional host space.
# Keep the same 16 GiB reserve after the image reaches its maximum size.
allocated_image_bytes=$(( $(stat -c '%b' "$IMAGE") * 512 ))
remaining_image_bytes=$((EXPECTED_IMAGE_BYTES - allocated_image_bytes))
if (( remaining_image_bytes < 0 )); then remaining_image_bytes=0; fi
readonly MIN_FREE_BYTES=$((remaining_image_bytes + 16 * 1024 * 1024 * 1024))''')
        # Test the identical shell arithmetic at sparse/partial/full/overallocated sizes.
        subprocess.run(['bash', '-c', '''set -euo pipefail
size=1610612736
for allocated in 0 362807296 1610612736 1610616832; do
  remaining=$((size - allocated))
  if (( remaining < 0 )); then remaining=0; fi
  minimum=$((remaining + 16 * 1024 * 1024 * 1024))
  (( minimum >= 17179869184 ))
  (( minimum + allocated >= size + 17179869184 ))
done
'''], check=True)
    destination = current / name
    destination.write_text(text, newline='\n')
    destination.chmod(0o500)
    subprocess.run(['bash', '-n', str(destination)], check=True)
    hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
report = dict(status='PREPARED_NOT_RUN', source_driver=str(driver),
              source_driver_sha256=expected_driver_sha, launcher=str(current),
              probe_parent=str(parent), source_hashes=hashes,
              space_guard='16 GiB reserve plus remaining possible image allocation',
              space_guard_arithmetic_checks=4, old_failed_receipts_retained=True)
(current / 'preparation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
