"""Create fresh mount receipts and a model-free filler mode; no execution."""
import hashlib
import json
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1')
out = root / 'launch-v4'
out.mkdir(mode=0o700)
(root / 'gpu-q4-v2').mkdir(mode=0o700)
pins = {'launch.sh': '7e624b14721e3cdedd1795bae0218a33dafa6b43f77c11d57f912e43480345ab',
        'private-mount.sh': '106f751c623beb722c5ca75e33cd4c8803c628084a0d0e063a9b4991f48c3c1c'}
hashes = {}
for name, expected in pins.items():
    data = (root / 'launch-v3' / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    changes = {b'( $mode == preflight || $mode == run )': b'( $mode == preflight || $mode == filler || $mode == run )'}
    if name == 'launch.sh':
        changes[b'INNER=$ROOT/launch-v3/private-mount.sh'] = b'INNER=$ROOT/launch-v4/private-mount.sh'
    else:
        changes[b'RECEIPTS=$ROOT/mount-$mode-v2'] = b'RECEIPTS=$ROOT/mount-$mode-v3'
        changes[b'$ROOT/gpu-q4-v1/campaign'] = b'$ROOT/gpu-q4-v2/campaign'
        changes[b'preflight=()'] = b'''if [[ $mode == filler ]]; then
  runuser -u dflanag3 -- env PYTHONDONTWRITEBYTECODE=1 \\
    "$PY" -B /home/dflanag3/spatial-artifacts/probe_canonical_filler_20261016.py \\
    "$driver" "$STORE" "$ROOT/filler-only-v1"
  exit 0
fi
preflight=()'''
    for before, after in changes.items():
        assert data.count(before) == 1, (name, before)
        data = data.replace(before, after)
    (out / name).write_bytes(data)
    (out / name).chmod(0o500)
    hashes[name] = hashlib.sha256(data).hexdigest()
dispatch = f'''#!/usr/bin/env bash
set -euo pipefail
ROOT={root}
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
mode=${{1:?preflight, filler or run}}
stage=${{2:?immutable stage}}
driver_sha=${{3:?driver SHA}}
image_sha=${{4:?image SHA}}
[[ $mode == preflight || $mode == filler || $mode == run ]]
[[ $stage =~ ^final-green-v[0-9]+$ && $driver_sha =~ ^[0-9a-f]{{64}}$ && $image_sha =~ ^[0-9a-f]{{64}}$ ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$ROOT/launch-v4/launch.sh" | cut -d ' ' -f1) == {hashes['launch.sh']} ]]
[[ $(sha256sum "$ROOT/launch-v4/private-mount.sh" | cut -d ' ' -f1) == {hashes['private-mount.sh']} ]]
SOURCE=$ROOT/$stage/source
DRIVER=$SOURCE/experimental/branching_context/scripts/run_q4_canonical_enospc.py
HELPER=$SOURCE/experimental/branching_context/scripts/run_q4_publication_fault.py
[[ ! -L $SOURCE && -d $SOURCE && -f $DRIVER && ! -L $DRIVER ]]
[[ $(sha256sum "$DRIVER" | cut -d ' ' -f1) == "$driver_sha" ]]
[[ -f $HELPER && ! -L $HELPER && $(sha256sum "$HELPER" | cut -d ' ' -f1) == d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695 ]]
[[ -z $(find "$SOURCE" -perm /222 -print -quit) ]]
exec bash "$RUNNER" "$SOURCE" "$ROOT/launch-v4/$mode-command-v1" bash "$ROOT/launch-v4/launch.sh" "$mode" "$DRIVER" "$driver_sha" "$image_sha"
'''
(root / 'dispatch-v3.sh').write_text(dispatch)
(root / 'dispatch-v3.sh').chmod(0o500)
hashes['dispatch-v3.sh'] = hashlib.sha256(dispatch.encode()).hexdigest()
receipt = dict(status='PREPARED_NOT_RUN', source_hashes=hashes,
               description='No-model filler mode plus fresh GPU/output/mount receipt paths; source/checkpoint/profile pins unchanged')
(out / 'preparation.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
(out / 'preparation.json').chmod(0o400)
print(json.dumps(receipt, sort_keys=True))
