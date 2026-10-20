#!/usr/bin/env bash
set -eu
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
test ! -e "$ART/target-v2/rewind.json"
test ! -e "$ART/cold-root-command-v2"
sha256sum -c - <<'PINS'
0130bd942bc965e6d0ad1eef603e5cacac177717181b0f0e5ee3cdd2a0a336e2  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/pool_campaign.py
30e8204ac2086d2a36a0f20702c14c1823b41257909f1b0f24d89faf4d5a454c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/native_request_probe.py
b537fd71b4924adeafcfabe6b1bc499259c0347edc0dcce7c7241a6e013a2aa6  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config-normal.json
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
PINS
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
record=json.loads((root/'target-v2/result.json').read_text())
assert record['status']=='PASS' and record['numerical_max_error']==0
assert len(record['comparisons'])==2
for phase in record['phases'].values():
    assert phase['exit_code']==0 and phase['native_exit_success']
    assert phase['native_exit_codes']=={'0':0}
for name in ('command.exit','wrapper.exit'):
    assert (root/'target-command-v2'/name).read_text().strip()=='0'
PY
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
available=$(df -B1 --output=avail "$ART" | tail -1 | tr -d ' ')
test "$available" -ge 12884901888
# Direct existing rewind phase: preserve A, restore the cold root, then ask B.
# Its result is separate from the completed prepare/resume campaign result.
exec bash "$RUN_COMMAND" "$API" "$ART/cold-root-command-v2" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$API/scripts-exit-v2/pool_campaign.py" rewind \
  --config "$ART/config-normal.json" --output "$ART/target-v2" \
  --prefix 8192 --suffix 512 --generated 128 --roots 1 --trees 2
