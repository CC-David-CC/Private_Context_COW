#!/usr/bin/env bash
set -eu
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
sha256sum -c - <<'PINS'
a0ef7d70718ef06dfbfae7ec3e5a7dbff7f55ab252471804ba3cd5ee5cf655fb  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/tiny-v1/profile.json
b537fd71b4924adeafcfabe6b1bc499259c0347edc0dcce7c7241a6e013a2aa6  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config-normal.json
2a946ae7f1d0502f68f01ac9e0e13c5d6231935f17b63e68d6aae7ea009c5f8f  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts/pool_campaign.py
94e6935cddac17acca6c1d75e511ac241d394bba81707313d7fdec2e1b35968f  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts/native_request_probe.py
78f26419ff389e70f6b219aa2c6fe88e8829b30ae260a402fecaa875f07f102c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts/durable_resume.py
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
PINS
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
result=json.loads((root/'tiny-v1/result.json').read_text())
assert result['status']=='PASS' and result['numerical_max_error']==0
assert len(result['comparisons'])==2 and not result['failures'] and not result['skips']
prepared=json.loads((root/'tiny-v1/prepare.json').read_text())
fixture=json.loads((root/'tiny-v1/fixture.json').read_text())
saved=prepared['roots'][0]
raw=Path(saved['frontier_file']).read_bytes()
width=4*saved['manifest']['logits_count']
assert len(raw)==width and hashlib.sha256(raw).hexdigest()==saved['frontier_sha256']
assert saved['manifest']['consumed']==len(fixture['prefixes'][0])
for tree,suffix in enumerate(fixture['suffixes']):
    oracle=root/'tiny-v1'/('fresh-tree-'+str(tree)+'.f32.requests.f32')
    vectors=oracle.read_bytes()
    assert len(vectors)==2*width and raw==vectors[:width], tree
    tokens=(root/'tiny-v1'/('fresh-tree-'+str(tree)+'.tokens')).read_text()
    assert list(map(int,tokens.split(',')))==fixture['prefixes'][0]+suffix
for name in ('command.exit','wrapper.exit'):
    assert (root/'tiny-command-v1'/name).read_text().strip()=='0'
for name in ('tiny-probe-v1','tiny-probe-command-v1','medium-v1','medium-command-v1'):
    assert not (root/name).exists(), name
PY
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
# Allow 3 GiB for these bounded outputs above the 10 GiB disk floor.
available=$(df -B1 --output=avail "$ART" | tail -1 | tr -d ' ')
test "$available" -ge 13958643712
bash "$RUN_COMMAND" "$API" "$ART/tiny-probe-command-v1" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$API/scripts/native_request_probe.py" \
  --profile "$ART/tiny-v1/profile.json" --output "$ART/tiny-probe-v1" \
  --prefix 128 --suffix 4 --generated 8 --check-boundaries every
# The shell exits on any probe failure; medium does not start in that case.
bash "$RUN_COMMAND" "$API" "$ART/medium-command-v1" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$API/scripts/pool_campaign.py" campaign \
  --config "$ART/config-normal.json" --output "$ART/medium-v1" \
  --prefix 1024 --suffix 64 --generated 16 --roots 1 --trees 2
