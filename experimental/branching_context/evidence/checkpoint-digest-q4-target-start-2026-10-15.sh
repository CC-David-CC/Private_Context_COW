#!/usr/bin/env bash
set -eu
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
FROM=/home/dflanag3/branching-g3/native-exit-status-luna-20261015/red/scripts-exit-v2
SCRIPTS="$API/scripts-exit-v2"
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
test ! -e "$SCRIPTS"
mkdir -m 700 "$SCRIPTS"
for file in pool_campaign.py native_request_probe.py durable_resume.py build_codebase_fixture.py; do
  cp "$FROM/$file" "$SCRIPTS/$file"
done
chmod 400 "$SCRIPTS"/*.py
chmod 500 "$SCRIPTS"
sha256sum -c - <<'PINS'
0130bd942bc965e6d0ad1eef603e5cacac177717181b0f0e5ee3cdd2a0a336e2  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/pool_campaign.py
30e8204ac2086d2a36a0f20702c14c1823b41257909f1b0f24d89faf4d5a454c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/native_request_probe.py
78f26419ff389e70f6b219aa2c6fe88e8829b30ae260a402fecaa875f07f102c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/durable_resume.py
3d4403bc03d539c46dc4db3a50b778b26e1a0769ae3f5884f1558825afc914a8  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/build_codebase_fixture.py
a0ef7d70718ef06dfbfae7ec3e5a7dbff7f55ab252471804ba3cd5ee5cf655fb  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/medium-v1/profile.json
b537fd71b4924adeafcfabe6b1bc499259c0347edc0dcce7c7241a6e013a2aa6  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config-normal.json
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
PINS
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
import hashlib,importlib,json,os,sys
api=Path('/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red')
art=Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
sys.path.insert(0,str(api/'scripts-exit-v2'))
import pool_campaign,native_request_probe
assert native_request_probe.API.resolve()==api
profile=json.loads((art/'medium-v1/profile.json').read_text())
result=json.loads((art/'medium-v1/result.json').read_text())
assert result['status']=='PASS' and result['numerical_max_error']==0
rows=[]
for path,expected in profile['native_build']['sha256'].items():
    p=Path(path)
    if p.parent!=api/'src/context_time_travel':
        continue
    name='context_time_travel'+('' if p.stem=='__init__' else '.'+p.stem)
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve()==p
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==expected
    rows.append(dict(module=name,path=str(p),sha256=actual))
assert len(rows)==17
for directory in ('medium-probe-v2','medium-probe-command-v2','target-v2','target-command-v2'):
    assert not (art/directory).exists(),directory
for name in ('command.exit','wrapper.exit'):
    assert (art/'medium-command-v1'/name).read_text().strip()=='0'
with (art/'target-import-witness-v2.json').open('x') as stream:
    os.chmod(stream.name,0o600)
    json.dump(dict(status='PASS',modules=rows,execution_profile_id=profile['execution_profile_id'],
        pool_script=str(Path(pool_campaign.__file__).resolve())),stream,indent=2)
    stream.flush();os.fsync(stream.fileno())
PY
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
# Bounded pilot forecast 6 GiB, retained free-space floor 10 GiB.
available=$(df -B1 --output=avail "$ART" | tail -1 | tr -d ' ')
test "$available" -ge 17179869184
bash "$RUN_COMMAND" "$API" "$ART/medium-probe-command-v2" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$SCRIPTS/native_request_probe.py" --profile "$ART/medium-v1/profile.json" \
  --output "$ART/medium-probe-v2" --prefix 1024 --suffix 64 --generated 16 --check-boundaries every
# set -e prevents expansion after a nonzero exit or failed comparison.
bash "$RUN_COMMAND" "$API" "$ART/target-command-v2" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$SCRIPTS/pool_campaign.py" campaign --config "$ART/config-normal.json" \
  --output "$ART/target-v2" --prefix 8192 --suffix 512 --generated 128 --roots 1 --trees 2
