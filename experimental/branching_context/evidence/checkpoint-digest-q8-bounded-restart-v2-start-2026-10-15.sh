#!/usr/bin/env bash
set -eu
umask 077
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q8-attestation-20261015
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
test ! -e "$ART/bounded8k-v2"
test ! -e "$ART/bounded8k-command-v2"
sha256sum -c - <<'PINS'
a919c49269b2547a16740827d47bbbd73db0777620030fb9d9feb20b2b955460  /home/dflanag3/spatial-artifacts/checkpoint-digest-q8-attestation-20261015/profile-unbound-transport-v1.json
03393e0d0a155d7f0ef517d04441366e212189d403c9555ef614a2fee9501ec3  /home/dflanag3/spatial-artifacts/checkpoint-digest-q8-attestation-20261015/derived-profile-candidate-v1/profile.json
b11987c458f1891b878e3ae4a2186895d150d7982f29cd8c7eab10d260dee9aa  /home/dflanag3/spatial-artifacts/checkpoint-digest-q8-attestation-20261015/config-unbound-v2.json
0130bd942bc965e6d0ad1eef603e5cacac177717181b0f0e5ee3cdd2a0a336e2  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/pool_campaign.py
30e8204ac2086d2a36a0f20702c14c1823b41257909f1b0f24d89faf4d5a454c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/native_request_probe.py
78f26419ff389e70f6b219aa2c6fe88e8829b30ae260a402fecaa875f07f102c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/durable_resume.py
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
PINS
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$API/src" python3 - <<'PY'
from pathlib import Path
import hashlib, importlib, json, os, shutil, sys
api=Path('/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red')
art=Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q8-attestation-20261015')
sys.path.insert(0,str(api/'scripts-exit-v2'))
import native_request_probe
assert native_request_probe.API.resolve()==api
profile=json.loads((art/'derived-profile-candidate-v1/profile.json').read_text())
assert profile['execution_profile_id']=='608bae645dd91c04f01d4cb42acccf8ff8cd6d0bf4da16d5dbecef8371661da3'
assert profile['execution']['GPU']==[0]
assert json.loads((art/'profile-transport-check-v1/profile.json').read_text())==profile
assert profile['execution']['capacity']==98304 and profile['execution']['prefill_chunk']==8192
rows=[]
for path,expected in profile['native_build']['sha256'].items():
    p=Path(path)
    if p.parent!=api/'src/context_time_travel':
        continue
    name='context_time_travel'+('' if p.stem=='__init__' else '.'+p.stem)
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve()==p
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    rows.append(dict(module=name,path=str(p),sha256=expected))
assert len(rows)==17
available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
assert available>=64*1024**3,available
assert shutil.disk_usage(art).free>=16*1024**3
with (art/'bounded8k-import-resource-witness-v2.json').open('x') as stream:
    json.dump(dict(status='PREFLIGHT_PASS_NOT_RESTART_RESULT',modules=rows,
        execution_profile_id=profile['execution_profile_id'],mem_available_bytes=available,
        scope='One bounded 8K/512/128 greedy restart check; no performance or cumulative gate qualification'),stream,indent=2)
    stream.write('\n');stream.flush();os.fsync(stream.fileno())
PY
bash "$RUN_COMMAND" "$API" "$ART/bounded8k-command-v2" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$API/scripts-exit-v2/pool_campaign.py" campaign \
  --config "$ART/config-unbound-v2.json" --output "$ART/bounded8k-v2" \
  --prefix 8192 --suffix 512 --generated 128 --roots 1 --trees 2
