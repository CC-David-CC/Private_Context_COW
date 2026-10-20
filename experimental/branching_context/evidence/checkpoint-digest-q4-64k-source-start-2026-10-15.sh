#!/usr/bin/env bash
set -eu
umask 077
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
SCRIPTS="$API/scripts-exit-v2"
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015
OUT=/dev/shm/context-cow-digest-q4-64k-source-20261015-v1
FIXTURE=/home/dflanag3/branching-g3/multi-code-contexts-20261014/fixtures/q4_64k.json
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
test ! -e "$OUT"
test ! -e "$ART/source64k-command-v1"
sha256sum -c - <<'PINS'
0130bd942bc965e6d0ad1eef603e5cacac177717181b0f0e5ee3cdd2a0a336e2  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/pool_campaign.py
30e8204ac2086d2a36a0f20702c14c1823b41257909f1b0f24d89faf4d5a454c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/native_request_probe.py
78f26419ff389e70f6b219aa2c6fe88e8829b30ae260a402fecaa875f07f102c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/durable_resume.py
3d4403bc03d539c46dc4db3a50b778b26e1a0769ae3f5884f1558825afc914a8  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-exit-v2/build_codebase_fixture.py
c661084b390720c1cbb1ca2465734cd5a099d366f2811704b4661251f5ca1903  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fixture.json
b537fd71b4924adeafcfabe6b1bc499259c0347edc0dcce7c7241a6e013a2aa6  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config-normal.json
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
PINS
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
import hashlib, importlib, json, os, shutil, sys
api=Path('/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red')
art=Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
sys.path.insert(0,str(api/'scripts-exit-v2'))
import pool_campaign, native_request_probe
assert native_request_probe.API.resolve()==api
profile=json.loads((art/'target-v2/profile.json').read_text())
assert profile['execution_profile_id']=='4ac77a63cfb8bd0d153c8cb5fb340ab79ae4d3cc23e554d39bd967d693adfdd1'
result=json.loads((art/'target-v2/result.json').read_text())
assert result['status']=='PASS' and result['numerical_max_error']==0
for phase in ('prepare','resume'):
    record=json.loads((art/'target-v2'/f'{phase}.json').read_text())
    assert record['native_exit_codes']=={'0':0}
for name in ('command.exit','wrapper.exit'):
    assert (art/'target-command-v2'/name).read_text().strip()=='0'
modules=[]
for path, expected in profile['native_build']['sha256'].items():
    p=Path(path)
    if p.parent!=api/'src/context_time_travel':
        continue
    name='context_time_travel'+('' if p.stem=='__init__' else '.'+p.stem)
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve()==p
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==expected
    modules.append(dict(module=name,path=str(p),sha256=actual))
assert len(modules)==17
from build_codebase_fixture import normalize_context_fixture
raw_fixture=Path('/home/dflanag3/branching-g3/multi-code-contexts-20261014/fixtures/q4_64k.json')
raw=json.loads(raw_fixture.read_text())
fixture=normalize_context_fixture(raw,98304,128)
old=json.loads(Path('/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fixture.json').read_text())
for key in ('prefixes','suffixes','tree_roots'):
    assert fixture[key]==old[key],key
raw_fixture_sha256=hashlib.sha256(raw_fixture.read_bytes()).hexdigest()
assert list(map(len,fixture['prefixes']))==[65536]*3
assert len(fixture['suffixes'])==6 and fixture['tree_roots']==[0,0,1,1,2,2]
available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
tmp=shutil.disk_usage('/dev/shm'); disk=shutil.disk_usage(art)
# Old comparable full campaign archive was 41.84 GB. Reserve 56 GiB in tmpfs
# before this source-only campaign. Coordinator also enforces 16 GiB available.
assert tmp.free>=56*1024**3,(tmp.free,'tmpfs source forecast')
assert available>=112*1024**3,(available,'initial host RAM reserve')
assert disk.free>=10*1024**3,(disk.free,'real disk floor')
witness=dict(status='PREFLIGHT_PASS_NOT_CAMPAIGN_RESULT',modules=modules,
    execution_profile_id=profile['execution_profile_id'],mem_available_bytes=available,
    tmpfs_free_bytes=tmp.free,real_disk_free_bytes=disk.free,
    scope='RAM-backed source preparation only; not durable-disk or timing qualification',
    raw_fixture_sha256=raw_fixture_sha256,
    fixture_tokens_equal_previous_baseline=True,
    ram_pool_bytes=103079215104,disk_pool_limit_bytes=137438953472,
    min_mem_available_bytes=17179869184)
with (art/'source64k-import-resource-witness-v1.json').open('x') as stream:
    json.dump(witness,stream,indent=2); stream.write('\n');stream.flush();os.fsync(stream.fileno())
PY
bash "$RUN_COMMAND" "$API" "$ART/source64k-command-v1" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$SCRIPTS/pool_campaign.py" campaign \
  --config "$ART/config-normal.json" --output "$OUT" \
  --fixture "$FIXTURE" --prefix 65536 --suffix 8192 --generated 128 --roots 3 --trees 6
