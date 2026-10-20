#!/usr/bin/env bash
set -eu
umask 077
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
TOOLS="$API/scripts-p4-v1"
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015
SOURCE=/dev/shm/context-cow-digest-q4-64k-source-20261015-v1
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
test ! -e "$ART/p4-disk-v1"
test ! -e "$ART/p4-disk-command-v1"
sha256sum -c - <<'PINS'
0d8fa63dcbde960db10e8a998bcc6e4d5d18d7ed9049e4780958e1b192b5b423  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts-p4-v1/p4_return_latency.py
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
ccd73b3810db4a8010528aa0464f82ee543c2bd6a5568d21fd9a150bec0938ff  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/source64k-independent-audit-v1.json
d6378ca796ffdac1516add3705755312fecc8c65dbf94fdae4d1c1c9d1556221  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/result.json
c59f69d9091925c9c99fb8db11c44b9bb86dee7fcfe1729cdbc08c3a54c806ed  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/prepare.json
a0ef7d70718ef06dfbfae7ec3e5a7dbff7f55ab252471804ba3cd5ee5cf655fb  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/profile.json
c661084b390720c1cbb1ca2465734cd5a099d366f2811704b4661251f5ca1903  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/fixture.json
5213ef8cec07cd002a462a66b9b5f22d872a2c49a993bdd42508b8beea2ff695  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/fresh-tree-0.stdout
f85dce6eb714486afc0dc69a850f546ce454eb2933ca1dcbcbc1243053ce632a  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/fresh-tree-2.stdout
04f99ff22136a8508296db7a8c9386c7533f775fc57e1cb3bb429bf9194cbac5  /dev/shm/context-cow-digest-q4-64k-source-20261015-v1/fresh-tree-4.stdout
PINS
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$API/src" python3 - <<'PY'
from pathlib import Path
import hashlib, importlib, json, os, shutil
api=Path('/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red')
art=Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
source=Path('/dev/shm/context-cow-digest-q4-64k-source-20261015-v1')
profile=json.loads((source/'profile.json').read_text())
audit=json.loads((art/'source64k-independent-audit-v1.json').read_text())
assert audit['status']=='PASS' and audit['generated_tokens_compared']==768
assert profile['execution_profile_id']==audit['execution_profile_id']=='4ac77a63cfb8bd0d153c8cb5fb340ab79ae4d3cc23e554d39bd967d693adfdd1'
modules=[]
for path,expected in profile['native_build']['sha256'].items():
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
largest=max(p.stat().st_size for p in (source/'store/checkpoints').glob('*.state'))
forecast=15*largest+10*1024**3
disk=shutil.disk_usage(art)
assert disk.free>=forecast,(disk.free,forecast)
available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
assert available>=48*1024**3,available
with (art/'p4-disk-import-resource-witness-v1.json').open('x') as stream:
    json.dump(dict(status='PREFLIGHT_PASS_NOT_TIMING_RESULT',modules=modules,
        largest_source_payload_bytes=largest,required_free_bytes=forecast,
        actual_disk_free_bytes=disk.free,mem_available_bytes=available,
        execution_profile_id=profile['execution_profile_id'],
        validation_enabled=True,throughput_loss_reporting_target_percent=8,
        throughput_target_is_rejection_gate=False,
        source_scope='tmpfs source; arm copies and fsyncs roots onto real disk before timing',
        measurement='uninstrumented, warm filesystem, three warmups and nine measured returns'),stream,indent=2)
    stream.write('\n');stream.flush();os.fsync(stream.fileno())
PY
mkdir -m 700 "$ART/p4-disk-v1"
bash "$RUN_COMMAND" "$API" "$ART/p4-disk-command-v1" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$TOOLS/p4_return_latency.py" \
  --source-store "$SOURCE/store" --source-result "$SOURCE/result.json" \
  --prepare "$SOURCE/prepare.json" --profile "$SOURCE/profile.json" \
  --fixture "$SOURCE/fixture.json" --reference-dir "$SOURCE" \
  --output "$ART/p4-disk-v1/output" --arm disk --arm-order disk,resident \
  --order-seed p4-q4-arm-order-draw-2026-10-15-01 \
  --ram-bytes 103079215104 --disk-bytes 137438953472 --min-available-bytes 17179869184
