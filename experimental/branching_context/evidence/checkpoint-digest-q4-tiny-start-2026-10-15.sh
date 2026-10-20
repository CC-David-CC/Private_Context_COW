#!/usr/bin/env bash
set -eu
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
ART=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
test ! -e "$ART/tiny-v1"
test ! -e "$ART/tiny-command-v1"
sha256sum -c - <<'PINS'
dcba2359c7e625ef5d8494cfcaf2394e352201bb8f2eae104a757c73b5a55c54  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/derived-profile/profile.json
b537fd71b4924adeafcfabe6b1bc499259c0347edc0dcce7c7241a6e013a2aa6  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config-normal.json
282dbf2af31ea60ba5bfa68d10022b19ae75f387b65ddf9f47063f145270f6a8  /home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/source-import-witness.json
2a946ae7f1d0502f68f01ac9e0e13c5d6231935f17b63e68d6aae7ea009c5f8f  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts/pool_campaign.py
94e6935cddac17acca6c1d75e511ac241d394bba81707313d7fdec2e1b35968f  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts/native_request_probe.py
78f26419ff389e70f6b219aa2c6fe88e8829b30ae260a402fecaa875f07f102c  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts/durable_resume.py
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
PINS
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
# Tiny output forecast: 2 GiB, in addition to the retained 10 GiB disk floor.
available=$(df -B1 --output=avail "$ART" | tail -1 | tr -d ' ')
test "$available" -ge 12884901888
chmod a-w "$API"/src/context_time_travel/*.py "$API"/scripts/*.py
chmod a-w "$API/src/context_time_travel" "$API/src" "$API/scripts"
chmod a-w "$ART/derived-profile/profile.json" "$ART/config-normal.json"
exec bash "$RUN_COMMAND" "$API" "$ART/tiny-command-v1" \
  env PYTHONPATH="$API/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$API/scripts/pool_campaign.py" campaign \
  --config "$ART/config-normal.json" --output "$ART/tiny-v1" \
  --prefix 128 --suffix 4 --generated 8 --roots 1 --trees 2
