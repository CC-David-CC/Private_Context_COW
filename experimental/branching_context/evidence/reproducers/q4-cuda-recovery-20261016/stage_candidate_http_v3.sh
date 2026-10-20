#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1
STAGE=$ROOT/http-control-v3
[[ ! -e $STAGE ]]
mkdir -m 700 "$STAGE"
mkdir -m 700 "$STAGE/source"
tar -xf "$ROOT/candidate-http-source-v2.tar" -C "$STAGE/source" --no-same-owner
chmod -R u+rwX "$STAGE/source"
API=$STAGE/source/experimental/branching_context/src/context_time_travel
GREEN=/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/replacement-root-green-v4/source/src/context_time_travel
cp "$GREEN/coordinator.py" "$API/coordinator.py"
cp "$GREEN/strata_backend.py" "$API/strata_backend.py"
[[ $(sha256sum "$API/coordinator.py" | cut -d ' ' -f1) == bb6a1f8d9582c35dfaec69391658a033320e25d717b61ebfb9b0bec1f6c6c32b ]]
[[ $(sha256sum "$API/strata_backend.py" | cut -d ' ' -f1) == d692e00310984dcea2332d86862fde0010ff06f50b7f62b35a97d25110443de1 ]]
cp /home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime/serve/chat_template.jinja "$STAGE/source/serve/chat_template.jinja"
chmod -R a-w "$STAGE/source"
exec bash /home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh \
 "$STAGE/source" "$STAGE/run" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/http-cpu-v3.py" "$STAGE/result.json"