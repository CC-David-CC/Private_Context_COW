#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1
STAGE=$ROOT/integrated-62b4e466-v2
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
[[ $(sha256sum "$ROOT/integrated-62b4e466-v2.tar" | cut -d ' ' -f1) == f9ed3c599825499922e51b862d1c81a4afba424e3b0d687612f5ea9e8c4131b0 ]]
mkdir "$STAGE"
mkdir "$STAGE/source"
tar -xf "$ROOT/integrated-62b4e466-v2.tar" -C "$STAGE/source"
chmod -R a-w "$STAGE/source"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$STAGE/source:$STAGE/source/experimental/branching_context/src:$STAGE/source/serve:$STAGE/source/experimental/branching_context/scripts
# Focus81 already passed from identical API/test bytes; this package adds HTTP dependencies.
bash "$RUNNER" "$STAGE/source" "$STAGE/http-command" /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/run-integrated-controls-v2.py" http
cd "$STAGE/source/experimental/branching_context"
bash "$RUNNER" "$STAGE/source" "$STAGE/g0-command" /home/dflanag3/src/Strata/.venv/bin/python -B "$STAGE/source/experimental/branching_context/scripts/verify.py" --gate G0 --no-write
