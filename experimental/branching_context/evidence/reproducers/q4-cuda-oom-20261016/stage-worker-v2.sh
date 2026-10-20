#!/usr/bin/env bash
set -euo pipefail
BASE=/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1
NAME=${1:?fresh stage name}
[[ $NAME == replacement-root-red-v4 || $NAME == replacement-root-green-v4 ]]
STAGE=$BASE/$NAME
[[ ! -e $STAGE ]]
mkdir -m 700 "$STAGE"
mkdir -m 700 "$STAGE/source"
tar -xf "$BASE/worker-recovery-root-v4.tar" -C "$STAGE/source" --no-same-owner
if [[ $NAME == replacement-root-red-v4 ]]; then
  cp "$BASE/replacement-root-red-v3/source/src/context_time_travel/coordinator.py" "$STAGE/source/src/context_time_travel/coordinator.py"
fi
chmod -R a-w "$STAGE/source"
[[ -z $(find "$STAGE/source" -perm /222 -print -quit) ]]
exec bash /home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh \
 "$STAGE/source" "$STAGE/run" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$BASE/worker_recovery_cpu_review_v1.py" "$STAGE/source" "$STAGE/report.json"