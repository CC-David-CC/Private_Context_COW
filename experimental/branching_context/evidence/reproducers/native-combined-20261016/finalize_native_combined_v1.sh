#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
for attempt in $(seq 1 120); do
    if [[ -f $ROOT/gpu-campaign-reference-command-v1/wrapper.exit ]]; then break; fi
    sleep 5
done
for name in command.exit wrapper.exit; do
    [[ $(cat "$ROOT/gpu-campaign-reference-command-v1/$name") == 0 ]]
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
bash "$RUNNER" "$ROOT/source-v1" "$ROOT/primary-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/collect_native_combined_v1.py"
exec bash "$RUNNER" "$ROOT/source-v1" "$ROOT/plot-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1/chart-env-v1/bin/python -B "$ROOT/plot_native_combined_v1.py"
