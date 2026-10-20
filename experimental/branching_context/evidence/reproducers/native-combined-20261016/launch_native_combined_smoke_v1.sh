#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1
CANDIDATE=$ROOT/gpu-candidate-v2
RUNTIME=$CANDIDATE/runtime
PROFILE=$CANDIDATE/profile.json
HARNESS=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1/gpu-harness-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
PYTHON=/home/dflanag3/src/Strata/.venv/bin/python
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == fb2d32373103536d13186abd9051ae0600cbeefa6012c84d080e40d5f21c2ff6 ]]
[[ $(sha256sum "$HARNESS/run_q4_cli_trees.py" | cut -d ' ' -f1) == 3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2 ]]
[[ $(sha256sum "$HARNESS/verify_q4_cli_tree_frontiers.py" | cut -d ' ' -f1) == a265d9bc8dc069a1a59390b912a3674a3ff95e132b83323564b4f28a1b01b8d0 ]]
[[ $(cat "$ROOT/cpu-command-v1/command.exit") == 0 && $(cat "$ROOT/g0-command-v1/command.exit") == 0 ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ ! -e $ROOT/gpu-smoke-v1 && ! -e $ROOT/workspace-store-v1 ]]
# This bounded smoke creates one ~0.3GiB root, one child and short reference
# arrays. It is not the three-tree/128-output campaign's 8GiB capacity guard.
available=$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d '[:space:]')
(( available >= 2*1024*1024*1024 ))
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
bash "$RUNNER" "$HARNESS" "$ROOT/gpu-smoke-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 "$PYTHON" -B "$HARNESS/run_q4_cli_trees.py" \
 --runtime-root "$RUNTIME" --profile "$PROFILE" --fixtures "$CANDIDATE/fixture-set.json" \
 --store "$ROOT/workspace-store-v1" --run-dir "$ROOT/gpu-smoke-v1" --phase smoke --outputs 8
exec bash "$RUNNER" "$HARNESS" "$ROOT/gpu-smoke-reference-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 "$PYTHON" -B "$HARNESS/verify_q4_cli_tree_frontiers.py" \
 --runtime-root "$RUNTIME" --profile "$PROFILE" --fixtures "$CANDIDATE/fixture-set.json" \
 --campaign "$ROOT/gpu-smoke-v1/result.json" --run-dir "$ROOT/gpu-smoke-reference-v1"
