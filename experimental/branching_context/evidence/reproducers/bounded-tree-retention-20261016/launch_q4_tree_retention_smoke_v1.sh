#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
CANDIDATE=$ROOT/gpu-candidate-v1
RUNTIME=$CANDIDATE/runtime
PROFILE=$CANDIDATE/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == 107cab965436707d13db92271aa629e7829950030dbe60b884308076f8e31253 ]]
[[ $(sha256sum "$ROOT/run_q4_cli_trees.py" | cut -d ' ' -f1) == 3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2 ]]
[[ $(sha256sum "$ROOT/verify_q4_cli_tree_frontiers.py" | cut -d ' ' -f1) == a265d9bc8dc069a1a59390b912a3674a3ff95e132b83323564b4f28a1b01b8d0 ]]
[[ $(cat "$ROOT/g0-command-v7/command.exit") == 0 && $(cat "$ROOT/g0-command-v7/wrapper.exit") == 0 ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d '[:space:]')
(( available >= 8*1024*1024*1024 ))
[[ ! -e $ROOT/gpu-smoke-v1 && ! -e $ROOT/gpu-smoke-command-v1 && ! -e $ROOT/workspace-store-v1 ]]
mkdir -m700 "$ROOT/gpu-harness-v1"
cp "$ROOT/run_q4_cli_trees.py" "$ROOT/verify_q4_cli_tree_frontiers.py" "$ROOT/gpu-harness-v1/"
chmod -R a-w "$ROOT/gpu-harness-v1"
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
bash "$RUNNER" "$ROOT/gpu-harness-v1" "$ROOT/gpu-smoke-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/gpu-harness-v1/run_q4_cli_trees.py" \
 --runtime-root "$RUNTIME" --profile "$PROFILE" --fixtures "$CANDIDATE/fixture-set.json" \
 --store "$ROOT/workspace-store-v1" --run-dir "$ROOT/gpu-smoke-v1" --phase smoke --outputs 8
exec bash "$RUNNER" "$ROOT/gpu-harness-v1" "$ROOT/gpu-smoke-reference-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/gpu-harness-v1/verify_q4_cli_tree_frontiers.py" \
 --runtime-root "$RUNTIME" --profile "$PROFILE" --fixtures "$CANDIDATE/fixture-set.json" \
 --campaign "$ROOT/gpu-smoke-v1/result.json" --run-dir "$ROOT/gpu-smoke-reference-v1"
