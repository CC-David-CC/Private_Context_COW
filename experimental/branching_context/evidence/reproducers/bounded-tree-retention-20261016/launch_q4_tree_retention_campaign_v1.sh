#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
CANDIDATE=$ROOT/gpu-candidate-v1
RUNTIME=$CANDIDATE/runtime
PROFILE=$CANDIDATE/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
OBSERVER=/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v17/source/experimental/branching_context/scripts/observe_cow_cli.py
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == 107cab965436707d13db92271aa629e7829950030dbe60b884308076f8e31253 ]]
[[ $(sha256sum "$ROOT/gpu-harness-v1/run_q4_cli_trees.py" | cut -d ' ' -f1) == 3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2 ]]
[[ $(sha256sum "$ROOT/gpu-harness-v1/verify_q4_cli_tree_frontiers.py" | cut -d ' ' -f1) == a265d9bc8dc069a1a59390b912a3674a3ff95e132b83323564b4f28a1b01b8d0 ]]
[[ $(sha256sum "$OBSERVER" | cut -d ' ' -f1) == 20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28 ]]
[[ $(cat "$ROOT/gpu-smoke-reference-command-v1/command.exit") == 0 && $(cat "$ROOT/gpu-smoke-reference-command-v1/wrapper.exit") == 0 ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d '[:space:]')
(( available >= 8*1024*1024*1024 ))
[[ ! -e $ROOT/gpu-campaign-v1 && ! -e $ROOT/gpu-campaign-command-v1 ]]
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
bash "$RUNNER" "$ROOT/gpu-harness-v1" "$ROOT/gpu-campaign-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/gpu-harness-v1/run_q4_cli_trees.py" \
 --runtime-root "$RUNTIME" --profile "$PROFILE" --fixtures "$CANDIDATE/fixture-set.json" \
 --store "$ROOT/workspace-store-v1" --run-dir "$ROOT/gpu-campaign-v1" --phase campaign --outputs 128 --repetitions 1 \
 --observer "$OBSERVER" --observer-sha256 20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28
exec bash "$RUNNER" "$ROOT/gpu-harness-v1" "$ROOT/gpu-campaign-reference-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/gpu-harness-v1/verify_q4_cli_tree_frontiers.py" \
 --runtime-root "$RUNTIME" --profile "$PROFILE" --fixtures "$CANDIDATE/fixture-set.json" \
 --campaign "$ROOT/gpu-campaign-v1/result.json" --run-dir "$ROOT/gpu-campaign-reference-v1"
