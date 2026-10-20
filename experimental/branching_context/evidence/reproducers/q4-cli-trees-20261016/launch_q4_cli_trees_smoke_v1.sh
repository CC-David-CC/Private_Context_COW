#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1
RUNTIME=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime
PROFILE=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SCRIPT=$ROOT/campaign-source-v1/run_q4_cli_trees_v1.py
[[ $(sha256sum "$SCRIPT" | cut -d ' ' -f1) == 1434178ed852d7a38cee7c3bfccd5cc1b8d6eb9d19522cecbe2f75f584e98460 ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755 ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
(( available >= 8*1024*1024*1024 ))
[[ ! -e $ROOT/smoke-v1 && ! -e $ROOT/smoke-command-v1 ]]
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
 bash "$RUNNER" "$ROOT/campaign-source-v1" "$ROOT/smoke-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" --runtime-root "$RUNTIME" --profile "$PROFILE" \
 --fixtures "$ROOT/fixtures-v1/fixture-set.json" --store "$ROOT/workspace-store-v1" \
 --run-dir "$ROOT/smoke-v1" --phase smoke --outputs 8
