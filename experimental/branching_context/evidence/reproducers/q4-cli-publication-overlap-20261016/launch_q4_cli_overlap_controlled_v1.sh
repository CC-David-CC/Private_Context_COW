#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cli-overlap-20261016-v1
BASE=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1
RUNTIME=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime
PROFILE=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SCRIPT=$ROOT/guard-green-source-v1/run_q4_cli_overlap.py
OBSERVER=/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v17/source/experimental/branching_context/scripts/observe_cow_cli.py
[[ $(sha256sum "$SCRIPT" | cut -d ' ' -f1) == 9e146f3cc6abed0df2ef4d30ccf53b8f3289172ab7f60067a0866af3a2b41e3a ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755 ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$OBSERVER" | cut -d ' ' -f1) == 20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28 ]]
[[ $(sha256sum "$BASE/campaign-reference-v1/result.json" | cut -d ' ' -f1) == 9048df92e236e243bb152b64568cb767c3a80a60b2de6d168eef862cbac06d6d ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ $(cat "$ROOT/guard-green-command-v1/command.exit") == 0 ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
(( available >= 8*1024*1024*1024 ))
[[ ! -e $ROOT/controlled-v1 && ! -e $ROOT/controlled-command-v1 ]]
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
 bash "$RUNNER" "$ROOT/guard-green-source-v1" "$ROOT/controlled-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" --runtime-root "$RUNTIME" --profile "$PROFILE" \
 --fixtures "$BASE/fixtures-v1/fixture-set.json" --store "$BASE/workspace-store-v1" \
 --reference-result "$BASE/campaign-reference-v1/result.json" --run-dir "$ROOT/controlled-v1" --mode controlled --outputs 8 \
 --observer "$OBSERVER" --observer-sha256 20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28
