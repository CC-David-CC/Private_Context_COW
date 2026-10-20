#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1
RUNTIME=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime
PROFILE=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SCRIPT=$ROOT/guard-green-source-v1/run_q4_cli_trees.py
OBSERVER=/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v16/source/experimental/branching_context/scripts/observe_cow_cli.py
[[ $(sha256sum "$SCRIPT" | cut -d ' ' -f1) == 3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2 ]]
[[ $(sha256sum "$OBSERVER" | cut -d ' ' -f1) == 294c360e94f2261f9c6e2fb33c1bf14e5b44491f147cf3ab634acb836c420aa2 ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755 ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ $(cat "$ROOT/smoke-reference-command-v2/command.exit") == 0 ]]
[[ ! -e $ROOT/smoke-observed-v1 && ! -e $ROOT/smoke-observed-command-v1 ]]
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
 bash "$RUNNER" "$ROOT/guard-green-source-v1" "$ROOT/smoke-observed-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" --runtime-root "$RUNTIME" --profile "$PROFILE" \
 --fixtures "$ROOT/fixtures-v1/fixture-set.json" --store "$ROOT/workspace-store-v1" \
 --run-dir "$ROOT/smoke-observed-v1" --phase smoke --outputs 8 \
 --observer "$OBSERVER" --observer-sha256 294c360e94f2261f9c6e2fb33c1bf14e5b44491f147cf3ab634acb836c420aa2
