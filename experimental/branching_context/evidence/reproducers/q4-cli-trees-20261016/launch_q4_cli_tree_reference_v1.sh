#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1
RUNTIME=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime
PROFILE=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SCRIPT=$ROOT/reference-source-v1/verify_q4_cli_tree_frontiers_v1.py
[[ $(sha256sum "$SCRIPT" | cut -d ' ' -f1) == e7a78e6363a0f38bc03d3d7a82674cd52611a32802e5287de7ce45166ada01fa ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755 ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ ! -e /proc/2865483 && ! -e /proc/2865489 ]]
[[ $(cat "$ROOT/smoke-command-v2/command.exit") == 0 && $(cat "$ROOT/smoke-command-v2/wrapper.exit") == 0 ]]
[[ ! -e $ROOT/smoke-reference-v1 && ! -e $ROOT/smoke-reference-command-v1 ]]
chmod -R a-w "$ROOT/reference-source-v1"
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
 bash "$RUNNER" "$ROOT/reference-source-v1" "$ROOT/smoke-reference-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" --runtime-root "$RUNTIME" --profile "$PROFILE" \
 --fixtures "$ROOT/fixtures-v1/fixture-set.json" --campaign "$ROOT/smoke-v2/result.json" \
 --run-dir "$ROOT/smoke-reference-v1"
