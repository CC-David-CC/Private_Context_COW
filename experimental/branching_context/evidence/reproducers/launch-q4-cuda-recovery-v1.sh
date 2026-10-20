#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1
SOURCE=$(realpath "${1:?frozen script directory}")
SCRIPT_SHA=${2:?runner SHA256}
FIXTURE=${3:?tiny or target}
MODE=${4:?preflight or run}
LABEL=${5:?unique attempt label}
[[ $SOURCE == /home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/*/source/experimental/branching_context/scripts ]]
[[ $SCRIPT_SHA =~ ^[0-9a-f]{64}$ && $LABEL =~ ^v[0-9]+$ ]]
[[ $FIXTURE == tiny || $FIXTURE == target ]]
[[ $MODE == preflight || $MODE == run ]]
RUNTIME=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime
PROFILE=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
HELPER=/home/dflanag3/spatial-artifacts/q4-cuda-allocation-helper-20261016-v1/build-v1/branch_pressure_helper
SCRIPT=$SOURCE/run_q4_cuda_recovery.py
[[ $(sha256sum "$SCRIPT" | cut -d ' ' -f1) == "$SCRIPT_SHA" ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ -z $(find "$SOURCE" -perm /222 -print -quit) ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
if [[ $FIXTURE == tiny ]]; then
  SNAPSHOT=96e083885945e0cf0517aa1709a1d80db3029bf04bcc4028cc8cb55a4a74e0d8
else
  SNAPSHOT=$(cat "$ROOT/target-source-snapshot.txt")
fi
[[ $SNAPSHOT =~ ^[0-9a-f]{64}$ ]]
STORE=$ROOT/$FIXTURE-input-store-v1
OUTPUT=$ROOT/$FIXTURE-recovery-$LABEL/campaign
COMMAND=$ROOT/$FIXTURE-recovery-$MODE-command-$LABEL
[[ ! -e $OUTPUT && ! -e $COMMAND ]]
ARGS=(--runtime-root "$RUNTIME" --profile "$PROFILE" --store "$STORE"
 --api-src "$RUNTIME/experimental/branching_context/src" --api-scripts "$SOURCE"
 --branch CliA --expected-source-snapshot "$SNAPSHOT" --run-dir "$OUTPUT"
 --expected-profile-sha256 a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755
 --expected-native-sha256 4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451
 --expected-http-sha256 9f6839a359e05cbb47e0850046c6f32f00e7468265dff80fedc3d72d9f720f31
 --expected-helper-sha256 a3d2690d104f192ca0a25ee224db4b3f7fcef45d0cad4de4a491c755937cd61b
 --pressure-helper "$HELPER" --recovery-branch "Restart${FIXTURE}${LABEL}"
 --ram-gib 96 --disk-gib 128 --max-tokens 4 --timeout 180)
if [[ $MODE == preflight ]]; then
 ARGS+=(--preflight-only)
 exec bash "$RUNNER" "$SOURCE" "$COMMAND" env PYTHONDONTWRITEBYTECODE=1 \
  /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" "${ARGS[@]}"
fi
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
(( available >= 17*1024*1024*1024 ))
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
 bash "$RUNNER" "$SOURCE" "$COMMAND" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" "${ARGS[@]}"
