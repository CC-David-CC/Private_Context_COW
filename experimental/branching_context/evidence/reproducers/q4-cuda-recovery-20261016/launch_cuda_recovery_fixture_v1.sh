#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1
SOURCE=$ROOT/fixture-source
SCRIPT=$SOURCE/prepare_cuda_oom_fixture_v1.py
RUNTIME=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime
PROFILE=/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
FIXTURE=${1:?tiny or target}
MODE=${2:?prepare or reference}
[[ $FIXTURE == tiny || $FIXTURE == target ]]
[[ $MODE == prepare || $MODE == reference ]]
[[ $(sha256sum "$SCRIPT" | cut -d ' ' -f1) == 34d1b4b91aa9f16651b975326b127c1b5e681650eb33b9ab677834dea690aa3b ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755 ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ -z $(find "$SOURCE" -perm /222 -print -quit) ]]
[[ -z $(find "$RUNTIME" -perm /222 -print -quit) ]]
if [[ $FIXTURE == tiny ]]; then
  EXPECTED=f64cc0c251fcfdcc4be7c46b41ad3253593b3f95207dbfa3b15baf76f31b7278
else
  EXPECTED=e9e261d7b2175f004635538024dff39bc794dd5d7f476967b31113dce463782f
fi
[[ $(sha256sum "$ROOT/inputs/$FIXTURE.json" | cut -d ' ' -f1) == "$EXPECTED" ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
(( available >= 17*1024*1024*1024 ))
if [[ $MODE == prepare ]]; then OUTPUT=$ROOT/$FIXTURE-prepared-v1; else OUTPUT=$ROOT/$FIXTURE-reference-v1; fi
[[ ! -e $OUTPUT && ! -e $ROOT/$FIXTURE-$MODE-command-v1 ]]
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
 bash "$RUNNER" "$SOURCE" "$ROOT/$FIXTURE-$MODE-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$SCRIPT" "$MODE" \
 --profile "$PROFILE" --runtime "$RUNTIME" --fixture "$ROOT/inputs/$FIXTURE.json" \
 --output "$OUTPUT" --prepared "$ROOT/$FIXTURE-prepared-v1" \
 --expected-profile-sha256 a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755
