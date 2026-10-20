#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cuda-oom-20261016-v1
RUNTIME=/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/runtime-v4
PROFILE=/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/profiles-v2/q4/profile.json
NATIVE=/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/build-green-v1/strata-batched-branch-lazy
HELPER=/home/dflanag3/spatial-artifacts/q4-cuda-allocation-helper-20261016-v1/build-v1/branch_pressure_helper
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
PY=/home/dflanag3/src/Strata/.venv/bin/python
mode=${1:?check or run}
SOURCE=${2:?immutable source path}
driver_sha=${3:?reviewed driver SHA-256}
[[ $(id -u) == 1000 && ( $mode == check || $mode == run ) ]]
[[ $SOURCE == "$ROOT"/*/source && -d $SOURCE && ! -L $SOURCE ]]
[[ $driver_sha =~ ^[0-9a-f]{64}$ ]]
DRIVER=$SOURCE/experimental/branching_context/scripts/run_q4_cuda_oom.py
SCRIPTS=$SOURCE/experimental/branching_context/scripts
[[ -f $DRIVER && ! -L $DRIVER && $(sha256sum "$DRIVER" | cut -d ' ' -f1) == "$driver_sha" ]]
[[ -z $(find "$SOURCE" -perm /222 -print -quit) ]]
[[ $(sha256sum "$SCRIPTS/run_q4_publication_fault.py" | cut -d ' ' -f1) == d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695 ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$PROFILE" | cut -d ' ' -f1) == f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c ]]
[[ $(sha256sum "$NATIVE" | cut -d ' ' -f1) == 4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451 ]]
[[ $(sha256sum "$HELPER" | cut -d ' ' -f1) == a3d2690d104f192ca0a25ee224db4b3f7fcef45d0cad4de4a491c755937cd61b ]]
[[ $(sha256sum "$RUNTIME/serve/cow_http.py" | cut -d ' ' -f1) == 9f6839a359e05cbb47e0850046c6f32f00e7468265dff80fedc3d72d9f720f31 ]]
[[ $(sha256sum "$RUNTIME/serve/cow_workspace.py" | cut -d ' ' -f1) == ed1a3e693fa0be423cd5e1413f59dc08c5fccfe6052483ded62298c2dbec9c72 ]]
[[ -x $HELPER && -z $(find "$HELPER" -perm /222 -print) ]]
[[ -d $ROOT/input-store-v1 && -d $ROOT/gpu-q4-v2 && ! -e $ROOT/gpu-q4-v2/campaign && ! -e $ROOT/gpu-command-v2 ]]
[[ $(sha256sum "$ROOT/input-store-v1/objects/808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85.json" | cut -d ' ' -f1) == 808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85 ]]
[[ $(sha256sum "$ROOT/input-store-v1/checkpoints/423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810.state" | cut -d ' ' -f1) == 423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810 ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
(( available >= 17*1024*1024*1024 ))
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPTS:$RUNTIME:$RUNTIME/experimental/branching_context/src:$RUNTIME/experimental/branching_context/scripts"
args=(--profile "$PROFILE" --store "$ROOT/input-store-v1"
  --runtime-root "$RUNTIME" --api-src "$RUNTIME/experimental/branching_context/src" --api-scripts "$SCRIPTS"
  --branch CliA --run-dir "$ROOT/gpu-q4-v2/campaign" --pressure-helper "$HELPER"
  --expected-profile-sha256 f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c
  --expected-native-sha256 4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451
  --expected-helper-sha256 a3d2690d104f192ca0a25ee224db4b3f7fcef45d0cad4de4a491c755937cd61b
  --expected-http-sha256 9f6839a359e05cbb47e0850046c6f32f00e7468265dff80fedc3d72d9f720f31
  --expected-source-snapshot 808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85
  --ram-gib 96 --disk-gib 128 --max-tokens 4 --timeout 180)
if [[ $mode == check ]]; then
  "$PY" -B "$DRIVER" "${args[@]}" --preflight-only
  [[ ! -e $ROOT/gpu-q4-v2/campaign ]]
  exit 0
fi
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
  bash "$RUNNER" "$SOURCE" "$ROOT/gpu-command-v2" "$PY" -B "$DRIVER" "${args[@]}"
