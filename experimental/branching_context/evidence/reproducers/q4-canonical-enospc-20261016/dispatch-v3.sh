#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
mode=${1:?preflight, filler or run}
stage=${2:?immutable stage}
driver_sha=${3:?driver SHA}
image_sha=${4:?image SHA}
[[ $mode == preflight || $mode == filler || $mode == run ]]
[[ $stage =~ ^final-green-v[0-9]+$ && $driver_sha =~ ^[0-9a-f]{64}$ && $image_sha =~ ^[0-9a-f]{64}$ ]]
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$ROOT/launch-v4/launch.sh" | cut -d ' ' -f1) == b493de7cfc8082d0278de14d60a84d242ff7c256adc9ab4019ff0bb0f4846848 ]]
[[ $(sha256sum "$ROOT/launch-v4/private-mount.sh" | cut -d ' ' -f1) == 529ce950ec6b94c86f2841e4b19170998d8ea4108f8f7355beaed0095059ebe7 ]]
SOURCE=$ROOT/$stage/source
DRIVER=$SOURCE/experimental/branching_context/scripts/run_q4_canonical_enospc.py
HELPER=$SOURCE/experimental/branching_context/scripts/run_q4_publication_fault.py
[[ ! -L $SOURCE && -d $SOURCE && -f $DRIVER && ! -L $DRIVER ]]
[[ $(sha256sum "$DRIVER" | cut -d ' ' -f1) == "$driver_sha" ]]
[[ -f $HELPER && ! -L $HELPER && $(sha256sum "$HELPER" | cut -d ' ' -f1) == d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695 ]]
[[ -z $(find "$SOURCE" -perm /222 -print -quit) ]]
exec bash "$RUNNER" "$SOURCE" "$ROOT/launch-v4/$mode-command-v1" bash "$ROOT/launch-v4/launch.sh" "$mode" "$DRIVER" "$driver_sha" "$image_sha"
