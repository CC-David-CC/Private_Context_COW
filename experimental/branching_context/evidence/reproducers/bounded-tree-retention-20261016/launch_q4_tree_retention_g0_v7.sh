#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SOURCE=$ROOT/g0-source-v7
[[ ! -e $SOURCE && ! -e $ROOT/g0-command-v7 && ! -e $ROOT/g0-evidence-v7 ]]
cp -a "$ROOT/green-stage-v4/source" "$SOURCE"
mkdir -m700 "$ROOT/g0-evidence-v7"
chmod u+w "$SOURCE/experimental/branching_context"
mkdir -m700 "$SOURCE/experimental/branching_context/evidence"
ln -s "$ROOT/g0-evidence-v7" "$SOURCE/experimental/branching_context/evidence/runs"
chmod a-w "$SOURCE/experimental/branching_context" "$SOURCE/experimental/branching_context/evidence"
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v7" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B scripts/verify.py --gate G0
