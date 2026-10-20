#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SOURCE=$ROOT/baseline-branch-source-v1
[[ ! -e $SOURCE && ! -e $ROOT/baseline-branch-command-v1 ]]
cp -a "$ROOT/green-stage-v4/source" "$SOURCE"
chmod u+w "$SOURCE/experimental/branching_context/src/context_time_travel/coordinator.py"
cp "$ROOT/red-stage-v3/source/experimental/branching_context/src/context_time_travel/coordinator.py" "$SOURCE/experimental/branching_context/src/context_time_travel/coordinator.py"
chmod -R a-w "$SOURCE"
exec bash "$RUNNER" "$SOURCE" "$ROOT/baseline-branch-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH="$SOURCE:$SOURCE/tools:$SOURCE/experimental/branching_context:$SOURCE/experimental/branching_context/src:$SOURCE/experimental/branching_context/tests/implementation:$SOURCE/experimental/branching_context/scripts" \
 /home/dflanag3/src/Strata/.venv/bin/python -B -m unittest discover \
 -s experimental/branching_context/tests/implementation -p test_pool_branch_actions.py -v
