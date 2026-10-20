#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SOURCE=$ROOT/green-stage-v4/source
[[ ! -e $ROOT/focus-command-v5 && ! -e $ROOT/g0-command-v5 ]]
bash "$RUNNER" "$SOURCE" "$ROOT/focus-command-v5" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/run-cpu-v4.py" "$SOURCE" "$ROOT/focus-v5.json" \
 test_bounded_tree_retention test_canonical_branch_actions test_branch_checkpoint_eviction
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v5" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B scripts/verify.py --gate G0
