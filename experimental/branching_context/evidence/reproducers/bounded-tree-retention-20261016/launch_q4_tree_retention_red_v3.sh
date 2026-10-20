#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SOURCE=$ROOT/red-stage-v3/source
[[ $(sha256sum "$ROOT/test_bounded_tree_retention_v2.py" | cut -d ' ' -f1) == a93064a33a14b3b4d611243e16944cdfb5c5453c7e96301006e1338fb839f896 ]]
[[ ! -e $SOURCE && ! -e $ROOT/red-command-v3 ]]
mkdir -m700 "$ROOT/red-stage-v3"
cp -a "$ROOT/red-stage-v2/source" "$SOURCE"
chmod u+w "$SOURCE/experimental/branching_context/tests/implementation/test_bounded_tree_retention.py"
cp "$ROOT/test_bounded_tree_retention_v2.py" "$SOURCE/experimental/branching_context/tests/implementation/test_bounded_tree_retention.py"
chmod -R a-w "$SOURCE"
exec bash "$RUNNER" "$SOURCE" "$ROOT/red-command-v3" env PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH="$SOURCE/experimental/branching_context/src:$SOURCE/experimental/branching_context/tests/implementation" \
 /home/dflanag3/src/Strata/.venv/bin/python -B -m unittest discover \
 -s experimental/branching_context/tests/implementation -p test_bounded_tree_retention.py -v
