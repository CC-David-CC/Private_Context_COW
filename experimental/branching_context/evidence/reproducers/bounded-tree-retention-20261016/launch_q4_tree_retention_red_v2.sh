#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
STAGE=$ROOT/red-stage-v2
SOURCE=$STAGE/source
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(stat -c %s "$STAGE/source.tar") == 4198400 ]]
[[ $(sha256sum "$STAGE/source.tar" | cut -d ' ' -f1) == 100e71dd54364673b8d7958b094dcd08ac46eed4223953e247452bcc3cc16922 ]]
tar -tf "$STAGE/source.tar" >/dev/null
[[ ! -e $SOURCE && ! -e $ROOT/red-command-v2 ]]
mkdir -m700 "$SOURCE"
tar -xf "$STAGE/source.tar" -C "$SOURCE"
cp "$STAGE/test_bounded_tree_retention.py" "$SOURCE/experimental/branching_context/tests/implementation/"
chmod -R a-w "$SOURCE"
exec bash "$RUNNER" "$SOURCE" "$ROOT/red-command-v2" env PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH="$SOURCE/experimental/branching_context/src:$SOURCE/experimental/branching_context/tests/implementation" \
 /home/dflanag3/src/Strata/.venv/bin/python -B -m unittest discover \
 -s experimental/branching_context/tests/implementation -p test_bounded_tree_retention.py -v
