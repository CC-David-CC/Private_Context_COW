#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
PY=/home/dflanag3/src/Strata/.venv/bin/python
for stage in guard-red-source-v1 guard-green-source-v1; do
 chmod -R a-w "$ROOT/$stage"
done
set +e
bash "$RUNNER" "$ROOT/guard-red-source-v1" "$ROOT/guard-red-command-v1" env PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m unittest -v test_q4_cli_trees
red=$?
set -e
[[ $red == 1 ]]
grep -q 'cannot import name.*validate_observation' "$ROOT/guard-red-command-v1/stderr.log"
exec bash "$RUNNER" "$ROOT/guard-green-source-v1" "$ROOT/guard-green-command-v1" env PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m unittest -v test_q4_cli_trees
