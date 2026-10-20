#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
STAGE=$ROOT/green-stage-v1
SOURCE=$STAGE/source
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$STAGE/coordinator.py" | cut -d ' ' -f1) == 7042b31f1f4bb23580cd21213052ad27e8ab861b011dca98a95ede9db2602f62 ]]
[[ $(cat "$ROOT/red-command-v2/command.exit") == 1 && $(cat "$ROOT/red-command-v2/wrapper.exit") == 1 ]]
[[ ! -e $SOURCE && ! -e $ROOT/green-command-v1 && ! -e $ROOT/g0-command-v1 ]]
cp -a "$ROOT/red-stage-v2/source" "$SOURCE"
chmod u+w "$SOURCE/experimental/branching_context/src/context_time_travel/coordinator.py"
cp "$STAGE/coordinator.py" "$SOURCE/experimental/branching_context/src/context_time_travel/coordinator.py"
chmod -R a-w "$SOURCE"
bash "$RUNNER" "$SOURCE" "$ROOT/green-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/run-cpu-v1.py" "$SOURCE" "$ROOT/cpu-v1.json" \
 test_bounded_tree_retention test_checkpoint_resource_policy test_pool_coordinator \
 test_pool_cold_cache_ownership test_publication_overlap
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v1" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B scripts/verify.py --gate G0
