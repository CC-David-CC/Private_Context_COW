#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
STAGE=$ROOT/green-stage-v2
SOURCE=$STAGE/source
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$ROOT/green-stage-v1/coordinator.py" | cut -d ' ' -f1) == 7042b31f1f4bb23580cd21213052ad27e8ab861b011dca98a95ede9db2602f62 ]]
[[ $(cat "$ROOT/red-command-v3/command.exit") == 1 && $(cat "$ROOT/red-command-v3/wrapper.exit") == 1 ]]
[[ ! -e $SOURCE && ! -e $ROOT/green-command-v2 && ! -e $ROOT/g0-command-v2 ]]
mkdir -m700 "$STAGE"
cp -a "$ROOT/red-stage-v3/source" "$SOURCE"
chmod u+w "$SOURCE/experimental/branching_context/src/context_time_travel/coordinator.py"
cp "$ROOT/green-stage-v1/coordinator.py" "$SOURCE/experimental/branching_context/src/context_time_travel/coordinator.py"
chmod -R a-w "$SOURCE"
bash "$RUNNER" "$SOURCE" "$ROOT/green-command-v2" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/run-cpu-v2.py" "$SOURCE" "$ROOT/cpu-v2.json" \
 test_bounded_tree_retention test_checkpoint_resource_policy test_pool_coordinator \
 test_pool_cold_cache_ownership test_publication_overlap test_pool_root_residency \
 test_pool_close_release test_poisoned_worker_replacement test_root_computation_reuse \
 test_pool_source_trace test_pool_failed_lru test_pool_spill_reload_trace \
 test_pool_disk_lru_integration test_pool_branch_actions
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v2" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B scripts/verify.py --gate G0
