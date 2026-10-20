#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SOURCE=$ROOT/green-stage-v2/source
[[ ! -e $ROOT/green-command-v3 && ! -e $ROOT/g0-command-v3 ]]
bash "$RUNNER" "$SOURCE" "$ROOT/green-command-v3" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/run-cpu-v3.py" "$SOURCE" "$ROOT/cpu-v3.json" \
 test_bounded_tree_retention test_checkpoint_resource_policy test_pool_coordinator \
 test_pool_cold_cache_ownership test_publication_overlap test_pool_root_residency \
 test_pool_close_release test_poisoned_worker_replacement test_root_computation_reuse \
 test_pool_source_trace test_pool_failed_lru test_pool_spill_reload_trace \
 test_pool_disk_lru_integration test_pool_branch_actions
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v3" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B scripts/verify.py --gate G0
