#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
SOURCE=$ROOT/green-stage-v4/source
ARCHIVE=$ROOT/red-stage/source.tar
[[ $(sha256sum "$ARCHIVE" | cut -d ' ' -f1) == cd280367bf59106d2de1316058d9f8caa3a877482903c4584e180ee9ea9782aa ]]
[[ ! -e $SOURCE && ! -e $ROOT/green-command-v4 && ! -e $ROOT/g0-command-v4 ]]
mkdir -m700 "$ROOT/green-stage-v4"
cp -a "$ROOT/green-stage-v2/source" "$SOURCE"
chmod u+w "$SOURCE"
tar -xf "$ARCHIVE" -C "$SOURCE" --wildcards 'serve/*.py' 'tools/*.py'
chmod -R a-w "$SOURCE"
bash "$RUNNER" "$SOURCE" "$ROOT/green-command-v4" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B "$ROOT/run-cpu-v4.py" "$SOURCE" "$ROOT/cpu-v4.json" \
 test_bounded_tree_retention test_checkpoint_resource_policy test_pool_coordinator \
 test_pool_cold_cache_ownership test_publication_overlap test_pool_root_residency \
 test_pool_close_release test_poisoned_worker_replacement test_root_computation_reuse \
 test_pool_source_trace test_pool_failed_lru test_pool_spill_reload_trace \
 test_pool_disk_lru_integration test_pool_branch_actions
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v4" env PYTHONDONTWRITEBYTECODE=1 \
 /home/dflanag3/src/Strata/.venv/bin/python -B scripts/verify.py --gate G0
