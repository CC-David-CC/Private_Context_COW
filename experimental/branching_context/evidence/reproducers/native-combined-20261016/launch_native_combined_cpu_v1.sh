#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1
SOURCE=$ROOT/source-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
PYTHON=/home/dflanag3/src/Strata/.venv/bin/python
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$ROOT/native-combined-source-20261016-v1.tar" | cut -d ' ' -f1) == 7a596c12351e458f4e849884a6daccbdf50302023aafb46691181cf282c742eb ]]
[[ ! -e $SOURCE && ! -e $ROOT/cpu-command-v1 ]]
mkdir -m700 "$SOURCE" "$ROOT/g0-evidence-v1"
tar -xf "$ROOT/native-combined-source-20261016-v1.tar" -C "$SOURCE"
mkdir -m700 "$SOURCE/experimental/branching_context/evidence"
ln -s "$ROOT/g0-evidence-v1" "$SOURCE/experimental/branching_context/evidence/runs"
chmod -R a-w "$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE:$SOURCE/tools:$SOURCE/experimental/branching_context:$SOURCE/experimental/branching_context/src:$SOURCE/experimental/branching_context/tests/implementation:$SOURCE/experimental/branching_context/scripts"
bash "$RUNNER" "$SOURCE" "$ROOT/cpu-command-v1" "$PYTHON" -B "$ROOT/run_native_combined_cpu_v1.py" "$SOURCE" "$ROOT/cpu-v1.json" \
 serve.test_cow_context_root_capability serve.test_cow_text_adapter serve.test_cow_presentation_adapter \
 serve.test_cow_presentation_flow serve.test_cow_namespace_http serve.test_cow_http \
 serve.test_cow_disconnect_durability serve.test_cow_workspace test_store_presentation_catalog \
 test_root_presentation_query test_publication_handoff_ownership test_bounded_tree_retention \
 test_checkpoint_resource_policy test_pool_coordinator test_pool_cold_cache_ownership \
 test_publication_overlap test_pool_root_residency test_pool_close_release \
 test_poisoned_worker_replacement test_root_computation_reuse test_pool_source_trace \
 test_pool_failed_lru test_pool_spill_reload_trace test_pool_disk_lru_integration \
 test_canonical_branch_actions test_branch_checkpoint_eviction
bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v1" "$PYTHON" -B scripts/verify.py --gate G0
# Keep the known baseline expectation conflict separate, visible and unmodified.
set +e
bash "$RUNNER" "$SOURCE" "$ROOT/baseline-conflict-command-v1" "$PYTHON" -B "$ROOT/run_native_combined_cpu_v1.py" "$SOURCE" "$ROOT/baseline-conflict-v1.json" test_pool_branch_actions
baseline_exit=$?
set -e
[[ $baseline_exit == 1 ]]
