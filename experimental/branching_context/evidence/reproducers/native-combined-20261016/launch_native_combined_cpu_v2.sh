#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1
SOURCE=$ROOT/source-v2
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
PYTHON=/home/dflanag3/src/Strata/.venv/bin/python
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$ROOT/native-combined-notice-overlay-v1.tar" | cut -d ' ' -f1) == dd28b5e07790dbde42566fb5edc357814ed3c12cf248c416d90731e14ae5c2c9 ]]
[[ ! -e $SOURCE && ! -e $ROOT/cpu-command-v2 && ! -e $ROOT/cli-state-v2 ]]
cp -a "$ROOT/source-v1" "$SOURCE"
chmod u+w "$SOURCE/tools" "$SOURCE/experimental/branching_context/tests/implementation" \
 "$SOURCE/tools/cow_cli.py" "$SOURCE/experimental/branching_context/tests/implementation/test_pool_branch_actions.py" \
 "$SOURCE/experimental/branching_context/evidence"
tar -xf "$ROOT/native-combined-notice-overlay-v1.tar" -C "$SOURCE"
[[ -L $SOURCE/experimental/branching_context/evidence/runs ]]
[[ $(readlink "$SOURCE/experimental/branching_context/evidence/runs") == "$ROOT/g0-evidence-v1" ]]
unlink "$SOURCE/experimental/branching_context/evidence/runs"
mkdir -m700 "$ROOT/g0-evidence-v2" "$ROOT/cli-state-v2"
ln -s "$ROOT/g0-evidence-v2" "$SOURCE/experimental/branching_context/evidence/runs"
chmod -R a-w "$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export COW_CLI_STATE_DIR="$ROOT/cli-state-v2"
export PYTHONPATH="$SOURCE:$SOURCE/tools:$SOURCE/experimental/branching_context:$SOURCE/experimental/branching_context/src:$SOURCE/experimental/branching_context/tests/implementation:$SOURCE/experimental/branching_context/scripts"
bash "$RUNNER" "$SOURCE" "$ROOT/cpu-command-v2" "$PYTHON" -B "$ROOT/run_native_combined_cpu_v2.py" "$SOURCE" "$ROOT/cpu-v2.json" \
 serve.test_cow_context_root_capability serve.test_cow_text_adapter serve.test_cow_presentation_adapter \
 serve.test_cow_presentation_flow serve.test_cow_namespace_http serve.test_cow_http \
 serve.test_cow_disconnect_durability serve.test_cow_workspace test_store_presentation_catalog \
 test_root_presentation_query test_publication_handoff_ownership test_bounded_tree_retention \
 test_checkpoint_resource_policy test_pool_coordinator test_pool_cold_cache_ownership \
 test_publication_overlap test_pool_root_residency test_pool_close_release \
 test_poisoned_worker_replacement test_root_computation_reuse test_pool_source_trace \
 test_pool_failed_lru test_pool_spill_reload_trace test_pool_disk_lru_integration \
 test_canonical_branch_actions test_branch_checkpoint_eviction test_pool_branch_actions \
 tools.test_cow_cli tools.test_cow_cli_retry tools.test_cow_cli_recomputation_notice
exec bash "$RUNNER" "$SOURCE/experimental/branching_context" "$ROOT/g0-command-v2" "$PYTHON" -B scripts/verify.py --gate G0
