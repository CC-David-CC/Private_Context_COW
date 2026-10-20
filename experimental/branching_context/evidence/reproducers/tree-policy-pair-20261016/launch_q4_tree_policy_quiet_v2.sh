#!/usr/bin/env bash
set -euo pipefail
PARENT=/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1
CONTROL=/home/dflanag3/spatial-artifacts/q4-tree-policy-control-20261016-v1
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
PY=/home/dflanag3/src/Strata/.venv/bin/python
HARNESS=$PARENT/gpu-harness-v1
OBSERVER=/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v17/source/experimental/branching_context/scripts/observe_cow_cli.py
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f1) == 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 ]]
[[ $(sha256sum "$PARENT/gpu-candidate-v1/profile.json" | cut -d ' ' -f1) == 107cab965436707d13db92271aa629e7829950030dbe60b884308076f8e31253 ]]
[[ $(sha256sum "$CONTROL/profile.json" | cut -d ' ' -f1) == 555ce6df8ee24928fc539961d25bab5f4329f9ab5feae70521450201178dfb68 ]]
[[ $(sha256sum "$HARNESS/run_q4_cli_trees.py" | cut -d ' ' -f1) == 3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2 ]]
[[ $(sha256sum "$HARNESS/verify_q4_cli_tree_frontiers.py" | cut -d ' ' -f1) == a265d9bc8dc069a1a59390b912a3674a3ff95e132b83323564b4f28a1b01b8d0 ]]
[[ $(sha256sum "$OBSERVER" | cut -d ' ' -f1) == 20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28 ]]
[[ $(cat "$PARENT/pair-eviction-command-v1/command.exit") == 0 && $(cat "$PARENT/pair-eviction-command-v1/wrapper.exit") == 0 ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ ! -e $PARENT/gpu-quiet-v3 && ! -e $CONTROL/campaign-quiet-v2 ]]
exec 9>/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
flock -n 9
"$PY" -B - <<'PY'
import json,os
from pathlib import Path
p=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
cleanup=json.loads((p/'pair-child-eviction-v1.json').read_bytes())
assert cleanup['status']=='PASS' and cleanup['canonical_bytes_unchanged'] and cleanup['original_prefix_bytes_unchanged']
available=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize
need=cleanup['selected_allocated_bytes']+512*1024**2+1024**3
print(json.dumps(dict(available_bytes=available,estimated_required_bytes=need,estimate_scope='Measured prior child copies for both arms plus512MiB metadata and1GiB headroom')),flush=True)
assert available>=need
PY
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
cat /proc/loadavg
date --iso-8601=seconds
run_arm() {
 local root=$1 runtime=$2 profile=$3 fixtures=$4 store=$5 run=$6 command=$7 audit=$8 audit_command=$9 refs=${10}
 bash "$RUNNER" "$HARNESS" "$root/$command" env PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -B "$HARNESS/run_q4_cli_trees.py" --runtime-root "$runtime" --profile "$profile" --fixtures "$fixtures" \
  --store "$store" --run-dir "$root/$run" --phase campaign --outputs 128 --repetitions 3 \
  --observer "$OBSERVER" --observer-sha256 20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28
 bash "$RUNNER" "$HARNESS" "$root/$audit_command" env PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -B "$HARNESS/verify_q4_cli_tree_frontiers.py" --runtime-root "$runtime" --profile "$profile" --fixtures "$fixtures" \
  --campaign "$root/$run/result.json" --run-dir "$root/$audit" --existing-references "$refs"
}
# Reverse the preceding pair's arm order. CPU fault workers are explicitly held.
run_arm "$PARENT" "$PARENT/gpu-candidate-v1/runtime" "$PARENT/gpu-candidate-v1/profile.json" "$PARENT/gpu-candidate-v1/fixture-set.json" \
 "$PARENT/workspace-store-v1" gpu-quiet-v3 gpu-quiet-command-v3 gpu-quiet-reference-v3 gpu-quiet-reference-command-v3 "$PARENT/gpu-campaign-reference-v1"
run_arm "$CONTROL" "$CONTROL/runtime" "$CONTROL/profile.json" "$CONTROL/fixture-set.json" \
 "$CONTROL/store-v1" campaign-quiet-v2 campaign-quiet-command-v2 reference-quiet-v2 reference-quiet-command-v2 "$CONTROL/reference-v1"
exec bash "$RUNNER" "$HARNESS" "$PARENT/pair-export-command-v2" "$PY" -B "$PARENT/export_q4_tree_policy_pair_v1.py" \
 --control-run campaign-quiet-v2 --control-audit reference-quiet-v2 --control-command campaign-quiet-command-v2 --control-audit-command reference-quiet-command-v2 \
 --candidate-run gpu-quiet-v3 --candidate-audit gpu-quiet-reference-v3 --candidate-command gpu-quiet-command-v3 --candidate-audit-command gpu-quiet-reference-command-v3 \
 --output "$PARENT/policy-pair-v2.json" --timing-scope candidate-then-control-fault-test-workers-held-three-repetitions-per-arm
