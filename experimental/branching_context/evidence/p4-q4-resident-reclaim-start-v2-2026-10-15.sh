#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01
REVIEW=$BASE/resident-reclaim-readonly-20261015-01
CHECKOUT=/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context
RUNNER=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
RUN_DIR=$BASE/command-receipts/resident-reclaim-20261015-01
EXEC_DIR=$BASE/resident-reclaim-execution-20261015-01
LOCK=$EXEC_DIR/resident-output.lock
RECEIPT=$EXEC_DIR/reclaim-receipt.json
HELPER=$REVIEW/reclaim-resident-v1.py
MAP=$REVIEW/resident-reclaim-map.json
PROOF=$REVIEW/archive-match-proof-v2.json
OUT=$BASE/resident-parent/output

test -f "$RUNNER"
test -f "$HELPER"
test -f "$MAP"
test -f "$PROOF"
test ! -e "$RUN_DIR"
test ! -e "$RECEIPT"
test "$(sha256sum "$HELPER" | cut -d' ' -f1)" = 9f65322f27d006ab5ab157da4070dc2906c63cff6d7fa56e21d845e33c6c8d58
test "$(sha256sum "$MAP" | cut -d' ' -f1)" = 1d80892dc4deee21897b22dd37ddbfd14843e8af3fcbe0fd92dd241f80e001ea
test "$(sha256sum "$PROOF" | cut -d' ' -f1)" = 3ab78750a2b599ad66a58a4d163f0a008af1733eb770798208b62b59f6e5ad3c

exec 9>"$LOCK"
flock -n 9 || { echo "LOCK_BUSY"; exit 75; }
echo "STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FREE_BEFORE_BYTES=$(df -PB1 --output=avail "$OUT" | tail -n1 | tr -d ' ')"

set +e
bash "$RUNNER" "$CHECKOUT" "$RUN_DIR" python3 "$HELPER" --unlink --receipt "$RECEIPT"
runner_exit=$?
set -e

echo "RUN_COMMAND_EXIT=$runner_exit"
echo "FREE_AFTER_BYTES=$(df -PB1 --output=avail "$OUT" | tail -n1 | tr -d ' ')"
echo "FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$runner_exit"
