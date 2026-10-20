#!/usr/bin/env bash
set -euo pipefail
umask 077

STAGE=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-map-audit-20261015-01
RUN=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-20261015-v1
RUN_COMMAND=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
RUN_ROOT=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-reclaim-command-20261015-01
LOCK=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-map-audit-20261015-01/reclaim-unlink.lock
RECEIPT=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-map-audit-20261015-01/reclaim-receipt-01.json

printf '%s  %s\n' \
  91ec8f54db1fc53fb1983dfc6722c86a898edbde79483f992fe86ff5ee462291 "$STAGE/map.json" \
  6b0e5f59311cbaa3d896706363601c264f0e9eea11993f664d52cdfff2c7c35e "$STAGE/verified-final.json" \
  af855ebc37962a0177671b4f19d0e898e5637f7cd6bc104815be36cc3b17fc5a "$STAGE/reclaim_states.py" \
  7b493b88e0ac9931ab9ccbc4300d41c5dea9a9adf1bc271aa8ebffe611fde273 "$STAGE/precheck_receipt-02.json" \
  7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 "$RUN_COMMAND" \
  | sha256sum -c -

test ! -e "$RUN_ROOT"
test ! -e "$RECEIPT"
printf 'FREE_BEFORE_BYTES='
df -B1 --output=avail "$RUN" | tail -n 1
exec bash "$RUN_COMMAND" "$STAGE" "$RUN_ROOT" \
  flock -n "$LOCK" python3 "$STAGE/reclaim_states.py" \
  --unlink --receipt "$RECEIPT"
