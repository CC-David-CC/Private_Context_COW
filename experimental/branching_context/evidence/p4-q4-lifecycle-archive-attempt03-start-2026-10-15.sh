#!/usr/bin/env bash
set -u

STAGE=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-20261015-03
RUN_DIR=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-command-20261015-03
RUN_COMMAND=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
SENDER="$STAGE/archive_p4_q4_lifecycle_attribution.py"
LOCK="$STAGE/archive-stream.lock"

test -s "$SENDER" || exit 2
test -f "$RUN_COMMAND" || exit 2
test ! -e "$RUN_DIR" || exit 2
exec 9>"$LOCK"
flock -n 9 || exit 3
exec bash "$RUN_COMMAND" "$STAGE" "$RUN_DIR" python3 "$SENDER" --stream-reviewed
