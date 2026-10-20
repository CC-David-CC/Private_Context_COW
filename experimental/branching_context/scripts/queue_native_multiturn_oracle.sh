#!/usr/bin/env bash
set -u
set -o pipefail

runner=/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts/run-command.sh
checkout=/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts
profile=/home/dflanag3/branching-g3/request-stop-20261014/text-next-user-q8-v1/profile.json
fixture=/home/dflanag3/branching-g3/request-stop-20261014/text-next-user-q8-v1/oracle-fixture.json
binary=/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/multiturn-build-v5/strata-batched-fresh
runner_py=/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts/run_native_multiturn_oracle.py
artifact_root=/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014
lock=/home/dflanag3/branching-g3/native-client-speed.gpu0.lock

run_case() {
    local name=$1
    shift
    local log=$artifact_root/${name}.heartbeat.log
    bash "$runner" "$checkout" "$artifact_root/$name" python3 "$runner_py" \
        --profile "$profile" --fixture "$fixture" --binary "$binary" \
        --run-dir "$artifact_root/$name/output" "$@" >"$log" 2>&1 &
    local child=$!
    while kill -0 "$child" 2>/dev/null; do
        printf 'heartbeat=%s case=%s pid=%s\n' "$(date -u +%FT%TZ)" "$name" "$child" | tee -a "$log"
        sleep 30
    done
    wait "$child"
    local result=$?
    printf 'case=%s exit=%s completed=%s\n' "$name" "$result" "$(date -u +%FT%TZ)" | tee -a "$log"
    return "$result"
}

exec 9>"$lock"
flock -x 9
printf 'lock_acquired=%s pid=%s\n' "$(date -u +%FT%TZ)" "$$"
run_case eos-rejection-v2 --rejection || exit $?
run_case full-q8-two-turn-v2 || exit $?
printf 'sequence_complete=%s\n' "$(date -u +%FT%TZ)"
