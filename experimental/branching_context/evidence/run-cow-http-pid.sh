#!/usr/bin/env bash
set -u

: "${RUN:?Set RUN to the private run directory}"
: "${COW_RUNTIME_API:=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api}"
: "${COW_PROFILE:=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json}"
: "${COW_PORT:=8099}"
: "${COW_PYTHON:=python3}"

cd "$COW_RUNTIME_API" || exit 90
export PYTHONPATH="/home/dflanag3/issue-2026-20261012/deps:$PWD/src:$PWD"
export NATIVE_EXIT_RECEIPT="$RUN/native-exits.jsonl"
# Bash marks asynchronous jobs SIGINT-ignored. Clear that disposition in the
# short-lived child shell where possible; the Python key-file entrypoint's
# explicit handler reset is authoritative for exact-PID SIGINT cleanup.
bash -c 'trap - INT; exec "$@"' cow-http-child "$COW_PYTHON" \
  "$RUN/cow_http_keyfile.py" "$RUN/api-key" \
  --profile "$COW_PROFILE" --store "$RUN/store" \
  --host 127.0.0.1 --port "$COW_PORT" --text --ram-gib 96 --disk-gib 128 \
  >"$RUN/server.stdout.log" 2>"$RUN/server.stderr.log" &
server_pid=$!
printf '%s\n' "$server_pid" > "$RUN/server.pid"
set +e
wait "$server_pid"
rc=$?
printf '%s\n' "$rc" > "$RUN/server.exit"
exit "$rc"
