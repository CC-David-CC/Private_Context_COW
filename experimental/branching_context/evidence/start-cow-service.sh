#!/usr/bin/env bash
set -eu
RUN=${RUN:-/home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2}
if [[ -e "$RUN/browser" || -L "$RUN/browser" ]]; then
  echo "smoke output directory must not exist before service launch" >&2
  exit 64
fi
GPU_LOCK=/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
if [[ -L "$GPU_LOCK" ]]; then
  echo "GPU0 lock path is a symlink; refusing launch" >&2
  exit 73
fi
exec 9>>"$GPU_LOCK"
if ! flock -n 9; then
  echo "GPU0 lock is held; refusing launch" >&2
  exit 75
fi
if [[ "${1:-}" == "--lock-check-only" && $# -eq 1 ]]; then
  exit 0
fi
if [[ "${1:-}" == "--test-launch-marker" && $# -eq 2 ]]; then
  printf 'service-launch-reached\n' > "$2"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "unexpected arguments" >&2
  exit 64
fi
export RUN
exec bash "$RUN/run-cow-http-pid.sh"
