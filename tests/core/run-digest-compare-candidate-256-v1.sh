#!/usr/bin/env bash
set -eu
LOCK=/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
RUNNER=/home/dflanag3/spatial-artifacts/device-seal-compare-20261015-v1/run-command.sh
CHECKOUT=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/checkout
RUN=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/runs/compare-candidate-256m-v1
BIN=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/build/candidate-v1-fulltests/branch_digest_compare_bench
PY=/home/dflanag3/src/Strata/.venv/bin/python
HARNESS=$CHECKOUT/benchmark_branch_digest_compare_scale.py
flock -n "$LOCK" bash "$RUNNER" "$CHECKOUT" "$RUN" "$PY" "$HARNESS" --binary "$BIN" --mib 256 --output "$RUN/measurement.json"