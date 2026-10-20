#!/usr/bin/env bash
set -eu
SRC=/home/dflanag3/spatial-artifacts/device-seal-compare-20261015-v1/source
OUT=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/build/candidate-v1-fulltests
CHECKOUT=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/checkout
PY=/home/dflanag3/src/Strata/.venv/bin/python
/usr/bin/g++ -std=c++20 -O3 -DNDEBUG -I"$SRC/include" "$SRC/tests/core/branch_digest_host.cpp" -o "$OUT/branch_digest_host"
CTT_DIGEST_GPU_BINARY="$OUT/branch_digest_gpu" CTT_DIGEST_HOST_BINARY="$OUT/branch_digest_host" "$PY" "$SRC/tests/core/branch_digest_test.py"
CTT_DIGEST_PAGES_BINARY="$OUT/branch_digest_pages_gpu" "$PY" "$CHECKOUT/branch_digest_pages_test.py"
CTT_DIGEST_VIEW_BINARY="$OUT/branch_digest_view_gpu" "$PY" "$SRC/tests/core/branch_digest_view_test.py"
CTT_DIGEST_TIMING_BINARY="$OUT/branch_digest_timing_gpu" "$PY" "$SRC/tests/core/branch_digest_timing_test.py"
"$OUT/branch_prepare_test"
"$OUT/branch_device_seal_test"