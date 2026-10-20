#!/usr/bin/env bash
set -eu
C=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/checkout
SRC=/home/dflanag3/spatial-artifacts/device-seal-compare-20261015-v1/source
OUT=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/build/baseline-v1/compare
NVCC=/usr/local/cuda-13.2/bin/nvcc
mkdir -p -m 700 "$C/tests/core"
test ! -e "$C/src"
ln -s "$SRC/src" "$C/src"
mv "$C/branch_digest_compare_bench.cu" "$C/tests/core/branch_digest_compare_bench.cu"
mkdir -p -m 700 "$OUT"
"$NVCC" -std=c++20 -O3 -DNDEBUG --generate-code="arch=compute_120,code=[compute_120,sm_120]" -I"$SRC/include" -Xptxas=-v -c "$C/tests/core/branch_digest_compare_bench.cu" -o "$OUT/branch_digest_compare_bench.o" 2>"$OUT/ptxas.log"
"$NVCC" --generate-code="arch=compute_120,code=[compute_120,sm_120]" "$OUT/branch_digest_compare_bench.o" -o "$OUT/branch_digest_compare_bench"
sha256sum "$OUT/branch_digest_compare_bench.o" "$OUT/branch_digest_compare_bench" "$OUT/ptxas.log" > "$OUT/build-manifest.sha256"