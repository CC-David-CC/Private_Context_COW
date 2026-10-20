#!/usr/bin/env bash
set -eu
C=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/checkout
BASE=/home/dflanag3/spatial-artifacts/device-seal-compare-20261015-v1/source
BUILD=/home/dflanag3/spatial-artifacts/device-seal-compare-20261015-v1/build
OUT=/home/dflanag3/spatial-artifacts/branch-digest-cuda-20261015-v1/build/candidate-v1-fulltests
CUDA=/usr/local/cuda-13.2/targets/x86_64-linux
NVCC=/usr/local/cuda-13.2/bin/nvcc
mkdir -p -m 700 "$C/source-overlay-v1/src/core" "$OUT"
cp "$C/branch_digest.candidate-v1.cu" "$C/source-overlay-v1/src/core/branch_digest.cu"
test -L "$C/src"
rm "$C/src"
ln -s "$C/source-overlay-v1/src" "$C/src"
"$NVCC" -std=c++20 -O3 -DNDEBUG --generate-code="arch=compute_120,code=[compute_120,sm_120]" -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_VERSION='"0.1.43"' -I"$BASE/include" -Xptxas=-v -c "$C/source-overlay-v1/src/core/branch_digest.cu" -o "$OUT/branch_digest.o" 2>"$OUT/ptxas.log"
for pair in 'branch_digest_gpu.cpp branch_digest_gpu' 'branch_digest_pages_gpu.cpp branch_digest_pages_gpu' 'branch_digest_view_gpu.cpp branch_digest_view_gpu' 'branch_digest_timing_gpu.cpp branch_digest_timing_gpu'; do
  set -- $pair
  "$NVCC" -std=c++20 -O3 -DNDEBUG --generate-code="arch=compute_120,code=[compute_120,sm_120]" -I"$BASE/include" -c "$BASE/tests/core/$1" -o "$OUT/$2.driver.o"
  "$NVCC" --generate-code="arch=compute_120,code=[compute_120,sm_120]" "$OUT/branch_digest.o" "$OUT/$2.driver.o" -o "$OUT/$2"
done
"$NVCC" -std=c++20 -O3 -DNDEBUG --generate-code="arch=compute_120,code=[compute_120,sm_120]" -I"$BASE/include" -Xptxas=-v -c "$C/tests/core/branch_digest_compare_bench.cu" -o "$OUT/branch_digest_compare_bench.o" 2>"$OUT/compare-ptxas.log"
"$NVCC" --generate-code="arch=compute_120,code=[compute_120,sm_120]" "$OUT/branch_digest_compare_bench.o" -o "$OUT/branch_digest_compare_bench"
/usr/bin/g++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_VERSION='"0.1.43"' -I"$BASE/include" -I/home/dflanag3/branching-g3/llama-pinned/ggml/include -isystem "$CUDA/include" -c "$C/branch_device_seal_test.cpp" -o "$OUT/branch_device_seal_test.o"
/usr/bin/g++ -O3 -DNDEBUG "$OUT/branch_device_seal_test.o" "$OUT/branch_digest.o" -o "$OUT/branch_device_seal_test" -L"$CUDA/lib/stubs" -L"$CUDA/lib" -Wl,-rpath,"$CUDA/lib" "$BUILD/libstrata_engine.a" "$BUILD/libstrata_kernels.a" "$BUILD/libstrata_core.a" "$BUILD/libstrata_kernels_cpu.a" "$BUILD/ggml/src/libggml-cpu.a" "$BUILD/ggml/src/libggml-base.a" -lm "$CUDA/lib/libcudart.so" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -Wl,-Map,"$OUT/branch_device_seal_test.map"
/usr/bin/g++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -I"$BASE/include" -I/home/dflanag3/branching-g3/llama-pinned/ggml/include -isystem "$CUDA/include" -c "$BASE/tests/core/branch_prepare_test.cpp" -o "$OUT/branch_prepare_test.o"
/usr/bin/g++ -O3 -DNDEBUG "$OUT/branch_prepare_test.o" "$OUT/branch_digest.o" -o "$OUT/branch_prepare_test" -L"$CUDA/lib/stubs" -L"$CUDA/lib" -Wl,-rpath,"$CUDA/lib" "$BUILD/libstrata_engine.a" "$BUILD/libstrata_kernels.a" "$BUILD/libstrata_core.a" "$BUILD/libstrata_kernels_cpu.a" "$BUILD/ggml/src/libggml-cpu.a" "$BUILD/ggml/src/libggml-base.a" -lm "$CUDA/lib/libcudart.so" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -Wl,-Map,"$OUT/branch_prepare_test.map"
sha256sum "$C/source-overlay-v1/src/core/branch_digest.cu" "$OUT/branch_digest.o" "$OUT/branch_digest_gpu" "$OUT/branch_digest_pages_gpu" "$OUT/branch_digest_view_gpu" "$OUT/branch_digest_timing_gpu" "$OUT/branch_digest_compare_bench" "$OUT/branch_device_seal_test" "$OUT/branch_prepare_test" "$OUT/ptxas.log" "$OUT/compare-ptxas.log" "$OUT/branch_device_seal_test.map" "$OUT/branch_prepare_test.map" > "$OUT/build-manifest.sha256"