#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "usage: $0 SOURCE_ROOT BASELINE_INCLUDE_ROOT /boot/Strata-data/context-cow-model-free-luna-hip-RED_ID" >&2
  exit 2
fi
source_root=$(realpath "$1")
baseline_include=$(realpath "$2")
run_root=$3
case "$run_root" in
  /boot/Strata-data/context-cow-model-free-luna-hip-*) ;;
  *) echo "run root must be a unique directory below /boot/Strata-data/context-cow-model-free-luna-hip-*" >&2; exit 2 ;;
esac
if [[ -e "$run_root" ]]; then echo "run root already exists: $run_root" >&2; exit 2; fi
mkdir -p "$run_root/build"
{
  sha256sum \
    "$baseline_include/strata/core/vmm.hpp" \
    "$source_root/src/core/vmm.cpp" \
    "$source_root/tests/model_free/hip_vmm_test.cpp" \
    "$source_root/tests/model_free/run_hip_vmm_baseline.sh"
} > "$run_root/source.sha256"
hipcc --version > "$run_root/hipcc-version.txt" 2>&1
hipcc --offload-arch=gfx1100 -std=c++17 -DSTRATA_USE_HIP -DSTRATA_VMM_BASELINE \
  -I"$baseline_include" -I"$source_root/include" \
  "$source_root/src/core/vmm.cpp" \
  "$source_root/tests/model_free/hip_vmm_test.cpp" \
  -o "$run_root/build/hip_vmm_baseline_test" \
  > "$run_root/build/build.log" 2>&1
sha256sum "$run_root/build/hip_vmm_baseline_test" > "$run_root/build/binary.sha256"
set +e
"$run_root/build/hip_vmm_baseline_test" > "$run_root/test.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$run_root/test.exit"
if [[ $status -eq 0 ]] || ! grep -q 'vmm_available' "$run_root/test.log" || ! grep -q 'failed' "$run_root/test.log"; then
  echo "baseline did not fail at the expected unsupported VMM assertion (exit=$status)" >&2
  exit 1
fi
printf 'Expected red: existing HIP stub reports VMM unavailable.\nsource hashes: %s\nbuild log: %s\ntest log: %s\n' \
  "$run_root/source.sha256" "$run_root/build/build.log" "$run_root/test.log"
