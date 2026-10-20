#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: $0 SOURCE_ROOT /boot/Strata-data/context-cow-model-free-luna-hip-RUN_ID" >&2
  exit 2
fi
source_root=$(realpath "$1")
run_root=$2
case "$run_root" in
  /boot/Strata-data/context-cow-model-free-luna-hip-*) ;;
  *) echo "run root must be a unique directory below /boot/Strata-data/context-cow-model-free-luna-hip-*" >&2; exit 2 ;;
esac
if [[ -e "$run_root" ]]; then
  echo "run root already exists: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root/build"
source_hashes="$run_root/source.sha256"
{
  sha256sum \
    "$source_root/include/strata/core/vmm.hpp" \
    "$source_root/include/strata/core/branch_pages.hpp" \
    "$source_root/include/strata/core/branch_digest.hpp" \
    "$source_root/include/strata/core/branch_sha256.hpp" \
    "$source_root/include/strata/core/branch_gpu_compat.hpp" \
    "$source_root/src/core/vmm_hip.cpp" \
    "$source_root/src/core/branch_pages.cpp" \
    "$source_root/src/core/branch_digest.cu" \
    "$source_root/tests/model_free/hip_vmm_test.cpp" \
    "$source_root/tests/model_free/run_hip_vmm_test.sh"
} > "$source_hashes"
hipcc --version > "$run_root/hipcc-version.txt" 2>&1
hipcc --offload-arch=gfx1100 -x hip -std=c++17 -DSTRATA_USE_HIP \
  -I"$source_root/include" \
  "$source_root/src/core/vmm_hip.cpp" \
  "$source_root/src/core/branch_pages.cpp" \
  "$source_root/src/core/branch_digest.cu" \
  "$source_root/tests/model_free/hip_vmm_test.cpp" \
  -o "$run_root/build/hip_vmm_test" \
  > "$run_root/build/build.log" 2>&1
sha256sum "$run_root/build/hip_vmm_test" > "$run_root/build/binary.sha256"
"$run_root/build/hip_vmm_test" > "$run_root/test.log" 2>&1
printf 'HIP VMM and native BranchPages COW probe PASS\nsource hashes: %s\nbuild log: %s\ntest log: %s\nbinary: %s\n' \
  "$source_hashes" "$run_root/build/build.log" "$run_root/test.log" "$run_root/build/hip_vmm_test"
