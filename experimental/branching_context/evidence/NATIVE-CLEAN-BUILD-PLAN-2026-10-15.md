# Clean current-source native build plan — 2026-10-15

## Pinned inputs and readiness

The selected source is the integrated Context COW commit
`52208d1a896a01585f9f86b354fe987820b4990a` (subject: “docs: advance handoff
past terminal archive and support checks”). This is the build source pin; the
worker branch is not the source pin. As supporting evidence only,
`CMakeLists.txt`, `src/core/`, and `include/strata/core/` are byte-identical
between that commit and worker commit `a66e594a44e490ee59c2fc28d3ac1a66ae3bea70`,
which introduced tested explicit external `--build-dir` wiring. The integrated
tree carries that wiring. Build admission must recheck a clean checkout at the
integrated commit and its builder test/source fingerprints.

Use fresh paths, not yet created:

- Source: `/home/dflanag3/branching-g3/context-cow-clean-52208d1a`
- External CMake build: `/home/dflanag3/spatial-artifacts/cow-build-52208d1a-sm120-01`
- Generated native probes: `/home/dflanag3/spatial-artifacts/cow-probe-52208d1a-sm120-01`

llm-60 was observed with CMake 3.28.3, GCC 13.3.0, CUDA 13.2.86
(`/usr/local/cuda/bin/nvcc`), 32 logical CPUs, and an RTX PRO 6000 Blackwell
at compute capability 12.0. The latest read-only capacity sample showed
31,032,041,472 bytes free on `/` (99% used), so no build is admitted now.
Historical planning sizes are 1.1 GiB for an old full build, 624 MiB for the
pinned llama.cpp source, and 137 MiB for probe binaries. Budget 3–5 GiB and
require at least 10 GiB actually free immediately before any future build.
P4 resource ownership and frozen binaries remain untouched.

## Dependency and configuration

The integrated source pins `https://github.com/ggml-org/llama.cpp.git` at
`3cf03257f219afbe7334045ff7c6a06ac68c627d` in
`third_party/ggml/VERSION.txt`. llm-60 has no suitable clean full llama.cpp
checkout. A future clean CMake configure must fetch this exact commit in the
new external build tree, or use a separately verified clean checkout at that
same commit through `STRATA_GGML_DIR`; never reuse an old `_deps` directory or
the dirty `context-cow-public-20261013` checkout/cache.

For CUDA compute capability 12.0, the intended core settings are Release, C++20, CUDA enabled,
`CMAKE_CUDA_ARCHITECTURES=120`, native experts enabled, and tests enabled.
The configure cache option `STRATA_PREFILL_MMQ` must remain `OFF`: CMake
rejects it when HIP is disabled. This is distinct from the effective CUDA
target compile definitions `STRATA_PREFILL_MMQ=1` and
`STRATA_PREFILL_FUSED=1`, which CMake adds to CUDA `strata_prefill` regardless
of that HIP-only option. `STRATA_PREFILL_FUSED` is not a configure option.
The measured Q4 full and Q8 HTTP profiles both reference the same native build
receipt (`e1b151aa37ddc4a75b9f6e121ccd7d73412e9a545105f1ffbe8cfc6ba053898f`)
and the same branch/fresh probe binaries (`7bb17b47d2dc8b6252530f377891380613fc2c43352c34ea6f0319f386bf4a2b`,
`a60314b8aff6c3954ec55aabce18c58196231b57f7cf99b340b68b747689100c`). The
effective measured setting is `STRATA_MMQ_KQUANTS=OFF`: the historical cache
records OFF and `strata_mmq.dir/flags.make` has no `-DSTRATA_MMQ_KQUANTS=1`.
The branch compile flags do contain `STRATA_PREFILL_MMQ=1` and
`STRATA_PREFILL_FUSED=1`; CMake sets those for CUDA independently of the
optional `STRATA_MMQ_KQUANTS` Q4_K/Q5_K/Q5_1 instances. Preserve this exact
combination for a comparable clean build. The pinned ggml integration disables
GGML CUDA, OpenMP, dynamic backend loading, tests, and examples.

The intended base configure arguments are:

```bash
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_MMQ_KQUANTS=OFF \
  -DSTRATA_PREFILL_MMQ=OFF
```

Do not pass `-DSTRATA_PREFILL_FUSED`; it is not a CMake option. Verify the
resulting cache and per-target `flags.make` after configuration.

## Build closure and fingerprints

First build the CMake targets `strata`, `strata-branch`, and `strata-fresh`
from the clean external build directory. This rebuilds `strata_engine` and all
consumers of the COW class layouts. Then run the integrated
`build_digest_reuse_probe.py` with its explicit `--build-dir`: it compiles
fresh `branch_digest.cu` and `branch_pages.cpp` objects, then the chained
request builder regenerates/compiles `generate.cpp`, `branch_driver.cpp`,
`fresh_prefill.cpp`, `branch_session.cpp`, `conversation_snapshot.cpp`, and
`conversation_state.cpp` before linking `strata-batched-branch` and
`strata-batched-fresh` from that same build's `strata-branch` link recipe.
Do not mix old class-layout objects or archives into these links.

The receipt should bind the integrated commit and clean status; ggml commit;
compiler/tool versions; full CMake cache; `strata-branch` `flags.make` and
`link.txt`; source and generated-source hashes; rebuilt object hashes; linked
archive/response-file hashes; and final executable SHA-256 values. Preserve
stdout, stderr, argv, working directories, and numeric exit codes for every
configure/build/probe command. Existing provenance deliberately reports
reproducibility unproven while external compiler, system-header, dynamic
library, and host-state closure is not pinned.

This plan covers a clean CUDA sm_120 build only. It excludes HIP, other CUDA
architectures, GPU execution, inference, and automatic profile or numerical
qualification. A successful build receipt is not runtime admission.
