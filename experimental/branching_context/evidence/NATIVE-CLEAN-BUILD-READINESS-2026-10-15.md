# Native clean-build readiness — 2026-10-15

## Decision

Prefer a fresh build on llm60 after the Q8 campaign’s raw backup and capacity
reclaim. It matches the intended Blackwell runtime host: CUDA 13.2.86, GCC
13.3.0, CMake 3.28.3, RTX PRO 6000 Blackwell (compute capability 12.0). The
read-only check found 15 GiB free on `/` and no build tree in its existing
734 MiB repository checkout. The filesystem was at 100%; this is not a start
signal. Recheck free space and campaign ownership immediately before a build.
No build, download, installation, GPU operation, server launch, or inference
was performed for this review.

llm49 has CUDA 13.3.73, GCC 15.2.0, CMake 4.2.3 and 1.4 TiB free. Its installed
CUDA headers accept GCC 15, but its driver reports CUDA 13.2 and its RTX 4090
is sm89. It is a viable compile host, but a poorer match for the llm60
deployment/runtime closure. Its existing ggml checkout/build artifacts are
from an older source/configuration and must not be reused as a clean build.

## Source and build-path facts

The fleet checkout was clean at `17bee11bd5f09f0358c760b09233f10b456dd4cf`
when inspected. This is later than the earlier readiness ticket’s 4f529753
baseline; use the exact selected clean source commit for any future build and
record it in the receipt. Current `CMakeLists.txt` defaults CUDA architecture
to 120, requires CMake 3.24 and C++20, defines `strata-branch` and
`strata-fresh`, and pins fetched ggml to
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. CMake can take a clean llama.cpp
checkout via `STRATA_GGML_DIR`; otherwise a genuinely fresh build directory
must fetch the pinned revision. Do not point it at an old `_deps` cache.

The current probe builders are layered on a CMake tree. They read
`<source>/build/CMakeFiles/strata-branch.dir/flags.make` and `link.txt`, and
the native request builder passes `<source>/build` as the provenance build
directory. They do not currently accept an external build path. A clean
external build therefore needs either a narrowly reviewed ignored
`<source>/build` symlink to that build directory (the repository ignores
`/build/`) or a follow-up adding explicit `--build-dir` support. Until one is
in place and verified, this is a proposed recipe, not a currently turnkey
command.

The clean build must rebuild the libraries and all changed COW class users,
including `strata`, `strata-branch`, and `strata-fresh`; the probe builder
then regenerates the batched branch/fresh drivers. `strata_engine` owns the
shared COW classes. Do not reuse old objects or archives. A clean build also
needs a clean, pinned ggml source at the CMake revision above. No suitable
llm60 dependency checkout was found; llm49’s detached checkout is exact but
should not be silently copied or treated as the llm60 build input.

## Proposed build outline — not executed

After resource approval, materialize a clean source checkout at the selected
commit and allocate a new external build/output directory. Configure with the
toolchain observed on llm60, then build from scratch:

```bash
SRC=/home/dflanag3/branching-g3/context-cow-clean-<commit>
BUILD=/home/dflanag3/spatial-artifacts/cow-build-<commit>-sm120-01
OUT=/home/dflanag3/spatial-artifacts/cow-native-probe-<commit>-sm120-01
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON
cmake --build "$BUILD" --target strata strata-branch strata-fresh -j"$(nproc)"
```

This omits the external-build wiring described above and dependency fetching
will occur only during an authorized configure. The probe step should follow
only after the build tree, source root, and actual link inputs are verified:

```bash
PYTHONPATH="$SRC/experimental/branching_context/scripts:$SRC/experimental/branching_context/src" \
  python3 "$SRC/experimental/branching_context/scripts/build_digest_reuse_probe.py" \
  "$SRC" "$OUT"
```

The existing profile-preparation script can make a candidate from a verified
base Q8 profile and a successful `BUILT_NOT_EXECUTED` receipt, with capacity
98,304 and chunk 8,192. It does not create the base profile’s text/template
metadata, certify numerical behavior, or admit the candidate for serving.
The current profile helper also labels its source as a staged driver over a
pinned base; that wording must be corrected or explicitly reviewed before a
fresh-source candidate is treated as current-source provenance. Build
provenance and numerical qualification remain separate; a new clean build
does not inherit the historical Q8 result automatically.

## Space and next gate

Observed historical sizes were about 1.1 GiB for an old full CMake build,
624 MiB for its pinned llama.cpp source, and 137 MiB for native probe
binaries. These are planning references, not a measured clean-build forecast.
Budget roughly 3–5 GiB for a fresh checkout/dependency/build/probe artifacts
and require at least 10 GiB free before starting; remeasure rather than rely
on this estimate. llm60’s observed 15 GiB free was below the preferred margin
while the filesystem was full, so wait for the Q8 backup/reclaim and then
recheck. The example uses the host's available CPU threads (`-j"$(nproc)"`),
not a two-core cap. Before starting, confirm available memory can support
that parallel build; if it cannot, reclaim resources or defer, and record any
reviewed parallelism choice instead of silently lowering the job count.

Before build execution, review the selected source commit, exact ggml
checkout/hash, external build location wiring, full CMake configuration,
provenance receipt, and available disk. This note records readiness only; it
does not establish a successful clean build or a newly qualified profile.
