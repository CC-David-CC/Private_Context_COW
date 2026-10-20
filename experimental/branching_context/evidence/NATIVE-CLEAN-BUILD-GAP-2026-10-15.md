# Native clean-build provenance review — 2026-10-15

## Scope and outcome

This was a read-only review of the native request-probe build and profile path at
source revision `e3bc0dda4e4c8a5ad370e395e90bfa151aeb8c2d`. No native build,
server launch, model access, inference, or profile/binary rewrite was performed.
The existing llm60 Q8 serving qualification remains evidence for those exact
historical binaries and profile; it does not demonstrate a clean build from this
source revision.

The current build helpers rebuild the required COW class users, but they are
probe builders layered on an existing CMake build tree. They are not a
reproducible source checkout-to-qualified-profile recipe. A new provenance
receipt must state that limitation unless it can bind the complete source,
generator, toolchain, and link-input closure.

## Existing qualified profile and artifacts

The reviewed Q8 profile is
`/home/dflanag3/spatial-artifacts/namespace-q8-http-luna-20261014/gpu-8k-512-128-v1/result/profile.json`,
SHA-256 `14b3d16b9ba65e84e42c16352ffcd88304e4ff3ea2302a235d75ef1ecb164e59`,
execution ID `b62b4e1842ae4c9b5a0b9c6ea372cb9dcaa2333206adf4b2dd9b7a747f7323d1`.
It selects physical GPU `[0]`, context capacity 98,304, chunk size 8,192,
FP16 KV, and `enable_thinking: false`.

The profile references the existing binaries:

- `/home/dflanag3/branching-g3/native-request-probe-20261014/binaries-digest-v1/strata-batched-branch`
  SHA-256 `7bb17b47d2dc8b6252530f377891380613fc2c43352c34ea6f0319f386bf4a2b`
- `/home/dflanag3/branching-g3/native-request-probe-20261014/binaries-digest-v1/strata-batched-fresh`
  SHA-256 `a60314b8aff6c3954ec55aabce18c58196231b57f7cf99b340b68b747689100c`
- Their `build.json` is 16,454 bytes, SHA-256
  `e1b151aa37ddc4a75b9f6e121ccd7d73412e9a545105f1ffbe8cfc6ba053898f`;
  it records `BUILT_NOT_EXECUTED` and scope “private resident COW request
  probe; not ordinary --serve integration”.

These are recorded as a historical qualification, not regenerated or modified
by this review.

## Build path observed in source

`scripts/build_native_request_probe.py` delegates to
`build_fast_prefill_probe.build()` and adds request protocol metadata. The
builder generates and compiles its generated `generate.cpp`, `branch_driver.cpp`,
and `fresh_prefill.cpp`, plus support sources, then uses the existing CMake
`strata-branch` `link.txt` after substituting the new objects. It reads
`build/CMakeFiles/strata-branch.dir/flags.make` and `link.txt`; it does not
configure a fresh CMake tree or build the full product from a clean checkout.

`scripts/build_digest_reuse_probe.py` first compiles `branch_digest.cu` with
`nvcc -std=c++17 -arch=sm_120` and `branch_pages.cpp` with the CMake C++ flags,
then invokes the native request builder with those added objects. This rebuild
is needed to avoid linking changed COW class layouts against stale object
users. The receipt captures those commands and selected source hashes, but
does not by itself bind all imported generator helpers or prove a clean source
revision.

The actual CMake cache observed for this historical build is under
`/home/dflanag3/branching-g3/native-request-probe-20261014/source-digest-v1/build`;
it resolves to the earlier `durable-resume-20261013/source/build` tree. Its
`CMAKE_HOME_DIRECTORY` is the older
`/home/dflanag3/branching-g3/context-cow-public-20261013` checkout at commit
`eeb12f3ca2c488a3b6526d1c3e711af8a2527227`, which was dirty and had untracked
files. The staging source tree itself is a copied, non-Git tree. Thus neither
the staged code nor the CMake source tree is a clean checkout of `e3bc0dda`.

The observed configuration was CMake 3.28, Unix Makefiles, Release, `/usr/bin/c++`,
`/usr/local/cuda/bin/nvcc`, CUDA architecture 120, CUDA root
`/usr/local/cuda`, `STRATA_ENABLE_CUDA=ON`, `STRATA_NATIVE_EXPERTS=ON`, and
`STRATA_BUILD_TESTS=ON`. C++ flags were `-O3 -DNDEBUG -std=c++20 -Wall -Wextra`.
The command definitions included `STRATA_BRANCH_DRIVER=1`,
`STRATA_NATIVE_EXPERTS=1`, `STRATA_PREFILL_FUSED=1`,
`STRATA_PREFILL_MMQ=1`, and version 0.1.43. The captured link closure included
`libstrata_engine.a`, `libstrata_prefill.a`, `libstrata_spec.a`, CUDA runtime,
engine repeat, kernels/core/kernels_cpu archives, ggml CPU/base archives,
cublas/cublasLt, culibos, MMQ, CUDA stub/device runtime, `m`, `rt`, `pthread`,
and `dl`.

## Profile preparation boundary

`scripts/prepare_native_request_profile.py` transforms an existing base profile
using a completed build receipt. It accepts explicit capacity and optional
chunk size (default 8,192), validates the existing base profile and binary
receipt, replaces executable paths, and hashes linked archives into the output
profile. It neither creates a base profile nor performs a build. It also does
not establish the text/template metadata needed by the qualified HTTP path.
`gpu_http_probe.py` supplies text metadata while executing a real request, so
it is not a non-inference profile-creation tool.

Consequently the present materials document an existing-host qualification,
not a reproducible fresh source build and profile workflow. A fresh source
build recipe and a separate non-inference text-profile creator remain distinct
follow-up tasks.

## Narrow provenance enhancement approved for implementation

The next change is limited to additive provenance fields in newly generated
build receipts. It will capture source revision and dirty state, CMake cache and
toolchain inputs, imported generator source hashes, and the actual final link
inputs (including response files where supported). A receipt will not claim
reproducibility if any required input is missing or the source is dirty or
non-Git. It will fail closed on malformed or unresolved declared build inputs.
The change will not hash model data, rewrite qualified receipts or binaries,
create profiles, run builds, or perform inference. Numerical qualification
remains separate from build provenance.

## Provenance enhancement check

The additive receipt implementation and its tests are recorded in
`NATIVE-BUILD-PROVENANCE-2026-10-15.json`. The focused llm60 CPU suite passed
seven tests covering the actual `{argv, exit_code}` receipt shape, terminal
failure status on provenance errors, clean/dirty/non-Git source state, a
transitive local Python import, linked archive mutation, relative link paths,
response-file resolution, and failure on malformed or missing inputs. Python
syntax compilation also passed. A read-only pass over the historical 10-command
receipt found the expected structured command rows and hashed 14 linked inputs;
the copied source root was correctly labeled `non_git` and not reproducible.
The implementation intentionally sets `provenance_complete: false` and
`reproducible: false` while external compiler, system-header, dynamic-library,
and host-state closure is unpinned. No native build, profile generation, GPU
execution, or inference was run.
