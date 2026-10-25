# Native dependency patch provenance

`llama-cuda-out-prod-singletons-pr30142.patch` imports the CUDA outer-product
leading-dimension fix from **edenfunf**, [llama.cpp PR #30142](https://github.com/ggml-org/llama.cpp/pull/30142),
commit `fdb4a0553f6ecb00450f04cc8da5b2b38dd9146d`.
The mail patch preserves the original author and commit description. The full
upstream change also adds regression tests to `tests/test-backend-ops.cpp`.
llama.cpp's source and license remain with the dependency.

Local checkpoint: the patched operator passes 103/103 CPU-reference cases on
RTX4090, and an actual A100 MiniCPM rank-one update completes with finite
gradients. Independent reload also passes (1.26e-7 error, unchanged 2e-4
bound); broader acceptance remains pending.
[Local evidence](../../../validation/cuda-rank1-20261025-be/README.md) records
passing checks, the reload-probe error, metadata limits and numerical failures.
Upstream author test results are not PCC test results. No running
study library has been changed. The earlier rank-1 failure remains preserved.

The dependency revision used by the current study is
`972d2313bc0bf0a45f634f77d95c9fb03aeab12c`; applying this patch does not upgrade
that entire dependency to upstream master. Record the exact source, native
library and executable identities when qualifying a patched build.

## Building PCC against the imported fix

Use a new private dependency tree at the recorded native pin, retain the
existing qualified F32/model-support patches, apply this mail patch, and build
its CUDA backend. Do not alter a source tree or library used by active workers.
Then invoke the existing native builder with the additional explicit flag:

```bash
python experimental/native_training/build.py \
  --llama /owned/isolated-patched-llama \
  --output /owned/new-native-build \
  --cuda-out-prod-singletons
```

The flag refuses an unrecognized CUDA source file and records the original
upstream credit, exact patched-source hash, and actual selected CUDA-library
hash in `build.json`. It does not certify that a source file produced a binary
or replace execution tests. Qualify the selected binary with the upstream
CPU/CUDA operator cases, gradients, adapter reload/continuation and actual-model
training/inference before using it for the rank-one study. The ordinary builder
path stays unchanged when the flag is absent.
