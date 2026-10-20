# Run the bounded Context COW experiment

This page covers the comparison probes and their experimental profile. To
launch the existing qualified Q8 browser service on llm60, see the separate
[native Context COW service guide](NATIVE_SERVICE.md). The service guide uses a
frozen executed profile; it does not replace this page's comparison recipe or
claim that a fresh native build/profile is reproducible yet.

The demonstrated path is Linux + CUDA, one active request, Qwen3.8-Flash-Next
Q8_0 and FP16 KV. It uses real batched prefill and ordinary one-token decode.
The tested machine had a 96 GB RTX PRO 6000 Blackwell and 124 GiB system RAM.
The scripts use its full-resource policy (40 GiB CPU expert budget and automatic
GPU expert cache). Other memory budgets/models require a separately measured
profile; no universal hardware or performance claim is made.

No Docker, database, GUI or web service is needed. The native `--serve` control
used by the comparison runner communicates through stdin/stdout.

## Build

Start with an existing Q8_0 model and the matching Strata native pack, including
its tokenizer. Model acquisition/packing follows the retained
[Strata instructions](../../STRATA.md) and [engine details](../DETAILS.md).
The cache files and weights are not part of this Git repository.

```bash
git clone https://github.com/CC-David-CC/Public_Context_COW.git
cd Public_Context_COW
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.5.3 regex==2026.9.29

cmake -S . -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j16 --target strata strata-branch strata-fresh \
  branch_prepare_test branch_page_batch_test branch_pages_test \
  branch_session_test branch_integrity_test branch_lifecycle_test \
  branch_digest_host branch_digest_gpu branch_digest_pages_gpu

python experimental/branching_context/scripts/build_fast_prefill_probe.py \
  "$PWD" "$PWD/build-batched" --long-demo
```

Architecture 120 is the measured Blackwell target. Select the architecture of
the actual GPU when building elsewhere. The isolated probe builder currently
reads CMake Makefile build receipts, so use `Unix Makefiles`, not Ninja.
CMake normally fetches the pinned ggml source; the measurement used the existing
local `STRATA_GGML_DIR` snapshot and records its library/build hashes. CUDA 13.2,
GCC 13.3 and Python 3.12.3 were tested. Native Windows execution of the history
store is currently blocked on its POSIX `fcntl` locking implementation. Use
`python`, never the Microsoft Store `python3` alias, for Windows-side scripts.

## Prepare a truthful local profile

Set the two paths to existing local files/directories. `MODEL` is the first
GGUF shard, not an alternative smaller quantization of the measured model.

```bash
MODEL=/absolute/path/to/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
PACK=/absolute/path/to/q8-pack
python experimental/branching_context/scripts/prepare_speed_profile.py \
  --source "$PWD" --binaries "$PWD/build-batched" \
  --model "$MODEL" --pack "$PACK" \
  --expert-profile "$PWD/data/expert-profile.bin" \
  --output "$PWD/demo-profile"
```

This checks model/tokenizer geometry, hashes the real model/pack inputs, records
the binaries, build receipts and GPU, and creates the exact 8K A/B fixture.
The first full inventory can take time. On a later run, `--model-receipt`
may point to a **trusted local** previous receipt; only unchanged file
size/mtime/ctime entries are reused. The profile is explicitly
`EXPERIMENTAL_UNQUALIFIED`; this command does not calibrate numerical tolerances.
Never use a path-redacted report from `docs/context-cow/results` as a local
execution profile. Use a new output directory for each run.

## Run COW, RAM, and the serving control

```bash
python experimental/branching_context/scripts/cow_speed_8k.py \
  --api "$PWD/experimental/branching_context" \
  --profile "$PWD/demo-profile/profile.json" --binaries "$PWD/build-batched" \
  --fixture "$PWD/demo-profile/fixture.json" --output "$PWD/demo-cow"

python experimental/branching_context/scripts/ram_speed_8k.py \
  --api "$PWD/experimental/branching_context" \
  --profile "$PWD/demo-profile/profile.json" --binaries "$PWD/build-batched" \
  --fixture "$PWD/demo-profile/fixture.json" --output "$PWD/demo-ram" \
  --oracle-evidence "$PWD/demo-cow"

python experimental/branching_context/scripts/speed_8k.py \
  --source "$PWD" --profile "$PWD/demo-profile/profile.json" \
  --engine "$PWD/build/strata" --resources normal \
  --fixture "$PWD/demo-profile/fixture.json" --output "$PWD/demo-serving"
```

COW executes resident A/B, RAM A and disk B. RAM executes A/B/A with private
GPU restoration; it retains the parent GPU image only for the same byte-audit
check. The first COW invocation starts independent fresh-oracle processes.
`--oracle-evidence` reuses only already executed matching traces; omit it to
rerun fresh inference. `result.json` is updated during execution, including
failures. Read the output count and status rather than inferring success from
a progress message. The final model run took 164 seconds when reusing matching
oracle traces; the clean first run with fresh processes took 329 seconds on
the measured host, excluding build and profile preparation.

Serving also offers `--fixed-dispatch` and `--threaded-launch` comparison
options. Its token-parity failures are documented, so it is a performance
control rather than the independent numerical oracle. To compare with a
different Strata revision, build that revision in a separate checkout and
pass its actual binary path; the runner records the binary hash and tokens.

## Bounded checks

```bash
ctest --test-dir build \
  -R '^(branch_prepare_test|branch_page_batch_test|branch_pages_test|branch_session_test|branch_integrity_test|branch_lifecycle_test)$' \
  --output-on-failure
CTT_DIGEST_HOST_BINARY="$PWD/build/branch_digest_host" \
CTT_DIGEST_GPU_BINARY="$PWD/build/branch_digest_gpu" \
  python tests/core/branch_digest_test.py
CTT_DIGEST_PAGES_BINARY="$PWD/build/branch_digest_pages_gpu" \
  python tests/core/branch_digest_pages_test.py
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/branch_prepare_test

cd experimental/branching_context
python scripts/verify.py --gate G0
python examples/branching_demo.py
python scripts/mutation_smoke.py
```

These commands reproduce the bounded checks. They do not certify G1–G5. The
registered suites and their frozen assertions remain in the repository.
