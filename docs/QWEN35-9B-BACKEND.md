# Qwen3.5-9B through Context COW

This is the dense-model preview using a pinned llama.cpp adapter. Flash keeps its existing adapter and expert-pack setup. Both use the existing COW immutable history, branch references, controller records, checkpoint store and RAM/disk pool implementation. A checkpoint from one model/profile is not interchangeable with another.

## Current scope

The tested Qwen file is Qwen3.5-9B-Q8_0.gguf, 9,527,502,048 bytes, SHA-256 `809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4`. NVIDIA qualification uses an RTX 4090 with FP16 KV. Complete native state copying is the initial implementation; this is not a claim of physical GPU page sharing. Some model tensors remain CPU-mapped, so this is not an all-tensors-on-GPU claim.

Exact continuation is bound to the model, execution profile and ordered native batch boundaries. Cross-arrangement equivalence remains outside this accepted prototype contract. Cumulative G1-G5 certification is not established.

Text fixtures at 128/4/8, 1,024/64/16 and 8,192/512/16 passed 178 full-logit comparisons and nine process-separated strict restores, with zero saved-prefix forward tokens. Mixed-input fixtures passed the corresponding bounded comparison campaign. The independent oracle uses fresh native contexts and the matching operator schedule, without loading checkpoints. These are bounded tests, not arbitrary-context certification.

BranchWrite on a separate VM created a website and completed a follow-up edit. PDF trials exercised actual ReportLab builds, page rendering and image ingestion, but failed document-quality/syntax checks. A CAD environment reference passed execution, reopened geometry and rendering; the model-driven baseline repeatedly guessed nonexistent export APIs and produced no accepted STEP. Toolchain functionality does not establish model competence in these domains. Preserve these failures for training and evaluate held-out variants separately.

## Build the pinned native adapter

Use Linux and an installed NVIDIA CUDA toolchain. Supply actual model files separately; the existing one-click Flash model downloader is not a dense-model download command. The following variables are example installation locations chosen by the operator:

```sh
COW_ROOT=/path/to/ContextCOW
LLAMA_ROOT=/path/to/llama.cpp
BUILD_ROOT=/path/to/qwen35-mixed-build
MODEL=/path/to/Qwen3.5-9B-Q8_0.gguf
COW_PYTHON=/path/to/cow-venv/bin/python
```

Use a clean llama.cpp checkout at `972d2313bc0bf0a45f634f77d95c9fb03aeab12c`. The adapter builder rejects another revision or a dirty checkout. The executed RTX 4090 build used Release, shared libraries, GGML_CUDA=ON, GGML_NATIVE=ON, tools/examples enabled and CUDA architecture 89. Other hardware needs its own build/profile and tests.

```sh
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$LLAMA_ROOT/build" --parallel
"$COW_PYTHON" "$COW_ROOT/experimental/branching_context/scripts/build_qwen35_mixed.py" --llama "$LLAMA_ROOT" --output "$BUILD_ROOT"
```

The current server Python environment uses the repository requirements plus llguidance 1.9.1. The native build produces the adapter, separate mixed oracle and build.json with executable/library/source hashes. Keep these with the profile. Do not silently substitute binaries in an existing profile.

## Configure and launch through the normal COW tools

Choose a new, absent output directory for initial setup. Do not supply a Flash expert pack for this model.

```sh
SERVER_DATA=/path/to/new-qwen35-server
"$COW_PYTHON" "$COW_ROOT/tools/cow_server_setup.py" --model "$MODEL" --native "$BUILD_ROOT/libqwen35ctt.so" --output "$SERVER_DATA" --context 65536 --prefill 512 --ram-gib 8 --disk-gib 32 --min-available-gib 4 --host 0.0.0.0 --port 18109
"$COW_PYTHON" "$COW_ROOT/tools/cow_server.py" "$SERVER_DATA"
```

These are the tested pool/context/batch settings, not a universal hardware recommendation. Use authentication on the trusted LAN. Setup creates a private api-key.txt. Give its value privately to the client operator; never put it in model context, source control or command-line arguments.

### Selecting physical GPUs

Rebuild the native adapter with this source before opting into explicit device
selection. Add `--gpu 4` to `cow_server_setup.py` to select physical GPU 4, or
`--gpu 4,6` for two independent dense workers using the existing shared RAM/disk
pools. IDs follow PCI order. The selected topology and CUDA visibility are pinned
in the new execution profile. Each model is loaded onto its own selected device;
workers do not change the process environment while running.

Omitting `--gpu` retains the legacy GPU-0 setup and accepts its existing native
library. An existing profile cannot be silently rebound: create a separate server
directory for a different selection. Startup locks every selected physical GPU
against other configured COW launchers under the same account. These locks do not
reserve GPUs against unrelated users or programs; inspect actual use before launch.

The coordinator's nested topology, such as `GPU=[[0,1],2]`, still describes one
grouped worker plus one independent worker. Native grouped checkpoint capture
remains unsupported and fails before dense model loading. Independent multi-GPU
configuration support does not establish numerical equivalence across different
GPUs; qualify the actual hardware/profile before relying on cross-worker restores.

### Temporary server sessions

Use `cow_server_setup.py --ephemeral` for sessions that should last only while the
server runs. `cow_server.py SERVER_DIRECTORY --ephemeral` also opts an existing
configuration into a temporary run. History, checkpoints and admitted grammars
are placed in a fresh private session directory and removed on normal exit or
an exception. Downloaded model files, source, configuration and existing durable
stores are preserved. Closing a BW client alone does not stop the server.

An uncatchable termination or power loss cannot run shutdown cleanup. The next
ephemeral launch removes only marked abandoned session directories whose lease
locks are no longer held. Active sessions and unmarked directories are preserved.

For a shared A100 server, `--ram-gib 100 --disk-gib 100` sets the shared numerical
spillover pool budgets to 100 GiB each. These are not total process-memory or
whole-directory size limits. Model weights, live GPU state, canonical history,
temporary staging and operating-system overhead need their own headroom.

### Wen naming support

For a Wen GGUF declaring architecture/pre-tokenizer `wen1`, apply the inspected
dependency naming patch before building:

```sh
python experimental/branching_context/scripts/prepare_wen_llama.py --llama "$LLAMA_ROOT"
```

The patch maps Wen to the compatible hybrid implementation while reading actual
`wen1.*` metadata keys and the Wen vocabulary/merges. Qwen remains supported.
The builder accepts only the exact inspected changes on the pinned base and
records their hashes; arbitrary dirty dependency trees are rejected. Build the
dependency after applying the patch, then build the COW adapter as above.

For restart, stop the owned server cleanly and launch the SAME server directory again. Do not rerun setup or change the profile. Model loading and transfers are allowed during restore; recomputing saved prefix tokens is a separate measured property.

## BranchWrite client

Install the selected BranchWrite source/wheel on another Linux client. To use this integration from a BranchWrite checkout of `main`, run `BW_REF=main sh install.sh`. The installer otherwise selects its older frozen preview tag; updating a source branch does not move that tag. Keep its configured server/model/execution identities together; use the actual server discovery rather than guessed IDs. The validated website path used `branchwrite code PROJECT --project-files --connection CONNECTION.json --state RUN_STATE --task-file PROMPT.txt`. The normal interactive path is `branchwrite session PROJECT`; `BW_COW_CONNECTION` selects a connection file without changing another user's default. Keep `BW_COW_API_KEY` in process memory unless saving it is explicitly requested.

The unchanged-answer retry is a read of the original durable result. It must retain the source receipt and must not move the branch or compute another answer. Lost acknowledgments require reconciliation under the original operation ID. A cancelled request without a durable result is not silently resent.

## Vision and state boundaries

The matching tested projector is mmproj-F16.gguf, 918,166,080 bytes, SHA-256 `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f`. Build its producer with scripts/build_qwen35_encoder.py and use a pinned image-producer configuration through the normal setup tool's `--vision-config`. The mixed adapter consumes actual image embedding rows with explicit grid positions. Producer identities and original-image hashes remain distinct from logical token positions. Image-bearing restart and unchanged-image retry have bounded real-backend evidence; general visual task success is not implied.

Native checkpoints preserve the hybrid model state, consumed position/proofs and frontier. RNG, grammar/parser progress, sampler history and decoder state belong to canonical COW controller records; they are not assumed to be supplied by llama.cpp state export. Broader controller combinations, AMD, complete Flash serving regressions and full performance qualification remain outstanding.

## Research variants and producer configuration

The tested user research GGUF uses the same compatible qwen35 geometry with a
changed vocabulary and BF16 tensors. Select that file with `--model` and use a
new server directory. The setup discovers its own vocabulary, EOS IDs, tokenizer,
weights and execution identity. Do not reuse another model's connection pins,
store or checkpoints. The research GGUF itself is not distributed by this repo.

For image input, the producer configuration's `pinned_sha256` map must include
the actual decoder GGUF selected by that server, plus the encoder executable and
projector. Copying a Qwen configuration without replacing its decoder-model pin
correctly fails admission. Keep separate producer definitions for separate models.
The projector's ability to produce compatible rows does not establish the model's
visual competence. Custom weights may need training despite successful ingestion.

A tiny research mixed-input fixture (128 text tokens, 48 real image rows, two
4-token suffixes and eight generated tokens per branch) passed 38 exact full-logit
comparisons and three process-separated strict restores. This is a bounded runtime
and preservation check, not a visual-answer benchmark or broad vision certification.

The tested coding CLI supports `/explore` for separate alternative files, `/switch`
for matching saved files/context, and `/edit` for a new workflow replacing an earlier
request. `/edit` is distinct from replaying an unchanged COW operation receipt.

The research model also completed a real encoder + authenticated HTTP image
request from the separate BranchWrite client: 48 image rows consumed, 183
generated tokens, durable result published. Both models use the same adapter
code; their model, execution, tokenizer and image-producer identities stay separate.

## Additional bounded runtime checks

Original Qwen3.5-9B BF16 weights (17,920,697,312 bytes, SHA-256
`daebe40eeea7057c1cdf35ac56d13f507d8bf12171bbb7a6b6b0d3f05439159a`)
passed the same tiny/1K/8K mixed-input stages with FP16 KV: 178 exact full-logit
comparisons, nine strict process-separated restores and zero prefix forward work.
The larger cases use 16 generated tokens per branch. This original BF16 file is
different from the user's research BF16 model. Select its actual model file in
new setup; do not reuse Q8 or research-model checkpoints.

The existing Flash Q8 native request regression also passed 32 exact comparisons
using current main Python source and the previously pinned Flash executables.
Parent-byte integrity, zero prefix work on activation, cancellation and observed
successful native shutdown were checked. That probe does not qualify the complete
canonical store/publication or serving stack. It reused 11 verified model hash
receipts without reading all model weights again. See the committed run summaries.
