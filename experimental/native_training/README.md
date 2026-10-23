# PCC native training experiment

This is a research preview with opt-in BW/CLI lesson controls and offline helpers.
Mechanics checks do not establish useful task learning or an always improving
model. No serving model or adapter has been promoted.

Both LoRA updates and direct base-weight backpropagation use native GGML without
PyTorch. Direct full-weight training is distinct from merging a LoRA into a base.
The [learning-loop guide](../../docs/LEARNING-LOOP.md) explains checked lessons,
single/batch requests, resource configuration and candidate selection.

## Ordinary backpropagation is the default

The standard path builds the complete causal model graph, forms the masked
completion loss and invokes one end-to-end reverse pass. Multi-GPU tensor
placement must preserve that path; adding devices must never silently select a
section-boundary training experiment. All selected parameters must have checked
gradients. No unsupported operation may silently stop gradient propagation.

Passing adjoints between independently executed model sections is a separate
experimental, explicitly selected path. Its existence cannot substitute for
ordinary model backpropagation. Its API admission requires an experimental flag.

## Current files

- `native_train.cpp`: explicit GGML recurrent/attention/MLP graph primitives,
  small layer gradient gates, a complete small hybrid-model standard graph,
  LoRA parameters, direct weights, native SGD, model/adapter loading and separate
  GGUF candidate export. Real-model evidence is listed below; long sequences and
  mid-step crash recovery remain unqualified.
- `build.py`: a private copy/build of the pinned GGML dependency, kept separate
  from every serving executable. A strided-noninplace-scale materialization
  patch is recorded explicitly. Its necessity and coverage are still under test.
- `tools/pcc_native_training.py`: offline admission and candidate-version helpers.
  Verified example provenance, explicit completion masks, batching weights,
  content identity and non-promotion are checked without importing a tensor
  training framework.
- `tools/test_native_training.py`: admission/version/default-backprop tests.
- `reference.py`: independent NumPy/FP64 forward and finite differences of
  exported tiny fixtures. Verification does not import GGML or PyTorch.

The initial compiler and backward-layout failures are retained in the private
operator run directory. A compile or a tiny loss reduction is not a gradient,
real-model, task-learning or multi-GPU pass.

## Acceptance sequence

1. Tiny recurrent and attention VJPs: directional finite differences through every
   parameter tensor. Fixed F32 perturbation 0.002; absolute error below
   0.0003 + 0.02 times the absolute numerical derivative.
2. Ordinary complete hybrid graph: check parameter directions in every stage,
   including embeddings and the head for full-weight mode. Fixed perturbation
   0.003; absolute error below 0.0004 + 0.03 times the numerical derivative.
3. Compare ordinary one-device and multi-device forward losses and gradients.
   Prove actual tensor allocation and backend execution on each selected GPU.
4. Validate the actual Wen graph/weights against the pinned native forward,
   keeping the execution/rounding schedule and any discrepancies explicit.
5. Real LoRA and full-weight native updates: finite gradients through the complete
   model, immutable original weights, explicit target scope and masked loss.
   Exercise single examples, accumulated batches and the partial final batch.
6. Save/reload and interrupted-step recovery: exact data/model/adapter/step
   identities, optimizer state, no silent repeat after an uncertain completion,
   new immutable candidate artifacts and old-cache incompatibility.
7. Repairs, independent held-out variants and preservation checks through PCC.
   Loss reduction alone never promotes a candidate. Do not claim an always
   improving adapter from a mechanics test or a failed task experiment.

## Capacity and ownership

Inspect actual A100 use before admission, identify selected physical devices,
retain per-device headroom and record capacities. Aggregate free VRAM is not one
automatic allocation. Shard weights/gradients/optimizer state explicitly and
measure transfers and memory. Leave other accounts and serving processes alone.

Only owned, verified training examples supply gradients. Held-out and protected
evaluation answers stay out of the training split and public artifacts. Candidate
model bytes, prompts and raw traces remain outside the source repository.

## How this fits PCC and BW

A **LoRA adapter** is a small additional set of learned weights: the model's
"hat." Updating it keeps the original model weights fixed. **Direct full-weight
training** changes active decoder weights themselves, without training an adapter
and merging it afterward. Full mode includes embeddings, layers, norms and head.
LoRA adapts linear projections and the head, not embeddings, convolutions or norms.

The C++/GGML engine performs forward, backward and SGD updates. Python checks
offline requests and records candidates; NumPy is used only for independent
verification. COW's saved context is neither a training example nor a weight
checkpoint. New model/adapter identities require separate compatible caches.

The intended later workflow is: capture a failure and an independently verified
correction; choose owned examples to learn from now or in a later batch; train a
separate candidate; evaluate repairs, held-out variants and protected behaviour;
then explicitly select a qualified version for new requests. Existing contexts
keep their original pinned weights. BW now supplies explicit lesson controls;
automatic candidate promotion and live-context weight replacement remain absent.
Successful tests and lower training loss do not automatically install an adapter.

## Measured scope

Short owned arithmetic fixtures exercise machinery. They are not protected test
answers, a useful new skill, a day of inference/training load, a large dataset
campaign, or a GPU-versus-CPU throughput comparison.

| Check | Result and limit |
| --- | --- |
| Offline admission and version rules | Eleven tests pass: masks, provenance, immutable examples, partial-batch weights, ordinary-backprop default, distinct devices, candidate identity and non-promotion. |
| Individual recurrent/attention gradients | CPU and CUDA directions through every parameter tensor pass. Independent FP64 finite differences check all 386 scalar derivatives on each backend. |
| Whole tiny hybrid model | Ordinary full and LoRA backprop pass on CPU and two actual A100 devices, with a checked parameter direction in every stage. |
| Actual Wen 1.1 forward | All 32 layers agree with fresh PCC native FP32 inference: maximum logit error 0.00001717, RMS 0.00000355, same top token. |
| Actual Wen LoRA | Two ordinary two-GPU steps, batches of two and one, finite gradients for 498 adapter tensors across all layers. Saved adapter reloads and agrees with fresh PCC inference. |
| Actual Wen direct full-weight update | One ordinary two-GPU step updates all 427 active decoder parameter tensors, each with finite nonzero gradients. Loss falls from 2.18680 to 1.42010; the separate full GGUF reloads and agrees with fresh PCC inference. |
| Completed LoRA continuation | The saved adapter completes another two-example update, with the ordered example cursor advancing from three to five. Mid-step recovery and equality with an uninterrupted run remain unqualified. |
| CPU-only build | Builds without CUDA and passes the small layer/whole-graph checks. Real Wen CPU training, HIP, SYCL, P4s and the home-hardware matrix remain unqualified. |
| BW/CLI CPU wiring | Real isolated BW failure/correction/check records, native single-example and partial-final-batch updates, frozen retry selection, LoRA/full continuation and candidate reload pass on the generated tiny hybrid model. No GPU or language-quality claim. |

Small complete-graph checks also compare every accumulated gradient against the
mean of independently computed per-example gradients in both modes, on CPU and
two GPUs. The real full-weight run uses a single-example batch; its large-model
accumulated-batch path has not separately been measured. Sampled process memory
peaked around 33,244 and 34,350 MiB on the two GPUs during the full-weight run;
two-second samples are observations, not exact allocator high-water marks.

Read the [dated results and fingerprints](../../validation/native-training-20261023/README.md).

The implementation currently targets dense Wen/Qwen3.5 hybrid recurrent and
full-attention models, not every model PCC can serve. BF16/F32 inputs become F32
master weights. Quantized training inputs are refused. Full exports save active
decoder weights in F32 and preserve inactive source extras unchanged.

## Building and running small checks

Use a separate source/build directory and the inspected GGML/llama.cpp revision
`972d2313bc0bf0a45f634f77d95c9fb03aeab12c`. Linux, a C++17 compiler and CMake are
required. The helper verifies the pin and records sources/dependencies in
`build.json`. For CPU-only development with that checkout already available:

```sh
python experimental/native_training/build.py \
  --llama /path/to/pinned-llama --output /path/to/private-training-build --cpu-only
/path/to/private-training-build/pcc-native-train --self-test CPU
/path/to/private-training-build/pcc-native-train --chain-test CPU
```

For CUDA, prepare a separate pinned Wen/FP32 engine using
[`prepare_wen_llama.py`](../branching_context/scripts/prepare_wen_llama.py)
with `--f32-activations`, build its CUDA shared libraries, then omit `--cpu-only`.
Keep the exact private GGML and inspected CUDA libraries with the receipt.
Training sets `GGML_CUDA_F32_ACTIVATIONS=1` and `NVIDIA_TF32_OVERRIDE=0` before
CUDA initialization. A CUDA compiler and driver are required for that build.

```sh
/path/to/private-training-build/pcc-native-train --gradient-diagnostic CUDA0 > layer-check.json
python experimental/native_training/reference.py layer-check.json
/path/to/private-training-build/pcc-native-train --chain-test CUDA0,CUDA1
python -m unittest discover -s tools -p test_native_training.py -v
```

GPU indices refer to the explicitly selected `CUDA_VISIBLE_DEVICES` list.
Two devices mean ordinary backprop, not the section experiment. The offline
execution entry point currently accepts only the standard complete-graph path.
NumPy is required for FP64 verification, not for native training.

## Offline requests and completed-candidate continuation

Use `tools/pcc_native_training.py`'s `execute_native()` from an offline operator
script with a fresh private run directory. Supply pinned model SHA-256, explicit
devices/per-device budgets, `mode="lora"` or `mode="full"`, examples, learning
rate, batch size and steps. Only SGD without momentum is implemented; other
optimizer requests fail rather than silently substituting.

Examples contain exactly `id`, `tokens`, `targets`, `loss_mask`, `verified` and
`provenance_sha256`. Tokenize with the exact base tokenizer. Each target is the
next token; the boolean mask selects predictions contributing to completion
loss. Masked prompt tokens still supply context. IDs are unique and provenance is
a SHA-256. Preserve independent verification evidence privately; setting a JSON
boolean does not verify a task. Evaluator answers must not become training data.

Each example has equal weight within its actual batch, including a shorter final
batch. Gradients accumulate before one SGD update. A separate `candidate.gguf`,
private request, launch fingerprint, exit status, native receipt and
content-addressed version manifest are recorded. The original model is checked
unchanged. Candidates are registered with `promoted=false`.

For another LoRA update, pin the same base, the saved `initial_adapter` path and
`initial_adapter_sha256`. Carry the completed receipt's `example_cursor` when
continuing the same ordered dataset. Loading checks adapter/base identity, rank,
tensor coverage and shapes. For direct full-weight continuation, use the
completed full candidate as the new model and pin its hash; do not supply an
adapter to full mode. Dataset, cursor, learning rate and settings must come from
the exact completed run, not a guessed restart position.

Exports are **completed-run candidates**, not periodic transactional checkpoints
in the middle of an optimizer step. Interrupted or uncertain runs are retained
for reconciliation; the helper refuses to reuse their run directories and never
blindly issues another update. SGD has no moment state, but weights, data order
and completed-update identity still matter.

Actual-model cross-graph logits require maximum error at most 0.0002, RMS at
most 0.00005 and the same top token, declared before comparison. FP64 derivative
checks use steps 0.000001 and 0.000002, native agreement `atol=0.00005,
rtol=0.0005`, and step sensitivity `atol=0.0000001, rtol=0.00001`. These checks
are distinct from COW's exact same-profile state/cache contracts.

Independent repair, held-out and preservation evaluations remain necessary
before using trained versions for real work. Outstanding qualification also
includes longer sequences/datasets, true mid-step crash recovery, hardware routes
and sustained resource pressure. Never promote solely because training loss fell.

The [completed 1,000-variation Wen study](../../validation/wen-learning-quality-20261023/README.md)
measures substantial selection-pattern learning, preservation regressions and
weak blind transfer. It qualifies no candidate and changes no live model.

## Bounded dense sequence profile

`dense-short-v1` is an opt-in complete-graph path for up to 512 consumed tokens
and 256 selected target positions, within the explicitly checked dense model
geometry. It uses balanced sequence concatenation, bounded graph capacity and
actual live-tensor counts for multi-device scheduler bookkeeping. It retains
ordinary end-to-end backpropagation; it does not use section adjoints.

An owner can set `block_devices` to place the embedding, every decoder layer and
the head on explicit visible device indices. Omission retains modulo placement.
`host_reserve_bytes` protects available host RAM as well as per-device memory
budgets. `timeout_seconds` in the owner runner sets both its subprocess deadline
and the native child's independent alarm. Native requests name this value
`deadline_seconds`; its range is 1–3600 seconds. Client lesson requests cannot
choose these resources.

`--probe-gradients REQUEST.json` checks one exact owned example and reports
finite parameter gradient norms without applying an optimizer update or writing
a candidate. `--admit-training` remains a planner-only check. Neither establishes
language learning or a release-ready model.

Measured on 2026-10-23: the tiny CPU fixture's old/new paths agreed within 0.00001
at 64 and 375 tokens for LoRA and full updates, including candidate reloads.
The same cases agreed with the CPU reference on two A100s, including explicit
placement. A real Wen 1.1, 178-token LoRA gradient-only probe then passed on four
A100s: 498 finite gradient tensors, 249 initially nonzero, loss 0.882686615.
The other half of the LoRA factors begin with zero gradients because their paired
factors are initialized to zero. This run took 72.42 seconds including its model
integrity check; shared-host timing is not a performance benchmark. Original
weights were unchanged, no candidate was written, and all owned GPU allocations
and locks settled. This does not qualify 375-token Wen training or full-weight
Wen updates under the new profile.

Run the reproducible CPU checks with:

```sh
python -m experimental.native_training.test_dense_profile --executable /path/to/pcc-native-train --output /fresh/private/dense-proof
python -m experimental.native_training.test_memory_admission --executable /path/to/pcc-native-train --output /fresh/private/admission-proof
```

See [the sanitized evidence](../../validation/foundations-20261023/dense-short-profile.json).
