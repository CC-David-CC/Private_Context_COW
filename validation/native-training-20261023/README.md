# Native training mechanics — 2026-10-23

**Bounded mechanics PASS; task learning and production qualification remain open.**
No PyTorch training, adapter promotion, model replacement or BW/interactive CLI
integration occurred. Ordinary complete-graph backprop was used throughout,
including on two physical A100 80 GB GPUs. Section-boundary gradient passing was
not used. Machine addresses, prompts, candidates and raw logs remain private.

## What ran

The actual Wen 1.1 BF16 model has 32 hybrid recurrent/full-attention layers.
Its ordinary training forward agreed with a separately built fresh PCC native
FP32 inference oracle before updates. The maximum logit difference was
0.00001717, RMS 0.00000355, with the same top token. The declared bounds were
0.0002 maximum and 0.00005 RMS, plus equal top token; they were not relaxed.

Three short owned arithmetic fixtures supplied completion-only losses. They
exercise mechanics and are not evidence of a natural model failure, useful new
knowledge, general skill improvement or a real day of inference load.

| Actual-model execution | Result |
| --- | --- |
| LoRA step 1: accumulate two examples | Loss 2.20562935 → 2.15538096. All 498 adapter tensors have finite gradients; 249 have nonzero gradients, as expected with initially zero B matrices. |
| LoRA step 2: final single-example batch | Loss 1.88410997 → 1.83374321. All 498 adapter gradients are finite and nonzero. |
| LoRA export/reload | Separate 45,426,096-byte GGUF. Reloaded loss matches the completed run within 0.000001. Fresh PCC inference agrees: maximum logit difference 0.00001621, RMS 0.00000250, equal top token. |
| Continue completed LoRA candidate | Another two-example update lowers loss 2.11993837 → 2.07373619; ordered cursor advances from three to five. This is not a mid-step crash or uninterrupted-run equality test. |
| Direct full-weight step | Loss 2.18679857 → 1.42009914. All 427 active parameter tensors have finite, nonzero gradients, including embeddings, both layer types, norms, convolution and head. No adapter was merged. |
| Full export/reload | Separate 31,876,841,504-byte GGUF. Reloaded loss matches within 0.000001. Fresh PCC inference agrees: maximum logit difference 0.00001812, RMS 0.00000304, equal top token. |

All candidates remain unpromoted. The original BF16 model's SHA-256 is unchanged.
The full run used a one-example batch. Large-model full-weight accumulation was
not separately measured; both modes' accumulation is checked on the tiny graph.

## Independent and integration checks

- Tiny recurrent/attention gradients: CPU and CUDA directions through every
  parameter tensor pass. Independent NumPy/FP64 finite differences check all 386
  scalar derivatives per backend, including a perturbation-step sensitivity
  check. This oracle imports neither GGML nor the native backward implementation.
- Whole tiny hybrid graph: ordinary full and LoRA updates pass on CPU and two
  A100 GPUs, with checked directions in every stage. GPU graphs assert that no
  tensor math silently runs on the administrative CPU backend.
- Native batch accumulation: every scalar gradient matches the average of
  separately computed example gradients. Maximum observed CUDA error is
  0.00000004471 in full mode and 0.000000000233 in LoRA mode.
- CPU-only build: small gradients and standard graph pass without a CUDA
  backend. A CUDA request explicitly fails; current available RAM is used for
  CPU admission. Duplicate devices and mixed CPU/CUDA placement are refused.
- Eleven offline contract tests pass. The existing Linux G0 reference selection
  passes 51 checks at the unchanged serving-core baseline. An earlier Windows
  G0 invocation failed on unavailable `fcntl`; it is not counted as a pass.

The exact numerical records, candidate hashes and dependency fingerprints are
in [results.json](results.json). Three bring-up source versions are identified:
the independent layer fixtures, the actual-model proof, and the final source.
Changes after the actual-model proof add admission guards and batch tests; they
do not change forward/backward mathematics or the optimizer. The final source
was rebuilt and checked on CPU and CUDA; the full large-model run was not repeated
under that final executable fingerprint.

## Resource observations and limits

Each GPU was admitted against its own free memory and explicit cap. Account
locks were held only while tests ran. Weights and gradients occupied both GPUs;
aggregate free VRAM was not treated as one allocation. Two-second process samples
observed maxima of 33,244 and 34,350 MiB during full-weight training. These are
sampled observations, not exact allocator peaks or a speed comparison.

At final audit no owned training/oracle processes remained and account GPU locks
were released. Other users' workloads were not stopped.

Initial compile/layout/CUDA failures and an intentionally stopped slow loader
remain in private evidence. Corrections included dense adjoints at sliced
normalization inputs, a differentiable causal mask and explicit FP32 CUDA math.
One final negative-test harness expected exit code 1; the executable correctly
returned its error code 2. The remaining negative checks then verified
both that code and the intended error message. No failed execution is represented
as a successful update.

Open qualification includes useful repairs, held-out generalization,
preservation, long sequences/datasets, real CPU and other hardware training,
large-model accumulated full updates, mid-step recovery, equality of resumed
and uninterrupted updates, full-candidate continuation, sustained shared-resource
pressure and BW/CLI feedback integration. Native SGD is the only optimizer.
