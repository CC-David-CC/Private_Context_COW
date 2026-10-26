# Selected-output head for native action training

`dense-llama-action-selected-head-8k-v1` is an opt-in ordinary-Llama native
training profile. It keeps the whole context and ordinary full backprop, but
computes vocabulary predictions only at positions selected by the loss mask.
It also shares the non-trainable causal mask across layers and uses GGML's
existing scale/mask/softmax operation and its ordinary adjoint. The previous
512-token and 8k profiles retain their original paths.

Long BW action lessons contain thousands of context tokens but select only the
current action through EOS. Computing a full vocabulary vector for every context
position creates avoidable head tensors and adjoints. This change gathers the
selected hidden columns before the column-local RMS norm and output projection.
Gather backward scatters the adjoints into the complete hidden sequence; the
attention graph still differentiates through the original full history.

For selected positions S, the loss remains
`L = (weight / |S|) sum(t in S) CE(W * RMS(H[:,t]), target[t])`.
The head's vocabulary tensor changes from `vocabulary × context_length` to
`vocabulary × selected_positions`. Labels, ordering, weighting, context and
optimizer rules remain the same. Attention still has quadratic storage here;
this does not promise that an entire real model fits any particular GPU.

## Qualification

Generated tiny weights passed CPU and RTX 4090 checks:

* Independent FP64 finite differences covered all 212 scalar weights on a
  scattered-label fixture, including an early context label. Maximum derivative
  error was 5.33e-6 under the existing unchanged tolerances. A 513-token fixture
  checked one scalar per full tensor with scattered labels at 0, 3, 241 and 512.
* Full and rank-one LoRA saved updates were compared with the dense-head path.
  All CPU compared weights were identical. CUDA maximum differences were about
  9.32e-10 for the full update and 1.82e-12 for two LoRA updates.
* The existing long-action controls passed on this new profile: actual 8,192-token
  tiny forward/backward, independent full/LoRA derivatives, 42-layer standard
  backprop with all 381 full gradient tensors at 513 and 1,024 tokens, 256 selected
  targets, adaptive state continuation and unchanged admission refusals.
* Actual HTTP/lesson/native jobs passed for full AdamW, full Muon and AdamW LoRA,
  with completion recovery enabled before dispatch, independent saved-model
  reload, exact-state continuation and parent resolution. Duplicate submission
  reused the original operation; changed-stream resume was refused. These use
  synthetic tiny digit-token lessons, not production S³ approval.
* Separate CPU/CUDA comparisons against the previous qualified executable found
  identical old-profile forward logit bytes and all saved full/LoRA tensor values
  on the checked short and 513-token fixtures. Existing short-profile independent
  CUDA qualification and 68 Python admission/optimizer/completion controls passed.

The [qualification receipt](../validation/selected-head-native-20261026.json)
records source, executable, dependency and result hashes. These are tiny-model
mechanics, not real MiniCPM memory qualification, BW learning or model acceptance.
No HIP/SYCL or new backend qualification is claimed. The native path is CPU/CUDA.
The existing imported [llama.cpp PR 30142](https://github.com/ggml-org/llama.cpp/pull/30142)
by edenfunf remains credited for the CUDA singleton outer-product fix.

## Reproduce and use

Build the native executable in an isolated directory with the existing build
tool and qualified dependency. Use a fresh output directory and the appropriate
PCC GPU lease. For CUDA, explicitly map the selected physical device and set
`GGML_CUDA_F32_ACTIVATIONS=1` and `NVIDIA_TF32_OVERRIDE=0`.

```sh
export PYTHONPATH="$PWD:$PWD/tools:$PWD/experimental/branching_context/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m experimental.native_training.test_llama_selected_head \
  --executable /absolute/path/pcc-native-train \
  --output /private/fresh-cpu-proof --devices CPU
python -m tools.test_selected_head_jobs \
  --executable /absolute/path/pcc-native-train --output /private/fresh-service-proof
```

Repeat the numerical fixture on a leased CUDA device with `--devices CUDA0`.
`tools.test_native_profile_compatibility` separately accepts both the new
`--executable` and prior `--reference-executable` for old-profile comparison.

For a qualified real learning job, select the new sequence profile in the
existing NativeRunner/LessonService owner configuration. Keep explicit native
allocator budgets/reserves, the 8,192 context and 256 selected-target bounds,
installed S³ practice membership, immutable parent identity and completion
recovery. No new training controller or tensor framework is introduced.

An explicit `logits_output` diagnostic still returns all context logits using
the original dense projection. The receipt calls this `full-logits-diagnostic-v1`;
training and loss-only probes report `selected-hidden-columns-v1`. Full-logit
diagnostic memory can therefore differ substantially from ordinary masked-loss
training memory. Both paths keep the full original context.

Real-model gradient residency, adaptive optimizer state, the entire lesson block,
actual BW workflows, mixed-review retention and acceptance remain required before
claiming a successful action-learning round. The [original action profile](DENSE-LLAMA-ACTION-TRAINING.md)
and [real-model memory refusals](MINICPM-ACTION-MEMORY.md) remain original evidence.
The frozen 68-update optimizer/rank comparison is not migrated to this profile.

## Initial real-model finding and attention follow-up

The first selected-head-only build was tested on the learned MiniCPM parent,
without any weight update. Its 3,029-, 3,663- and 4,278-token rank-16 LoRA
gradient probes passed; each exposed 590 finite tensor gradients with the expected
295 nonzero B gradients at zero-initialized B. The 4,873-token LoRA graph still
required 96,270,881,792 bytes and exceeded the card's guarded graph budget.
Full 4,873-token backprop was also resource-refused. Those original outcomes
remain in the [initial real-model receipt](../validation/selected-head-real-initial-memory-20261026.json).
No production lesson or learning-quality approval follows from these probes.

At 3,663 tokens, full standard backprop used a 63,547,054,080-byte planned graph
and a sampled native-process peak of 80,544 MiB. Its loss differed from the saved
dense-head result by 1.49e-7. All 381 gradient-tensor norms agreed under unchanged
rtol=1e-4, atol=1e-5; the largest absolute norm difference was 1.72e-5. This compares
norms, not every component. Both probes used the same unchanged learned weights.
Optimizer persistent-only plans in that record are not adaptive training proofs.

The follow-up keeps the selected head and shares one non-trainable causal mask
per graph instead of copying an identical input for every layer. The fused
`ggml_soft_max_ext` operation incorporates the same scale and additive mask;
its existing derivative supplies the scale during ordinary whole-graph backward.
No custom CUDA kernel, precision reduction, CPU fallback or section-gradient
training is introduced.

The complete generated-fixture CPU/CUDA and actual CPU HTTP/native qualifications
passed again for this changed native implementation, including independent FP64
checks, full/LoRA saved-update agreement and strict old-executable compatibility.
The 42-layer, 1,024-token CUDA fixture's planned buffer fell from 537,187,456 to
376,640,640 bytes. That tiny-fixture reduction is not a real-model memory estimate
or an end-to-end speed claim. The [follow-up receipt](../validation/selected-head-shared-mask-native-20261026.json)
binds the new source and results; the earlier source/results remain recorded.
The complete real lesson block must still qualify on this updated implementation.
