# Native whole-graph backprop with attention recomputation

`dense-llama-action-recompute-8k-v1` is a separate opt-in ordinary-Llama profile.
It keeps the original full context, FP32 operators and ordinary whole-graph
backprop. It adds the selected output head and shared causal mask from the
[preceding profile](SELECTED-HEAD-ACTION-TRAINING.md), then recomputes attention
probabilities during backward instead of keeping every layer's probabilities.
The earlier profiles retain their original paths. Bounds remain 8,192 input
tokens, 256 selected targets and the inspected ordinary-Llama geometry.

## What the engine does

GGML first builds the complete ordinary reverse graph. The implementation keeps
every original derivative expression and parameter-gradient accumulator. Only
backward reads of saved attention probabilities are replaced by equal values
recomputed from the original Q/K tensors, causal mask and scale. A topological
rebuild includes all original nodes, preserving accumulator side effects and
the original reverse-mode root seed. Recomputed nodes keep their original GPU
ownership. The native receipt still reports `backprop_mode="standard"`, with
`attention_profile="scaled-masked-softmax-rematerialized-v1"` identifying this
explicit computation strategy.

This is activation-value recomputation inside the complete reverse graph. It
does not supply hand-written section adjoints, detach context, change the loss,
reduce precision or silently fall back to CPU. It spends additional QK/softmax
compute to reduce saved attention activations. The existing separate experimental
section/checkpoint mode is not used. Real-model memory and timing still require
measurement; a smaller tiny fixture is not a deployment guarantee.

## Numerical and pathway qualification

Generated tiny CPU and RTX 4090 checks passed. Independent FP64 finite differences
covered all 212 scalar weights on a scattered-label fixture and one direction in
each of 12 full tensors at 513 tokens. Existing independent rank-one B-direction
checks, actual 8,192-token forward/backward, 42-layer standard gradients at 513 and
1,024 tokens, selected-target/context refusals and adaptive continuation passed.
Full and rank-one LoRA saved updates agreed with the dense-head control within
the unchanged numerical bounds. CPU compared weights were identical; the largest
CUDA differences remained at floating-point rounding scale.

Actual tiny CPU HTTP/lesson/native jobs passed for full AdamW, full Muon and
AdamW LoRA, including completion recovery, saved-model reload, parent resolution,
exact-state continuation, idempotency and changed-stream refusal. The fixture
also verifies nonzero gradients and actual saved tensor changes in its first and
continued known learning updates. It does not approve production S³ lessons.

CPU and CUDA compatibility against the preceding executable checked all three
older ordinary-Llama profiles: returned logit bytes and saved full/LoRA tensor
values were identical on the checked fixtures. Another 72 Python admission,
optimizer and completion controls passed. The 42-layer, 1,024-token CUDA fixture
used a planned graph buffer of 32,915,712 bytes, versus 376,640,640 in the
shared-mask-only fixture. These are generated tiny-model memory measurements,
not real-model savings or an end-to-end speed claim.

The [qualification receipt](../validation/attention-recompute-native-20261026.json)
binds the source, dependencies, executables, controls and original failure.
Only CPU/CUDA are qualified here. The previously imported
[llama.cpp PR 30142](https://github.com/ggml-org/llama.cpp/pull/30142) by edenfunf
remains credited for the CUDA singleton outer-product fix.

## A failure the independent check caught

The first tiny prototype rebuilt topology without preserving the old graph's
loss-gradient accumulator map. Its embedding derivative was zero where an
independent FP64 difference was approximately -0.1494257724. The numerical gate
refused it before CUDA or real-model dispatch. Lifecycle/reload checks alone had
passed: finishing and saving a job did not prove that the expected learning update
happened. The original failed receipts were retained. The corrected implementation
preserves the actual root-one seed tensors and initializes them after allocation;
the numerical and strengthened service checks then passed. No trained real model
or original fixed-study result came from that failed prototype.

## Reproduce

Use a fresh private output directory and the existing isolated native build tool
with the qualified dependency. Select a physical CUDA device through the actual
PCC GPU lease before using CUDA0; enable the qualified FP32 activation environment.

```sh
export PYTHONPATH="$PWD:$PWD/tools:$PWD/experimental/branching_context/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m experimental.native_training.test_llama_selected_head \
  --executable /absolute/path/pcc-native-train --output /private/fresh-proof \
  --devices CPU --profile dense-llama-action-recompute-8k-v1
python -m tools.test_selected_head_jobs \
  --executable /absolute/path/pcc-native-train --output /private/fresh-service-proof \
  --profile dense-llama-action-recompute-8k-v1
```

Use `--devices CUDA0` for the numerical CUDA fixture, with
`GGML_CUDA_F32_ACTIVATIONS=1` and `NVIDIA_TF32_OVERRIDE=0`. The compatibility helper
accepts a prior executable and `--include-selected-head` to cover all three older
profiles. Build/test logs and raw operational records remain on the qualification
host; no model weights are uploaded.

Select this profile through the existing NativeRunner/LessonService owner
configuration only after the entire real lesson block and chosen optimizer's
memory have qualified. Install independently checked practice through S³'s
registry and enable completion recovery before dispatch. Target skill, actual
BW workflow, mixed-review retention and acceptance remain separate learning-loop
gates. The frozen 68-update comparison is not changed by this profile.

The [real MiniCPM memory report](REAL-ACTION-RECOMPUTE-MEMORY.md) records all
four action-context gradient probes passing on one 96 GB-class card, including
the 4,873-token full-weight and LoRA cases. No real weights were updated; adaptive
optimizer residency and the actual S³/BW learning round remain separate gates.
