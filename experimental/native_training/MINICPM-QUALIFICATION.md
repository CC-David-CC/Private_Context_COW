# MiniCPM native training qualification — 2026-10-24

The new opt-in `dense-llama-short-v1` profile implements ordinary dense Llama
attention and full-graph backpropagation for the inspected MiniCPM architecture.
**Tiny generated fixtures and the newer short actual-model forward controls pass;
real MiniCPM training is not yet qualified.**
The first checkpoint used SGD. The subsequent [native optimizer
qualification](OPTIMIZER-QUALIFICATION.md) adds opt-in AdamW/Muon and actual-model
short FP64 forward evidence. Unsupported optimizer requests fail explicitly. No MiniCPM weights have been updated or
candidate accepted. This path uses C++/GGML; independent verification uses NumPy
FP64, without PyTorch or GGML differentiation.

## What changed

The ordinary Llama route has plain Q projections, grouped-query attention,
ordinary RoPE, no Q/K normalization or attention output gate, and `ffn_norm`.
It accepts F16/F32/BF16 source weights as F32 master weights, with explicit
parameter coverage. Extra/tied parameter layouts and unqualified rotary scaling
are refused. F16 admission is specific to this route; hybrid-model and adapter
format rules stay unchanged. The existing default hybrid fixture is byte-identical.

The bounded profile admits at most 512 consumed tokens, 256 selected targets and
42 layers, within explicit geometry/memory limits. The 42-layer check uses tiny
hidden dimensions; it is not a memory or throughput proof for the 2B model.
Ordinary end-to-end backpropagation remains distinct from experimental section
adjoints. Serving tokenization and native context checks are documented separately
in [MiniCPM inference](../branching_context/MINICPM-NATIVE.md).

## Measured evidence

[Machine-readable mechanics evidence](../../validation/minicpm-native-training-20261024/mechanics.json)
binds the source hashes and separates the final CPU revision from an earlier
CUDA revision. No protected questions, solutions, model bytes or raw traces are
included in that public summary.

| Check | Final CPU | Earlier two-A100 CUDA revision |
| --- | --- | --- |
| Independent scalar derivatives, complete full graph | 364; max absolute error 3.20e-7 | 364; max error 2.31e-7 |
| Independent scalar derivatives, complete LoRA graph | 278; max error 9.63e-10 | 278; max error 1.13e-9 |
| Independent layer derivatives | 152 per layer, two layers | 152 per layer, two layers |
| F16/F32 tiny forward, batches of two then one, immutable base, candidate reload | Pass | Pass |
| Tiny 42-layer ordinary reverse graph | 381 parameter tensors, no update | Same scope, pass |
| Negative admission/replay checks | Six | Five, before final F16-format guard |

The final F16-format guard is CPU-qualified. **Its exact CUDA revision still
needs requalification**; the earlier CUDA result is retained under its own source
hash. Its supervised two-GPU allocation was 53.89 seconds (0.02994 allocation
GPU-hours), not an end-to-end training benchmark. CPU-only and CUDA are the
implemented training backends; HIP/SYCL training remains unqualified.

An instructive failure: the tiny full-weight fixture's SGD step at .01 increased
loss from 1.74270 to 1.74925 despite independently correct derivatives. FP64
predicted the same overshoot. The fixed fixture rate .001 reduces loss; this is
not a changed production default or a relaxed derivative tolerance. A successful
backward pass and a useful update are separate requirements.

## Reproduce the small checks

Build the separate training executable using the pinned dependency and FP32
instructions in [README](README.md). Use a new private output directory:

```sh
python -m experimental.native_training.test_llama_profile \
  --executable /path/to/private-training-build/pcc-native-train \
  --output /path/to/new-private-proof
```

For an admitted two-device CUDA check, append `--devices CUDA0 CUDA1`; those
indices refer to the explicitly selected physical devices. Retain build.json,
source hashes, original diagnostics and independent-oracle results. The suite
uses generated owned fixtures and no downloaded language-model weights.

## Remaining gates

1. Recheck the exact current source on CUDA.
2. Compare actual MiniCPM forward/loss against independently pinned native
   inference, making F16 operand rounding versus F32 master execution explicit.
3. Qualify optimizer math, state serialization and resumed updates before using
   adaptive/momentum optimizers. Completed SGD candidates are not transactional
   mid-step optimizer checkpoints.
4. Train separate immutable candidates, then run frozen independent, composed,
   transfer and mixed-review tests through S³. Keep harness improvements separate
   from weight changes; loss alone cannot accept a candidate.

[Current optimizer comparison plan](../../docs/OPTIMIZER-COMPARISON.md) prioritizes
Muon and its defined variants at a fixed compute budget. Broad ablations are
stretch goals, not a reason to delay the main comparison.

A separate opt-in [8k action-training profile](../../docs/DENSE-LLAMA-ACTION-TRAINING.md) now has tiny CPU/CUDA and actual CPU service qualification. Existing 512-token studies remain unchanged; this does not establish real MiniCPM long-context training quality.
