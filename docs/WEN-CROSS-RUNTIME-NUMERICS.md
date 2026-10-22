# Wen: native COW and PyTorch numerical differences

## Purpose and current conclusion

The practical goal is for a LoRA learned in PyTorch to produce the same useful
changes when served through native COW. Bit identity between independent forward
implementations is a separate, stronger requirement. No LoRA update has yet been
qualified by this investigation, and no backpropagation occurred in these probes.

Three sources of reduced precision were isolated: BF16 activation downcasts in
the custom matrix kernel, FP16 attention/cache math, and explicit TF32 instructions
in the custom F32 matrix kernel. The opt-in `f32-reference` profile removes those
sources for the tested dense CUDA path. The original GGUF is unchanged.

The remaining short-prefix discrepancy is about 5e-6 to 8e-6. Intermediate traces
and isolated replays demonstrate FP32 reduction-order differences, including
normalization and a value-head permutation before an output projection. This
does not prove that every remaining bit, input, layer or device has been explained.

## Model, runtimes and measurement

* Original model: Wen BF16 GGUF, SHA256
  `40f048a00bc2a2a8c2db163999f0af90613d136376e9b3c9372457b1769b44ca`.
* llama.cpp pin: `972d2313bc0bf0a45f634f77d95c9fb03aeab12c`, with the inspected
  Wen alias and optional precision recipes in `prepare_wen_llama.py`.
* PyTorch: `2.14.1+cu130`; Transformers: `5.20.2`; GGUF: `0.19.0`.
* Measured device: NVIDIA RTX PRO 6000 Blackwell, 96 GB. Native CUDA toolkit:
  13.2.86, architecture 120a. These results do not qualify Ada, A100 or HIP.
* Training representation: reconstructed directly from the original GGUF into
  F32 parameters, with no missing/unexpected/mismatched keys. There is no original
  safetensors checkpoint. GGUF conversion transforms, including norm offsets,
  recurrent head permutations and the A-log representation, remain relevant.
* PyTorch uses evaluation/inference mode, eager full attention, reference
  recurrent/convolution fallbacks, TF32 disabled and highest F32 matmul precision.
  There is no dropout, sampling, optimizer update or generated-token feedback.
* Each comparison covers all 127991 next-token logits after the exact same input
  IDs. Top-1 is derived from those logits; no sampler can cause their differences.

The one/two/three-token fixtures are `[759]`, `[759,3796]`, and
`[759,3796,13256]`: `The`, `The quick`, `The quick brown`. These are fixed prefixes,
not three successive autoregressive predictions. Longer fixtures extend the same
fixed text through 64 tokens. Six independent text/code prompts are also measured.

Final reference probes use native capacity 512, prefill limit 64, threads 8,
FP32 KV, non-Flash attention, disabled fusion and disabled TF32. Initial default
probes used the existing serving profile. Batched/serial controls compare both
schedules within the same profile. Exact default regression checks use the
original profile and final patched build with all new options disabled.

## What the controls established

| Control or change | Observation |
| --- | --- |
| Original native BF16 versus source-F32 PyTorch | Max error 0.0316353 at two tokens, 0.0498664 at three |
| Original native batched versus serial | About the same large discrepancy; PyTorch's own equivalent discrepancy was about 1e-5 |
| cuBLAS precision override and fusion disable only | Original 1/2/3-token vectors unchanged: the custom MMF path bypasses cuBLAS settings |
| Lossless F32-storage GGUF control | Every expanded tensor value independently verified; the large BF16-path discrepancy shrank without changing source values |
| FP32 products, FP16 attention/KV retained | Final-profile errors 0.0005970, 0.00112593, 0.00197601 at 1/2/3 tokens |
| FP32 attention/KV, before fixing custom F32 MMF | 1/2/3-token errors about 1e-5, but 8/16-token errors 0.00443912/0.00636554 |
| Custom F32 MMF bypassed as well | Final 8/16-token errors 0.0000076890/0.0000109673 |
| Final reference, eleven prefixes through 64 tokens | Worst error 0.0000357628; all eleven top-1 IDs agree |
| Final reference, six other text/code prompts | Worst error 0.0000200272; all six top-1 IDs agree |
| Final patched build, options off | Six full logit vectors at 1/2/3/8/16/32 tokens are bit-identical to the original native build |

Pinned source explains the large dispatch effects: `mmvf.cu` limits the BF16
vector path to a single column on the measured architecture; `mmf.cuh` casts F32
activations to the weight type; `mma.cuh` contains explicit F32/TF32 MMA. NVIDIA's
cuBLAS TF32 override cannot change instructions already emitted by a custom kernel.
See [WEN-PRECISION.md](WEN-PRECISION.md) for the build/profile controls.

Native checkpoint restoration is checked separately. Both adapter variants pass
three suffix comparisons each with exactly zero error against fresh native
execution using the same schedule, and zero prefix-forward work during restore.
The HTTP check adds two exact native comparisons, two concurrent generations,
unchanged parent state, clean exit and zero remaining ephemeral sessions.

## First-divergence trace

Native evaluation callbacks capture F32 tensors without modifying them. PyTorch
forward hooks capture corresponding semantic boundaries. Native shapes and byte
strides are respected; a last-position-only native tensor is compared with that
position in PyTorch. Each initial trace compares 154 boundaries. The instrumented
oracle's final logits are bit-identical to the uninstrumented oracle for all
three fixtures, so these observations do not introduce a new final-logit result.

| Prefix length | Embedding | Layer 0 input RMS | Layer 0 recurrent output | Final max logit difference |
| --- | --- | --- | --- | --- |
| 1 | Exact | Exact | 0.000000476837 | 0.00000524521 |
| 2 | Exact | 0.00000381470 | 0.000000476837 | 0.00000667572 |
| 3 | Exact | 0.00000381470 | 0.000000476837 | 0.00000810623 |

### Normalization replay

The first norm's effective weights are bit-identical after reversing the GGUF
offset. Both epsilons are exactly `9.999999974752427e-7`. Replaying only the captured
embedding, original norm weight and normalization formula reproduces the observed
difference without the rest of the model.

For the same first token, PyTorch's mean square is `0.00017896325152833015` when
processed alone and `0.00017896326608024538` in a two/three-row operation. They
are one F32 representable step apart. The reciprocal-root scales are respectively
`74.543212890625` and `74.54320526123047`, also one F32 step apart. Native's first
row stays bit-identical across these sizes; PyTorch's first row does not.

The native kernel accumulates partial square sums and performs a CUDA block
reduction. PyTorch evaluates square, mean and reciprocal square root as separate
operations. The isolated mean change demonstrates a reduction-arithmetic effect;
the exact compiled instruction tree has not been disassembled. There is no
evidence of a TF32 or FP16 norm calculation in this control. A rounded-F64 replay
also shows that neither implementation is an exact real-arithmetic oracle.

### Shared native normalization experiment

A diagnostic wrapper calls the exact native RMS forward helper, compiled with
the native CUDA flags, on PyTorch CUDA tensors and the current stream. It is used
for ordinary and gated RMS forward calls. This makes layer 0 input RMS bit-exact
for every fixture, but final errors remain 0.00000524521, 0.00000596046 and
0.00000810623. Fixing that boundary alone therefore does not make the full model
bit-identical. The wrapper explicitly rejects gradient-enabled execution: it has
no qualified backward and is not part of the training or serving implementation.

### Recurrent head order and projection replay

The GGUF importer changes value heads from native tiled order to PyTorch grouped
order. Comparisons must undo that permutation; comparing the raw arrays would
incorrectly report a large semantic mismatch.

For the one-token fixture, layer 0 QKV, z, beta and alpha projections match exactly
after canonicalizing head order. The pre-output-projection activation differs by
at most 0.000000238419. That residual has not been fully traced inside the
recurrent/gated-normalization calculation.

Using the captured native projection input in PyTorch, with the weight columns
restored to native order, reproduces native output exactly. Permuting those same
inputs and weight columns into grouped head order changes the result by
0.000000476837. The mathematical dot product is equivalent, but its F32 summation
order changes. This isolates another rounding contribution without a dtype change.
It does not explain away the separate pre-projection residual.

## What is good enough for the LoRA objective

Native restore/fresh consistency continues to require exact equality for the
same recorded schedule. Cross-runtime numerical limits must be validated on the
actual training and evaluation inputs rather than inferred from three tokens.
The observed short-prefix bounds are measurements, not a universal tolerance or
a claim that all near-tie decisions are safe.

Before adapter promotion:

1. Verify the base, adapter tensors, export/readback, scale and immutable identity.
2. Measure both base and adapted forwards in each runtime. Compare adapter deltas
   `L_native(adapter)-L_native(base)` and `L_torch(adapter)-L_torch(base)`, including
   elementwise error, decision changes and effects near ties. Small improvements
   must not be claimed when they are indistinguishable from the numerical gap.
3. Run the same independently verified task repairs, held-out variants and passing
   preservation cases through both. Agreement on these short next-token probes
   does not establish matching generated sequences or BW task competence.
4. Verify adapter-bound save/restore and lazy prefill rebuilding under the new
   identity. Never relabel an old adapter's KV state as compatible.

This investigation does not establish training memory requirements, long-context
stability, real adapter effects, original-checkpoint reconstruction equivalence,
other device parity or cumulative G1-G5 release qualification. It also does not
justify a broad rewrite solely to remove the last few rounding bits.

The raw bounded measurements and diagnostic source/binary hashes are in
`validation/wen-precision-20261022/results.json` and `layer-trace.json`. Instrumented
tensor dumps and operational scripts are retained with the private diagnostic
run; they contain no changes to the production forward path. The time-limited
shared-kernel experiment was stopped after establishing the measured boundaries.
