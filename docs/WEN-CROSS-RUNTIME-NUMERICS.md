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
normalization and a value-head permutation before an output projection. A focused
follow-up identifies native fast-math SiLU as the first divergence inside layer
0's one-token recurrent block. The published reference profile still uses the
native compiler's fast-math setting for these elementary functions. Strict SiLU
is a diagnostic control, not a qualified whole-model serving build. Not every
remaining bit, input, layer or device has been explained.

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
at most 0.000000238419. The initial trace did not separate the internal stages;
the focused follow-up below identifies the first divergence there.

Using the captured native projection input in PyTorch, with the weight columns
restored to native order, reproduces native output exactly. Permuting those same
inputs and weight columns into grouped head order changes the result by
0.000000476837. The mathematical dot product is equivalent, but its F32 summation
order changes. This isolates another rounding contribution without a dtype change.
It does not explain away the separate pre-projection residual.

## Focused convolution, activation and recurrence inspection

An additional one-token trace captures the actual convolution/activation and
recurrent boundaries. Both observers preserve their respective uninstrumented
final logits bit-for-bit. Native value heads are canonicalized before comparison;
Q/K repetition is accounted for separately.

| Layer 0 boundary | Maximum absolute difference |
| --- | --- |
| Raw depthwise convolution | 0.0, exact |
| Convolution followed by SiLU | 0.000000476837 |
| Q and K after L2 normalization | 0.000000119209 each |
| Recurrent output before gated RMS | 0.0000000745058 |
| Gated output before output projection | 0.000000238419 |

The first new difference is therefore activation, not convolution, for this
fixture. Later stages already receive different inputs. These boundary maxima
cannot be subtracted or added to assign independent error contributions: they
can occur at different coordinates and pass through nonlinear transformations.

### Identical-input SiLU and compiler controls

The exact native helper computes `x / (1 + expf(-x))`. Compile it with the native
flags and feed it the 8192 captured raw-convolution values: it reproduces all
native activation bits. Remove `-use_fast_math`, enable precise division and
square root, and disable subnormal flushing: the helper matches PyTorch SiLU
exactly over the same values. Both controls retain F32 inputs and outputs.

The native and strict outputs differ at 2878/8192 elements, worst absolute error
0.000000476837 and worst distance four F32 representable steps. Factor controls
separate the compiler options:

| Compiler control | Elements differing from PyTorch SiLU | Maximum absolute difference |
| --- | --- | --- |
| Native `-use_fast_math` | 2878 | 0.000000476837 |
| Fast math, only subnormal flushing disabled | 2878, identical to native | 0.000000476837 |
| Fast math, precise division enabled | 629 | 0.000000238419 |
| Standard exponential, approximate division retained | 2598 | 0.000000476837 |
| Strict control | 0 | 0.0 |

Both exponential evaluation and division contribute in this fixture. Subnormal
flushing is a negative control here; it is not a general proof that flushing
never matters. Square-root precision does not explain this SiLU operation,
which contains no square root.

The emitted native PTX uses a multiply by the log2(e) factor followed by
`ex2.approx.ftz.f32`, addition and `div.approx.ftz.f32`. The strict helper uses
extra range-reduction/correction FMAs around an `ex2.approx` instruction and
finishes with `div.rn.f32`. Thus an accurate library exponential can still use
approximate hardware internally, with software correction. PTX excerpts and
kernel hashes are recorded; the native-versus-strict distinction is not inferred
from dtype names alone.

At input `6.3273725509643555`, native produces `6.316087245941162` and PyTorch
SiLU produces `6.31608772277832`. The rounded-F64 control agrees with native at
that particular input. Over the entire vector neither F32 implementation equals
the rounded-F64 control. Matching PyTorch and measuring closeness to higher
precision are different checks; strict arithmetic is not automatically the
closest rounded answer for every composite expression.

These controls isolate the activation difference. They do not yet determine the
independent contribution of recurrent arithmetic or gated normalization after
supplying identical upstream activations. No production kernel was changed by
this follow-up. A complete strict reference build must be measured separately;
merely selecting F32 dtypes does not remove the compiler's fast-math behavior.

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
`validation/wen-precision-20261022/results.json`, `layer-trace.json` and
`activation-trace.json`. Instrumented
tensor dumps and operational scripts are retained with the private diagnostic
run; they contain no changes to the production forward path. The time-limited
shared-kernel experiment was stopped after establishing the measured boundaries.
