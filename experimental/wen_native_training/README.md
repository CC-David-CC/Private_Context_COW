# Experimental native Wen learning and sparse chunk composition

**EXPERIMENTAL. Keep this work on its experimental branch, including if tests
succeed. No production promotion, general training support or full-model
backward correctness is claimed. Failed experiments and unsupported operations
must remain explicitly recorded.**

## Objective

Train the computation COW actually executes. A PyTorch model with matching logits
is not a prerequisite. PyTorch autograd may be an independent small-operation
gradient oracle; it is not the source of the native forward or training scores.

The model is the published `davidflanagancomm/Wen` file `Wen-1.1-BF16.gguf`,
revision `7328e9fab25b207523b5236aa29c81be24eed924`, SHA-256
`49425fd802b04b400a1725e8afe527c405bb15e828d0275ff295e38396839f34`.
The existing native backend pin is `972d2313bc0bf0a45f634f77d95c9fb03aeab12c`.
The model, private BW prompts, corrective answers and adapters stay outside Git.

Start with previously observed BW failures and lightly guided variants, not new
out-of-distribution skills. Evaluate guidance before adding training complexity.
Keep training, held-out variants and preservation cases separate.

For each training case, reuse its meaning across 32 **separate** distance trials.
There is one answer per trial; repeated correct answers are not inserted into one
stream. Initially leave the gap empty: move the Reader's logical positions with
RoPE math, without allocating padding or inventing recurrent updates. Accumulate
one trial's gradients at a time and take one update from their mean. Distant
examples must not receive extra weight accidentally through both loss scaling
and boundary-adjoint accumulation. Optional distance weights are normalized to
mean one; equal weighting is the initial experiment.

## Separate contracts

1. **Existing causal execution:** whole and split prefill can select different
   floating-point kernels. Record the native schedule. Cache restore must match
   fresh execution under that same schedule. Native-versus-PyTorch parity is a
   different measurement.
2. **Independent Writer/Reader graph:** each Writer chunk is computed at local
   positions starting at zero. Reader attention composes immutable occurrences
   `(chunk_id, origin)` using one shared softmax. It is a different computation
   from full-causal prefill. Gradient equivalence means splitting the backward of
   this same graph, not matching gradients of ordinary concatenated text.
3. **First bounded native backward:** only an output-head LoRA is differentiated.
   Its hidden features come from native Wen and remain frozen. This exercises a
   real native loss/update/export path without claiming backward through Wen's
   24 recurrent layers. It is not a Writer adapter or proof of chunk composition.

Wen has eight attention layers and 24 recurrent/convolution layers. Rotating
attention keys neither relocates nor combines recurrent states. A multi-chunk
Reader needs an explicit recurrent-state contract (for example, fresh Reader
state, with chunk content exposed only through the eight attention memories).
That changes model behavior and needs its own implementation and training.
A one-prefix native sparse-position continuation retains that prefix's recurrent
state; it must not be labelled the multi-Writer composition implementation.

## Positional and cache invariants

- Freeze the complete YaRN schedule, rotary layout/dimensions, attention gain,
  model, tokenizer, Writer adapter, cache precision and execution identity.
- For fixed phase frequencies, use `K_global = R(origin) K_local` or the
  equivalent `Q_local = R(-origin) Q_global`. Values remain unchanged.
- Relocation applies the **unit phase rotation**. YaRN's attention gain may
  already be present in cached Q/K; multiplying it again breaks composition.
- Partial RoPE leaves unrotated channels unchanged. Text MRoPE uses the proper
  model layout; the small reference supports split-half and interleaved layouts.
- Logical positions determine causal visibility. A gap contains no token and
  consumes no cache entry. Large logical distances do not establish learned
  long-context behavior or make positional floating-point error vanish.
- The same chunk can occur at multiple origins. Its Writer adjoint is the sum
  of inverse-rotated key adjoints and value adjoints across occurrences/trials.
- No shared chunk is mutated. Writer updates require a new chunk identity and
  lazy regeneration on the next request; old cache bytes are never relabelled.

## Backward policy and acceptance

F32 rounding is discrete. The first output-head backward explicitly uses the
smooth LoRA/softmax derivative and treats intermediate F32 rounding as identity.
It does not differentiate the frozen BF16 base and does not supply a derivative
for an integer quantizer. Future quantized/trainable paths require an explicit
STE or another approximation, validation and a separate identity.

The native backward implementation must pass small FP64 finite differences and
appropriate adjoint tests. On native Wen, use predeclared finite-difference step
sizes, report precision sensitivity, then measure actual native loss before and
after an accepted step. Do not call a loss reduction proof of gradient correctness.

Check zero-adapter behavior and traced/untraced forward agreement. Preserve and
reload immutable GGUF adapter bytes, rerun native evaluation, and verify that the
improvement survives. Evaluate held-out examples and preservation cases without
using their answers or gradients for optimization. Report full BW task outcomes
separately from a selected-token loss or syntax repair.

Backward through earlier adapters requires all downstream operations, including
recurrent updates, causal convolution, normalization, attention and masking.
No unsupported operation may silently detach gradients. Synthetic boundary
gradients are deferred until exact split gradients are established for the
implemented graph. No online HTTP training or automatic adapter promotion is
enabled by this experiment.

## Files and checks

- `math_reference.py`: inspectable NumPy/FP64 relocation, shared-softmax attention,
  cache identity, LoRA and explicit vector-Jacobian products. Not the Wen runtime.
- `test_math_reference.py`: finite differences, inverse-query versus explicit-key
  relocation, repeated-chunk gradient accumulation, causal masks, stale-identity
  rejection and a per-chunk-softmax counterexample.
- `head_backward.cpp` / `head.py`: portable C++ output-head derivative and ctypes
  binding, GGUF adapter export/readback and small-operation gradient checks.
- `native_probe.cpp`: fresh native Wen contexts and actual native head inputs and
  scores; bounded physical inputs with explicit sparse logical positions.
- `build.py`: build the isolated probe against the exact installed llama pin.
- `probe.py`: bounded native-process lifetime and explicit sparse positions.
- `run_guidance.py` / `bw_validation.py`: first run the already observed BW cases
  with and without a light procedural hint, then test logical gaps. Validate
  actual tool proposals/file effects in isolated workspaces. These scripts take
  private owned cases externally and do not bundle their answers in Git.
- `prepare_bw_boundaries.py`: derive one missing-brace target from each actual
  failed writing receipt, plus finish-valid-JSON counterexamples. Keep original
  word-count failures visible; adding a brace is not a prose repair.
- `run_boundary_training.py`: native directional checks, equal averaging over
  32 independent distances, native-loss line search, immutable adapter export,
  held-out/preservation evaluation and fresh-process reload. This selected-token
  CE experiment is not self-OPD and is not full BW task training.
- `verify_split_autograd.py`: optional independent CPU PyTorch-autograd oracle
  for the same small 32-distance graph; no native forward is replaced by PyTorch.

Run the CPU algebra checks:

```sh
cd experimental/wen_native_training
python -m unittest -v test_math_reference.py
```

Build and check the portable backward on a Linux host with the pinned runtime:

```sh
python build.py --llama /path/to/llama.cpp --output /path/to/private-build
python head.py /path/to/private-build/libwen-head-backward.so
```

## Evidence status

Eight CPU algebra/finite-difference tests pass on Windows and Linux. The initial
native probe compiles against the pinned CUDA runtime. Its portable head backward
passes small-operation finite differences: maximum absolute A/B gradient errors
are approximately 4.57e-8 and 1.67e-8; the zero-B control passes. These tests do not
qualify native Wen, the whole recurrent backward, million-position inference or
held-out BW learning. Native forward/training qualification is pending.

The first native light-guidance run on A100 uses one fixed YaRN factor of four,
8,192 physical context slots, FP16 KV/Flash attention and 64-token prefill chunks.
The traced and untraced scores are bit-identical. Three preservation cases pass.
The four known writing cases fail both without and with the light procedural
hint (0/4 in each group), still omitting the outer JSON brace. Word counts are
checked separately and also remain a concern. This is not a newly discovered
out-of-distribution skill; it reproduces observed BW behavior under this
experimental profile. No training occurred in that guidance run.

The conceptual references are [RoFormer](https://arxiv.org/abs/2104.09864),
[YaRN](https://arxiv.org/abs/2309.00071), and
[Gated Delta Networks](https://arxiv.org/abs/2412.06464). They motivate the
experiment; they do not certify this implementation.
