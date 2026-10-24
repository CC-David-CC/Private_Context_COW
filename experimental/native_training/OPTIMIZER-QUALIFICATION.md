# Native AdamW and Muon qualification — 2026-10-24

Native Muon now exists in **PCC's experimental trainer**, with native AdamW as a
control and auxiliary optimizer. Both use C++/GGML on CPU/CUDA; no PyTorch is
imported or linked. The subsequent real MiniCPM pilot now passes full-weight updates, fresh reload
and completed Muon continuation. **No learning winner or accepted candidate is
established.** SGD remains the default path.

| Component | Responsibility and current status |
| --- | --- |
| PCC experimental/native_training | Native forward/backprop, AdamW/Muon updates, immutable candidate export and completed optimizer state. Mechanics qualified below. |
| PCC modified Strata/COW serving | Inference and pinned context/version identities. No optimizer runs during ordinary inference. |
| BW | Existing task ledger and lesson controls. The new Muon settings are not exposed through its CLI. |
| S³ | Skills, practice/evaluation separation and acceptance/retention tests. Evidence is recorded; the MiniCPM Muon learning campaign has not run. |

## Implemented mechanics

`native_optimizer.h` builds explicit native F32 update graphs. The Muon matrix
momentum and finite quintic projection follow the pinned
[creator reference](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806).
The declared F32 implementation is not bitwise equivalent to its BF16 kernel.

Hidden Q/K/V, attention output and FFN projection matrices use Muon. Embeddings,
prediction head, normalization gains and other auxiliary roles use AdamW.
A one-row projection is still a matrix; its role, rather than a dimension-size
shortcut, determines grouping. FFN gate projections and scalar/gating parameters
are different roles. The inspected MiniCPM architecture has no mHC gates.

The standard scaling is `sqrt(max(1, rows/cols))`. The separate RMS-scaled option
uses `0.2 * sqrt(max(rows, cols))`; decoupled decay and shape scaling are established
prior work in [scalable Muon](https://arxiv.org/abs/2502.16982), not a new invention.
All coefficients, parameter orientation, precision, rates and decay are recorded.

The explicit `fast-refine-v1` experiment runs the fast Jordan quintic iterations,
then, from a declared global optimizer step, divides by 1.5 and performs a fixed
number of cubic `1.5X - 0.5(XXᵀ)X` refinements. This is a defined experimental
schedule, not an implementation of Polar Express or evidence of better learning.
The [comparison plan](../../docs/OPTIMIZER-COMPARISON.md) keeps broader ablations
as stretch goals and prioritizes frozen-test outcomes at a fixed compute budget.

## Measured qualification

[Results, fingerprints and retained discrepancies](../../validation/native-optimizer-20261024/results.json).
The CUDA checks ran on an RTX PRO 6000 Blackwell. These are not A100 results or
a hardware-generation comparison.

| Gate | Evidence and limit |
| --- | --- |
| Independent optimizer math | Six cases, 4,264 scalar comparisons per backend, covering AdamW, momentum/Nesterov, rectangular/zero/rank-deficient matrices, one-row/column matrices, decay, scaling and refinement. Worst error: CPU 8.40e-8; CUDA 6.76e-8. |
| Independent oracle | NumPy FP64 AdamW and SVD-based scalar singular-value recurrence. It does not copy the native Gram-matrix multiplication graph. |
| Ordinary complete tiny-model training | AdamW, Muon, RMS-scaled Muon and refinement all pass full backprop with batches of two then one. |
| Completed save/resume | All four arms produce bit-identical final weights and moments to uninterrupted training on each backend, including the refinement transition across resume. |
| State refusal | Fourteen reuse/identity/configuration/corruption/coverage/truncation/trailing/nonfinite checks per backend. No automatic repeated update. |
| Existing route | Tiny ordinary-Llama SGD checks, six negative cases and twelve offline admission/version checks still pass. |
| Actual MiniCPM forward | A lazy independent FP64 calculation checks all positions on two inputs of ten and eight tokens: 2,350,080 logit values. Maximum errors 3.94e-5 and 4.45e-5; last-position top tokens agree. This is forward-only evidence; actual-model update evidence follows below. |

## Subsequent real MiniCPM optimizer pilot

A bounded S³-owned pilot now runs three independently checked practice examples
through this qualified native trainer on the RTX PRO 6000 Blackwell. It passes
finite/nonzero gradient coverage for all 381 parameter tensors, fresh Muon and
AdamW updates, immutable candidate/state export and fresh-process candidate
reload. Muon uses 294 hidden matrices plus 87 auxiliary AdamW tensors. Ordinary
complete backprop is used; this does not exercise the experimental sectionwise
route.

After two updates, resumed versus uninterrupted Muon matches every weight and
moment bit for bit: 381 weight tensors, 468 moment tensors and 5,568,460,800
scalar values. Candidate/state hashes differ because their lineage headers
are different; complete arrays, counters and configuration were compared.
Reload-loss errors are at most 1.20e-7. All owned GPU jobs are settled.

Muon state is 12.21 GB versus AdamW's 20.13 GB, but optimizer-only update time
is about 2.63 seconds versus 0.061 seconds on this small batch. The pilot used
0.16943 Blackwell allocation-hours; checkpoint I/O is substantial. It is not a
fixed-compute learning comparison, and lower training loss is not acceptance.
No protected behavioral scores or promotion were produced. RMS-scaled and
fast-refine actual-model learning tests remain pending.

[Runnable S³ integration, practice TXT and full results](https://github.com/CC-David-CC/sss/blob/feat/minicpm-optimizer-pilot-20261024/docs/MINICPM-OPTIMIZER-PILOT.md).

## Additional A100 qualification

The same native trainer and optimizer/state headers now pass on an A100-SXM4-
80GB with a separately built strict F32 CUDA backend. Six independent tiny
optimizer cases cover 4,264 scalar comparisons, worst error 8.40e-8. All four
tiny full-backprop/save-resume arms and 14 negative state checks pass. An actual
MiniCPM 120-token gradient-only probe covers all 381 parameters with finite,
nonzero gradients; it performs no real-model update. This is coverage evidence,
not an independent full-model derivative-value comparison.

The owned GPU phases used 0.00967 A100 allocation-hours on a shared card. The
owner permits admission based on free VRAM despite active shared compute;
contention remains part of the evidence. This is not a controlled performance
comparison against the separate Blackwell pilot. All these owned jobs settled.
The larger S³ set now passes actual token/lesson admission, with 379-token
maximum training inputs and separate 96-case-per-leaf evaluation splits.
[Qualification, checked dataset and remaining fixed-budget gates](https://github.com/CC-David-CC/sss/blob/feat/minicpm-fixed-budget-20261024/docs/FIXED-COMPUTE-PREPARATION.md).

## Why an earlier comparison differed by 0.011

The earlier independent inference comparison failed the fixed 0.0002 maximum
absolute / 0.00005 RMS contract. Switching attention/cache precision to F32
changed the errors, but left maximum differences of 0.01019 and 0.00447.
Those original failures remain in the evidence.

A controlled output-selection experiment isolates that remaining discrepancy.
With the same binary, GPU, weights and token IDs, requesting every position's
logits rather than only the final position changes maximum logits by 0.01019
and 0.00447. The inference graph can prune earlier positions near its output;
that changes matrix execution shapes. With full-output selection and the same
whole-input prefill schedule, inference agrees with training to **1.94e-5 and
2.26e-5**, below the original limits. Identical fresh repeats are exact in the
measured modes. Changing the prefill schedule can produce still larger drift.

This identifies an output-selection/execution-shape effect on these inputs.
The exact low-level kernel algorithm has not been profiled. The probability
changes were small here (default/full-output total variation about 0.00055 and
0.00029), but same argmax alone does not establish equivalence on arbitrary tasks.
**130,560 is the vocabulary size; these were not 130K-token contexts.**

Keep the existing tolerance for the matched-profile qualification. Do not treat
it as a universal error limit across different output selection, batching,
precision or hardware. No tolerance was widened to turn the original failures
into passes. BW's remaining verification/termination failure is a separate issue.

## Reproduce the bounded checks

Build the separate training binary as described in [README](README.md), then use
new private directories:

```sh
python -m experimental.native_training.test_optimizer \
  --executable /path/to/pcc-native-train --output /new/math --backend CPU
python -m experimental.native_training.test_optimizer_training \
  --executable /path/to/pcc-native-train --output /new/continuation --devices CPU
```

For an admitted CUDA run use `CUDA0` instead of `CPU`, a qualified FP32 native
backend, recorded physical device ownership/headroom and the GPU lease. GPU
indices refer to the explicitly selected visible-device list. No silent CPU
compute fallback is admitted in the native graphs.

The independent fresh oracle's default remains final-position output. Its new
explicit `PCC_ORACLE_ALL_LOGITS=1` control requests every position during decode,
while retaining the existing returned-last-vector protocol. Use a separately
fingerprinted oracle build. For the F32 comparison also set
`GGML_CUDA_F32_ACTIVATIONS=1`, `NVIDIA_TF32_OVERRIDE=0` and
`COW_QWEN35_F32_ATTENTION=1`. Consume the same token IDs in the same prefill chunks;
never reinterpret a saved state under another execution profile.

The training evaluate-only request can export all logits with `logits_output`.
Compare those retained artifacts against the lazy independent FP64 implementation:

```sh
OPENBLAS_NUM_THREADS=8 python -m experimental.native_training.probe_llama_forward_fp64 \
  --model /path/to/model.gguf --requests /private/request0.json /private/request1.json \
  --native-logits /private/logits0.f32 /private/logits1.f32 --output /new/fp64-proof
```

Use F16/F32 ordinary-Llama inputs; this oracle is forward-only. It reads one
tensor at a time as FP64 rather than expanding the entire model in RAM.
Retain the exact request, source/build/model hashes and original diagnostics.

## Offline update request and state contract

The adaptive route is explicit ordinary-Llama **full-weight** training with
`sequence_profile="dense-llama-short-v1"`. Other architectures, LoRA optimizer
use and hardware routes need separate qualification. Select `optimizer="adamw"`
or `"muon"`, and `optimizer_state_mode="fresh"` for a new optimizer. A minimal
Muon settings object is:

```json
{"schema": 1, "precision": "native-f32-v1", "momentum": 0.95,
 "nesterov": true, "ns_steps": 5, "projection": "jordan5",
 "scaling": "original", "weight_decay": 0.0,
 "aux_learning_rate": 0.0003, "beta1": 0.9, "beta2": 0.95,
 "epsilon": 1e-8, "aux_weight_decay": 0.0}
```

Set the main `learning_rate`, examples and resources in the existing offline
operator request. These are mechanics defaults, not tuned MiniCPM settings.
For refinement explicitly select `projection="fast-refine-v1"`,
`refine_start_step` and `refine_steps`. Rates, schedules and grouping must remain
fairly budgeted and frozen for the learning comparison.

The helper writes a separate candidate and `optimizer-state.bin` into a new run
directory. State includes moments, global step, tensor roles/coverage, canonical
configuration, ordered-example identity and cursor, and binds to the candidate's
SHA-256. A completed continuation pins that candidate as `model`, its hash,
`optimizer_state_mode="resume"`, `initial_optimizer_state`, its
`initial_optimizer_state_sha256` and the exact `example_cursor`. The loader
refuses mismatched data, configuration, model, cursor or state bytes.

This is completed-update continuation, not mid-step transactional recovery.
A failed/uncertain run is preserved, never blindly rerun. Model/adapter/context
identities remain separate, and no successful loss or mechanics test promotes a
candidate. State follows the selected block owners, but multi-device optimizer
qualification, within-matrix gathers, live offload and compute-efficiency claims
remain outstanding. The next gate is an actual-model gradient/update/reload pilot,
then the fixed-budget S³ learning/retention comparison.


## Completed-update training budgets

The ordinary-Llama full-update route now supports an opt-in
`training_budget_seconds` below `deadline_seconds`. Its steady training-phase
clock starts after model/optimizer load and state restoration, and includes
complete gradient/coverage computation, optimizer work and post-update batch
loss checks. Loading, candidate/state serialization and operator reload are
outside that clock. The hard native deadline still bounds the job. At least one
whole update completes; the receipt exposes elapsed time, remaining/overrun time,
completed updates, step cap, stop reason and per-update phase timings. Overrun is
bounded by the last completed update's span, not silently discarded.

The Python helper requires the executable's `--capabilities` contract before
launch and validates returned budget/cursor/state accounting. Default unbudgeted
receipts retain their prior shape, including SGD's omitted-step default. This is
completed-state continuation, not mid-update transactional recovery. No budget
or loss receipt promotes a candidate.

CPU and CUDA checks pass exact tiny AdamW/Muon resumed-versus-uninterrupted
weights/moments, global refinement continuation, partial batches, step caps and
19 refusal checks per backend. [CPU evidence](validation/completed-update-cpu-budget.json),
[CUDA evidence](validation/completed-update-cuda-budget.json). CUDA allocated
3.73 seconds on the dedicated Blackwell GPU; this is not a shared A100 timing.
The new time-budget option has not yet been qualified on A100, HIP or SYCL.

```sh
python -m experimental.native_training.test_training_budget \
  --executable /path/to/new/pcc-native-train --output /new/budget-proof \
  --devices CPU --old-executable /path/to/previous/pcc-native-train
```

For CUDA, use the qualified FP32 backend and the existing owner lease/admission
procedure, with explicit physical-to-visible device identity. The optional old
executable verifies refusal before it can ignore a budget. The tiny test's
1e-12-second budget deliberately exercises the first whole-update boundary; it
is not a recommended training setting.

S3's [current MiniCPM baseline and fixed-budget study](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/MINICPM-NATIVE-LEARNING.md)
uses this control for four full-update candidates, with held-out learning and
retention measured independently of optimizer mechanics. The earlier next-step
text above is historical; real-model update/reload mechanics are already qualified.
