# Native Muon comparison plan

Status: mechanics implemented, 2026-10-24. Native AdamW and Muon now pass bounded
CPU/CUDA math and completed-state continuation checks. SGD remains the default.
[Qualification and actual-model forward evidence](../experimental/native_training/OPTIMIZER-QUALIFICATION.md)
separate that implementation from the planned fixed-budget learning comparison.
No real MiniCPM weights have been updated or accepted.

## Main question and priority

At a declared, equal compute budget, how does ordinary full-weight Muon training
perform on the frozen S³ tests, compared with the controls and defined variants?
Finish publishing current evidence and investigating BW's shell failures first.
Then qualify native optimizers and actual MiniCPM math, train candidates and
measure independent skills, composition, transfer and mixed-review retention.
A broader AdamW ablation study is a stretch goal. It must not displace the main
Muon work. No candidate is automatically promoted and no model upload is implied.

## Define the arms before running

- Native AdamW provides a qualified optimizer control and the auxiliary optimizer
  needed by Muon. A large AdamW parameter ablation is outside the first comparison.
- Pin the [creator's Muon reference](https://github.com/KellerJordan/Muon) at
  `f98f1cacc0263b04290753e32be8d498c1efc806`: momentum, declared Nesterov choice
  and finite Newton–Schulz steps. Record the exact equations, coefficients,
  precision, epsilon and parameter orientation in every run.
- Separately identify scalable Muon with decoupled weight decay and shape/RMS
  update scaling. Those are established features in
  [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982), not a new
  contribution here. Do not label two equivalent implementations as new arms.
- Define the requested fast-to-refine variant with explicit early/late polynomial
  coefficients, iteration counts and transition rule before seeing evaluation
  outcomes. [Polar Express](https://arxiv.org/abs/2505.16932) is relevant prior
  work on efficient accurate polar approximation. More exact orthogonalization
  is a hypothesis to test, not a guarantee of better learning.

Muon transforms matrix momentum using an approximate polar update. AdamW's
adaptive scaling is coordinate-wise, but the gradients themselves remain coupled
through the model. Neither description establishes a universal quality advantage.
LoRA is a parameterization, not an optimizer: a full-weight Muon versus LoRA
comparison changes more than one variable and must be labelled accordingly.

Use role-aware parameter groups. Embeddings, prediction head, biases, norm gains
and scalar/gating parameters use explicitly configured auxiliary AdamW unless a
separate qualified arm says otherwise. Do not classify every 2D tensor as a
hidden-layer matrix. MiniCPM's inspected architecture has no mHC gates; support
for such roles in other architectures cannot be inferred from this model.

## Budget and fairness

Freeze initial weights, tokenizer, practice examples/order, masks, harness
versions and independent tests. Permit explicit thinking-target and answer/action
masks as distinct configurations. Tune numerical learning rates within equal,
small declared budgets rather than assuming equal numeric rates are equivalent.
No different-seed model-repeat campaign is authorized for the current study.

Report training steps, examples, selected tokens, wall time, allocated GPU-hours,
optimizer time, communication, memory and contention. Separate training cost from
evaluation and diagnostic cost; report both totals. Stop arms at predeclared
budget checkpoints and compare the same frozen tests. Missing/infrastructure
results are not model failures. A faster kernel alone is not a learning gain.

Shared A100 throughput reflects contention. Use the dedicated LAN GPU for claims
as small as 1% compute savings, with repeated timing samples and uncertainty.
Fixed-seed timing repetition is distinct from different-seed model training runs.
Use approved devices only, preserve other jobs and account for all reserved time.
State sharding, full-matrix momentum gather/scatter and disk offload must include
their transfer/stall costs; aggregate free VRAM is not a unified memory pool.
The [Gram Newton–Schulz project](https://github.com/Dao-AILab/gram-newton-schulz)
is relevant implementation research, but its advertised kernel hardware/software
requirements do not establish an A100 speedup for this native engine.

## Geometry measurements and acceptance

Higher effective rank is a hypothesis about useful learning. Distinguish ranks of
weights, updates and held-out activations. On declared matrices/checkpoints,
record singular values or a bounded diagnostic and explicitly define:

- stable rank: `sum(sigma^2) / max(sigma)^2`;
- energy-entropy rank: `exp(-sum(p * log(p)))`, with
  `p = sigma^2 / sum(sigma^2)`;
- participation rank: `sum(sigma^2)^2 / sum(sigma^4)`.

Define zero-matrix handling and activation centering, normalize by available rank
where appropriate and report diagnostic overhead. Use exact SVD/polar references
on small matrices; expensive full-model SVD is not required at every step. A
high-rank noisy update is not evidence of improved capabilities. The proposed
“parallax” idea has no mathematical definition yet and is not an implemented arm.

Before real updates, independent small-matrix oracles must check momentum,
Nesterov, decay, shape scaling, orthogonalization, role grouping and finite
precision, including rectangular/zero/rank-deficient inputs. Persist step counters,
moments, exact config/data-order/model identity and verify save/resume against
uninterrupted updates. Uncertain completion never triggers a blind repeat.

Candidate selection belongs to S³'s frozen independent/composed/transfer and
retention gates. Report gains, regressions and unresolved results separately.
Preserve the incumbent if those gates do not pass. Improving a BW shell harness
must be evaluated with unchanged weights before attributing a repair to training.
