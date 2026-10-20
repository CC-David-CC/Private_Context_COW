# Sampler extensions after DRY

DRY passed bounded real Q4/Q8 tests on2026-10-16. User authorized five separate
development branches, including starting Mirostat2 and Adaptive-P independently.
Implementation may proceed in parallel. Integration and GPU validation remain
ordered: Typical plus DynaTemp, XTC, Mirostat2, Adaptive-P. These are four waves
and five samplers; they do not replace the original G0-G5 gate requirements.

All following paths are under C:\Users\dflanag3\Documents\fleet. Each starts
at5864d386d90e1bf42af6c1014fbc9ac2b061ed7b. Private remote is
box-git:/srv/data/git/private_context_cow.git. No public/GitHub pushes.

| Sampler | Worktree | Branch | Owner and current bounded task |
|---|---|---|---|
| Locally Typical | context-cow-sampler-typical | work/sampler-typical-20261016 | Luna native: pure Python reference, separate C++ helper, CPU tests |
| DynaTemp | context-cow-sampler-dynatemp | work/sampler-dynatemp-20261016 | Luna native: pure Python reference, separate C++ helper, CPU tests |
| XTC | context-cow-sampler-xtc | work/sampler-xtc-20261016 | Root: independent selection reference and activation-draw contract |
| Mirostat2 | context-cow-sampler-mirostat2 | work/sampler-mirostat2-20261016 | Luna fixture: independent arithmetic and explicit proposed feedback state |
| Adaptive-P | context-cow-sampler-adaptive-p | work/sampler-adaptive-p-20261016 | Luna hip: complete distribution transform and deferred EMA acceptance |

At this plan checkpoint these are implementation assignments, not integrated
or GPU-qualified features. Root alone owns shared controller/backend/serving
integration; workers initially own new standalone files in their own checkout.
Never copy a live worker's mutable output into qualification. Wait for its
frozen handoff, inspect code and hashes, then combine sources in an isolated
integration worktree. CPU execution belongs on llm-60; no Windows test/build
execution. Root alone launches GPU jobs, one at a time, GPU=[0]. Chat stays off.

## Requested independent presets

| Sampler | Preset |
|---|---|
| Typical | temperature=.80, typical_p=.95, min_p=0, top_k=0, top_p=1 |
| DynaTemp | temperature=.55, dynatemp_range=.35, dynatemp_exponent=1, min_p=.06, top_k=0, top_p=1 |
| XTC | temperature=.90, min_p=.02, xtc_probability=.12, xtc_threshold=.15; unspecified ordinary filters use documented new-request defaults |
| Mirostat2 | temperature=.80, mirostat=2, mirostat_tau=4, mirostat_eta=.05; standalone feedback mode, ordinary probability truncators off |
| Adaptive-P | adaptive_target=.55, adaptive_decay=.90, min_p=.05; temperature/default-order behavior must be recorded before integration |

These are starting examples, not hard-coded allowed values or promises of
better prose. Settings remain fixed per generation and adjustable on later
partial continuations. Document inheritance and disabling explicitly.

## Arithmetic and controller boundaries

Keep old v1-v3 default arithmetic/commands unchanged. New numerical transforms
operate on private score copies, never KV or stored raw logits. Use finite
binary64 arithmetic, token-ID-order sums, deterministic token-ID tie breaks
and stable log probabilities. This is our pinned contract; it is not a claim
of bit-compatible sampling with another engine's float32 implementation.

Typical retains candidates near the entropy after legal support, DRY, Min-P
and Top-K. Retention uses strict cumulative probability greater than typical_p,
minimum one survivor; typical_p=1 is an exact no-op. DynaTemp maps entropy of
its retained support to [max(0,T-range),T+range], using the positive exponent.
One candidate means normalized entropy zero. Range zero returns T exactly.
Base T=0 preserves existing greedy/no-draw behavior. Positive base T still
uses one draw if effective dynamic temperature becomes zero.

XTC needs an explicit activation draw separate from categorical selection.
Define disabled/one-candidate cases and record draw accounting before coding
the shared controller. It must never eliminate all legal candidates.

Mirostat2 holds mu: initialize2*tau, truncate by surprise in bits, renormalize,
sample, then propose mu-eta*(selected renormalized surprise-tau). Commit that
proposal only after the selected token is consumed. Adaptive-P transforms the
distribution around its feedback-adjusted target; an EMA alone is incomplete.
Its weighted sum and total weight update from the selected token's ORIGINAL
probability only after consumption. It is a terminal selection stage after
other enabled filters, not a filter placed before a second categorical draw.

Keep RNG cursor and feedback in the one canonical controller. Never publish
pending selected-token proposals. Forks detach mutable feedback; restart
restores it exactly. No global mutable sampler and no second history.
Reject conflicting terminal samplers and unsupported capabilities explicitly.

## Test and integration sequence

1. Add failing deterministic tests, run RED on llm-60, implement the pure
   reference/native helper, and preserve GREEN plus all earlier failures.
2. Check exact boundaries, uniform/tied and extreme distributions, masking,
   singleton support, finite overflow/underflow behavior, disabled byte
   identity and input immutability. Stateful tests need forked feedback,
   JSON round trips, rejected/mismatched consumption and no pending export.
3. Integrate each reviewed feature with canonical validation, per-generation
   overrides, capability checks before work, CLI/HTTP forwarding and operation
   ID binding. Add regressions before shared behavioral changes.
4. Freeze source/build/profile/model hashes. Run combined CPU regressions and
   G0; then a tiny Q4 real-backend case before a1K Q4 campaign. Compare every
   checked frontier and complete controller with fresh matching-profile runs.
5. Test fork/retry, partial continuation, grammar masks, disk restart and zero
   saved-prefix work. XTC must prove activation and nonactivation; Mirostat and
   Adaptive-P must restart after nonzero feedback progress.
6. Validate the supported winner on Q8. Stop a failing configuration, keep the
   smallest reproducer and fix it before expanding. CPU or Q4 results cannot
   substitute for missing Q8 evidence.
7. Measure resident throughput and request/first-token latency separately from
   audit/startup costs. Report when overhead exceeds the user's8% preference;
   it does not automatically discard a correct feature. No quality or speed
   claim without the relevant measurement.
8. Commit source and compact evidence, back raw diagnostics privately to
   box-git with streamed hash/member readback, update status and this plan.

Original cross-arrangement disagreement, broad controllers/fault campaigns,
long contexts and cumulative G1-G5 qualification remain separate. Do not alter
registered tests, tolerances or reference equations to qualify these features.
