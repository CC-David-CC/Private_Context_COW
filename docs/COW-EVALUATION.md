# Shared COW execution for S³ evaluation and BW

`tools/pcc_cow_evaluation.py` connects bounded evaluations to the existing COW
`BranchingModel` and native dense backend. It owns no scorer, training policy,
task ledger, alternate cache, or alternate sampler.

Both this adapter and BW's COW coordinator call `run_greedy_request`; on dense
MiniCPM it delegates to the existing consumed sampled/state request. That
implementation owns native selection, one live decode context, stop handling,
controller state, checkpoint restoration and immutable snapshot publication.
The adapter uses `create_with_reuse` for the root and evicts RAM materializations
between independent benchmark cases. It does not modify COW's normal retained
context, fork, rewind or recovery behavior.

The caller must hold the existing COW GPU lease and supply a pinned profile.
The optional S³ adapter does this through `tools.cow_server.gpu_locks`, observes
the selected physical GPU again under the lock, and checks memory/utilization
limits before loading. It records admission before native loading so a failed
load remains allocated work in S³'s settlement.

The profile's source closure must include this adapter and the existing COW
implementation. Model, native binaries, numerical settings and source files
are checked; sources and model weights are checked again at close. Every
request verifies that the returned snapshot consumed the complete prefix and
all sampled output, including EOS. Its counters must agree with that boundary.
A failed request poisons the evaluation session; it cannot be silently retried.

S³ owns the CLI and scoring records. See its
[usage and identity guide](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/COW-EVALUATION.md).
BW continues using its existing COW client and workflow ledger; no second BW
execution engine or inference dependency was added to S³'s importable core.

## Actual validation

- Six focused PCC adapter tests cover the consumed EOS/length boundary,
  invalid inputs, bad native receipts, refusing retries after failure, source
  corruption and an incomplete or out-of-root source closure.
- Eight S³ adapter/reference tests cover exact model/GPU/policy binding,
  multiline output, original stop tokens, infrastructure errors and runtime
  comparability across candidates while preserving math-policy changes.
- The real shared-A100 F32 probe passed 18 full-vocabulary frontier comparisons
  on two owned practice prefixes plus eight continuation tokens each, within
  the existing maximum/RMS bounds of `2e-4`/`5e-5`. All selected tokens matched.
- Saved-state restoration was bit-exact. Canonical COW requests repeated with
  identical tokens and no prior-case tail contamination.

The full numerical report is
[cow-evaluation-20261024.json](../experimental/native_training/validation/cow-evaluation-20261024.json).
Its small forward-work timings are not a production-serving benchmark or an
end-to-end training-loop measurement. Shared-GPU contention affects timing.
Full frozen 864-case behavioral comparisons are a separate assessment.

This first connection qualifies dense MiniCPM F32 at a 512-token context, not
all models, precision formats, long contexts or attached adapters. The native
fresh-context oracle remains independent and unchanged as the reference.

## Shared roots in the evaluation adapter

`CowEvaluationSession.prepare_shared_prefix(tokens)` materializes one existing
COW root before any request. `complete()` then verifies each question extends
that input, calls the same consumed request with its suffix, and retains only
the shared root between questions. Native work counters include root setup
separately from suffix/decode work. Closing or poisoning a session evicts the
remaining materializations. There is no second cache or sampler.

Nine PCC adapter tests now cover the original boundary/failure checks and this
wiring. S3 owns prefix selection, grading, runtime policy and optional case
partition/merge. Existing COW branch proofs are reused rather than recertified.
The actual 864-case shared-root baseline and migrated owner continuations are
running; no completed speedup for that new mode is claimed yet.

The full earlier COW path matched every sampled token for both original and
trained F32 MiniCPM. Observed evaluation ratios were 6.1x (original) and 2.7x
(AdamW update 1); shared loads and different cards limit timing conclusions.
See [the graphs and evidence in S3](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/COW-FIRST-DATA-BLOCK.md).

## Allocation audits

`ResourceBudget.reconcile()` accepts pinned private owner evidence for an
already settled unknown allocation. It preserves the original settlement and
appends an idempotent adjustment, under the existing budget lock. It cannot
change active allocations, measured settlements or the campaign ceiling.
The owner must establish actual allocation from original phase/process evidence;
the API verifies evidence identity, not its interpretation. Tests cover retained
history, changed evidence, duplicate/conflicting audits and invalid allocations.

## Bind a candidate profile before loading it

Owner drivers can use 	ools.pcc_cow_profile.bind_dense_profile()` to bind a
new dense FP32 model artifact and an observed GPU to a copied execution profile.
It preserves the template and its numerical policy, updates the weight contract,
physical placement and execution fingerprint, and returns the copied profile.
It does not select a GPU, grant authority, load weights or qualify a build.
Normal COW source, artifact and memory checks still apply.

An integration driver accidentally used the compact S3 record digest to generate
COW execution fingerprints. The existing native validator rejected three such
assessments before model loading, with zero model calls. Those engineering
failures and their original checkpoints were retained. The corrected drivers
reuse those exact native checkpoints and call the original CPU-only native
validator before starting the evaluator; no training update is repeated.

COW execution fingerprints use the established sorted JSON representation in
NativeBackend.validate_execution_profile(). S3 record identities use a separate
canonical representation. Both can be valid SHA256 values while identifying
different bytes. The new binding helper has CPU tests against that original
native validator, including changed model/GPU bindings, policy tampering and
invalid placement inputs. The validator test constructs an uninitialized backend
and calls only its metadata validation method, so no model or library is loaded.

Attached dense LoRA candidates have a separate immutable profile and an explicit
opt-in native build declaration. See [the owner workflow and qualification limits](DENSE-LORA-COW.md).
