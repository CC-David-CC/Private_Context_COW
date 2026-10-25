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
