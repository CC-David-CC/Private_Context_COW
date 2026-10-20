# DRY with configurable generation settings

Implemented as an opt-in sampler extension. Bounded Q4/Q8 restart and
HTTP/CLI checks passed on llm-60; this is not cumulative G1-G5 certification.
The earlier temperature/Min-P/Top-K/Top-P baseline does not include DRY.

The requested preset is:

```json
{
  "temperature": 0.70,
  "min_p": 0.05,
  "top_k": 60,
  "top_p": 0.95,
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 2,
  "dry_penalty_last_n": 256
}
```

DRY discourages completing a sequence that has already occurred in the last
256 consumed tokens, including prompt tokens. It considers tokens in order,
not just token counts. For each candidate, it finds the longest matching
suffix preceding an earlier occurrence of that candidate. If that matching
length is at least the allowed length, its logit penalty is
`multiplier * base ** (matching_length - allowed_length)`. Multiple matches
use the maximum length; their penalties are not added. Overlapping matches
count. The prospective selected token is not already part of the history.

The fixed default sequence breakers are newline, colon, double quote and
asterisk. A token containing any of those bytes is a breaker. The latest
breaker limits the current suffix length; earlier occurrences elsewhere in
the window still count. Breaker candidates are exempt. Custom sequence
breakers are not part of this delivery. A zero multiplier or zero lookback
disables DRY. An allowed length of zero gives earlier following candidates a
zero-match minimum; longer actual matches still take precedence.

The sampling order is legal grammar support, DRY, Min-P, Top-K, temperature,
Top-P and categorical selection. With temperature zero, choose the highest
legal DRY-adjusted score, breaking ties by token ID, without drawing RNG.
DRY changes a private score copy; raw model logits and native checkpoints stay
unchanged. Finite binary64 penalties saturate at binary32 maximum, subtraction
is bounded at negative binary32 maximum, and adjusted scores round once to
binary32. This explicit arithmetic is not a claim of bit-identical sampling
to another engine.

All fields are fixed for one generation call. Changing them at a later partial
continuation preserves RNG cursor, grammar progress and output decoder state.
The CLI sends only explicitly supplied continuation fields; other fields keep
their saved values. An explicit `--dry-multiplier 0` disables the penalty.
Multiplier must be finite and nonnegative; base must be finite and at least
one; allowed length and lookback must be exact unsigned 32-bit integers.

The immutable controller stores its policy and RNG/parser/decoder progress.
Ordered repetition history is derived from the branch's canonical consumed
tokens at restore, then advanced privately after consumption. No separate
persisted repetition transcript is created. Forks share immutable definitions,
and each continuation derives its own ordered window. The native sampler
checks the supplied window against its actual consumed token IDs.

Native DRY requires `full-vocab-dry-min-p-u53-v3`. Active DRY is rejected on
older native capabilities before evaluation. Existing requests without DRY
retain the v1/v2 sampler paths and canonical policy fields. New DRY policy
fields are additive within controller schema 3; older readers reject them
rather than silently ignoring them. Execution fingerprints bind implementation
and breaker semantics to saved computational checkpoints.

The repetition rules follow the primary
[llama.cpp DRY implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)
for the four single-character default breakers. A separate brute-force Python
reference checks the native reverse-Z matcher. Native and controller tests,
real-GPU continuation checks and performance results have separate statuses;
a successful CPU fixture does not establish model or restart correctness.

Executed on 2026-10-16: CPU524/524, G051/51; Q4 128-prefix/64-suffix/8-output
and Q4/Q8 1024/64/16 campaigns passed106 full-vector checks with zero error.
Tokens/controllers matched independent references, saved prefixes were not
recomputed after restart, and CLI changed DRY strength/lookback without
resetting other settings. Natural short answers had no qualifying repeats;
separate deliberately repeated histories changed scores on both real models.
Three native choices per model matched independent Python selection, kept raw
logits/parent bytes unchanged, and respected single-token legal support.

No throughput qualification for DRY is claimed. Full GBNF-plus-DRY state
continuity is currently CPU-tested; the real-GPU legal-mask fixture is narrower.
Only Linux/CUDA was built/executed. Complete machine-readable fingerprints,
test IDs, counters, failures and private archive receipts are under
`experimental/branching_context/evidence/DRY-QUALIFICATION-2026-10-16.json`.
