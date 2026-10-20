# Full-vocabulary Min-P milestone: worker interface

User requested actual Context COW support for temperature0.70, min_p0.05,
top_k0, top_p1, standalone and combined with GBNF; keep greedy working.
One request per GPU. Q4 routine tests, Q8 final bounded spot check.

Keep one canonical immutable history/controller representation. New controller
schema/capability must be explicitly versioned; schemas1/2 remain readable and
their default greedy behavior unchanged. No ordinary native CUDA sampler top64
shortlist is acceptable as unrestricted Min-P-only.

Sampling semantics v1: legal support -> Min-P -> temperature -> categorical.
This follows existing Strata's Min-P-before-temperature order with top_k/top_p
disabled. Min-P compares exp(raw_i-max_legal) >= min_p. Implement the inclusive
logit cutoff raw_i >= max_legal + log(min_p), in binary64. min_p0 retains all
legal tokens. Always reject nonfinite raw scores, including illegal positions.
Among retained support calculate binary64 exp((raw_i-max_legal)/temperature),
sum in increasing token-ID order, and select the first cumulative weight greater
than u*total. If rounding hits the total, return the last positive-weight legal
token. Never select an underflowed zero-weight token. Positive finite temperature
required for Min-P path; greedy remains its existing separate selection policy.
min_p finite in[0,1]. No repetition penalty/adaptive sampling/topK/topP in v1.

Uniform source: existing hmac-sha256-u53-v1, branch stream_key/cursor. The host
draws an exact integer draw53 in[0,2^53), passes it to the native sampler, and
updates canonical draw progress only with a successfully consumed/published
boundary. Native performs no hidden RNG draw. Sampling selection is read-only
on native model state; consumed token enters through verified decode operator.
Same stream/cursor/profile produces a reproducible draw; different streams may
produce different answers. No cross-model or cross-batch equivalence claim.

Native protocol: optional capability sampling_protocol=full-vocab-min-p-u53-v1.
New command: sample_mask EXPECTED_CONSUMED TEMPERATURE MIN_P DRAW53 MASK_PATH.
Numeric parsing must reject trailing junk/nonfinite/out-of-range values.
MASK_PATH follows select_mask's quoted-path convention; inspect actual driver.
Result statusok, operatorfull-vocab-min-p, protocolfull-vocab-min-p-u53-v1,
token, consumed, logits_count, mask_bytes, legal_count, kept_count, draw53;
optional total_weight diagnostic. Counters separate sampling calls, scored
logit/mask bytes and timing. Existing greedy counters/protocol/defaults unchanged.

Root owns integration in controller/grammar_first/real/strata_backend and new
cross-layer tests. Worker reference owns only new min_p.py and its new tests.
Native worker owns only new min_p_select.h, new builder, new native tests. Each
works in its own worktree; no one launches GPU jobs except root. CPU/builds on
llm-60 only, Windows is control/source/Git. Use verified remotevenv, external
artifacts and wrapper receipts, frozen sources, test-first RED/GREEN. Keep hashes.
No registered tests/spec equations/tolerances may change. No commits/push by
workers until root review; all eventual publication private box-git only.
