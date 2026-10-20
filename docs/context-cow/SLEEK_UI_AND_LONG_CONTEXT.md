# Compact interface and cached long-context tests — 2026-10-13

The live interface now defaults to icons. Fork uses a splitting arrow; rewind
uses a curled arrow. Tooltips appear after 850 ms of hover or keyboard focus.
The book button toggles teaching labels. Archive is an × on each chat row;
archived chats remain in SQLite and can be restored from Show archived.

UI checks passed **20/20 in each of Chromium, Firefox and WebKit**. Storage and
HTTP tests passed **20/20**, including forced checkpoint suppression for an
archived request even when a client asks to save a checkpoint. The previous
383-method regression run predates this compact-interface change.

## Caches are enabled on the live review server

- Six prompt checkpoints, an interval of 1,024 tokens.
- An 8,192 MiB conversation cache with eight slots.
- Ordinary batched prefill, chunk limit 4,096; Q8_0 model, FP16 KV.
- llm-60, RTX PRO 6000 Blackwell; 40 GiB CPU expert budget and automatic GPU expert cache.

Archive disables saving new conversation checkpoints when an archived chat is
used again, through the existing `strata_checkpoint: false` / `ckpt=0` primitive.
Previously parked entries remain subject to native cache eviction. There is no
per-chat selective native cache deletion API; archive does not claim an immediate
release of those entries. No native C++ source was changed for this UI update.
An actual archived-chat GPU request completed with its text saved and effective
`strata_checkpoint: false`, despite the client sending `true`; its receipt and
native binary fingerprint are retained in `archive-native-check.json`.

## Real 1K and 4K cases

The fixture creates a saved reference, forks A and B from its completed answer,
adds about 512 tokens of context to each, requests up to 128 output tokens,
compares each with ordinary serving of the same history, and rewinds/retries A.
Actual prefix prompt sizes were **1,047 and 4,122 tokens**, including template
overhead. All twelve requests completed. Shorter answers are retained honestly.

| Prefix | Call | Prompt / cached tokens | Outputs | Wall seconds | Decode tokens/s |
|---|---|---|---|---|---|
| 1K | A | 1,585 / 1,048 | 106 | 2.179 | 71.8 |
| 1K | B | 1,585 / 1,040 | 121 | 2.007 | 101.8 |
| 4K | A | 4,660 / 4,123 | 128 | 1.924 | 101.5 |
| 4K | B | 4,660 / 4,115 | 105 | 1.931 | 104.2 |

Cold prefix preparation took 1.599 seconds at 1K and 1.817 seconds at 4K, with
zero reported cached tokens. At 4K, A consumed 537 fresh prompt tokens and B
consumed 545. Rewind/retry A took 1.411 and 1.439 seconds, respectively.
These are bounded serial whole-serving measurements, not an isolated GPU COW
benchmark or a comparison against a cache-disabled baseline.

## Exact agreement failed

All branch-path checks passed: shared node IDs, separate suffixes and unchanged
Main. All **six initial-output equality checks failed**: A and B versus their
same-history serving controls, and rewound A versus its original answer, at both
sizes. The campaign status is **FAIL**, not a numerical or gate PASS.

To separate UI replay construction from engine reuse, a fresh server process
received the exact same 4,660-token HTTP payload three times. The first request
reported zero cached tokens; each repeat reported 4,653. **Cold versus warm output
differed, while warm versus warm matched.** The payload SHA and full responses are
retained. This places the difference below the branch UI, but does not determine
whether it is numerical sensitivity to different prefill partitions or a state
restore defect. No full-logit oracle was executed; logit error is unmeasured.
Frozen equations and tolerances were not changed. Caches remain enabled as requested.

Raw reports, failures and screenshots:
[sleek-workspace-20261013](results/sleek-workspace-20261013/).
Runners: `tools/test_workspace_long_context.py` and
`tools/check_cached_payload_repeat.py`. The latter requires the server to be
restarted before its first inference request, and verifies zero initial reuse.
Run with the verified host interpreter and supply the key through `STRATA_API_KEY`.

Current live Python PID: `1291435`, loopback port `8097`; browser tunnel
`http://127.0.0.1:18097/`. Source and private configuration remain under
`/home/dflanag3/branching-g3/web-review-20261013`. The long test chats are archived
in SQLite, not deleted. Native GPU COW handles remain separate from this interface.
