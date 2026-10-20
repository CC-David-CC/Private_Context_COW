# Process-separated text continuation

Added `/cow/text/resume`: branch, revision, operation_id and max_tokens only.
It reads the original request-bound canonical token history/controller, requires
an active zero-temperature controller, and submits identical token IDs (empty
suffix). It never renders the conversation or re-encodes generated text. Native
policy validation still rejects unsupported controller features. Pending UTF-8
bytes initialize the text view; stopped/idle controllers reject continuation.

HTTP RED: resume route returned 404. GREEN: 39 supplemental CPU/real-socket
tests, 106 existing pool/control regressions and G0 51 passed on llm-60.
Machine IDs/hashes: SERVING-TEXT-RESUME-CONTROL and SERVING-TEXT-RESUME-POOL JSON.

## Executed real Q8 result

Campaign `request-stop-20261014/text-resume-q8-v2` PASS. Producer Python1697282
and native1697288 exited 0; consumer Python1697684 and native1697690 exited 0.
Each branch saved after four consumed outputs, then restored in new processes
and produced another 124 outputs with zero prefill tokens and forward work equal
only to those outputs. Durable retries returned the existing reply/receipt.
Both 128-token sequences and final vectors matched independent fresh native
references exactly. Full canonical greedy controller records matched an
independent ordered CPU fold over the fresh reference's actual token IDs/bytes.
This qualifies draw cursor 0, grammar None and the tested greedy profile only.

Executed additional byte comparisons on llm-60: saved prefix frontier and each
four-token/final frontier matched the corresponding independent full vectors.
Six checks (prefix checked against each independent reference), zero error.
Parent payload hashes, including four-token snapshots, remained unchanged.
Peak GPU concurrency was one; no skips or mocked computation. Evidence with
native build/source/model fingerprints and all counters:
`SERVING-TEXT-RESUME-Q8-2026-10-14.json`.

Resumed received generation: A 42.34 TPS, B 52.29 TPS. TTFT 7.41/6.51 s includes
loading/activation but excludes engine/model startup. These are two single
124-output measurements, not medians or a 2K-output performance qualification.
Raw request duration, native generation, capture and work breakdowns remain in
the recorded profiles. Both replies remained active at the 128-output bound.
No claim that a complete answer was generated or judged correct.

The first resume launch failed before starting native work because the fixture's
new CPU source-relevance status differed from the reader's expected CPU-only
status. It created no phase result; its unchanged log/empty output are retained
as `text-resume-q8-v1`. The corrected reader requires the explicit source-
relevance status and exact-prefix/tokenizer parity; actual inference is still
proved by independent references, never by that CPU label.

## Source relevance correction

The earlier text HTTP result (`text-q8-v2`) remains a valid numerical/performance
measurement: 128 outputs each, zero final-vector error, received 58.26/56.84 TPS,
TTFT5.65/2.98 s, durable requests14.82/14.99 s. Its input selection was poor for
code-review claims: the native-labelled 73K body began with .github files and
had zero COW symbols. The model correctly said the requested implementation
was absent. That evidence is retained, not rewritten or presented as successful
native-COW code review.

New source-prioritized fixtures put relevant native implementations/headers,
canonical Store/controller files and the frontend first. Required markers were
checked inside consumed code; ordered receipts and per-file full/partial/not-
consumed coverage disclose clipping. All three Q8 tokenizer/template parity
checks and A/B prefix checks passed on CPU. Manifest and fixture are committed
as `q8-relevant-code-manifest-2026-10-14.json` and
`COW-Q8-RELEVANT-CODE-2026-10-14.json`. The restart campaign uses context0 from
that fixture; the other two views remain CPU-only in this new campaign.

Original mathematical contract/equations/tolerances and registered tests remain
unchanged. This is bounded execution-profile/native-operator validation, not
cumulative G1-G5 qualification. Chat server stays off; private Git only.
Next: longer text replies on the meaningful-code fixture, then ordinary chat/UI
projection and rendering behavior, without introducing a second branch authority.

Long reply campaign launched separately as `text-resume-q8-2k-v1`, tmux
`cow-text-resume-q8-2k-v1`, using the same frozen serving-v5 control/backend and
relevant source fixture, up to 2,048 outputs per branch (4 before restart plus
up to2,044 afterward). `--generated` changes the bounded experiment length,
not operator/layout/tolerance or checkpoint identities. Natural EOS may shorten
the run. This run is now terminal PASS, independently of the earlier short run.

## Long-reply terminal result

A produced 2,042 total outputs and B produced 2,048. Both independent fresh
references matched the complete generated token lists, final logits and full
greedy controller records exactly. The post-run audit also compared prefix,
four-output and final full-logit vectors on each branch: six exact comparisons,
maximum error zero. Resume performed zero prefix prefill; forward-token deltas
equaled the respective resumed output counts. Saved parent payload hashes were
unchanged. Both producer and consumer Python/native processes exited zero.

Received resumed generation measured 46.163/45.908 TPS; TTFT 6.388/6.188 seconds
excluded model startup and included activation. These are single measurements,
not performance medians. Complete timings, counters and source/build/model
fingerprints are in SERVING-TEXT-RESUME-Q8-2K-2026-10-14.json. This audit samples
three full vectors per branch, not every generated frontier. Greedy only;
nonzero RNG, grammar and original cumulative gate qualification remain separate.
