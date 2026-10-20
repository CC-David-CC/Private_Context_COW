# Three source contexts: harness ticket

Goal: exercise useful independent source contexts through one native GPU worker,
shared RAM and disk, revision-checked rewind, and continued inference without
recomputing their saved prefixes. This extends the completed 64K repeated
client measurements; it does not replace ordinary serving integration or gates.

Added fixture normalization for exactly three distinct tokenized source contexts,
ordered A/B questions, provenance, and capacity rejection. Existing synthetic
fixtures retain their original round-robin mapping. Added an optional source
fixture input and rewind phase to `pool_campaign.py`. The rewind phase retains
the prior A tip as a sibling, moves only the working reference to the root,
and runs changed question B. Independent matching references compare the B
frontier and tokens again. Worker concurrency and actual RAM eviction counters
are recorded; a config with `require_clean_evictions` fails without eviction.

Executed CPU-only on llm-60, Python 3.12.3:

- Fixture RED: 7 methods, 11 errors because normalization was absent.
- Fixture GREEN: 7 methods, zero errors/failures/skips.
- Source-trace RED: import failure for missing `tree_root`/`validate_rewind`.
- Source-trace GREEN: 4 methods, zero errors/failures/skips.
- G0 on the new API package: 51/51, no skips; evidence timestamp
  `20261014T203315384195Z-G0.json`.

The first packaged G0 invocation failed before testing because `reference/` was
omitted from the source ZIP. Added the original reference files, unchanged;
locked-reference validation and G0 then passed. No tolerance, reference equation,
registered test or existing assertion was changed.

Remote new API package:
`/home/dflanag3/branching-g3/multi-code-contexts-20261014/api-v1`.
This is a separate source tree; frozen successful candidates remain untouched.
No new real-GPU campaign has executed at this checkpoint. Source-derived
fixtures and pinned profile preparation are in progress. Planned progression:
1K source prefixes/16 outputs, then 8K/128 outputs, Q4 first and Q8 validation.
Actual suffix sizes and EOS output counts will be recorded, without padding.

Remaining gaps: real multi-context execution/pressure evidence, larger useful
workloads and ordinary serving integration. A100 remains disconnected; llm-60
chat remains off. No GitHub/upstream publication. Goal active.

## Actual launch

Source-derived fixture generation completed on llm-60 CPU using its existing
tokenizer/template venv (no package installation, model forward or GPU use).
`fixtures/contexts_combined.json` SHA-256:
`971acbcc064e0508011eb5f8c1e2df24fee21dfca0f9aad25cb85253aea4170a`.
Each variant has native COW, canonical history and scheduler/pool source bundles.
The 1K variant contains three 1,024-token prefixes with source excerpts of
1,536 tokens; the 8K variant contains three 8,192-token prefixes and 8,704-token
source excerpts. Actual A/B suffix counts are 576-583. Full per-file/source,
tokenizer and template receipts are retained remotely. Exact-token-input model
contract remains distinct from the renderer used to construct those input IDs.

Frozen API `api-v2` includes the committed harness and original reference files;
G0 passed 51/51 there (timestamp `20261014T203515456891Z`). Native binaries
remain the successful Q4 digest-v1 build; added Python-source fingerprints form
a new execution profile, without changing weights or tolerances.
Base derived profile ID:
`500dadd2a63d44f7588101d1e657f330b8ac4921d7d48129653a7e4f5320c915`.
The campaign additionally binds GPU selection into its final profile.

Q4 1K campaign launched under tmux `cow-three-code-q4-1k` with client PID
1658506 and prepare PID 1658521, both authoritatively observed live at 27s.
GPU 0 was idle with 15 MiB used before launch; no compute-app process existed.
At 15s GPU residency was 81,891 MiB. Config: GPU=[0], 1 GiB accounted RAM,
32 GiB disk budget, 16 GiB minimum available host RAM, required clean eviction.
Output: `multi-code-contexts-20261014/q4-1k-v2`; output bound 16.
Prepare/resume/rewind and then independent references run serially. No campaign
PASS is claimed yet. Do not restart on observation timeout; inspect the recorded
PIDs/session and phase/result files. Chat stays off.

Previous completed native-client raw evidence was backed up directly llm-60
to private box-git, excluding checkpoint/store bodies, at
`/srv/data/git/private_context_cow.evidence/20261014-native-client-resume/native-client-resume-20261014-r3-evidence.tgz`.
SHA-256 matched on both hosts:
`c63521f3f2b29886bd3feee0ceb4a119262b6031047e19a6adc7dea985887da9`.

## Admission failure and bounded-profile retry

The first Q4 1K run (`q4-1k-v2`) terminated FAIL, exit 1, zero completed root
requests. Prepare startup took 32.936 seconds; the native process exited and
its GPU allocation was released. Error: `Checkpoint/staging reservation exceeds
RAM pool`, raised before checkpoint admission. It used a 96K-capacity profile
with only 1 GiB accounted RAM. This is not a numerical failure or passed pressure
test. Preserve its result and stderr files.

Retry `q4-1k-v3` uses the same frozen API/native binaries/weights, with both SUT
and reference native capacity changed to 12,288 and a 2 GiB accounted RAM pool.
This is a new matching execution profile, not reuse of an incompatible saved
checkpoint. All contexts are prepared anew. Clean eviction remains required.
Derived base profile ID:
`2863ff712ca05a49ac0212745decb8cd24721aea16190f483e6faf3d963eaac2`.
tmux `cow-three-code-q4-1k-v3` was launched after observing the first run terminal
and no compute-app process. No new numerical result is claimed yet.

Retry preparation subsequently completed in 42.046 seconds. Three roots and
six initial A/B continuations were durable, with four outputs per branch. The
RAM pool observed four clean evictions, peak accounted bytes 2,115,726,628
against limit 2,147,483,648. Accounted disk files occupied 1,403,966,507 bytes.
Producer native exit was observed. New resume process PID 1659401 and campaign
PID 1659194 were observed live; the original prepare PID had exited. Numerical
references and rewind are still outstanding, so no full campaign PASS yet.

## Rewind evidence serialization correction

`q4-1k-v3` terminated exit 1 after preparation and all six restart continuations
completed. Resume produced 12 further outputs for each branch, with six clean
evictions; producer/consumer native exits were observed. Rewind completed its
first changed question, then the harness failed writing the next evidence
record: `TypeError: Object of type BranchHead is not JSON serializable`.
`ContextCoordinator.rewind` returns the canonical branch head, which the
harness had placed unconverted in its JSON receipt. No numerical reference
campaign completed; this run is FAIL, not a backend correctness PASS.

Added a regression using the actual BranchHead type and an evidence-only
`rewind_record` converter. RED was a missing-helper import error; GREEN on
llm-60 was 5/5 source-trace methods, zero failures/errors/skips. Existing
assertions remain unchanged. Converted the returned branch head with `asdict`.
Also derive nominal source fixture sizes from actual arrays rather than CLI
synthetic defaults; full per-context counts remain recorded.

New frozen API `api-v3` passed source-trace checks 5/5 and G0 51/51, timestamp
`20261014T204131359806Z-G0.json`. A fresh campaign `q4-1k-v4` was launched only
after observing the previous run terminal and no GPU compute apps. Same Q4
native binaries, 12K capacity, 2 GiB RAM and required eviction; no checkpoint
reuse across the new Python/profile identity. Base profile ID:
`0d10986222d045878dada309714f3d39113c7d65c626275be37f5c655d07aa54`.
tmux: `cow-three-code-q4-1k-v4`. Independent numerical/reference and rewind
results remain outstanding. Keep both earlier failed output trees untouched.

## User-directed full-resource workload

The user rejected continued use of the iteration-sized 2 GiB/12K configuration
for the main workload. Retain that as a bounded pressure regression, not the
normal serving/resource plan. At this check llm-60 had 127,396,200,448 available
RAM bytes (about 118.6 GiB), 32 available CPUs and 179,081,596,928 free filesystem
bytes. Main config now uses 96 GiB accounted RAM, 128 GiB disk budget, 16 GiB
host headroom, GPU=[0], and 98,304 native capacity. No additional CPU affinity
or thread restriction is imposed. Normal reuse is not forced to evict; deliberate
pressure campaigns remain separate. This does not remove an integrity or
correctness assertion; all numerical, byte-audit and zero-prefix-work checks remain.

The small `q4-1k-v4` campaign finished exit 0, six tree comparisons and three
rewind comparisons, numerical max error 0, ten actual clean evictions, no
failures/skips/xfails. Compact evidence: `SOURCE-CONTEXT-Q4-1K-2026-10-14.json`.
The final fresh-reference checks were already executing after its pooled phases
ended; no additional small pooled campaign was launched.

Full fixture: `fixtures/q4_64k.json`, SHA-256
`9fb32731440a794234eae1f622b34415d6a80c433df79f30aea6c37d0f9cb5c2`.
Three distinct native COW, canonical API/store and serving/tool/template source
bundles contain 48/38/18 files, source SHA receipts and actual 73,728-token
source excerpts. Each stored prefix is 65,536 tokens. A/B suffixes:
8,336/8,311; 8,320/8,311; 8,315/8,318. No repeated/padded source. Output limit
2,048, early EOS to be reported. These are source contexts from the project,
not a claim of three unrelated external repositories.

Full campaign `q4-64k-full-v1` launched under tmux `cow-three-code-q4-64k-full-v1`,
client PID 1662132 and prepare PID 1662147, both observed live at 1m16s.
New base profile ID:
`e995ae490db606f4fa0ec685af8e3890dd30b36be705a1a565eb3d6ad52feaa4`.
Same frozen api-v3/native digest-v1 binaries, native capacity restored to 96K.
All three prefixes were durably prepared by this observation; RAM held
5,491,806,696 payload bytes, peak accounted 11,797,386,368 bytes, zero forced
evictions. GPU reached 90,747 MiB used and 100% utilization during preparation.
Restart, all continuations, rewinds and fresh references remain outstanding.
Do not report a full workload PASS from prefix preparation alone.

Serving integration review found two distinct boundaries still to implement:
exact token-prefix validation after normal template rendering, and active
request cancellation at the native consumed-token ACK boundary. Current workspace
SQLite branch heads and COW branch refs must not become unsynchronized authorities.
The native request protocol supports boundary cancellation, but the coordinator
only exposes queued cancellation today. No serving integration was claimed or
enabled as part of the source-context campaign.

## Full Q4 terminal result and Q8 launch

`q4-64k-full-v1` completed exit 0: all three native phases exited, six tree
comparisons and three rewind comparisons PASS, zero error at checked frontiers,
no failures/skips/xfails. Outputs per tree: 2,048 / 507 / 2,048 / 2,048 / 2,048 /
2,048. The 507-output branch stopped at EOS and matched its fresh reference;
do not describe every branch as reaching its bound. Frontier sampling is two
generated boundaries per tree (initial four and final), plus one final boundary
per rewind, 15 full-vector checks total. It is not every-token oracle certification.
Restart worker generation TPS per branch: 64.518 / 63.910 / 63.772 / 63.519 /
62.668 / 62.952. These exclude restore/capture/durability and are not received
client/network TPS. Full timing/work/memory/build fingerprints are in
`SOURCE-CONTEXT-Q4-64K-2026-10-14.json`.

Q8 fixture built independently with the actual Q8 tokenizer. Its inputs agree
with Q4 token IDs, but generated Q8 outputs need not agree with Q4. Fixture SHA:
`d8f44b8706a982b491b91848d4eb59ee2029de9f934e613de529337cbbb2ce09`.
Same source hashes, prefixes/suffixes, capacity and output limits.

Q8 `q8-64k-full-v1` launched after Q4 terminal/no compute-app observation, under
tmux `cow-three-code-q8-64k-full-v1`. Client PID 1669147 and prepare PID 1669162
were authoritatively observed live. Same 96 GiB RAM, 128 GiB disk, 96K capacity,
GPU=[0] and unrestricted available CPU resources. Two prefixes were durable
by the 2m15s heartbeat; GPU peaked at 92,637 MiB used. New base profile ID:
`e8e2b7a9f844800c1718faa5fd18968825ab34b392d2001b5e71efc14a554c59`.
Uses the separately frozen request-stop api-v1 candidate whose 106 control,
G0 51 and four mutant checks passed; default continuation exercised here,
not cancellation. Full Q8 results remain outstanding. Observe the same job,
not a replacement on timeout. Chat off, A100 disconnected, private only.

## Full Q8 terminal result

The same `q8-64k-full-v1` runner completed exit 0. All three native phases
exited, all six tree comparisons and three rewind comparisons passed, numerical
max error 0 at 15 checked full-vector frontiers, no failures/skips/xfails.
Actual output counts: 2,048 / 996 / 2,048 / 2,048 / 2,048 / 2,011; the two
shorter branches ended at EOS and matched the independent references.
Restart worker generation TPS: 45.931 / 46.514 / 46.143 / 42.870 / 45.818 /
45.190. Counts/time reflect the remaining continuation after the four prepared
outputs. These exclude restore/capture/durable publication, and are not a
received-client or whole-request benchmark.

Full work, transfer, memory, phase timing and model/build/profile identities:
`SOURCE-CONTEXT-Q8-64K-2026-10-14.json`. Each continuation's native work assertion
excluded saved-prefix recomputation. Parent payload audits and one-active-worker
assertions passed. Terminal observation showed no compute-app process.

This advances the native-speed goal, but does not complete ordinary serving
integration or qualify active cancellation on real GPU. Keep the chat server
off; there is no queued automatic server restart. A100 remains disconnected.
Original numerical-gate and complete stochastic/grammar qualification remain
separate from this pinned greedy/matching-operator workload.
