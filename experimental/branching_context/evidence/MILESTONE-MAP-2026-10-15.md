# Native-speed Context COW milestone map

## Combined native candidate: measured Q4 gain; bounded Q8 restart PASS

Current combined v3 passed bounded Q4 tiny/1K/8K numerical/restart checks. Four
corrected Q4 actual-code timing arms now record numeric native exit zero, with
six measured requests each. Resident/disk streaming medians improved from
66.04/65.68 to 74.85/74.94 TPS. Durable times improved 5.47%/2.56%; disk
first-token wait increased 3.76%. Full fresh hashing remains enabled. These
are two COW profiles, not a stock-native overhead measurement. See
`figures/2026-10-15-q4-seal-stream-v2/README.md` for exact scope and raw pins.

The first current-v3 Q8 real-code 8K campaign made 13 exact comparisons at
zero error, but its old harness omitted numeric backend exit codes. The
corrected harness repeat passed all 13 comparisons, A/B 128 outputs each and
124 remaining tokens after A's generated checkpoint, with numeric-zero native
prepare/resume exits. Prefix restoration evaluated zero saved tokens. The
current combined-source 64K/8,335/2,048 campaign passed all 4,096 output IDs and
13 sampled frontier comparisons at zero error, with numeric-zero exits and
zero saved-prefix forward work after restore. Completed-suffix frontiers before
generation were not compared by this harness. Dedicated resident timing passed
six measured requests: A/B medians 57.47/45.06 streaming TPS and 48.44/39.37
durable request TPS. Disk timing also passed six measured requests: A/B medians
57.56/45.22 streaming TPS, 43.12/36.56 durable TPS, 8.02/7.55 seconds TTFT.
All requests reached 2,048 outputs; every disk restore evaluated zero saved
prefix tokens. These are CLI timings, not HTTP
or a matched ordinary-serving overhead measurement. Earlier phases below remain
evidence for their own pinned source/profile, not automatic qualification of
this native update. Original cumulative gates remain separate.

## Hash candidate: bounded real-Q4 correctness PASS; speed comparison next (October 15)

The integrated immutable-RAM digest optimization passed real Q4 checks on
llm-60, GPU=[0], FP16 KV, capacity 98,304 and 8,192-token prefill chunks.
Tiny and 1K canonical restart fixtures passed; independent native probes
checked every generated boundary (32 and 56 comparisons respectively), with
zero logit error. The 8K-prefix/two-512-suffix fixture passed matching-schedule
reference comparisons: A produced 128 tokens across restart; B stopped at EOS
after one token, so generated-B restart was not exercised. Target comparisons
sample prefix/suffix and generated-4/final frontiers, not every 8K output.
The v2 medium probe, target prepare/resume and cold-return native exits were
numeric zero; original tiny/medium canonical harnesses only observed exits.

An additional fresh-process cold return to P loaded 332,013,048 bytes and
computed exactly 512 changed input tokens plus one output, matching B's
independent reference token. The saved parent bytes stayed unchanged. This
extra check also matched all 993,280 frontier bytes against fresh B
(SHA d420a07c7878aaf0e927af5a29e3e2fe57ef2b207fedbcf47e89822933334536).
Command/wrapper/native
exits were zero. See CHECKPOINT-DIGEST-Q4-GPU-RESULT-2026-10-15.json for raw
artifact hashes and scope. Execution profile is
4ac77a63cfb8bd0d153c8cb5fb340ab79ae4d3cc23e554d39bd967d693adfdd1.

No hash-optimization speedup has been measured yet. Next is paired Q4 return
latency with newly prepared checkpoints under this exact profile, then Q8.
Old-profile checkpoints must not be relabelled. Disk hashes and GPU integrity
guards remain enabled. Chat stays off; evidence stays private. This does not
qualify cumulative G1-G5 or unrestricted controller continuation.

## Current milestone

Native-speed serving for one active request, retaining several independent
reasoning/codebase contexts through GPU, shared RAM and durable disk backing.
The current subphase is P4 controlled return-to-context latency: both Q4 arms
passed nine measured requests each; the Q8 latency arms remain unrun. Root
independently verified all 2,304 measured output tokens against retained fresh
references and suffix-only work. P5 bounded Q4/Q8 pressure campaigns and their
full raw backups passed.
P6 reusable-file browser correctness passed on Q8: an 8,189-token context,
two independent suffixes and 128 consumed outputs each match separate fresh
native references in all 256 output IDs and both final frontiers. Root and
full A/B canonical histories/payloads passed a post-shutdown audit. This is
bounded correctness, not a browser latency or full controller qualification. GPU cancellation already has bounded
real-Q8 evidence; it is not the current missing capability.

This map distinguishes implemented capabilities, bounded measured results and
remaining qualification. It does not replace the mathematical specification or
the original G0-G5 acceptance obligations.

| Phase | Executed evidence | Remaining work |
|---|---|---|
| P0: reproducible native baseline | Pinned model, executable, source and execution-profile identities; separate uncached native reference | Same-day stock-serving comparison if making new relative speed claims |
| P1: resident request execution | Native request loop, ownership, cancellation/recovery, fork and rewind; Q4/Q8 real-GPU runs | Controllers beyond the tested deterministic greedy scope |
| P2: durable computation resume | Earlier 64K Q8 process-separated restart, unchanged saved bytes and zero prefix work | Repeated restores, broader crash/OOM/controller qualification on the newest source |
| P3: reduce COW overhead | Fresh integrity hashes retained; digest scratch reuse measured on Q4/Q8; bounded Q8 continuation above 40 TPS in several cases | Repeated comparable measurements before new speed claims |
| P4: client latency | Both Q4 return arms PASS: three64K roots, three repetitions each,128 outputs each; independently checked2304 token IDs and suffix-only work. RAM/disk medians: first token4.80/7.56s, durable13.09/15.77s, steady consumer67.63/67.51 TPS | Resident raw backup/reclaim verified; qualify Q8 arms. These are coordinator timings, not browser/HTTP timing or new full-logit comparisons. Capture3.67s and publication2.47s remain measured RAM-arm overhead |
| P5: several retained contexts | Full-size Q4 and Q8 pressure PASS: each three64K roots, six A/B trees, three rewinds,15 sampled exact vectors and three causal same-process eviction/reload chains; actual Q8 canonical Store audit18 snapshots/15 payloads PASS; full raw backups verified | Repeat return-to-context latency; Q8 native payloads are archive-only after verified exact-file reclaim |
| P6: serving/browser integration | Canonical namespace/history, serialized requests, publication recovery, graph and previous tips; earlier real-Q8 browser8; reusable-file UI serving126/G051/Chromium regressions; new browser harness14 CPU support/audit tests | Bounded file-only Q8 UI correctness and full canonical A/B audit PASS; remaining rename/archive/clear controls, graph scalability, latency and broader workloads |
| P7: expand and package | Private source/raw-evidence backups; build-provenance receipt improvements; existing runners and documentation | Clean current-source build/profile admission, polished launch/tutorial/screenshots and broader workloads |

## Latest P5 result and remaining execution

Q8 completed all eight executed POOL-REAL IDs, six independent matching-native
token comparisons and three rewinds, with15 sampled full vectors at zero error.
The actual canonical Store readback also passed for18 snapshots and15 payloads.
See STATUS and P5-Q8-CANONICAL-STORE-AUDIT-2026-10-15.json for exact scope.
The complete Q8 private raw backup is verified:42,301,583,360 bytes, SHA-256
e478f85807fe330f6bd4fd5f4c38969365812014a5651eaf4f8e963a3693251f.
Its21 exact derived payload/dump files were reclaimed with protected canonical
metadata unchanged. Three Q4 root payloads were selectively restored from
their verified archive, without computation. The Q4 disk arm is now terminal
PASS; see P4-Q4-DISK-INDEPENDENT-AUDIT-2026-10-15.json.

Earlier superseded browser attempt: the reusable-file GPU browser at source981aaeaa/profilef39df09d failed
the harness completion check after the first ask was durably stored. It did
not run B or independent references. A CPU synthetic reproduction established
that Chromium can report canceled ERR_ABORTED despite the actual UI reader
reaching EOF and receiving a matching durable complete receipt. The corrected
harness requires this stronger client evidence and rejects truncated streams;
its full synthetic flow passed, but root review found a fork-button selection
bug requiring another focused fix. That attempt did not pass; the later v2 workflow and independent references
passed as recorded in COW-UI-Q8-RETRY-V2-RESULT.json and the branch audit.

P4's two arms required phased archive/reclaim. The disk raw backup is verified;
its exact nine derived state files were reclaimed with90 protected files
unchanged. The resident arm subsequently passed with clean0/0 receipts and
native exit0. Its raw backup is verified on box-git; no resident payload
reclaim is authorized yet. See both P4-Q4 independent audit JSONs.

## Earlier P5 observations (superseded chronology)

Latest checkpoint: Q4's terminal pressure result,43 retained-data audit checks
and complete private raw archive are verified. Its three exact same-process
eviction/reload chains establish the causal transition missing from earlier
unpressured runs. Q8 v2 is live; prepare, restart and rewind phase checks passed.
Two of six independent fresh-reference comparisons have passed; the terminal
campaign remains unqualified. P4 repeated-return harness is reviewed and
integrated, with19 CPU checks and G051; native arms remain unrun. The reusable-file
UI is integrated with126 serving tests,51 G0 checks and Chromium regressions;
the new UI's native GPU smoke remains pending. Its opt-in browser harness is
integrated with14 CPU support/audit tests and a pending minimal-runtime stage.
These supersede the older
running/prelaunch observations below, which are retained as chronology.

Cancellation is already bounded real-Q8 evidence, not the current missing
capability: GPU-STOP-001/002 and HTTP-CANCEL-001/002 passed consumed-boundary
abandonment and subsequent exact recovery. This does not qualify arbitrary
interruption races or all controller states. See REQUEST-STOP-GPU and
SERVING-HTTP-CANCEL-Q8 JSON for executed counters and scope.

Terminal update: Q4 campaign passed all8 executed POOL-REAL IDs, six token
comparisons and three rewinds, with15 sampled full vectors at zero error.
Command/wrapper exits were0/0 and GPU processes exited. Independent broader
controller/history/prompt audits, raw backup and Q8 pressure qualification
remain outstanding. See STATUS.md and P5-Q4-FULL96-RESULT-2026-10-15.json.
The live observations below document the preceding run stages.

The full-size Q4 pressure campaign is now running on llm-60 GPU0, frozen
source24b4fbbf85deec2b3c4beb8cc8af389c730ea32c. Capacity remains98,304,
native prefill chunks8,192 and FP16 KV. Three distinct65,536-token codebase
prefixes and six8,311-8,336-token questions use a deliberately constrained
16GiB RAM pool and64GiB disk budget; CPU/GPU resources are not constrained.

The completed prepare trace has three validated same-checkpoint chains of
positive RAM accounting release followed by successful disk read/admission in
the same process. Each releases and reloads1,830,602,232 bytes. This proves the
missing causal pool-transition witness, not OS RSS reclamation or campaign
numerical correctness. See P5-Q4-PREPARE-SPILL-WITNESS-2026-10-15.json.

All six restart continuations completed: additional output lengths were
2,044/503/2,044/2,044/2,044/2,044, each with zero prefill work and corresponding
generated-token forward work. Early EOS is retained. Rewinds are in progress;
independent fresh-reference comparisons and terminal audits remain pending.
P5-Q4-FULL96-RUNNING-2026-10-15.json records the pinned profile and handles.
The older unrun/space observations below describe the prelaunch state.

## Source and evidence boundaries

The assembled source for current browser qualification is
`ac6b8e6d465d866a0ae341d9d37a28400aea582f`, branch `feat/model-free-fleet`.
The checkout is `C:\Users\dflanag3\Documents\fleet\context-cow-model-free-fleet`.
Source and compact evidence are private on box-git; no new public publication
is authorized. The ordinary chat server remains off.

The separately qualified namespace HTTP Q8 case used 8,841 prepared prefix
tokens, suffixes of 456/512 tokens, and 128 actual output tokens per branch.
Four checked full-logit vectors matched independent references exactly;
tokens and greedy controllers matched, parent bytes were unchanged and native
work did not include prefix replay. Received continuation observations were
49.373/52.481 TPS, TTFT 1.465/0.949 seconds and durable completion 5.688/5.000
seconds. These are single observations, with model startup separate. They do
not establish every-token equivalence or qualify the later browser assets.
See `NAMESPACE-Q8-HTTP-2026-10-14.json` and its independent audit profiles.

The newest assembled CPU/filesystem checks and browser fixtures are recorded
in `NATIVE-GRAPH-FINAL-*`, `NATIVE-GRAPH-BROWSER-FINAL-V3-2026-10-14.json`
and `NATIVE-MATH-BROWSER-FINAL-2026-10-14.json`. Synthetic inference and
display-only rendering are explicitly separate from real-model execution.

The assembled ac6 candidate subsequently passed eight real-Q8 browser checks
on llm-60 with the same 8,806-source-token attachment. Both answers naturally
ended at two output tokens. V2 recorded command/wrapper exits0/0; its report is
`NATIVE-Q8-BROWSER-2026-10-15.json`. This adds real-model UI integration evidence,
not repeated performance or independent numerical comparisons. V1's malformed
shell exit receipt remains preserved separately.

## Accepted prototype scope

Exact continuation is bound to identical tokens, model/execution profile and
ordered native operator/request boundaries. Greedy/temp0 is the qualified
controller scope here. Existing cross-boundary mismatch evidence remains
recorded. This accepted prototype limitation does not modify the original
specification, reference equations, tolerances or registered tests.

Only G0 has current full reference-gate evidence. The bounded results above
are not cumulative G1, G2, G3 or G5 certification. Graph shared ancestry does
not imply physical GPU page sharing or RAM residency. Streamed output is not
an acknowledged durable save point.

The full real-codebase campaigns used 96 GiB accounted RAM and 128 GiB disk
budgets and recorded zero clean RAM evictions. They establish those workloads
and numerical comparisons, not pressure-driven spill/return. Existing aggregate
eviction and disk-read counters cannot prove the same checkpoint made both
transitions within one process. A bounded, identity-keyed event witness is the
next P5 ticket; existing fixtures and independent references can be reused.
That recorder and its17 CPU regression cases are now integrated at44f8729d.
The assembled source passed control221/serving118/G051; root compared executed
source fingerprints exactly. The real-model pressure campaign remains unrun.

The next substantial workload milestone after browser qualification is P5/P4:
measure pressure-driven transitions among the existing real codebases, then
repeat Q8 latency/throughput measurements. Additional copies of the same
unpressured prefix do not substitute for those tests. A current llm-60 sample
showed only about 30 GiB disk free against roughly 30.35 GB existing full-run
checkpoint state: a new full campaign has not been launched. The configured
16 GiB minimum is host MemAvailable, not a disk-free reserve.
After exact archive-member hash verification, eight old derived payload files
and two duplicate local tar spools were archived out of local storage. Observed
disk availability is now53,260,918,784 bytes. Metadata/history/refs and private
backups remain; PLAN/RESULT JSON retain exact file identities and restore map.
The historical complete Q4 output tree used41,198,726,395 bytes including fresh
reference dumps. Recheck capacity before launching rather than using only its
30.35GB checkpoint-pool total as the estimate.
