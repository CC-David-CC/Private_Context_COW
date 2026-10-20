# P4 three-root return latency plan - 2026-10-15

## Purpose and evidence boundary

Measure repeated returns to three already prepared Q4 code-context roots using
the pinned single-worker Q4 profile. The pressure campaign has already been
reported PASS by the primary worker; this plan reuses that campaign's canonical
root checkpoints and does not repeat its 65,536-token prefix construction. It
does not change the campaign output or claim a new qualification until the
specific source, profile, store, root IDs, fixture, and pressure receipts are
frozen and rechecked before execution.

The comparison is between (a) a checkpoint already resident in the coordinator's
RAM pool and (b) a checkpoint deliberately absent from that pool and reloaded
from its canonical checkpoint files. Both paths still import serialized
checkpoint bytes into the native backend. In particular, native
`loaded_file_bytes` and `checkpoint_import_bytes` are **not** a RAM-versus-disk
witness: `StrataBackend.restore()` writes the serialized image to its temporary
`cold.state` file before the native `load` command for either source tier.
Positive values are therefore expected in both arms. Classify the tier using
the exact snapshot key in pool membership and the `PoolTransitionTrace`
`disk_preload_started` / `disk_preload_succeeded` records.

This is a same-host coordinator-consumer timing experiment, not HTTP/SSE
latency. Producer event `time` is monotonic time when `Request.emit()` ran;
existing `pool_campaign.complete()` serializes those events later and records no
consumer-arrival time. Do not relabel producer times as client TTFT or queue
wait. A later HTTP claim needs a separately instrumented real HTTP client and
server path.

## Frozen inputs and isolation

Use the completed pressure run's three immutable roots and its exact Q4
fixture. The recorded fixture from the earlier source campaign has SHA-256
`9fb32731440a794234eae1f622b34415d6a80c433df79f30aea6c37d0f9cb5c2`; verify
that the completed pressure run used these same bytes before using the fixture
indices. For the six-tree fixture, select one fixed suffix per context
(`suffixes[0]`, `suffixes[2]`, `suffixes[4]`) and record the exact token-ID
arrays and their hashes. Record the root snapshot IDs, manifests, payload SHA-256
values, model-contract ID, execution-profile ID, binary/build identity,
tokenizer/operator contract, GPU UUID, and source receipt in the new run
manifest.

Create a new private run directory and a separate working Store for each arm,
both derived from the same three immutable roots. Use Store APIs to retain the
three root histories and derive the exact lineage metadata, manifests, and
checkpoint payload files they require. Copy mutable metadata; immutable
`.state` payloads may be hard-linked read-only or copied, with source and
working-store hashes verified. Never hard-link mutable refs, locks, indexes,
or journals, and do not copy the full campaign's checkpoint directory. If the
Store API has no supported way to create this minimal working Store, stop for a
small reviewed CPU-tested importer rather than mutating the frozen campaign
Store. All new branch refs, answers, traces, and reports belong to their arm's
new Store/run directory. Recheck original root manifests and payload hashes
after both arms.

Use one worker on GPU 0 and the exact pressure-run Q4 profile/build. Set greedy
policy and a 128-token output bound in every measured request. The completed
full-pressure campaign already covers the 2,048-token bound; this repeat is a
short return-latency experiment. Validate each output against the first 128
tokens (or earlier EOS) of that campaign's independent fresh reference. Record
actual output token counts and EOS; 128 is a bound, not a promised completion
length. Use one distinct branch per `(root, repeat, arm)`, each forked from the
unchanged root, so all requests have identical source history and question.
Use a fixed round-robin root order and unique operation IDs. Complete and
durably publish each request before the next; do not run concurrent requests.

## Experimental arms

Run each arm with one fresh coordinator process over its minimal Store clone:
two coordinator/model startups total per quantization. Exclude and
report startup separately from request timing. Choose the arm order once using
a recorded fixed randomization seed, then run both arms serially; two-arm order
bias remains and is not eliminated by this design. Each arm has a one-token,
unmeasured warmup per root, followed by three measured repeats per root (nine
measured requests per arm; 18 total). Warmups use separate branches and are not
included in latency summaries.

**Resident-pool arm:** use a RAM pool budget sufficient for the three root
checkpoints and their working outputs (the previously used 96 GiB campaign
budget is the reference setting). Warm each root once before measured returns.
Before every measured request, assert that the target
`(model_contract_id, execution_profile_id, snapshot_id)` is present in
`SharedCheckpointPool` and that it is not merely being served from the worker's
last-root `BranchingModel.cache`. Cycle among the three roots so the target is
not the immediately preceding source. Fail or mark the sample unclassified if
the key is absent, a clean eviction occurs, or an exact-key disk preload is
observed. A RAM-pool hit still deserializes/imports the checkpoint into the
native backend; record that cost and do not require native import counters to be
zero.

**Disk-return arm:** use the proven pressure-run RAM budget and trace mechanism
(16 GiB if that is the exact completed pressure configuration). Keep one
coordinator process alive for the full arm. Before each measured return, ensure
the target is not the immediately preceding root and is absent from the worker's
per-model cache. If it remains as a durable, unpinned pool entry, evict only that
exact key with the public `SharedCheckpointPool.remove(key)` operation; this
removes the RAM copy only and must leave the canonical checkpoint files intact.
Record before/after pool metrics, the key, successful remove result, and root
file hashes. If it is already absent after real pool pressure, record that
state. Then submit the request and require an exact-key
`disk_preload_started` followed by matching `disk_preload_succeeded` with a
positive byte count in the same process/phase/trace. If the pool key is present
at request start or a success witness is missing, the sample is not a disk
return. This arm measures a controlled disk reactivation; label it as such
rather than implying every natural user revisit would evict its checkpoint.
If the exact eviction was performed by `remove`, classify the sample only as a
controlled tier-return measurement, not as natural pressure eviction evidence
or a pass for `POOL-REAL-SAME-PROCESS-EVICT-RELOAD`. Only a naturally observed
`ram_clean_eviction` followed by a matching exact-key successful preload may
count for that existing pressure invariant. Removing the in-memory checkpoint
does not evict the host filesystem page cache; report disk returns as
filesystem-cache-warm-or-unknown and make no cold-device-latency claim.

Run Q4 first. Only after its checks pass, repeat this same two-arm protocol with
the completed Q8 pressure roots and the corresponding exact Q8 profile/build.
Keep pool budgets, worker count, request IDs/questions, output limit, and timing
method the same where compatible, while freezing the Q8-specific profile and
root identities. Report Q4 and Q8 as separate profile-specific results; do not
present their absolute timings as a controlled quantization comparison. No
2,048-token repeat is part of this ticket; a longer-output repeat is optional
later.

The event trace's sequence, process/phase, key, load ID, positive bytes, and
non-overflow checks establish ordering and identity, not elapsed read time. For
stage timings, the new harness must add same-process monotonic timestamps at
observation points: immediately before submit; when each request event is
dequeued; around the actual `_load_checkpoint` call when a disk preload runs;
around the disk `SharedCheckpointPool.preload` future from submission through
successful admission; around `backend.restore`; around suffix
`backend.consume`; and at snapshot and durable acknowledgment. The pool-future
timing includes disk read, decode/validation, reservation wait, and RAM
admission as one combined stage. Time `_load_checkpoint` separately if disk
read/validation needs isolation. The trace has no timestamps, so capture the
preload-future completion time in the harness while preserving the unchanged
trace for exact eviction/load identity. Do not edit the frozen campaign trace
or production sources for instrumentation.

## Required observations and checks

For every request, report:

- **Caller-to-first-consumed-token:** submit timestamp to first dequeued `token`
  event. Also report submit-to-`request_started`; the latter separates queue and
  dispatch delay from the subsequent preload/materialize/prefill/decode path.
- **Consumed-token gaps:** first and subsequent token dequeue times, median and
  p95 gaps, actual output count, and the event's EOS marker. These are
  coordinator-consumer arrival gaps, not browser-rendered token cadence.
- **Snapshot and durable completion:** submit-to-dequeued `snapshot` and
  `durable` events separately. Also report producer `snapshot.time` to
  `durable.time` as writer/publication elapsed time; keep it distinct from
  consumer dequeue delay. A snapshot event is not durable completion.
- **Restore and compute stages:** disk `_load_checkpoint`/validation time when
  invoked; pool preload/wait/admission time; `backend.restore` time (native
  import from temporary staging); suffix `backend.consume` time; native
  `generation_seconds`; `answer.request_seconds`; capture time; validation time
  if explicitly measured; and durable writer/CAS time if explicitly measured.
  Mark uninstrumented stages UNAVAILABLE rather than inferring them by
  subtraction.
- **Work and storage witnesses:** `forward_tokens == len(question_ids) +
  actual_outputs`; `prefill_tokens == len(question_ids)` (never the saved
  65,536-token prefix); checkpoint payload hashes unchanged; exact source root
  and branch receipt; RAM-pool key hit for resident samples; exact-key
  disk-preload start/success for disk samples; pool resident/reserved/entry
  metrics and disk `file_bytes`; native import/copy counters in both arms.

The request timeline API in `client_stream_speed.py` is a useful local pattern
for event-arrival TTFT/TPOT/snapshot/durable summaries, but it is not the
measurement implementation for this coordinator experiment: it covers a single
root and performs direct `restore_checkpoint()`, not shared-pool
eviction/preload. `pool_campaign.py` already covers three roots and records the
pool causal trace, but its persisted event timestamps are producer times and it
does not perform three measured returns per root. The minimal follow-up runner
should reuse the coordinator, Store, fixture, root state, and validation
contracts, adding only controlled arm setup, consumer/stage timestamp capture,
and 18 measured requests per quantization (Q4 first, then Q8 if Q4 passes).

## Claims deliberately excluded

Do not call these measurements HTTP TTFT, end-to-end browser latency, OS block
device latency, or full RAM-vs-disk native-import cost. Do not use
`loaded_file_bytes == 0` to qualify resident hits. Do not count warmups, branch
setup, process startup, output rendering, or background pressure work as
measured request latency. Do not claim every context remained resident or was
disk-reloaded unless the per-sample key and trace checks pass. Do not infer
performance from the already completed pressure campaign: it demonstrates the
pool/restore path but does not by itself provide these repeated-return timing
samples.
