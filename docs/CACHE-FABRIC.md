# Cache inventory and placement experiment

This experiment extends COW's existing checkpoint ownership and weighted queue.
It does not replace immutable histories, branch revisions, complete controllers,
checkpoint checks, RAM/disk reservations, or BranchWrite's workflow ledger.

The first bounded campaign passed 58 focused Linux CPU tests and the 51-test
G0 gate. On Wen 1.1, three complete responses with the features disabled and
three with them enabled all passed their checks and matched byte for byte.
The repeated prompt reused 14 tokens in both modes. Inventory observation did
not move branch heads. Both ephemeral servers exited cleanly with no session
caches left; the GPU was released. This was one RTX PRO 6000 Blackwell 96 GB,
not a multi-GPU speed comparison or a complete BW tool-workflow campaign.
The Wen run exposed RAM/disk locations; native-image inventory was not observed
on that adapter. See the [answer-free receipt](../experimental/branching_context/evidence/CACHE-FABRIC-20261022.json).

A broad collection ran 2,027 tests but was NOT PASS: 36 errors and one skip.
All 36 error names were reproduced on unchanged main. These include absent
native profile/calibration prerequisites, import-path problems, older diagnostic
source bindings and an incomplete coordinator fixture. That original run remains
failed. The subsequent [stack integration](STACK-CAPABILITIES.md) repairs the
applicable fixtures/imports and explicitly separates native prerequisites from
CPU checks. Its passing CPU campaign does not qualify the excluded native or
historical campaigns.

Enable `--cache-inventory --scheduling transfer-aware` when configuring a fresh
server. Existing configurations still use FIFO and expose no inventory. Normal
BW and standard CLI requests continue through the same COW coordinator.

The authenticated `GET /cow/cache/inventory` endpoint returns location hints for
serialized RAM checkpoints, disk manifests with an existing correctly sized
payload, and idle workers' native images. It never reads model logits, acquires
a GPU lease, exports history, or verifies the complete payload. A location can
disappear immediately after observation. Actual activation retains COW's full
identity, integrity, pinning and capacity checks.

Each process has a new epoch and a bounded 1,024-event journal. A consumer asks
for `?epoch=...&since=...`; missed events within the window produce a delta.
Older cursors or a restarted producer produce a complete snapshot instead.
`InventoryView` validates a complete packet before replacing its view, rejects
sequence gaps and retired epochs, and tolerates duplicate delta delivery.
The journal is derived from observations, not an audit log of every transient
cache mutation. Inventory is capped at 16,384 locations. Its bounded Python
metadata is separate from checkpoint payload budgets.

Optional `page_size=1..1024` requests use schema 2 frozen snapshot pages. Continue
with the returned `next_cursor` and the same page size. Each producer retains at
most four snapshots for 60 seconds; an expired cursor returns HTTP 409 and the
consumer starts recovery again. `InventoryView` stages pages without exposing a
partial view, tolerates exact duplicate pages, and rejects gaps, changed repeats
and conflicting identities. The final page atomically replaces the visible view.
Clients that omit paging retain the original schema 1 snapshot/delta protocol.

RAM and disk pools publish bounded metadata notifications (1,024 events). An
unchanged inventory read avoids rescanning disk manifests. A missed notification,
startup, changed worker identity or 30-second reconciliation interval triggers a
full scan. These events carry locations only and never call user callbacks while
pool locks are held. All notifications and recovery metadata are disabled when
`--cache-inventory` is absent.

Transfer-aware placement retains grouped-prefix admission, in-flight prefill
coalescing and two-second aging. It compares measured forward-work cost and
native restore cost per byte for compatible idle workers. A slow warm worker
can lose to a faster cold worker when measured restore plus compute is cheaper.
Missing rates retain stable warm affinity. Rates are process-local moving
averages; output work is a bound, not a prediction of actual response length.
Forward service observations include host orchestration and capture overhead;
the separately measured native restore interval is subtracted to avoid counting
it twice. Disk preload is a shared-server cost. Configured peer recovery
separately records successful transfer seconds per byte; it does not dispatch
inference remotely.

The status endpoint reports the last placement estimate and rates. Already
running native batches are not preempted or moved. A cache hint is never a
checkpoint reservation or a successful restore receipt.

## Integration sequence

1. Recoverable location metadata and measured restore/work placement.
2. Authenticated cross-server discovery and bounded checkpoint transfer, with
   model/profile compatibility, source pinning, checksums, interruption cleanup,
   backpressure and failures tested before use by BW.
3. Compare local reuse, transfer and reconstruction using measured complete
   response results. Add admission/eviction policy hooks only where measurements
   establish a useful behavior beyond the existing pool controls.
4. Separate prefill and decode placement after validated handoff exists.
5. Compare fastokens against Wen's actual tokenizer, including token IDs,
   literals, special tokens and long shared prefixes, before any default changes.

Passing metadata or synthetic scheduler tests does not establish GPU throughput,
native restore correctness, BW task quality, or cumulative G1-G5 certification.
Feature controls remain experimental and off by default; the bounded completed
checks below do not qualify the broader claims.

## Ownership and feature controls

Strata owns native execution, GPU memory and complete native model state. COW
owns canonical histories/controllers, checkpoint validation, placement and
transport. BW owns the workflow ledger, tool permissions, effects and task
verification. An inventory never becomes another history store; a remote worker
never advances a BW workflow; a scheduler never mutates a checkpoint.

| Feature | Control | Disabled behavior | Current scope |
| --- | --- | --- | --- |
| Inventory publication | `--cache-inventory` | No inventory journal; endpoint rejects observation | RAM/disk and idle native images within one server |
| Placement costs | `--scheduling transfer-aware` | FIFO remains default; weighted-prefix remains independently selectable | Measured local native restore and forward service costs |
| Peer export | `--cache-transfer --cache-inventory` | No checkpoint export | Existing, compatible disk checkpoint bytes; authenticated even on localhost |
| Peer recovery | `--cache-peers /absolute/private/peers.json` | No peer requests | Reuse for locally owned canonical histories, including an identical new root |
| Prefill/decode separation | `--prefill-gpus 3 --decode-gpus 0 --gpu 0,3` | Ordinary combined serving | Independent workers; root prefill versus continuation requests |
| Tokenizer backend | `--tokenizer-backend fastokens` | Existing Python tokenizer | Optional fastokens 0.3.4 for long ordinary pieces; original splitting and cache |

These options are accepted by both setup and the configured launcher. Compatibility
APIs and BW's native API remain independent of these switches. Missing optional
peer data must permit ordinary serving, but present incompatible or corrupt
checkpoint data remains an integrity error. A fallback must not silently replay
tools, reset a controller, change input tokens, or violate a strict restore
request. Reconstructing an evicted cache uses COW's existing explicit semantics.

## Configured checkpoint peers

The exporting server enables inventory and checkpoint export. A receiving server
uses a private, owned configuration file and separate private key files:

```json
{"schema":1,"peers":[{"name":"cache-a","url":"http://cache-host:8099","api_key_file":"/home/you/.config/cow/cache-a.key"}]}
```

Set these files to mode 600. Configuration accepts 1–16 explicit origins; it
does not follow redirects, use ambient HTTP proxies, or discover arbitrary peers.
The existing server API key authenticates the read-only export. Transport on an
untrusted network requires the existing trusted-LAN/tunnel or TLS boundary.

Inventories expire after five seconds and recover through bounded pages. Both
model and execution identities must match. Missing, busy or unreachable optional
cache locations permit COW's existing rebuild path. Wrong credentials, mismatched
identities and present corrupt payloads fail explicitly. A peer never supplies a
canonical history, moves a branch or publishes a BW result.

Transfers use the existing checkpoint envelope and payload. The source holds a
history read lease and disk pin; the destination reserves its existing RAM/disk
budgets, checks length, SHA-256, finite frontier and execution schedule, then
publishes atomically. Defaults permit two transfers, one-MiB chunks, a 64-GiB
checkpoint ceiling and a 300-second deadline. Retry of an already checked image
copies zero payload bytes. The existing collector recovers abandoned staging and
process reservations; timeout, cancellation and a real killed receiver are tested.

Measured peer copy costs can avoid a transfer when a supplied reconstruction cost
is lower. Unknown rates retain configuration order. This is cache-source selection,
not a distributed inference scheduler or a universal remote RAM/disk allocator.

## Execution roles and tokenizer

Both GPU role sets must be specified, contain independent workers and cover the
selected GPU set. They may overlap to permit a combined worker. Root creation and
input preparation use the prefill set; continuation requests use the decode set.
New suffix prefill within a continuation stays with that decode worker. Existing
shared checkpoints, controllers and publication rules perform the handoff. No
running native request is migrated or preempted by this option.

The optional tokenizer requires a separately installed `fastokens==0.3.4`. It
builds its BPE table from the model's own GGUF and downloads no tokenizer. COW
retains its Unicode splitting, protected literal spans, special-token rules and
bounded piece cache. Short pieces use the original implementation; long pieces
use native BPE. Text, grammar admission and workspace projections use the same
startup tokenizer. Missing dependencies or unsupported versions fail before GPU
startup; there is no silent backend fallback or process-wide monkeypatch.

## Bounded qualification

The [final receipt](../experimental/branching_context/evidence/STACK-UNIFICATION-FINAL-20261023.json)
records exact identities and measurements. On A100, a real Wen checkpoint moved
from GPU 3's completed prefill to GPU 0: complete tokens and controller matched,
with zero saved-prefix recomputation. An identical newly created root also reused
that peer image. A complete BW coding workflow passed with combined workers and
with explicit prefill/decode roles. That pair did not establish a speed advantage.

LAN storage tests used llm-60 with llm-49, llm-79 and r730, without WAN cache
transfer to A100. Two real Wen payloads (55,814,626 and 122,972,642 bytes) crossed
each link, passed validation and idempotent retry, and returned to llm-60. All six
native restored responses and controllers matched, with zero prefix recomputation.
Two simultaneous receivers also passed. The test's storage-only listeners needed
no remote GPU or weights; they are a bounded probe, not a shipped cache daemon or
proof of execution on the peers' different GPUs. All four services and their
temporary payload directories were cleaned up.

Fastokens matched 5,180 actual Wen/Qwen token-ID, literal, Unicode, prefix-cut and
chat-rendering checks. Both models passed complete BW code-repair/test/finish,
pause/resume and completed-replay checks with the option off and on. For the
54,000-byte long-piece fixture, cold encoding changed from 7.95 to 3.78 ms on Wen
and 34.34 to 3.29 ms on Qwen. Ordinary code remained approximately 9–10 ms;
warm long-piece cache hits were already fast. These are three-run median tokenizer
timings on one host, not whole-workflow throughput gains. The all-piece native
alternative was rejected because it slightly slowed ordinary code.

All work follows local edits, box-git publication, GitHub publication and remote
pulls. Each increment has its own commit and evidence; deployment never copies
uncommitted source to the serving machine. Promotion to main requires its
acceptance results, not merely an implementation or a passing synthetic model.

## Acceptance matrix

| Boundary | Required evidence |
| --- | --- |
| Feature off | Existing launcher arguments, API behavior, branch/controller identity and completed responses remain unchanged |
| Inventory recovery | Empty inventory, missed/duplicate/reordered events, journal overrun, producer restart, retired epoch and atomic rejection of malformed packets |
| Placement | Warm versus cold costs, checkpoint byte sizes, unknown measurements, stable ties, fairness, cancellation, failed leader and existing branch publication invariants |
| Peer transport | Authentication, exact semantic/execution identity, source lease, bounded staging, payload checksum, interrupted/duplicate transfers, disk full, process death and stale inventory |
| Prefill/decode handoff | Complete hybrid state and controller transfer; request retry, cancellation, sampler/grammar continuity and unchanged full-response/task outcomes |
| Tokenization | Exact IDs for literal controls, new vocabulary, Unicode, code, chat templates and shared-prefix cuts, followed by complete BW task outcomes |
| Performance | Same hardware/model/workload baseline; time to first output, full response latency, throughput, queue delay, transferred bytes, redundant prefill and peak memory/disk |

Tests that require a native profile or external fixture run only in a campaign
that supplies and records it. They are not converted to passes, nor are assertions
weakened. CPU tests establish metadata/queue behavior; real Wen runs establish
the bounded response behavior tested. Broader G1-G5 certification stays separate.

## Scale and failure boundaries

The peer manager holds location and scheduling metadata, not model payloads.
Workers retain local residency authority; cache bytes travel directly between
the selected source and destination. Peer inventories require freshness/expiry
and an explicit resynchronization protocol before influencing remote placement.
Queue size, concurrent transfers, per-peer staging, metadata and recovery packet
sizes must all have limits. Active DMA must settle before its memory can be
released, including after cancellation or timeout.

The current local inventory uses pool notifications and paged full recovery.
CPU tests cover frozen snapshots, duplicate/gapped delivery, notification overrun,
and unchanged-read scan avoidance. A real Wen/BW workflow recovered ten locations
over ten HTTP pages, each delivered twice. Large-store scan cost remains
unqualified. Local moving averages are not a distributed network cost model or
a throughput guarantee.

Disable/rollback uses a stopped, drained server and the prior pinned executable,
source and configuration. Changes to runtime source create a different execution
profile; checkpoints are never relabeled as compatible to make rollback work.
Retained canonical history remains governed by its original contracts. For
ephemeral runs, the existing launcher owns cleanup of session caches and shutdown
of the native worker. No auxiliary service may outlive that ownership silently.

## Inspiration and thanks

Thank you to the DeepInfra contributors, NVIDIA Dynamo and KVCR contributors,
and the fastokens contributors for publishing their work and explaining their
designs. In particular, this implementation was inspired by:

- [DeepInfra kv-local-indexer](https://github.com/deepinfra/kv-local-indexer):
  recovering missed location events through a complete inventory snapshot.
- [Dynamo](https://github.com/ai-dynamo/dynamo) and
  [DeepInfra's Dynamo fork](https://github.com/deepinfra/dynamo): considering
  cached work and worker load together when placing requests.
- [KVCR](https://github.com/ai-dynamo/kvcr) and
  [DeepInfra's KVCR fork](https://github.com/deepinfra/kvcr): keeping cache
  transport and residency separate from GPU execution ownership.
- [fastokens](https://github.com/crusoecloud/fastokens) and
  [DeepInfra's fork](https://github.com/deepinfra/fastokens): the optional tokenizer
  comparison and safe reuse of tokenized prefixes.

The inventory, scheduling, transport and tokenizer adapter code were written for
COW. No upstream repository was copied into this worktree. The optional fastokens
0.3.4 dependency remains a separately installed Apache-2.0 package, with its own
notices. The qualified native library hash is in the receipt. No upstream
contribution is required. Further selective code reuse must preserve its original
notices and license and identify its exact revision.
