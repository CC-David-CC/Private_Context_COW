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
The journal is derived from observations at polling time, not an audit log of
every transient cache mutation. Inventory is capped at 16,384 locations. Its
bounded Python metadata is separate from checkpoint payload budgets.

Transfer-aware placement retains grouped-prefix admission, in-flight prefill
coalescing and two-second aging. It compares measured forward-work cost and
native restore cost per byte for compatible idle workers. A slow warm worker
can lose to a faster cold worker when measured restore plus compute is cheaper.
Missing rates retain stable warm affinity. Rates are process-local moving
averages; output work is a bound, not a prediction of actual response length.
Forward service observations include host orchestration and capture overhead;
the separately measured native restore interval is subtracted to avoid counting
it twice. Disk preload is currently a shared-server cost, not a measured LAN
transfer. No network bandwidth estimate or cross-server placement is claimed.

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
Experimental code remains on its experimental branch pending those checks.

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
| Peer discovery/transfer | Separate explicit peer configuration, not yet implemented | No peer listeners, polling or transfer tasks | Planned |
| Prefill/decode separation | Separate execution-role option, not yet implemented | Ordinary combined serving | Planned |
| Tokenizer backend | Separate explicit backend choice, not yet implemented | Existing Wen tokenizer | Planned |

No future flag is advertised as currently accepted by the launcher. Compatibility
APIs and BW's native API remain independent of these switches. Missing optional
peer data must permit ordinary serving, but present incompatible or corrupt
checkpoint data remains an integrity error. A fallback must not silently replay
tools, reset a controller, change input tokens, or violate a strict restore
request. Reconstructing an evicted cache uses COW's existing explicit semantics.

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

The eventual manager holds location and scheduling metadata, not model payloads.
Workers retain local residency authority; cache bytes travel directly between
the selected source and destination. Peer inventories require freshness/expiry
and an explicit resynchronization protocol before influencing remote placement.
Queue size, concurrent transfers, per-peer staging, metadata and recovery packet
sizes must all have limits. Active DMA must settle before its memory can be
released, including after cancellation or timeout.

The current local inventory reconciles disk metadata during polling. It is
bounded, but its scan cost has not been qualified for large stores. Event-driven
pool notifications and a paged full-inventory protocol are candidates for that
qualification; neither is claimed implemented. Likewise, local moving averages
are not a distributed network cost model or a throughput guarantee.

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
  [DeepInfra's fork](https://github.com/deepinfra/fastokens): the planned tokenizer
  comparison and safe reuse of tokenized prefixes.

The inventory journal and placement code here were written for COW; no upstream
implementation was copied into this increment. No upstream contribution is
required. Further selective code reuse must preserve its original notices and
license, as well as identify its exact revision.
