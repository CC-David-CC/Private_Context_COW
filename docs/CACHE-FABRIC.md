# Cache inventory and placement experiment

This experiment extends COW's existing checkpoint ownership and weighted queue.
It does not replace immutable histories, branch revisions, complete controllers,
checkpoint checks, RAM/disk reservations, or BranchWrite's workflow ledger.

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
