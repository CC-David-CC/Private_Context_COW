# Shared checkpoint pools

The private prototype keeps one immutable history/controller store. GPU images,
RAM payloads and disk checkpoints are optional materializations of that history.
Clients select branches; the coordinator chooses storage tiers.

## GPU selection

`GPU` is the only worker-count setting. Its outer list selects execution workers.

| Configuration | Workers | Physical GPUs |
|---|---:|---:|
| `[0]` | 1 | 1 |
| `[1,5,6]` | 3 | 3 |
| `[[0,1,2]]` | 1 split worker | 3 |
| `[[0,1],5,6]` | 3, including one split worker | 4 |

Groups cannot overlap. Selection/routing and mixed-group scheduling have CPU
tests. Real model qualification currently uses only GPU 0 on llm-60.

**The pinned native Context COW adapter rejects split workers.** Stock Strata
supports layer splitting, but the private BranchSession adapter has no verified
multi-device snapshot protocol. Device-tagged ownership, capture/restore of all
layer states and synchronization across the group must be implemented and tested
before enabling it. The configuration syntax is not a claim of native split COW.

## Request lifecycle

1. Submit an operation ID, branch, input token IDs and output bound. An optional
   expected revision and authenticated principal preserve existing CAS/access rules.
2. The oldest eligible request takes one compatible execution worker. Requests
   sharing a branch wait for its preceding publication.
3. Activate GPU state, or restore immutable RAM bytes. A RAM miss reads and
   validates the disk checkpoint. Retained prefixes are not recomputed.
4. Stream consumed-token progress. Capture the completed continuation once.
5. Emit `snapshot`. The complete candidate is available, but its branch move is pending.
6. A background writer publishes and flushes the existing payload and manifest,
   validates saved bytes, then commits the branch and retry receipt. Emit `durable`.

Independent branches can execute while the writer finishes. A disk failure leaves
the previous branch committed and the candidate RAM bytes pinned. `retry_failed`
retries publication without generation; `discard_failed` explicitly abandons it.
Closing the coordinator abandons volatile failed publications, never acknowledging them.

Operation intents contain request fingerprints and source references, not another
history. After restart, a committed retry is answered from the Store's receipt,
with checkpoint validation and no new inference.

## Budgets and eviction

The shared RAM pool counts unique immutable payload/frontier extents plus reservations.
Capture, disk-read and GPU-upload/deserialization staging are reserved before use.
Native model/arena memory and Python object overhead are outside that extent budget;
host `MemAvailable` headroom is checked separately. It is not a whole-process RSS cap.

Evict only clean, unleased RAM entries in LRU order. An eviction does not write a
checkpoint. Dirty entries remain protected until publication; pressure causes
admission waiting or an explicit capacity error. The next eligible queued context
can preload from disk without touching an active GPU lease.

Disk retention protects roots, live tips, named saved points, fork points and
active/queued read leases. Numerical GC never removes canonical history. Exact
payloads can share one allocation/file; different prefix-related payloads remain
owned copies. No host prefix-page deduplication is claimed.

Only current source/tip GPU images are retained for the current root; switching
roots releases inactive handles. Fixed model and native arena allocations remain.

## Run the bounded demonstration

Run on Linux with the existing pinned engine/model profiles. These commands do
not start the chat server:

```sh
cd experimental/branching_context
python3 scripts/verify_pool_control.py --output evidence/pool-control.json
python3 scripts/pool_matrix.py --output evidence/runs/pools-new-run
```

The matrix uses two prefixes, eight trees/four simulated client streams, separate
producer/consumer Python and native processes, Q4/Q8 and independent uncached
matching-operator references. It checks four consumed outputs and the final
boundary; output limits are 8 and 128. EOS ends a turn honestly. Finished turns
are validated after restart without requesting more generation.

Edit the machine-specific plan's `GPU`, pool budgets and pinned profile path for
another installation. Device selection is bound into the execution profile.
Changing hardware, model, execution layout or topology requires qualification;
old checkpoints are not silently treated as interchangeable.

The Windows checkout uses `python`, but its Store still needs POSIX `fcntl`.
Native Windows execution is not supported by this milestone.

## Evidence boundaries

This is a greedy, execution-profile-bound CLI prototype. It does not qualify
original G1–G5, arbitrary batch partitions, stochastic/grammar generation,
ordinary serving integration or real multi-GPU operation. Initial pool fixtures
use synthetic token sequences, not measured codebase-answer quality.

Model-free CUDA allocation/COW fixtures can later test selected A100 devices
without transferring weights. They establish memory/transport behavior, not
model inference or numerical equivalence.

### Planned model-free data fixtures

Do not use a constant value as the sole fixture. Generate reproducible, bounded,
finite values from a recorded seed and logical layer/page/token/channel indices.
Identical logical data must have identical values across devices, so physical
placement does not change the reference. Use separate device-specific canaries
to detect routing mistakes and distinct markers around page boundaries.

Include zero, alternating signs and mixed magnitudes. Check copy/restore byte
identity against an independently generated CPU buffer. Check any arithmetic
against an independent CPU implementation, with precision-specific expectations
declared before execution. Test partial-page writes, sibling isolation, eviction,
restart and wrong-device rejection. These fixtures are planned, not executed GPU
evidence; they cannot establish real-model correctness.
