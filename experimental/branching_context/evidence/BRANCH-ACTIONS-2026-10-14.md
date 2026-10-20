# Branch action checkpoint

`ContextCoordinator.fork` keeps its legacy call path intact. Callers can opt into namespace association with `source_branch` and a stable `operation_id`; the coordinator derives the namespace from that branch, accepts only its current ancestry or a namespace-catalogued savedpoint, pins and validates the serialized checkpoint, then performs the idempotent Store fork. Identical retries use the catalog receipt; operation reuse with another target fails before creating a retention pin.

`rewind_presented` is a separate typed operation. It requires a namespaced branch, current expected revision, ancestor target, and valid durable checkpoint. It derives renderer identity from the published namespace root, records source snapshot/revision/history length and target turn completion in the Store move receipt, and replays the original typed receipt without model work after later branch movement. Pending work and stale revisions fail before a move.

Legacy `rewind()` retains its original condition lock across the pending-work check and Store move. A concurrency regression pauses that move and verifies same-branch admission waits, then fails its stale revision check.

The fork/rewind paths acquire read leases before rechecking the branch head; the initial run showed that nesting a lease acquisition under the Store flock deadlocked. Checkpoint verification calls the loaded model's CPU file/hash/frontier validator and fails closed on missing or corrupt data. No native lease is entered.

Verification and exact source hashes are in [BRANCH-ACTIONS-2026-10-14.json](BRANCH-ACTIONS-2026-10-14.json). llm-60 CPU focused tests: 7/7 passed. Existing G0: 51/51 passed. These are synthetic control results, not GPU qualification.

The Store fork receipt currently records target branch, snapshot, and namespace, but omits `source_branch`. Two source branches sharing a computational snapshot but carrying distinct presentation histories therefore cannot be distinguished by the durable fork receipt; this remains a provenance limitation for a later Store schema change.

Remaining: no automatic `retry_failed` or HTTP retry recovery was added; callers still need the existing explicit recovery path after a publication failure. The real native/GPU path is outside this checkpoint.
