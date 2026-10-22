# Weighted prefix admission

`tools/cow_server_setup.py --scheduling weighted-prefix` opts in to scheduling
independent configured GPU workers with shared completed RAM/disk checkpoints.
The default is `fifo`; existing server configuration need not change.

Jobs are grouped by their immutable source snapshot, exact root input spans or
prepared root input and preparer identities. While one slot is building a missing
prefix, matching jobs wait in the queue; unrelated prefixes may use other slots.
The derived prefix is admitted to the existing shared checkpoint pool before
followers are released. Rebuilt source histories are also published as derived
checkpoints, without moving their logical branches.

Among runnable jobs, queued group size divided by one plus the group's active
slot count determines priority. Requests waiting two seconds take FIFO priority
over newer weighted work. Compatible warm workers are preferred; measured
seconds per forwarded token breaks equivalent placement ties. This is an
admission heuristic, not a guarantee of optimal throughput. Native batches
already in progress are not migrated or preempted.

`/cow/status` includes scheduling policy, queued and active counts by hashed
prefix identity, in-flight prefill claims, waiting followers and measured worker
service rates. A failed leader releases its claim. A cancelled waiting follower
cannot release another request's claim. Existing pool reservations and limits
account for shared source checkpoint capture, transfer and publication.

Flat GPU selections represent independent workers. Grouped tensor-split native
checkpoints and a scheduler spanning separate HTTP servers are not qualified by
this option. CPU queue correctness does not establish GPU throughput improvement.
See `experimental/branching_context/evidence/WEIGHTED-PREFIX-20261022.md`.
