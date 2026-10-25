# Prospective working-storage phase reservations

This opt-in owner-side prototype accounts for bytes that cooperating work may
still allocate. It does not launch work, change optimizer or evaluation settings,
manage GPU time, select models, or modify an existing campaign ledger.
No live native-training or COW evaluation policy is qualified by this change.

A native update may reserve space for a new model, optimizer state, merge output,
and later assessment. Once those outputs exist, filesystem free space already
reflects them. Keeping the entire original reservation during assessment counts
those bytes twice conservatively. A phase transition can replace that claim with
a smaller, qualified bound on **additional** allocation. This creates admission
headroom; it does not free any memory or promise a measured speedup.

The guarded resource is usable filesystem capacity (`statvfs`), such as an owned
working directory on tmpfs. It is distinct from GPU VRAM and CPU `MemAvailable`.
The invariant for each admission is:

```
fresh usable filesystem free - all active prospective claims >= owner floor
```

Capacity is sampled inside the same cross-process ledger lock as the change.
Caller-provided free-space snapshots are not accepted. Sampling before waiting
for that lock is unsafe: another process may materialize outputs and lower its
claim while the old, larger free-space sample is in flight.

## Ownership and qualification

`PhaseReservations` lives with PCC's experimental native training owner code.
PCC owns resource leases, immutable artifact identities, and native execution.
S³ supplies assessment requirements and consumes evidence; BW invokes registered
owner policies. This primitive adds no workflow scheduler or acceptance policy.

The owner registers immutable `PhasePolicy` definitions when creating a ledger.
A `QUALIFIED` policy pins a bound-evidence SHA256 and a maximum additional byte
allocation. An `UNQUALIFIED` policy can be recorded but cannot admit work or
replace a claim. A successful completed run, average store size, or chosen safety
factor alone does not establish an upper bound. The hash identifies reviewed
owner evidence; this primitive does not interpret a hash as proof by itself.
The owner must validate that evidence before marking the registry qualified.

Qualification must establish either an enforced total allocation cap honored by
every writer, or a source-derived hard maximum for the frozen workload. Include
COW nodes, raw response files, merged models, optimizer output, temporary files,
and any other allocations on the guarded filesystem. Serialized byte lengths,
filesystem allocation behavior, alignment, and metadata overhead matter. Existing
native/evaluator processes do not acquire a cap merely by holding a lease.

**Current live budgets remain UNQUALIFIED.** Earlier full candidate/state sizes
and completed COW store sizes are observations rather than allocation peaks or
hard bounds. This prototype does not convert them into reduced reservations.
The controlled CPU tests use tiny synthetic qualified policies; those policies
have no authority over any model or real evaluation.

A service must use one designated ledger for all of its claims on a shared
capacity domain. Independently initialized ledgers do not coordinate. Other
accounts and uncoordinated writers can consume the same shared filesystem;
a userspace ledger cannot guarantee a host-wide floor against those writes.
A hard host-wide guarantee requires quota or a separately enforced pool.

## Transition and writer contract

1. Create a new private metadata directory and a private owned pool directory.
   Metadata must reside on a different filesystem from the guarded working pool.
   This keeps ledger temporary files and retained history outside its claims.
   Reopening pins the pool path/device/inode, floor, and entire phase registry.
2. Reserve an original operation ID and qualified phase. An existing or completed
   ID is refused rather than rerun. Reconciliation uses the original ID.
3. Enter `writer_lease(operation, revision=...)`. Pass the returned descriptor to
   **every** process capable of writing, including detached native/evaluator
   descendants, until all their writes are finished. A writer's actual allocation
   must stay within its qualified phase bound. This lease does not enforce bytes.
4. Validate native completion, optimizer/dataset/cursor and reload receipts using
   PCC's existing interfaces. This generic byte-accounting module does not replace
   those model-specific scientific checks. Close writers; make completed artifacts
   read-only; supply their exact relative paths, expected lengths and SHA256s.
5. `transition` obtains the writer lease exclusively and without blocking. It
   rejects any surviving inherited writer, wrong revision, missing/partial file,
   symlink, hardlink, writable/nonprivate output, changed bytes, or unknown next
   bound. It checks actual files, fsyncs them and their named directories, then
   atomically records the new claim under a fresh capacity sample. A larger phase
   is also possible, but must pass admission before acquiring a GPU lease.
6. `finish` performs the same sealed-materialization checks and removes only the
   prospective claim. Actual retained files still reduce filesystem free. It
   does not delete artifacts or rewrite GPU-time accounting.

The parent closes its writer descriptor without explicit `LOCK_UN`: flock is
associated with an open file description. Explicit unlock would release the
protection still needed by an inherited child descriptor. Writable existing lock
descriptors are used for NFS emulated exclusive flock. Lock files are never
replaced or unlinked during normal operation.

The metadata registry and policy API are owner-trusted, not an adversarial client
security boundary. Native and evaluator adapters must prevent bypass writers or
later mutation of sealed artifacts. The primitive cannot infer successful native
completion from file existence or prove an external allocation bound from bytes.
It also refuses a transition when current free space cannot satisfy the pinned
floor plus other claims; even a decrease is conservative under external pressure.

## Crash behavior and evidence

The original operation identity, pinned admission payload/digest, phase revisions,
manifests, previous/new byte claims and capacity samples remain in the bounded
ledger. Every event is checked against the preceding phase/bound/revision, and
the current active or completed state must match the complete transition chain.
A valid JSON record with a smaller qualified current phase alone is refused.
Event and admission digests detect structural corruption; they are not signatures
or protection against a malicious owner rewriting a wholly consistent history. Exact transition replay
returns its original event; changed replay or a stale writer is refused.
A malformed/truncated authoritative ledger is a refusal, not an empty budget.
Unique temporary ledger files are fsynced before atomic replacement; the metadata
directory is fsynced afterward. Abandoned partial files are never completion
receipts and do not override the last good ledger.

A process crash before replacement retains the original conservative claim.
After replacement, the validated materialization and next claim are recorded;
a retry reconciles that exact transition rather than dispatching work. The tests
exercise process crashes, not loss of power or a distributed-filesystem server.
If any original work is uncertain, leave the claim in place. There is no timeout,
PID-only cleanup, unconditional cancellation release, or automatic native retry.
The ledger is size-bounded; archival is an explicit later owner action.

## CPU validation and remaining work

Run from the repository root on an authorized Linux host:

```
python3 -m unittest -v experimental.native_training.test_phase_reservations
```

The tests use real separate filesystems, file descriptors, inherited locks and
concurrent processes. Tiny capacity values are synthetic where tests need exact
admission boundaries. One control invokes the real filesystem capacity sampler.
Coverage includes concurrent admission, sampling while locked, stale revisions,
inherited child writers, original identity/replay, partial output and metadata,
crashes before/after atomic replace, immutable manifests and pinned floor/registry.
No GPU/model work or live reservation mutation occurs.

Before production integration: qualify bounded native and COW writers; bind every
writer descendant to the inherited lease; validate the filesystem's durability
contract; review pool-wide coordination; and integrate only at a cold checkpoint
handoff. Re-run unchanged scientific comparisons through the new owner API and
measure scheduling idle time separately. No throughput claim follows from the
CPU machinery tests alone.
