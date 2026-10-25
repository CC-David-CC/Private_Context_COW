# Working storage for native learning experiments

`tools/pcc_training_workspace.py` adds owner-side resource accounting and
artifact-storage helpers for Linux native training. It does not implement an
optimizer, curriculum, workflow runtime, remote filesystem API or acceptance
policy. S3 still selects lessons and evaluates learning/forgetting; BW drives
the existing job/operation APIs. Native execution remains in PCC.

This is an opt-in owner/helper boundary used by the parallel curve experiment.
Automatic product orchestration and consolidation of inference/training forward
graphs remain unfinished. The helpers do not make the entire learning loop
durable automatically.

## Local working files and durable milestones

An owner may allocate a private RAM filesystem directory for active model and
optimizer-state files. The files are ordinary immutable native artifacts with
the same SHA identities; moving them within one host does not train or convert
them. CPU/GPU model loading and serialization can then avoid a slow shared
filesystem. Check both RAM-filesystem capacity and physical-memory headroom;
other users can consume either resource. Never put an unbounded checkpoint
history in RAM.

After a native update finishes, independently reload it and assess it. Preserve
the original requests, native receipts, scores and response evidence before
retiring a superseded cache entry. `preserve_evidence` writes those nonbinary
files into a private tar archive with individual source hashes. Model and
optimizer binaries are excluded from that archive and retain their own hashes.

`copy_checkpoint` persists the exact completed model and moments into a new
owner directory. It hashes the sources, copies and fsyncs the files, verifies
the copied bytes, makes them read-only and writes a bound verification receipt.
It does not merge an adapter, quantize weights, promote a model, or transfer it
to another host. The caller selects milestones and keeps their identities.

`retire_cached_candidate` accepts only exact owned cache roles, a later qualified
checkpoint and preserved evidence. A declared milestone additionally needs its
verified durable copy. Retirement writes and fsyncs its intent before unlinking.
An interruption follows that same intent; it does not choose different files or
invent another operation. Changed bytes, links, missing original files and paths
outside the private workspace are refused. Original bases and current/best
working candidates remain the caller's protected identities.

RAM files are volatile. After a machine failure, reconcile original native job
handles and use the last verified durable checkpoint. A missing RAM candidate
does not authorize claiming its bytes survived, silently resetting optimizer
moments, reusing its scores for a different model, or repeating an uncertain
training operation. A new recovery trajectory needs explicit provenance.

## Resource ceilings are separate from experimental training budgets

`ResourceBudget` reserves bounded device-allocation seconds and prospective RAM
bytes under an owner filesystem lock. It refuses GPU or RAM overcommit before
execution. The owner supplies the ceiling, prior conservative accounting,
physical resource observations and exact operation identities. Settlements use
actual allocated device wall time, including contention; uncertain work retains
the entire reservation as a conservative charge until owner reconciliation.
Repeated settlement of the same operation returns its existing receipt.

Its private directory contains `resource-budget.json` with `schema: 1`, the
owner's `ceiling_gpu_seconds`, `prior_upper_bound_gpu_seconds`, the sum of new
`charged_gpu_seconds`, and `active` / `settled` operation maps. Initialize empty
maps only for a genuinely new accounting scope. Reopening an experiment uses
its existing file. `reserve` binds an original operation ID to its bounded phase,
RAM requirement and owner process identity; `settle` reconciles that same ID.
These records account for resources and do not replace PCC's native job receipts.

An upper bound is labeled as such. It must not be reported as measured GPU use
or an exact remaining balance. Reconcile it from original allocation receipts
and live/terminal process identities before revising it. A stale lock or an
observation timeout does not prove a child stopped. Canonical per-GPU leases
remain owned by the native executor and inherited by its children.

These are lifecycle/resource guards. The learning comparison still uses
explicit completed updates, examples and supervised tokens, with no fixed
training-phase time budget. Different arms have independent phase operations;
a failed LoRA qualification does not disable full-weight jobs.

## Checks

```sh
python -m unittest tools.test_training_workspace -v
```

The tests use actual owned temporary files for reservation refusal, measured and
unknown settlement, verified checkpoint copying, evidence retention, corruption
and link refusal, and interruption after the first unlink followed by original
intent reconciliation. They establish storage mechanics, not language learning,
new GPU numerical correctness or resistance to hardware bit rot. The experiment
records its real-model execution evidence separately.
