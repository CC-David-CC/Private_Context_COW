# Recover an original native completion without training twice

This owner-driver pathway handles a narrow crash: native training and its
independent reload finished, but the service did not commit the completed job.
Inspection, adoption and completion reconciliation read the original evidence
and change service metadata. They never start native training, inference or an
evaluation, and never promote a model or approve a lesson.

A person, authorized agent or smaller model driving the owner harness can use
the same commands. The student model's tools and permissions are unchanged. The
explicit adoption flag records an already authorized owner decision; it does
not require a new human confirmation when that scope has been delegated.

## Owner-driver sequence

Use the existing authenticated COW connection and a private client journal.
Keep the original job, run, step and operation IDs. Do not invent replacement
IDs to get past an interrupted operation.

```text
python -m tools.cow_cli --state-dir OWNER_JOURNAL lesson inspect-completion JOB
```

The response is typed JSON. Follow `state`, `reason_code` and `next_actions`:

| State | Next action | Meaning |
| --- | --- | --- |
| `eligible` | `adopt_original_completion` | Original bytes, completion and reload gate are currently verified. |
| `adoption_pending` | `resume_original_adoption` | Finish the exact recorded metadata approval, using its existing operation ID and request. |
| `adopted` | `reconcile_learning_run` | Fresh verification still matches the adopted candidate. Reconcile its original learning round if applicable. |
| `unavailable` | `hold` | Work is live, admission is busy, or evidence is incomplete. Inspect again later; no skill or learning failure is inferred. |
| `ineligible` | `hold` | Original provenance is missing or changed. Do not reconstruct a success from guesses or start another update implicitly. |

`eligible` is true only in the `eligible` state. The separate
`current_evidence_verified` flag is true for verified eligible, pending and
adopted states. Hold states omit adoption hashes. Fixed hold reasons are
`legacy_provenance_missing`, `training_boundary_busy`, `original_process_live`,
`original_evidence_incomplete`, `original_evidence_invalid` and
`adoption_transaction_conflict`. Unknown states or older servers are HOLD.

A normal service-committed job needs no completion adoption. It returns
`ineligible` with `reason_code: original_job_already_complete` and
`next_actions: [inspect_original_job, learning_status]`. This is an ordinary
completed job, not corrupt evidence or a model failure; this inspection does not
claim a fresh native evidence check for it.

For an eligible completion, use its exact inspection hashes:

```text
python -m tools.cow_cli --state-dir OWNER_JOURNAL lesson adopt-completion JOB --operation-id ADOPTION_ID --expected-job-record-sha256 ORIGINAL_RECORD_SHA --expected-evidence-sha256 EVIDENCE_SHA --expected-candidate-sha256 CANDIDATE_SHA --approve-original-completion
```

These compare-and-swap hashes bind the original interrupted record, original
evidence and exact candidate. An intervening change refuses the commit. For
`adoption_pending`, use exactly `pending_operation_id` and `pending_request`;
changing the ID or any approval field refuses. The existing request journal
retains uncertain requests. Only this metadata approval can be replayed to
finish its original transaction; native training requests retain their ordinary
no-blind-retry behavior.

An adoption acknowledgment is historical. Replaying the same completed
operation returns the same acknowledgment even if an artifact later changes.
Inspect again to establish current eligibility. A historical acknowledgment
cannot by itself move a learning round forward.

For an associated learning run:

```text
python -m tools.cow_cli --state-dir OWNER_JOURNAL lesson learning-status RUN
python -m tools.cow_cli --state-dir OWNER_JOURNAL lesson reconcile-completion RUN --operation-id STEP_ID --expected-revision REVISION
```

The dedicated command requires the exact current revision, `needs_operator`,
`operator_reason_code: native_job_unresolved`, and the advertised `reconcile`
action. The server independently revalidates the original request, parent,
round, source, configuration, completion, reload gate, optimizer state and
candidate bytes. It changes only that original round to `candidate_pending`;
it does not advance, evaluate, train or report it. A revision race refuses.

On retry the CLI gives the original journal and operation acknowledgment
priority: the server may already have reached `candidate_pending` after a lost
reply. It does not reject that successful original acknowledgment merely
because the phase changed, create a new step ID, or advance twice. When there
is no acknowledgment, an exact metadata replay still requires the current
completion-specific phase/reason/revision/action preflight. Generic recovery
from `training` or `training_pending` is not this command.

The matching BW owner-driver extension exposes `lesson inspect-completion JOB
--output FILE`, `lesson adopt-completion JOB --inspection-file FILE
--operation-id ID --approve-original-completion`, and completion-specific
learning status/reconciliation. It binds the private inspection file to the
existing endpoint/model connection and durable operation journal. This is an
owner interface, not an added student tool or authority to approve its own
lesson. BW integration is qualified separately against the combined PCC source.

## What evidence is required

Recovery is opt-in **before the original training dispatch**. Owner native
configuration enables `completion_recovery` with schema 1 and an explicit
`source_closure` mapping of absolute source/native artifact paths to SHA-256.
The executable must be present. Use the actual qualified build dependency
closure, including linked runtime libraries, not merely an executable label.
The implementation also pins its training, optimizer, tokenizer, rendering and
recovery sources. This owner-declared closure is not a new universal kernel or
model-quality qualification.

The normal opted-in path saves private original request/configuration, exact
lesson rendering, source hashes, relevant runtime settings and native process
identity before committing completion. After the original native child exits,
it performs one separate forward-only reload on the actual final completed
batch, including partial batches and resumed cursors. It retains the original
native stdout unchanged and compares the reload mean loss to the original last
post-update loss with the existing absolute bound `2e-4`. A separate gate records
the result; the native receipt's `save_reload_qualified: false` is not rewritten.

The native and reload calls use the same executable, devices, runtime settings
and inherited worker/GPU lease descriptors. An explicit bounded CPU/CUDA
allowlist records numerical, backend-selection and loader settings such as
`GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_DISABLE_FUSION`,
`COW_QWEN35_F32_ATTENTION`, activation precision, TF32, device selection, graph
and CPU fusion settings, threading and loader paths. It does not record the
whole environment or credentials. Actual native/reload environment drift
refuses before dispatch; later configuration/source/runtime drift refuses
adoption. Unsupported backend semantics require their own reviewed binding.

Inspection verifies exact original native and reload PID/start/boot identities
are gone, zero exit codes, finite losses/gradients/state payloads, step and data
cursors, unchanged base/adapter/state bytes, original sources and the immutable
candidate envelope. It uses existing nonblocking record/worker locks without
creating missing lock files. Existing 0644 lock files inside the owned 0700 store
are accepted without mode or inode replacement; aliases and nonregular locks
are refused. Adoption commits a durable intent, immutable candidate version,
original job update and acknowledgment. Each crash boundary can resume that
same metadata transaction. Original errors and interruption evidence remain.

Legacy jobs lacking the original source/process/reload proof are refused.
Enabling recovery later cannot create historical evidence, and existing frozen
studies are not retrofitted. Inspection returns hashes of native configuration,
source closure and runtime settings; it does not invent a serving execution
profile for training metadata.

## Qualification bounds

CPU tests use clearly labeled saved **NONMODEL** bytes and fake child processes.
They cover explicit inspection/adoption, exact replay and commit crashes,
busy/inherited locks, strict legacy refusal, altered original/adopted bytes,
source/configuration/environment drift, numeric and optimizer-state corruption,
typed hold states, authenticated HTTP, journals, lost acknowledgments and
completion-only learning reconciliation. Native/model/GPU qualification of this
combined feature is separate. Default non-opted-in execution retains one native
call and its original request; no reload is inserted there.
