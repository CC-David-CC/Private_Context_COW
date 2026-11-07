# Backend extensibility: initial source audit

Implementation recommendations are superseded by [MODEL-EXTENSION-PLAN.md](MODEL-EXTENSION-PLAN.md). In particular, materialization residency is multidimensional, not the linear sketch below. Source observations remain the initial audit record.

Inspection only, 2026-11-07. No runtime behavior changed or conformance claimed.
Scope: backend setup/factories, leases, materialization, coordinator requests,
capabilities and training entry point; BW/SSS client identity boundaries sampled.
This is not an exhaustive concurrency or training-native-code audit.

## Observed coupling

- tools/cow_server_setup.py:117-156 dispatches on model architecture and has a
  dense-specific LoRA gate. serve/cow_http.py:1038 and real.py:load_backend repeat
  backend construction differently. HTTP falls through to Strata; load_backend
  rejects unknown names. New adapters need multiple synchronized edits.
- capabilities.py:describe_dense/describe_strata bind capabilities to backend
  names and protocol strings. Preserve validation, but let a checked backend
  descriptor own these facts. Dense/MoE is model topology, not a complete runtime
  capability: input types, state codecs, sampling and training differ separately.
- qwen35_backend.py:_acquire/_release/close represents lifecycle through model,
  _active, _owner and _close_pending. qwen35_mixed_backend.py adds _inputs and native
  poison state. Explicit transitions should reject use after close/poison, wrong
  lease/thread and partial consume before any native operation.
- coordinator.py:Request.state, snapshot/durable futures, pause_event and _Job
  source/prefill leases jointly encode execution and publication. These are
  separate lifecycles, not one giant enum. Preserve committed-but-not-yet-published
  recovery, idempotency and cancellation settlement.
- real.py:restore_checkpoint requires exact state and zero forward work;
  rebuild_materialization explicitly replays retained history. Keep these distinct.
  Reconstruction eligibility currently contains backend-name/protocol special
  cases. Move eligibility to validated operation contracts, not permissive flags.
- tools/pcc_native_training.py:run validates immutable base, LoRA/full mode,
  sequence/memory profiles, optimizer resume identity, dataset and cursor.
  Model inference support does not establish gradient or optimizer support.
  Training publishes a new candidate identity; never mutate a serving model or
  reuse old checkpoints after changing weights/adapters.

## Proposed boundaries and transitions

1. Checked backend registration: detect -> validate metadata -> resolve native
   binding -> qualify operation descriptors -> construct. Unknown is an error.
   One factory serves CLI/HTTP. Keep persisted profile formats compatible.
2. Native worker lifecycle: unloaded -> ready -> leased -> ready; a failed native
   mutation transitions to poisoned -> close/recreate. Closing a leased worker
   requests deferred close. Each transition owns exactly one cleanup obligation.
3. Context materialization: absent -> reconstructing -> resident -> exporting ->
   durable; resident/durable copies can be evicted independently. Canonical input,
   controller and immutable snapshot outlive every numerical materialization.
   Strict restore validates codec/profile/history before loading and proves no
   forward work. Reconstruction emits its own work/schedule receipt.
4. Request execution and durable publication are orthogonal transition tables.
   Keep current states/receipts; first centralize transition checks and traces.
5. Training: validated -> admitted -> running -> candidate written -> verified ->
   published, or cancelled/failed with cleanup. Capabilities independently state
   frozen-base LoRA inference, LoRA training, full-weight training, optimizer
   resume and supported quantizations/devices. Unsupported combinations fail
   before resource acquisition. Promotion creates a new model/profile namespace.

## Implementation order / acceptance

- First add characterization tests around current behavior, including both
  factories rejecting unknown backends consistently and unchanged known profiles.
- Introduce a small registry and shared factory, with descriptors validated against
  native facts. Avoid a general plugin framework or a new parallel storage model.
- Add lifecycle transition guards one component at a time. Test cancellation at
  every allocation/consume/export/publication boundary, poison recovery, wrong
  leases, duplicate retries, concurrent eviction, corrupt checkpoints and restart.
- Preserve LoRA/full training regression cases: unchanged base, adapter scaling
  and hash, model/dataset/cursor optimizer mismatch, resumed-vs-uninterrupted
  updates, cancellation and candidate promotion/cache incompatibility.
- Add Qwen3.6 backend through those boundaries; validate fresh oracle, forks,
  exact restore, reconstruction and memory pressure on real devices.
- Only then merge the reviewed/tested feature into main. Short generation checks
  already passed but do not qualify any of these lifecycle/training guarantees.

## BW and SSS (historical recommendation, superseded)

The owner subsequently removed unchanged BW/SSS compatibility as a requirement.
The current plan permits versioned PCC interface changes and future client
rewrites. The following records the earlier inspection, not an active gate.

### Original inspection

BW branchwrite/base_cli.py pins connection model/profile IDs and validates receipts;
branchwrite/capabilities.py already validates server capability requirements.
SSS sss/adapters/artifact_evaluation.py binds evaluation to model contract and
PCC renderer. Preserve these external contracts and try both clients unchanged.
Some SSS paths call PCC Python helpers directly, so HTTP compatibility alone is
insufficient. Run existing BW/SSS suites plus real serving/evaluation flows before
claiming no changes needed. No client modifications proposed from this inspection.
