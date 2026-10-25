# Native learning jobs for a person or agent

PCC now exposes owner-registered AdamW/Muon policies and private named GGUF
preservation through its existing authenticated lesson/job service and CLI.
The same operation receipts, native executor and immutable candidates are used;
there is no second training framework or parallel task ledger. This checkpoint
is qualified with real tiny native CPU updates through HTTP and CLI. A full-model
AdamW HTTP/CLI block has also completed, including native saved-weight reload,
864-case assessment and private named preservation; that block did not improve
on its immediate parent. The full
continuous-run coordinator, BW driver and shared forward-graph consolidation in
[the design](NATIVE-LEARNING-LOOP.md) remain unfinished.

## Owner configuration

The optional training configuration still has `schema`, `directory`, `max_tokens`,
`max_steps` and `runner`. Keep the executable, devices/physical GPU UUIDs, memory
reserves and timeout in owner configuration. Clients never submit those paths or
resource grants. Ordinary dense adaptive training uses
`sequence_profile="dense-llama-short-v1"`, `mode="full"`, a qualified native
trainer and at most512 consumed tokens. Longer inference contexts do not qualify
longer training examples.

Add `optimizer_profiles` to the runner: a map from each profile's SHA-256 to its
exact descriptor. For example, save this descriptor as `adamw-policy.json`:

```json
{"schema":1,"title":"AdamW mixed block, explicit update count",
 "optimizer":"adamw","learning_rate":0.000001,
 "optimizer_settings":{"schema":1,"weight_decay":0.0,"aux_weight_decay":0.0},
 "state_policy":"fresh","training_budget_seconds":null}
```

Calculate its ID with the existing canonical identity function:

```sh
python -c "import json; from tools.pcc_optimizer_profiles import identity; print(identity(json.load(open('adamw-policy.json'))))"
```

Use that exact ID as the `runner.optimizer_profiles` key. Any profile change
requires a different ID. `optimizer` can be `adamw` or `muon`; the settings pass
through the native optimizer's validation. A positive training-phase budget must
be below the owner's hard native deadline; null uses the explicit step cap.
The current learning study uses null: requested updates and actual practice
exposure organize the comparison, while compute cost is measured. An optional
time limit remains available for explicitly timed experiments; it is not the
quality stopping rule. The helper preflights completed-update budget support
before advertising timed profiles. Qualified runs set FP32 activations and
disable TF32.

`state_policy="fresh"` continues a registered parent's WEIGHTS with new moments
and a new data cursor. Use it when choosing a different lesson block. A separate
`state_policy="resume"` profile continues the exact saved optimizer state, step
and cursor; changed example order/content or optimizer configuration is refused.
No client supplies an optimizer-state path. Adaptive profiles currently use
explicit lesson blocks without the legacy multi-round `skill_guard_config`
wrapper. Combining those owner settings is refused rather than silently
misreporting a multi-round budget. S3's pure next-block policy and verified lesson
selection remain usable outside that legacy wrapper.

## Drive a block

Use the existing service URL and `COW_API_KEY` environment variable. Keep keys
out of lesson files, command arguments and published logs.

```sh
python -m tools.cow_cli --url http://127.0.0.1:8099 lesson optimizers
python -m tools.cow_cli --url http://127.0.0.1:8099 lesson submit checked-lesson.json \
  --operation-id admit-example-01
python -m tools.cow_cli --url http://127.0.0.1:8099 lesson train LESSON_ID \
  --operation-id learn-block-01 --mode full --batch-size 1 --steps 68 \
  --learning-rate 0.000001 --optimizer-profile-id PROFILE_SHA256
python -m tools.cow_cli --url http://127.0.0.1:8099 lesson status learn-block-01
```

For a batch, provide the ordered lesson IDs or use the existing queue selection.
Retain its exact selection in the client journal. For the next block include
`--parent-job-id learn-block-01` and the appropriate fresh/resume profile ID.
The job rate must match the advertised profile. A new operation ID is a NEW
update request. Reusing the same ID/bytes reconciles its existing receipt; a lost
response is not permission to train again with another ID.

If an explicitly timed profile is selected, it stops at a completed update and
may finish before visiting every
selected lesson. The cap must be large enough to cover the selection, but that is
not a guarantee the time budget will do so. Read actual completed steps, consumed
presentations and `consumed_lesson_ids_this_job` /
`unconsumed_lesson_ids_this_job`. `pending` in the lesson queue means not yet
selected; it is not proof that every selected lesson received a gradient update.
These fields distinguish completed native execution from curriculum coverage.

The HTTP equivalents are `GET /cow/training/optimizer-profiles`, the existing
lesson/job POSTs (with optional `optimizer_profile_id` on jobs), and existing job
and operation GETs. Default jobs without a profile retain the previous SGD path.

## Test, report, preserve and continue

Let S3 evaluate the candidate and compare both the original baseline and its
immediate parent. Keep a useful improved working checkpoint, practice regression
families in the next mixed block, and retain the original measurements. Formal
qualification and publication are separate decisions. The native job finishing
does not establish learning, forgetting resistance or model mastery.

Preserve the completed full-weight job using its exact SHA from job status:

```sh
python -m tools.cow_cli --url http://127.0.0.1:8099 lesson preserve learn-block-01 \
  --operation-id preserve-model-01 --name MiniCPM-Experiment-01-F32 \
  --expected-sha256 CANDIDATE_SHA256 --notes-file model-notes.txt \
  --report-sha256 S3_REPORT_SHA256
python -m tools.cow_cli --url http://127.0.0.1:8099 lesson exports
```

This creates a private same-filesystem hardlink at the owner's training store
`exports/NAME.gguf` and a bound JSON manifest, without copying the whole model.
It makes the checkpoint read-only. It does not bake an adapter, quantize a model,
change live serving weights, or upload anything. Only complete registered
full-weight jobs are supported. The label has no path or `.gguf` suffix. Notes
and report references are owner-provided metadata, not an independent quality
attestation. Different bytes/provenance cannot overwrite an existing label.
Repeating the identical preservation operation returns the same acceptance.
Listing verifies manifest identity and storage size; it does not rehash every
multi-gigabyte file on each list operation. Explicit preservation checks the full
candidate hash. Quantized deployment formats need their own
[behavioral study](MINICPM-QUANTIZATION-STUDY.md).

The HTTP preservation mutation is `POST /cow/training/exports` with operation ID,
registered job ID, label, expected SHA, notes and optional report SHA. It accepts
no arbitrary filesystem path. GET lists named private exports. There is no model
upload or automatic download operation in this feature.

Repeat **test → choose lessons → train → test → report → preserve → next block**.
A person or agent can drive these commands now; automatic durable multi-round
policy orchestration and its interruption/recovery qualification are still being
built. Do not describe the entire continuous pathway as finished.

## Evidence

[Real HTTP/CLI native CPU workflow](../experimental/native_training/validation/native-learning-jobs-cpu.json)
covers six actual AdamW/Muon jobs, unchanged-data optimizer resume, changed-block
fresh moments while retaining parent weights, exact uninterrupted-versus-resumed
weights/moments, partial-budget visibility, duplicate-operation handling and
idempotent named GGUF preservation.24 refusal/corruption cases pass.41 existing
CLI/lesson/schema/profile tests also pass. The CLI tests now isolate their
journals so a previous test run cannot collide with a user's persistent operations.
These are mechanics results, not full-model learning or new GPU qualification.

## Per-update studies and native LoRA

The current comparison uses matched updates and identical ordered practice.
Leave the profile training budget null; measure compute as an outcome. Optional
time budgets remain available for explicitly requested bounded jobs and historical
receipts, but they are not this study's organizing principle. The native
`record_timing` option collects per-update phase costs without a training-phase
time cap. An execution deadline remains a hang/lifecycle guard.

[Native per-update mechanics and experimental LoRA](NATIVE-PER-UPDATE-CURVES.md)
describes exact state continuation, the new AdamW LoRA executor route and native
F32 merged export. Registered HTTP/CLI profiles still select full-weight jobs;
the new LoRA route is an explicit experimental native-owner integration.
