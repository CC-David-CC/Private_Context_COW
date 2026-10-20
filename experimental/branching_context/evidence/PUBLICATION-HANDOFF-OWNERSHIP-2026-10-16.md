# Publication handoff ownership — 2026-10-16

Primary review accepted this as a CPU-qualified candidate, not an integrated
or GPU-qualified change. The unchanged final principal test was rerun RED
against the preceding coordinator and failed as intended. The private compact
backup is independently verified at
`/srv/data/git/private_context_cow.evidence/20261016-publication-handoff-ownership-controller-v1/evidence.tar`:
1,689,600 bytes, SHA-256
`809abde072cd8a866d01b722141f526d528506f771fdd02a06e90416a1e2ce84`.
Streaming readback checked all 240 members / 1,418,857 data bytes without
extraction; transfer/readback command and wrapper exits are all 0/0.
Model/native payloads are excluded; this ticket ran no model or GPU work.
Unchanged source closure is reconstructed from private Git base40ac335d plus
the archived owned-source overlays. See CPU and BACKUP JSONs beside this file.

Earlier worker packaging attempts remain preserved: v1 stream-member path
mapping failed, v2 refused an already-created bundle, and v3 found its manifest
builder missing from the declared backup-tools location. Root used the proven
compact archive/readback helper instead. These are packaging failures, not
failed ownership tests. Available failed-attempt receipts/scripts are archived;
the partial staging bundles remain on llm-60.

## Scope and result

Final CPU qualification: the ownership suite passes 8/8, the related regression set passes 37/37, and G0 passes 51/51; all have zero skips and command/wrapper exits 0/0. The fix keeps ownership of a prepared checkpoint, its shared-pool pin, and its canonical read lease with the worker until native cleanup and publication submission both succeed. If pre-submit work fails after a checkpoint was admitted, eligible bytes and the operation envelope remain in the existing failed-publication retry machinery as an ordinary unpinned LRU entry. The writer owns them only after `_submit_publication` accepts the gated task. The retry record also preserves the principal-bound model used for the original request.

The change is limited to `coordinator.py` and the new deterministic ownership test module. No public API, checkpoint representation, cache policy, tolerance, or existing test was changed. Real GPU/model behavior remains unqualified by this ticket.

## Test-first evidence

Frozen base: `40ac335dffa44b38817d665acdbfd3a19763682c`.

The first RED staging attempt (`red-v1`) was an authoring error: its capture-failure case assumed the snapshot future had succeeded and its synthetic history expected one extra token. It is retained but is not backend evidence. Corrected test-only stage `red-v2` ran 4 tests: the entry-limit 1/2 blocked-writer test and capture-failure test passed; the backend-release-failure and publication-submit-failure tests failed because one pool pin remained after each failed handoff. The frozen RED test SHA-256 was `50a4673d859eedd2e3c8a2187dca1c58742fba39a97ee8e431ae92f9c9dadd61`.

## Frozen GREEN stage and verification

Remote stage: `/home/dflanag3/spatial-artifacts/publication-handoff-ownership-20261016-v1/green-v1/source`.

- `coordinator.py`: SHA-256 `f054a23565316df842b07ff8464db87a2a6e7185dfd2c7a0fb2f0b824ad3fde3`
- `test_publication_handoff_ownership.py`: SHA-256 `50a4673d859eedd2e3c8a2187dca1c58742fba39a97ee8e431ae92f9c9dadd61`
- The source tree was frozen read-only before execution; its two owned files had the same hashes after all runs.
- Runner: `/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh`, SHA-256 `7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537`.
- Interpreter: `/home/dflanag3/src/Strata/.venv/bin/python` (Python 3.12 environment), `-B`, with bytecode writing disabled.

The new module passed 4/4 tests (command/wrapper exits 0/0) at `green-v1/run-tests-v2`. Existing regressions passed 25/25 (0/0) at `green-v1/regressions-v1`: publication overlap, cold-cache ownership, poisoned-worker replacement, and bounded tree retention. G0 passed 51 tests, with 0 required real-backend cases (0/0) at `g0-v1/run/runner`. These are historical, narrower results superseded by green-v2 and then principal-green-v1.

The first GREEN runner setup used a pre-created run directory, which the approved runner correctly refused. That setup refusal is preserved at `green-v1/run-tests`; the successful run used the fresh `run-tests-v2` directory. No test or runtime process was left running.

## Ownership details

On the success path, native checkpoint cleanup now completes before `_submit_publication`, so cleanup errors cannot occur after ownership transfer. If cleanup or submit fails first, the worker exits the pool pin and store lease, marks the nondurable pool entry failed while retaining eligible bytes in the ordinary LRU, and then invokes the existing job-created-image cleanup. The tests exercise blocked asynchronous publication under cache limits 1 and 2, canonical lease/pin retention during the block, capture failure after target admission, injected backend-release and submit failures followed by a successful retry, and principal-bound retry publication.

The tests use the existing synthetic `NativeImageModel`; they do not establish real CUDA behavior. The known baseline conflict around the old `test_pool_branch_actions` missing-manifest expectation was not run or modified.

## Retry-preserving handoff follow-up

Review required the pre-submit paths to preserve eligible prepared bytes in the existing failed-publication retry machinery, and required retry submission rejection to leave a terminal failed future. The final tests add same-operation retries after backend-release failure, publication submission rejection, and retry submission rejection; they check that native forward work does not increase on retry, that only the target branch advances, and that an identical already-durable pool alias remains durable. An executor stub also queues the writer callable and then raises; the queued orphan is executed after failure and must be a no-op. This exercises the submit acceptance gate rather than assuming `submit()` always raises before queue insertion.

The new baseline stage `red-v6` ran all 7 ownership tests: 5 failed and 2 passed (command/wrapper 1/1). The failures exposed the pin leak, nonretryable/lost bytes after pre-submit failures, retry submission raising without terminalizing the replacement future, and an orphaned queued writer being able to publish after the request had reported handoff failure. The separate retry-focused RED at `red-v5/run-retry-red` also failed 1/1 with command/wrapper 1/1. The full `red-v5` run entered all tests and printed 5 failures/1 error, then stalled in teardown; the owner terminated its exact process group. Its runner is marked RUNNING and is explicitly not counted as a terminal receipt. Earlier `red-v3` had the same teardown symptom; it too is not a terminal receipt.

The final principal-binding regression was first run against the preceding green-v2 coordinator and failed because retry publication called the owner model instead of the principal-bound model. The fix stores the already-bound model in `_FailedPublication`. A fresh stage then passed all 8 ownership tests, including the authorized-principal retry.

Final frozen stage: `/home/dflanag3/spatial-artifacts/publication-handoff-ownership-20261016-v1/principal-green-v1/source`.

- `coordinator.py`: SHA-256 `a45638c8aa159c3ab91213d085e70822b8433d16ec9ad826e9c2a963d19d48dd`
- `test_publication_handoff_ownership.py`: SHA-256 `38aa3012536e8573413498de4950ca03f4deac5f3aae3a3f78098246e4060076`
- Both hashes matched after execution; the source stage is read-only.
- New ownership module: 8/8 passed, 0 skipped, command/wrapper 0/0 at `principal-green-v1/run-tests`.
- Related regressions including failed-publication LRU: 37/37 passed, 0 skipped, command/wrapper 0/0 at `principal-green-v1/regressions`.
- G0: 51 passed, 0 required real-backend cases, command/wrapper 0/0 at `principal-green-v1/g0-v3` (`--no-write` because the source tree is frozen read-only).

The retry path keeps the failed record available while attempting handoff. If writer submission raises, it releases the retry pin and lease, restores that record, completes the replacement durable future with the submission error, and releases the branch lock. `_submit_publication` now submits a gated task; the task can enter `_publish` only after the submitting thread confirms acceptance. If submission raises after queue insertion, the gate rejects the orphaned callable and clears its prepared-checkpoint holder. On pre-submit worker failure, successfully admitted checkpoint bytes are marked failed only when nondurable, retained as an ordinary unpinned LRU entry with the original operation envelope, and are removed from the native cache by the existing created-image cleanup. An existing durable alias is not marked failed or removed.

Retries remain conditional on the existing operation/head/history checks and on the checkpoint still being resident; eviction continues to make the failed operation explicitly nonretryable rather than recomputing it. No real GPU/model qualification was run.

A follow-up RED used the exact final test file (`38aa3012…0076`) against a fresh read-only stage of the preceding coordinator (`581e5f41…eda6`). It failed the single principal-binding method (command/wrapper 1/1): retry publication called the owner model instead of the principal-bound model. The earlier `principal-red-v1` test hash (`00c0d150…a382`) differs only by three `addCleanup` registrations in this method, restoring `model.for_principal`, `model.save_prepared_branch`, and `coordinator._submit_publication`; its scenario and assertions are unchanged. The fresh run completed and its test process was absent afterward. This confirms the final cleanup-safe test itself is RED against the old coordinator and GREEN against the final coordinator.
