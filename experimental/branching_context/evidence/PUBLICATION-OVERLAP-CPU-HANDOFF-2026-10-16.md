# Publication overlap CPU controls — 2026-10-16

This isolated test-only change adds three deterministic controls for concurrent checkpoint publication. The checkout is based on `3ff345e93c6698bd68ccc269f6c879deb677f416`; no production source was changed.

`tests/implementation/test_publication_overlap.py` covers:

- `test_writer_overlap_computes_three_fifo_siblings_without_durable_ack_or_head_moves`: while the first writer is blocked, three distinct branches compute in FIFO order with a single active worker. No durable acknowledgement or branch-head move occurs before the writer completes. A same-branch stale-revision request cannot pass the pending publication and is rejected after the committed revision becomes visible.
- `test_first_writer_failure_leaves_siblings_durable_and_retry_reuses_computed_target`: an injected genuine `OSError(28)` affects only the first publication; siblings persist, the failed operation has no false durable acknowledgement, and retrying that operation publishes its saved target without additional model work.
- `test_cancelled_queued_job_does_not_consume_worker_or_reorder_remaining_jobs`: cancellation of a queued job does not start it or disturb the remaining FIFO work; event gates are released during cleanup.

These are synthetic CPU coordination tests. They do not qualify GPU behavior, model numerics, or production performance. The first v1 test attempt incorrectly expected the stale same-branch submission to fail at submit time. Source behavior queues it; the scheduler prevents execution until the active publication resolves, after which the stale revision is rejected. That assertion was corrected in v2 without changing production code.

## Frozen test evidence

- Base API/test source archive: `base.tar`, SHA-256 `603ee96cde7038b7ffb203892325333786093b93660c60d6fa6d75143a945a2c`.
- Added test source SHA-256: `d91ce8c65310e24b3ab5cd02e04014fb90ed0b6130b3bebe7223764d2b575f2d`.
- Frozen remote source stage: `/home/dflanag3/spatial-artifacts/publication-overlap-cpu-20261016-v1/green-v2/source`.
- Final runner record: `/home/dflanag3/spatial-artifacts/publication-overlap-cpu-20261016-v1/green-v2/run-tests`.
- Result: 3 tests passed, 0 skipped; command exit 0 and approved wrapper exit 0.
- Runtime used the approved `run-command.sh` at `/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh`, with `/home/dflanag3/src/Strata/.venv/bin/python -B` and the staged API source on `PYTHONPATH`.

The final source stage is made read-only after its hash verification. No GPU, service, inference, or commit was performed for this task.
