# CLI reseed receipt plumbing correction

The seed/stop probe now parses reseed's durable receipt from stderr through the existing `receipt=True` path. It no longer attempts to decode stdout as JSON. The CLI and production/runtime sources are unchanged.

Regression: `test_reseed_probe_reads_durable_receipt_from_stderr` exercises `SeedStopProbe.reseed_cli` and the real `SeedStopProbe.cli` receipt parser with empty stdout and the durable receipt marker/payload on stderr. The RED run against frozen v16 harness bytes failed because `reseed_cli` was absent; the corrected frozen harness passes. The preceding real GPU v2 setup attempt independently reached reseed and then failed when the probe tried to decode empty stdout as JSON; the preserved campaign result remains the authority for that failure.

Remote host: `llm-60`; runner SHA256 `7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537`; Python `/home/dflanag3/src/Strata/.venv/bin/python -B`.

- RED: `/home/dflanag3/spatial-artifacts/seed-stop-probe-receipt-fix-20261016-v1/runs/red-v1`, command/wrapper exit `1/1`, expected regression failure (`AttributeError` on missing `reseed_cli` in frozen pre-fix harness).
- GREEN focused (final frozen harness): `/home/dflanag3/spatial-artifacts/seed-stop-probe-receipt-fix-20261016-v1/runs/green-frozen-focused-v2`, 9 tests passed, command/wrapper exit `0/0`.
- Read-only harness overlay: `/home/dflanag3/spatial-artifacts/seed-stop-probe-receipt-fix-20261016-v1/frozen-v2`; both files mode `0444`. It imports the unchanged runtime at `.../combined-cpu-v16/source`.
- G0: frozen runtime `/home/dflanag3/spatial-artifacts/seed-stop-combined-20261016-v1/combined-cpu-v16/source`, `/home/dflanag3/spatial-artifacts/seed-stop-probe-receipt-fix-20261016-v1/runs/g0-v16`, 51 tests passed, 0 real-backend cases, command/wrapper exit `0/0`.

Harness-only overlay hashes:

- `scripts/run_cli_seed_stop_probe.py`: `1dad4abfa8e6cd192b105ef8f00266727a38e6c02285de6d78192a159db22542`
- `tests/implementation/test_cli_seed_stop_probe.py`: `bff0ff0d7530b08447b880e41c2bac4059b3627b5f9f50bd3f735bbde737316d`
- Frozen profile-v4 remains unchanged: `682b7f8ea85487c0617333636a274e33022c07c57567df9ee8762defb5c5740b`.

The failed GPU v2 campaign remains preserved separately. This CPU check does not qualify a GPU campaign or change the profile closure.
