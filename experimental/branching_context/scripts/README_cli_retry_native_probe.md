# CLI retry native probe

This harness exercises the installed `tools/cow_cli.py` through the real HTTP
service. Root owns the GPU launch; this worker ran CPU-only tests and prepared
the immutable runtime/profile stage.

The workflow covers a bounded eight-token greedy answer, exact replay (also
with explicit temperature zero), a changed-policy retry from the original
source, continuation and changed retry of that continuation, a stale revision
conflict, and exact replay from a fresh client journal after service restart.
For changed retries, the harness reads the completed CLI journal, verifies its
intent and record checksums, then resends the exact serialized request body,
including its originally submitted branch revision. It validates the actual
HTTP terminal event and canonical Store receipt; it does not synthesize stream
events or report native counters from the CPU fixture as GPU evidence.

The CPU fixture run at `cli-retry-native-probe-20261016-v1/cpu-final-v6/run`
passed 11 tests with command and wrapper exits 0/0. It uses the canonical
synthetic `RetryFlowModel`, so it validates CLI, HTTP, journal, and Store
orchestration only. Independent numerical qualification and GPU execution are
not included.

The reviewed launch candidate is
`/home/dflanag3/spatial-artifacts/cli-combined-20261016-v1/retry-gpu-stage-v3/source`
with sibling `profile.json`. Profile preparation passed `_validate_execution`
and `_validate_model`; it reused 15 cached model-file hash receipts and did not
load the model or start inference. The profile pins native SHA
`c674659df23ace94dfb29023d4f1dec1bd9f4f219df0ddb4802e0300b11ad87e` and
execution profile ID
`accffb2f6db32d037f2487e263e4d9eb0ed026dee04c18ab8f40a6cbe66d3e59`.

The retry runner entrypoint is
`experimental/branching_context/scripts/run_cli_retry_probe.py`. It requires a
fresh empty disposable Store, a new run directory, a bounded prompt, and
distinct branch and sibling names. It always requests eight outputs and stops
with `UNSUPPORTED_EARLY_EOS` if the fixture ends sooner. Any real GPU result
must be recorded separately from this CPU evidence.
