# COW CLI event timing observer handoff

Branch `work/luna-cli-event-timing`, based on `c4c958d783c34d976c11c6c92cf9038688f4e182`. This is a standalone harness only. It does not change `tools/cow_cli.py`, server code, profiles, or backend behavior.

Run it as:

```text
python observe_cow_cli.py --runtime-root <frozen-runtime> --trace-file <private-new-path> -- <normal cow_cli arguments>
```

The observer loads and hashes `<runtime-root>/tools/cow_cli.py`, invokes its existing `main()` with the remaining arguments unchanged, and patches only that module's `urlopen`. It wraps responses whose request advertises NDJSON. The proxy records request-send and response-header timestamps and buffers every raw line plus its monotonic arrival timestamp in memory. It passes each original byte line through unchanged. After the CLI returns, it writes one mode-0600 trace in a private directory; it does not fsync per event. Non-NDJSON requests and responses pass directly through.

The reducer reports request-to-first-token, per-token interval mean/median/p95/max, generation rate `(n-1)/(last-first)`, last-token-to-durable, request-to-durable, and process elapsed. Undefined metrics are `null`. It treats token IDs from `token`, `text`, and `turn_complete` events as output tokens; the latter carries the final EOS token in the current serving event shape. It records empty-text token events separately and checks captured tokens against `complete.answer.tokens`, receipt equality and branch identity/revision, the optional snapshot event's top-level tokens/tip, event order, duplicate terminals, malformed lines, and the full monotonic clock chain. The actual CLI still performs its own stream validation; observer validation only labels the captured trace and never changes the CLI return code. Invalid, truncated, and transport-error streams never receive generation TPS.

## CPU evidence

All tests ran on llm-60 through the approved wrapper at `/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh`, using `/home/dflanag3/src/Strata/.venv/bin/python -B`. No model, server, inference, or GPU job was started.

- Tests: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v11/source/experimental/branching_context/tests/implementation/test_observe_cow_cli.py`
- Observer: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v11/source/experimental/branching_context/scripts/observe_cow_cli.py`
- Actual CLI copy used by the CPU entrypoint test: `.../green-v11/source/tools/cow_cli.py`, SHA-256 `798fe8706b0e5bf8b6ebdf956d10d59ef6bc1e5647b98efe3590e098afb04b91`. It matches the selected runtime's `tools/cow_cli.py` byte-for-byte.
- Observer SHA-256: `d9475775b043e2b566d46f480a14e48a7368828b57a711764670150773baa51c`.
- Test SHA-256: `5ff03532c90837fe9001e3d53866e7a3ab2fd5a44bccb7fc3fb28ebd9d14f2ca`.
- Green receipt: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v11/run-green`; 10 tests passed, zero skips, command exit 0, wrapper exit 0. The cases cover transparent byte-preserving proxy behavior, actual `cow_cli.main()` and journal completion, event timing and empty-text/EOS/snapshot classification, zero/one-token null metrics, malformed/reordered/nonmonotonic/duplicate/mismatched streams, error and truncation labeling, read-error propagation, actual CLI rejection of duplicate completion, and preserving the CLI result when reducer or trace writing fails.
- The frozen `green-v11/source` tree is read-only. The observer, tests, and CLI-copy hashes matched local source before freezing; no `__pycache__` directories were created under the stage.
- Initial RED receipt: `.../red-v1/run-red-v2`, command/wrapper 1/1 because the observer module was absent. Preserved setup refusals and intermediate green failures are under the same artifact root (`red-v1/run-red`, `green-v1/run-green`, `green-v4/run-green`, `green-v6/run-green`, `green-v9/run-green`, and `green-v10/run-green`). These are not counted as qualification passes.

## Clock, snapshot, and receipt validation update

The follow-up is frozen at `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v16/source`. It uses the same pinned CLI copy (`tools/cow_cli.py`, SHA-256 `798fe8706b0e5bf8b6ebdf956d10d59ef6bc1e5647b98efe3590e098afb04b91`) and leaves `green-v11` evidence unchanged.

- Observer SHA-256: `294c360e94f2261f9c6e2fb33c1bf14e5b44491f147cf3ab634acb836c420aa2`.
- Test SHA-256: `061151a3766c2002e5546a09db43d6b029aac2b65e4215cd679a7419863a8ab4`.
- Green receipt: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v16/run-green`; 14 tests passed, zero skips, command exit 0, wrapper exit 0. The suite checks process-start/request-start/headers/event/EOF/process-end order, open/read errors, no TPS for invalid streams, nonnegative integer token IDs, request revision to receipt revision, structured branch name/tip/root, and the coordinator's actual `event='snapshot'` shape with top-level `tokens` and `snapshot`. The snapshot event is optional; when present it must match the completed answer and follow output tokens before durable publication.
- Behavioral RED against the previous observer plus the expanded tests: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/red-v13/run-red`; 14 tests ran, command/wrapper 1/1, with seven assertion failures and three missing-field errors. The earlier reducer accepted malformed clock, snapshot, and request-revision evidence; no production source was involved.
- Intermediate `red-v12`/`red-v13` and `green-v12` through `green-v16` remain preserved. An early snapshot fixture used a nested answer shape based on an incorrect source reading; it was replaced with the verified top-level event before the final RED/GREEN pair.
- The final green stage hashes matched the local observer/test before freezing. Its source is read-only with no `__pycache__`; no service, model, inference, or GPU process was started.

## Post-complete transport error guard

A final reducer edge case is frozen at `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v17/source`. A stream read error after an otherwise valid `complete` event, or any observer capture error, now invalidates the trace and prevents a generation TPS value.

- Observer SHA-256: `20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28`.
- Test SHA-256: `b95b3bcf51c639530c298d733843666cae6f2459756daea3a7dc8df434f60adc`.
- Green receipt: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/green-v17/run-green`; 16 tests passed, zero skips, command exit 0, wrapper exit 0.
- Behavioral RED against `green-v16` observer plus the two new regression cases: `/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1/red-v14/run-red`; 16 tests ran, command/wrapper 1/1, with two failures (read error after complete was labeled valid; observer capture error was labeled valid).
- The stage retains the same pinned CLI source SHA `798fe8706b0e5bf8b6ebdf956d10d59ef6bc1e5647b98efe3590e098afb04b91`; no service/model/GPU process was used.

## Scope limits

This establishes CPU behavior against synthetic streams and the real CLI entrypoint; it does not establish timing accuracy or observer overhead against a live service. `request_to_first_token` is measured from immediately before `urlopen` to the first token-bearing raw event. Generation timing uses token-bearing event arrival times, not text length. The three-source live campaign and any instrumentation-overhead comparison remain unrun. The trace includes base64-encoded raw NDJSON lines, so callers must keep its directory private.
