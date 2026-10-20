# Phase-aware P4 lifecycle diagnostics and activation plan

The external observer now captures `RunHeartbeat.update()` phase labels and attaches a bounded phase context to observer events and summaries. It retains phase, root index, and repeat context across worker threads. A phase with omitted root/repeat fields preserves the most recent explicit values. Existing whole-run stage totals remain available; `phase_stages` separates import/setup, root preload, warmup, measured-request, and finalization/cleanup observations. Request-specific contexts preserve the 9 measured samples instead of collapsing them into one warmup/campaign total. No production runner/API/native file is modified.

## CPU validation and provenance

Executed on llm-60 in scratch copy `/home/dflanag3/branching-g3/checkpoint-lifecycle-diagnostics-luna-20261015-v8`, with the verified runner helper, using:

```sh
env PYTHONPATH=. CTT_DIAGNOSTIC_SOURCE_ROOT=/home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015 \
  python3 -m unittest \
  experimental.branching_context.tests.implementation.test_checkpoint_lifecycle_diagnostics \
  experimental.branching_context.tests.implementation.test_checkpoint_lifecycle_pinned_binding
```

Receipt directory: `/home/dflanag3/spatial-artifacts/checkpoint-lifecycle-diagnostics-focused-v16`. `command.exit=0`, `wrapper.exit=0`; 10 tests passed in 0.046 s, no skips/errors/failures. The new phase tests check phase separation, worker-thread propagation, retained root/repeat fields, and explicit disk-gate pass/fail values. A CLI rejection test sets a deliberately unattainable free-space requirement and verifies it emits `PREFLIGHT_REJECTED` before importing the P4 runner or model modules. The test also checks that unpartitioned stage totals still count all three events.

The v11 run is retained as a test failure: after adding a third `finalizing` sample, the phase test still expected a total count of two. The assertion was corrected to three; no implementation behavior or runtime assertion was relaxed. The v12/v13 runs passed before the complete external preflight-before-import test was added. The earlier v1 test-fixture arity failure and its correction remain documented in the first evidence file. Several output-directory collisions were rejected by the verified runner before any test command started; those paths were preserved.

Local and remote v12 source hashes matched:

- Diagnostic wrapper: `4fd182338e4f9613c26b15b0d48aa18a18efb9b0479bd60e31726ba9bc071906`
- Phase and storage-gate tests: `d2616d48771671d3a2b60e748daee439e4f7c42e796afa2f9a9118986b6ec029`
- Pinned-binding tests: `b7d29e8dc68ed1b689144aec2a7f1d1662fb4b4b5443fe5913d2e2a94dfe880b`

The P4 runner in the staged source is SHA-256 `f3e1898ab70fb9125216fb477d756a0ca7cd8a543c55cdab7597ae816b4f8fc9`. The three runtime modules imported for the observer match their P5 pinned hashes: `real.py` `6ab1abe6d2e6e4d8c92b7af0cb896e55ede51edf721a61a6a746f8aa3ca4fddd`, `pools.py` `7c90645bf87a414e1686543a0b099e5c9648616c391e74bee754496fd9b2a78f`, and `strata_backend.py` `20d94f46a7b4d66fe110265219deef7f39fac400adb730b25fbe3eabb01ea692`.

## Read-only activation plan

Use the existing full Q4 roots, normalized fixture, P4 runner, and original profile. Do not rebuild or alter those inputs. Source case: `/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure`; staged runner root: `/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red`. Input identities are recorded in `P4-Q4-EXECUTION-PREP-2026-10-15.json`: profile SHA `f3c69ad908786548cdd325f33af03a623553134d9406483e7622edd62135ea4b`, execution profile `9b23df373eb42687a2eca77fb99a22409ada0ab4c3edac732da2674bd23d3763`, native-build receipt SHA `e1b151aa37ddc4a75b9f6e121ccd7d73412e9a545105f1ffbe8cfc6ba053898f`, and fixture SHA `c661084b390720c1cbb1ca2465734cd5a099d366f2811704b4661251f5ca1903`. The profile is capacity 98,304, chunk 8,192, FP16 KV, GPU 0; three source histories are 65,536 tokens, with three selected suffixes of 8,336, 8,320, and 8,315 tokens.

For capture/publication attribution alone, the smallest useful arm is the full resident P4 arm: all three original roots, three one-token warmups, and nine measured requests (three repeats per root). This is not a smaller-fixture speed comparison and yields no new RAM-versus-disk claim. Keep the existing 128-token output bound and all validation checks. If a paired P4 comparison is later requested, use both original arms and the existing seeded `disk,resident` order; do not infer a paired comparison from this attribution-only resident arm.

The resident runner arguments remain exactly:

```text
--source-store /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/store
--source-result /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/result.json
--prepare /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/prepare.json
--profile /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/profile.json
--fixture /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fixture.json
--reference-dir /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure
--output NEW_RUN_ROOT/output/resident
--arm resident
--arm-order disk,resident
--order-seed p4-q4-arm-order-draw-2026-10-15-01
--ram-bytes 103079215104
--disk-bytes 137438953472
--min-available-bytes 17179869184
```

Invoke the reviewed diagnostic wrapper with `--source-root /home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red`, and pass the runner arguments above unchanged after `--diagnostics NEW_RUN_ROOT/diagnostics/resident.json`. Keep command receipts outside `--output`. Stage the phase-aware wrapper as a separately hash-pinned external tool; do not modify the frozen P4 source to install it. The existing runner helper accepts the diagnostic invocation as its command, just as it accepts the direct runner invocation.

The P4 runner's built-in per-arm gate estimates 38,196,451,720 bytes using source-root-sized payloads. The root-reviewed disk-reclaim map `P4-Q4-DISK-RECLAIM-MAP-2026-10-15.json` (SHA-256 `6fdfdbd3e6ee8eae2926d0fb316bfa7eb047eb9d15f34bf974c491ac838f2a56`) identifies the largest actual output state at 2,051,648,056 bytes. Use the external observer gate conservatively at `15 × 2,051,648,056 + 10 GiB = 41,512,139,080` free bytes on the output filesystem. Invoke with `--storage-check-path NEW_RUN_ROOT --required-free-bytes 41512139080`; the observer records the measured available/required bytes and rejects before importing the P4 runner or model modules if the gate fails. Then apply the runner's own 16 GiB `MemAvailable` floor, GPU UUID/idle state, and exact root-payload hash gates immediately before launch.

The frozen LF launcher was adapted from the reviewed P4 v3 script at `context-cow-p4-return-luna/experimental/branching_context/evidence/p4-q4-resident-start-v3-2026-10-15.sh`. It preserves that script's source/reclaim/result/checkpoint pins and resource checks, but directs output to a new attribution run root and invokes the external lifecycle observer. The launcher itself is at `/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-start-v1/input/start.sh` on llm-60, SHA-256 `ada508425b33a4ae8d64e318224184c136c5ab1f96eb8a9f415a7f5ee34b06c9`.

Exact read-only preflight invocation, already executed through the verified run-command wrapper:

```bash
RUN_COMMAND=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
bash "$RUN_COMMAND" \
  /home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context \
  /home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-preflight-v2 \
  bash /home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-start-v1/input/start.sh --preflight-only
```

The script holds GPU0's established `/home/dflanag3/branching-g3/native-client-speed.gpu0.lock` on FD 9 from preflight through the nested P4 runner and native child. The actual tool and runner checks use `printf` with absolute paths, not a quoted heredoc containing `$TOOL`. The first preflight attempt was refused at 31,004,344,320 bytes free. Later read-only preflights passed with command/wrapper exits 0/0 after all source and restored-root hashes passed; immediately before launch `/` had 48,794,198,016 bytes free and `MemAvailable` was 129,735,655,424 bytes. The launcher verified the pinned GPU0 UUID and no active compute applications, then root authorized the single resident run.

The authorized invocation used fresh outer and lifecycle receipt directories in persistent tmux session `p4-q4-lifecycle-attribution-v1`. Both outer command/wrapper receipts and the nested P4 command/wrapper receipts are 0, and both terminal statuses are COMPLETE. The additional diagnostics JSON is 243,069 bytes, contains 267 events, with 0 dropped events and 0 instrumentation errors. The observer adds no CUDA events: timings are host `perf_counter_ns` intervals, inclusive and same-thread exclusive. Never add nested inclusive values together; cross-thread timing groups are observational phase associations, not causal parent/child sums.

## Resident attribution result

The frozen Q4 resident arm returned PASS for all 9 measured requests, after 3 warmups across the original 3 roots. Every request reached its 128-token bound without EOS; the native executable's recorded exit code is `[0]`, and the source checkpoints remained unchanged. The run used the pinned 98,304-token context, 8,192-token prefill chunks, FP16 KV, GPU 0, and original profile ID `9b23df373eb42687a2eca77fb99a22409ada0ab4c3edac732da2674bd23d3763`. Its result SHA-256 is `fa1f7b856f1808c8f8a109e248d5a5191f4fb0ab0b213a1ef4547c407976345c`; diagnostics SHA-256 is `4c5b67145a62b4eb9b62521719bb9e549427f86881b05d8c01476ba375e6de4f`.

Across the 9 measured requests, host-observed `backend_serialize` took 14.626 s inclusive / 1.638 s same-thread exclusive for 18.453 GB returned; `export_state_readback` took 6.149 s for the same returned byte count. `checkpoint_digest` totaled 20.550 s over 54 calls and 53.426 GB of digest input. `prepare_checkpoint` totaled 25.685 s inclusive / 4.177 s exclusive; `publish_prepared_checkpoint` totaled 24.856 s inclusive / 6.559 s exclusive. The 3 `payload_publish` calls accounted for 6.154 GB and 3.370 s inclusive / 2.876 s exclusive; their payload file fsyncs totaled 0.493 s. `validate_checkpoint` took 8.070 s across 9 calls and 18.462 GB. Native `write` and `load` command text accounted for 441 and 1,134 UTF-8 bytes and 6.840 s and 10.937 s respectively; those byte counts are not payload-transfer sizes. These values are per-stage aggregates and overlap by nesting, so they must not be summed as a request total.

The runner's existing request summary reports 32.493 s startup, median submit-to-first-token 4.921 s, median submit-to-durable 13.416 s, and median token gap 14.804 ms. This was one instrumented resident arm, not a disk-versus-resident comparison or a new general speed claim. Observer overhead was not separately measured. The host ended with 30,980,411,392 bytes free; the native/observer processes exited, the lock owner disappeared, and GPU0 returned to 15 MiB / 0% utilization.

## Byte and timing interpretation

The `_request` byte count is UTF-8 command text length for native `write`/`load`, not payload transfer size. `serialize` and `export.state` readback count actual returned bytes. `Checkpoint.digest`, pool digest/admission, prepared-checkpoint, load, and manifest validation callbacks derive byte extents from their actual pinned method arguments/results (`blob + float32 frontier` where applicable). Payload/manifest publication count source part/data bytes; file and directory fsync durations are nested in the parent publication stage. These are host wall times around the actual calls, not device-side time or disk-device service time.
