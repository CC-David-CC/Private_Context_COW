# Native terminal-error recovery proposal

This is a source-level proposal, not a production change or CUDA qualification.
The new CPU tests exercise the Python adapter using scripted protocol frames.

## Observed protocol boundary

The pinned generated driver emits `{"status":"error","message":"..."}` when a
native operation fails. The generate path in the pinned `branch_driver.cpp`
replies and continues its outer command loop, but this frame contains no
consumed position, generated count, work counter, or request identifier. The
Python `_NativeRequest.__next__` currently treats it as a malformed token,
raises a generic `IntegrityError`, and poisons the adapter. Retaining the exact
error text is useful diagnosis; it does not prove that CUDA state is reusable.
The safe immediate contract is to retain the cause, release the active request
owner, and keep the pipe poisoned. Do not issue stats, restore, or another
generation on that process after the error.

The inspected generated source was
`/home/dflanag3/spatial-artifacts/cow-seal-stream-candidate-20261015-v1/native-probe-v3/branch_driver.cpp`,
SHA-256 `9370d129ee83399ad88cceee6b696c4f2e04655a9de0f362b610904f99a86d75`.
This is source inspection of that generated driver, not a new build or runtime
execution.

## Coordinator restart implications

The coordinator creates one worker thread and one `ready` future per GPU.
That future describes initial startup only. `_dispatch` currently considers a
worker eligible when its original `ready` future succeeded, it is not busy,
and its model is compatible. An exception in the worker loop after readiness
does not update the completed future. Therefore restarting the backend inside
the current loop without an explicit health state could route more jobs to a
dead or poisoned model; a replacement-factory failure could also strand jobs
in the queue.

A future recovery change should be worker-owned and must not replay the failed
request. The failed job should retain its ordinary failure result, with no
snapshot, capture, or publication from partial generated output. After the
native lease exits, the worker should close and record the old child PID and
return code, discard only process-local `StrataImage` aliases, construct a
fresh model from the same pinned factory/profile, and publish a distinct
healthy-worker state with the replacement PID. A later client request may
restore from the existing canonical checkpoint; it must use normal source
validation and accounting.

The restart must preserve durable Store data and must not invalidate a
`PreparedCheckpoint` already handed to the asynchronous publication writer.
Publication owns portable checkpoint bytes, Store read leases, and pool pins
independently from the worker. Recovery must not clear those pins or delete
canonical files. Any cache cleanup should be limited to old process-local
handles after active leases have ended.

Dispatch needs explicit health/recovery state separate from the initial
`ready` future. Keep the worker busy while replacing it. If replacement fails,
mark that worker unavailable and either route to another healthy compatible
worker or fail queued requests promptly with a capacity/worker-unavailable
error; never leave requests waiting indefinitely. Record the failed operation,
old and new native process identities/exit codes, replacement startup result,
and whether each subsequent request loaded a canonical checkpoint. Do not
automatically retry the failed operation ID or imply recovery if no replacement
was started.

## Evidence limits

The generated C++ command loop's `continue` after an error reply only proves
that it accepts another text command syntactically. It does not establish
correct CUDA allocator state, request accounting, or safe restore after an
allocation failure. The actual CUDA allocation-failure experiment and any
coordinator restart implementation remain separate, unrun work. Existing
EFBIG and publication-restart results concern filesystem/publication failures,
not native CUDA allocation failure.

## CPU test provenance

The new regression and existing native-request protocol tests ran against an
exact copy of the b140 worktree's `context_time_travel` source package. The
coordinator file in the first attempted artifact stage was from an older
root-reuse source snapshot (`1d9fb342…`); that stage was superseded. The counted
`cpu-v3` run used the copied package and its coordinator hash is
`01a2b42d7bf23a16441563a24afb6b611ed46b12a4d7f379ccdd4d688ca6c3ad`. The new
test file hash is `6b91e008e83f67073cdd55b63a0e1b5fb1edf04733ce8c6548bf861e5ac53bdd`.
The tested `strata_backend.py` hash is
`20d94f46a7b4d66fe110265219deef7f39fac400adb730b25fbe3eabb01ea692`.

Command: `bash /home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh /home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/source-exact /home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/cpu-v3 env PYTHONPATH=/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/source-exact/src:/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/source-exact/tests /home/dflanag3/src/Strata/.venv/bin/python -m unittest test_native_request_protocol test_native_terminal_error_recovery -v`.
It ran 13 tests: 11 passed, 2 failed on the expected error-message preservation
assertions. `command.exit` and `wrapper.exit` were both `1`; stderr SHA-256 was
`c4096708279e9d908fdfbefbf26428a517427afeb5e8235f2d7eb4e0fef1a4e6`, and
stdout was empty (SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
This is CPU protocol-fixture evidence only, not a native CUDA failure result.

After the RED run, a narrow adapter change was tested: exact nonblank
`status:error` frames now raise `IntegrityError` with their original message;
the existing `_NativeRequest` exception path still clears the active iterator
and poisons the backend. No stats, restore, capture, or retry is attempted on
that pipe. The final patched `strata_backend.py` hash is
`d692e00310984dcea2332d86862fde0010ff06f50b7f62b35a97d25110443de1`; the final
regression test hash is
`261fb67d0a23bad044870936072263959f3d980be181a8f58e56afc6a5b1a88f`.

The fresh `cpu-v5` run used the same runner with checkout
`/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/source-green-v2`
and run directory
`/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/cpu-v5`.
It ran the existing 8 native-request protocol tests and 5 new tests: 13 passed,
0 failed, 0 skipped; `command.exit` and `wrapper.exit` were both `0`. The suite
and CPU fixture do not load a model or initialize CUDA. An earlier `cpu-v4`
attempt failed in the test double because its mocked stats helper bypassed the
backend's poisoned-pipe check; that test-only setup issue was fixed before
`cpu-v5` and is preserved in its own run directory.

## CPU-only coordinator replacement candidate

An isolated coordinator candidate now implements worker-owned replacement after
the failed job has been completed as a failure. Dispatch selects only workers
whose additive health state is `healthy`; replacement keeps that worker busy,
requires both the backend's closed marker and an observed process exit, and
checks model contract, execution profile, and GPU selection before making the
replacement eligible. If replacement cannot be proven safe, the worker becomes
unavailable and queued work fails promptly. Durable receipt lookup still uses
the retained model's Store/profile metadata while a worker is recovering or
unavailable. Failed publication payloads keep their old model reference and
pool pin, so publication-only retry remains separate from native-worker
replacement.

The candidate source files tested were `coordinator.py` SHA-256
`b79b862b7cfaa449cd3d23b6d2933a053be18ba453075a3b896719be26d9507a`, the
existing terminal-error adapter `strata_backend.py` SHA-256
`d692e00310984dcea2332d86862fde0010ff06f50b7f62b35a97d25110443de1`, and new
`test_poisoned_worker_replacement.py` SHA-256
`efb0547b30127211e4656c81ab0dca7b3b147dcef834dc9ca74decf7c0b4becc`. They
were staged together in
`/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/replacement-green-v2/source`.
The tests cover successful replacement with preserved publication retry,
failed replacement and prompt unavailability, close during replacement,
close-error process-exit proof, rejected model/profile mismatches, receipt-only
recovery while replacement is blocked or unavailable, and continued work on a
healthy peer GPU. Existing coordinator, native request protocol, terminal
error, spill/reload trace, and checkpoint receipt tests ran in the same
candidate stage: 59 passed, 0 failed, 0 skipped. Runner directory:
`/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/replacement-green-v2/run-focused-v4`;
`command.exit` and `wrapper.exit` were both `0`.

An expanded test-first run against the unchanged coordinator produced 8
failures across 7 tests, as expected because replacement and health states did
not yet exist. It is retained at
`/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/replacement-red-v2/run-v2`
with command/wrapper exits `1/1`. These are CPU fake-backend tests only. The
candidate has not been GPU-qualified, and the real CUDA allocation-failure
recovery campaign remains unrun. A fresh execution profile/source qualification
is required before native use.

## Primary correction and final CPU evidence

The earlier 59-test claim above does not qualify current source. The worker
copied later source into the already-executed replacement-green-v2 stage.
Its logs remain, but the executed test bytes cannot be independently recovered
from that stage. The later run-focused-v5 records 60 tests with one failure.
Do not restore or overwrite historical artifacts to make them appear frozen.

Primary review took ownership after the worker stopped. A fresh read-only
replacement-root-red-v3 source records 62 tests, two expected cleanup failures.
A subsequent fresh replacement-root-red-v4 records 63 tests, three expected
failures, including ownership of a replacement whose first close cannot prove
exit. The identical tests on replacement-root-green-v4 pass 63/63, exits 0/0;
G0 passes 51/51, exits 0/0. The primary runner records actual imported module
paths and hashes before execution, hashes the entire source before and after,
and confirms no source changes. All execution was on llm-60 without CUDA.
See NATIVE-WORKER-REPLACEMENT-CPU-2026-10-16.json for exact hashes and paths.

The final candidate closes rejected replacement children on malformed profile
or lifecycle-trace failure before making the worker unavailable. An unproven
child exit remains owned by the worker for final shutdown. Model metadata for
canonical receipt recovery remains available while dispatch is disabled.
Backend process IO and lifecycle recording occur outside the coordinator lock.
No normal request is automatically retried after native failure.

The unchanged production runtime was separately exercised on the real Q4 GPU:
actual cuMemCreate OOM, no failed-request forward work or branch advance,
poisoned same-service follow-up, and successful explicit service restart.
That baseline is FAIL-RECOVERY. It does not qualify this candidate. Main keeps
its existing API/profile unchanged until a new execution profile and fresh
prefix fixture qualify the candidate on the GPU.
