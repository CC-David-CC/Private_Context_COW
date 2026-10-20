# Q4 CUDA allocation failure harness handoff

This is a harness-only candidate for one Q4 HTTP greedy request held immediately before its native `generate` send, followed by a separate VMM pressure-helper allocation attempt. It does not change native/API/profile code and it has not been used for a GPU qualification.

The candidate records a zero-forward, zero-prefill gate; a same-pipe native error and CUDA VMM counter delta; the unmodified HTTP error stream; source checkpoint and branch-head checks; helper identity, explicit release and numeric exit; then a cleanly restarted service request. The same-service follow-up is reported separately. Only an explicit same-service error/failure can be labeled `FAIL-RECOVERY`; timeout/EOF remains `FAIL`. Every non-PASS runner result exits nonzero.

The current CPU source stage is `/home/dflanag3/spatial-artifacts/q4-cuda-oom-20261016-v1/cpu-lifecycle-v5/source`. It was made read-only before testing. Its files and hashes are:

- `experimental/branching_context/scripts/run_q4_cuda_oom.py`: `13fe3064e77202414139dd206e3ce0a7e3ddfd9a2e66ddaeef27268074b6e1ad`
- `experimental/branching_context/tests/implementation/test_q4_cuda_oom.py`: `d0653bc3a55c9b691cd2677fcd7fc1308b87d33c25369bd22e5ea1e0f72dfd60`
- Pinned sibling `experimental/branching_context/scripts/run_q4_publication_fault.py`: `d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695`
- CPU runner `run-cpu-tests.sh`: `ea54273a1eece6bcbb4d8a3da21e25e3c7b490299c7990c99032c498393ae1f5`

The approved llm-60 runner receipt is at `/home/dflanag3/spatial-artifacts/q4-cuda-oom-20261016-v1/cpu-lifecycle-v5/run/tests-v1`. The focused suite ran 16 tests, all passing; command and wrapper exit codes were 0/0, stderr was empty, and stdout ends with `Ran 16 tests in 0.065s` and `OK`. It used `/home/dflanag3/src/Strata/.venv/bin/python -B`, with bytecode writing disabled. The suite includes separate profile-API vs service-source pin checks, wrong-HTTP-pin rejection, and a preflight-only CLI control that never invokes the runtime path. This is CPU harness evidence only.

The first root-owned launch was stopped before model startup: the selected execution profile correctly did not include `serve.cow_http` in its native/API closure. That attempt is a preserved setup failure, not a CUDA or SUT OOM observation. The driver now checks the five API modules against the profile and binds `serve.cow_http` separately to `--expected-http-sha256`; it labels the latter as service-source provenance rather than claiming the profile pins it. A second CPU-stage attempt (`cpu-lifecycle-v4`) was also preserved; its test import failed on a Python syntax error before executing tests. The corrected v5 is a new immutable stage.

An earlier Windows controller attempt invoked the unit suite locally before the Linux-only runtime constraint was reiterated. Those attempts are uncounted setup errors and are not qualification evidence. All counted tests above ran on llm-60.

The v5 preflight passed on llm-60 with command/wrapper exits 0/0 (`ROOT/root-preflight-v2`). It independently checked the execution-profile API closure, HTTP source pin `9f6839a359e05cbb47e0850046c6f32f00e7468265dff80fedc3d72d9f720f31`, and workspace resolver pin `ed1a3e693fa0be423cd5e1413f59dc08c5fccfe6052483ded62298c2dbec9c72`; the profile itself does not claim to bind the HTTP module or workspace resolver. The selected Q4 profile SHA is `f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c`, and the native binary SHA is `4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451`. The pressure helper SHA is `a3d2690d104f192ca0a25ee224db4b3f7fcef45d0cad4de4a491c755937cd61b`. Root has started a root-owned Q4 campaign using this source; at handoff it is live at `start_q4_http_service` (HTTP PID 2823816, native PID 2823822). Its outcome is pending and is not reported here. No CUDA OOM observation is claimed until root audits the terminal evidence.
