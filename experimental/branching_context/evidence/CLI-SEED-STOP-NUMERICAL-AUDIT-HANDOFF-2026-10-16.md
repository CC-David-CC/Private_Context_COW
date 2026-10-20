# CLI seed/stop raw numerical audit handoff

This is an audit-only harness. It compares exact full-vocabulary FP32 frontiers from fresh raw-native Prefill/Verifier processes with the canonical durable COW checkpoints produced by the integrated seed/stop CLI workflow. The oracle does not import a checkpoint, adapter, or branch session. Forced token histories are checked exactly; grammar choice and RNG arithmetic are outside this audit's scope.

## Frozen inputs

| Item | Path | SHA-256 |
|---|---|---|
| Interleaved audit orchestrator | `experimental/branching_context/scripts/verify_cli_seed_stop_interleaved_numerics.py` | `eb51f32e43fb473c6ba5bc7ae71094de1ea3b853c50cc17d79fec198d9734e90` |
| Raw child runner | `experimental/branching_context/scripts/run_cli_seed_stop_raw_audit.py` | `2a037aa52b624ba64b0323590a5077b9a281e7dc53b43b92b18dacd9c171abb8` |
| Oracle build helper | `experimental/branching_context/scripts/build_cli_seed_stop_raw_oracle.py` | `9a34eecaa7f483730f0292687be5e703825e0324cd176ed130c38830e6f32d3d` |
| Schedule contract tests | `experimental/branching_context/tests/implementation/test_cli_seed_stop_numerical_audit.py` | `2106d5a53a2ae8522ad567f6af7966f50873f5528ddc485b631247724e75ca0d` |
| Orchestrator tests | `experimental/branching_context/tests/implementation/test_cli_seed_stop_numerical_orchestrator.py` | `3c7127895bb68b461139ce1d3b397dab0189142e7bae1872cae58a7c84f04a2f` |
| Audited native source | `experimental/branching_context/native/fresh_prefill_interleaved_raw_audit.cpp` | `fe8cce38146e66efc1c8d1eb0c13e53108502bc82a2cbbc6ee6e3ceb9b42a83e` |

The copied native file is byte-identical to the source compiled in build v6. Temporary `*_reference.py` files at the repository root are not part of this handoff.

The executed read-only stage used raw-runner SHA `2a037aa52b624ba64b0323590a5077b9a281e7dc53b43b92b18dacd9c171abb8`. Before this local commit, I removed one extra final LF from the committed copy to satisfy the staged whitespace check; the committed raw-runner SHA is therefore `a307a8a85a5cde0c677f8255925b0ed63fc85f4f06f81a46d8ce60eb2871afc1`. No executable code changed. The preflight and root-launched audit use the external stage SHA above.

## CPU controls and preflight

The remote llm-60 CPU contract suite passed 22 tests with command/wrapper exits 0/0 at:

`/home/dflanag3/spatial-artifacts/cli-seed-stop-numerical-audit-20261016-v1/runs/integrated-v1-controls-v2`

The tests cover absolute operator positions, append ordering after interleaved segments, environment scrubbing, no-append forward-work accounting, operation binding, exact finite frontier comparisons, controller-only reseed validation, appended-input span extraction, and recording a launched raw child even if it exits with failure.

The no-GPU preflight passed with command/wrapper exits 0/0 at:

- Runner receipt: `/home/dflanag3/spatial-artifacts/cli-seed-stop-numerical-audit-20261016-v1/runs/integrated-v1-preflight-command-v4`
- Result: `/home/dflanag3/spatial-artifacts/cli-seed-stop-numerical-audit-20261016-v1/runs/integrated-v1-preflight-result-v4/result.json`

Its status is `READY_NOT_RUN`, with `runtime_executed=false` and `gpu_launched=false`. It used the integrated runtime root `/home/dflanag3/spatial-artifacts/sampler-integration-1538b8b3-v1/source`, profile `.../profile-q4-v1/profile.json`, campaign `.../gpu-cli-results-v1/result.json`, and store `.../state/gpu-cli-store-v1`. The bound execution-profile ID is `efbe7aa44d8c4cbd9b712cad7490c6e8c29efc9fbd5cabc0bcac763e81ce3aaa`; model-contract ID is `ff923fd4e0fb068eea3a39a79f130860fdbdda8f686ecb0c470fa09a27808827`.

The canonical schedule and frontier mapping passed preflight:

- Sampled generation: root history 18; output segments 1, 16, and 10; checkpoint histories 19, 35, and 45. The controller-only reseed at history 19 changes only the RNG stream key and resets its cursor to zero; continuations must preserve the new key.
- Stop and appended input: root 18 + READ output 5 gives history 23; appended TEST input span starts at 23 and contains 18 tokens; TEST output 11 gives history 52. The append span digest was checked against the canonical token IDs.

## Raw-native build

Build v6 is recorded as compile-only; it was not executed by this worker:

- ELF: `/home/dflanag3/spatial-artifacts/cli-seed-stop-numerical-audit-20261016-v1/native-build-v6/output/strata-batched-fresh-interleaved-raw-audit`
- ELF SHA-256: `093c8012811e9d3bb32415f8610b3a93345604835b2402a72674cdb0e3b5e0d9`
- Build receipt SHA-256: `a4a87928d9df8e667c61097a50146b857d423c87199e95748138952834ac2668`
- Driver source SHA-256: `fe8cce38146e66efc1c8d1eb0c13e53108502bc82a2cbbc6ee6e3ceb9b42a83e`
- Input-verification report SHA-256: `9ec1bb03b22e8b18449dc62775c57b84658b84181adb9f8c2787ef75ce8af6e1` (53 pinned inputs; zero mismatches)
- Base serving/source receipt SHA-256: `d0fd5cc7981e29677f08c0bffadb1c951581d7e67cc8c3149562c83aab9f898f`
- Compile/link receipt SHA-256: `7b26945aef71ed8ad46e3c8c030b9eebe3381a0b5fe6dab9f152667c6ac6aa30`

## Preserved unsuccessful attempts

- `integrated-v1-preflight-command-v1` failed on the pre-fix validator's assumption that the RNG stream key stayed the same across the explicit reseed. The validator now checks the canonical zero-token direct-child transition and then checks continuity on the new stream.
- `integrated-v1-controls-v1` was an incomplete transfer that omitted the build helper imported by the contract test module; it failed with `ModuleNotFoundError`. No assertion was counted as passed from that attempt.
- `integrated-v1-preflight-command-v2` used an incorrectly assembled Python path and exited 127 before running the validator. The v4 preflight uses the verified interpreter path.

## Numerical audit status

The root-owned GPU audit was launched separately after the successful preflight. Its status was `RUNNING` at handoff; no numerical audit outcome is claimed here. The launch executes two histories twice each in fresh raw-native processes (four child processes), compares every recorded frontier byte-for-byte, and checks saved canonical checkpoint frontiers. A successful preflight alone is not a numerical pass.

Primary completion after that worker handoff: the four-process GPU audit passed
with command/wrapper exits 0/0 and all children independently absent. It made
51 fresh-repeat and seven fresh-versus-SUT full-vector comparisons, maximum
absolute error zero. The SUT comparisons cover six distinct durable frontiers;
they do not cover every SUT output frontier or independent sampled choice.
Exact raw result SHA-256:
`9c876f06abbed316528b5ebc9412deaeb133539233b4fcdc0d43b4504bab4fbb`.
See `CLI-INTEGRATED-Q4-INDEPENDENT-NUMERICS-2026-10-16.json` and
`CLI-INTEGRATED-AUDIT-RESIDENT-BACKUP-2026-10-16.json` beside this document.
