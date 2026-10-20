# Seeded model-free fleet campaign

This private campaign exercises actual native allocations, immutable page COW,
device ownership and the shared checkpoint pools with generated data. It does
not qualify model inference or original G1-G5. Windows handles source edits,
Git, SSH and orchestration only; tests and builds run on Linux.

## Ownership and source

Primary checkout: `C:\Users\dflanag3\Documents\fleet\context-cow-model-free-fleet`.
Primary branch: `feat/model-free-fleet`, base `dd2f996d`.
Private destination: `box-git:/srv/data/git/private_context_cow.git`.
GitHub and Strata upstream are untouched. llm-60 chat stays off.

Luna fixture work: `context-cow-model-free-fixture-luna`, branch
`work/model-free-fixture-luna`, generator and independent CPU oracle, followed
by the test-only backend adapter and shared-pool campaign.
Luna HIP work: `context-cow-model-free-hip-luna`, branch
`work/model-free-hip-luna`, HIP VMM, compatibility bridge and AMD witnesses.
Luna native work: `context-cow-model-free-native-luna`, branch
`work/model-free-native-luna`, seeded CUDA/page tests and pressure runner.
Each has its own writable worktree. Native build-input overlays are recorded by
hash; they are not silently included in a worker's owned source commit.

## Selected hardware

| Host | Configurations | Runtime |
|---|---|---|
| llm-60 | `[0]` | Blackwell, CUDA 13.2 |
| llm-49 | `[0]` | RTX 4090, CUDA 13.3 |
| llm-79 | `[0]` | RX 7900 XTX gfx1100, HIP 7.15 |
| r730 | `[[0,1]]`, `[[0,1,2]]`, `[[0,1],2]` | P4, CUDA 12 / g++-12 |
| a5500 | `[0]` | RX 5500 XT gfx1012, installed HIP 5.7 |

The two-card split remains mandatory after adding the three-card split.
P4 0/1 report peer access; card 2 requires explicit pinned host staging.
The mixed configuration represents two workers, not a three-card split.
llm-79 currently has an unrelated resident server; its executed probes are
small and do not claim full-capacity pressure results.

## Executed observations

- Linux G0 baseline: 51/51 passed on llm-60. An earlier Windows invocation
  ran nine toy checks but failed two module imports on missing `fcntl`;
  it is retained as a platform limitation, not qualification.
- A real r730 allocation witness went red: requesting device 1 allocated
  physical memory on device 0. Per-device allocation state then passed
  ownership checks for devices 0, 1 and 2. Source/build receipts are in
  `experimental/branching_context/evidence/model-free-fleet-20261014`.
- Fixture review caught per-cell offset resets that made short cells marker-only.
  Linux regressions reproduced the failure. Global offset correction passed
  eight fixture tests and three CPU contract tests. Python and independent C++
  reference hashes agree. See `tests/model_free/VALIDATION.md`.
- Instrumented llm-60 tiny page COW: 17/17 assertions passed. Independent CPU
  bytes, partial-page isolation and cleanup were exercised; multi-device stage
  cases were explicitly unrun on the one-device host.
- Instrumented r730 tiny: 58/58 assertions passed, including 18 two-card and
  23 three-card stage assertions. Two-card peer transfer: 262144 bytes. Third
  stage used 262144 bytes through pinned host memory. Stage COW copied two or
  three physical pages respectively; fixture allocations returned to baseline.
- llm-49 fixture daemon smoke: 16/16 checks passed, independent full saved-byte
  audit, exact synthetic frontier, separate exited producer/consumer processes,
  zero native prefix-step work on restore and isolated A/B suffixes.
- llm-79 bounded native HIP probes passed mapping, protection, copies, range
  cleanup and actual BranchPages COW. Protected-write child returned error 700.
  Wrong-device testing is unrun on its single-device installation. The initial
  missing-HIP-implementation failure and build-overlay mistakes are retained.
- a5500 was discovered and assigned for capability inspection; no result yet.

These are bounded component results. The larger state-machine/pool campaign,
mixed-worker routing, full-size pressure and full native engine regressions
are not yet qualified.

## Fixture scope and sizes

`cow-fixture-v1` is deterministic logical data; GPU identity appears only in
separate device canaries. Logical pages are 4096 bytes. The GPU implementation
and CPU streaming implementation are separate. Physical granularity is queried.

The small resident test backend uses 32-byte rows per logical stage and an
actual GPU integer recurrence: `r = (257*r + token + 1) & 65535`.
The full 64-value frontier is `(r ^ (197*k)) & 65535`, represented exactly as
FP32. This simple fixture is explicitly not the Qwen state layout or forward.
Its native counters count executed GPU fixture steps; model-token inference is
not being claimed. Native image `CFB1` binds seed, capacity, device topology,
consumed IDs, mapped page bytes and per-stage running state. Existing canonical
history/controller and checkpoint publication remain the source of truth.

Q4 and Q8 transport-size anchors are 111334654784 and 188225033248 bytes.
These are pinned GGUF file-span sizes, not resident memory measurements.
Pressure tests must separately report resident allocations, logical bytes
streamed, actual dense disk bytes, fsync time and bounded-window execution.
No sparse reservations, fake operator receipts or skipped cases count as passes.

## Immediate next checks

The shared-pool runner must pass tiny cases before 1K/8K expansion. Its fixtures
are 128/4/8, 1024/64/16 and 8192/512/128; context capacities use the explicit
Q4/Q8 profiles. Keep native access on worker-owner threads. After the corrected
runner passes, test pressure with independent chunk verification and actual
charged allocations, preserving all red evidence and reporting host blockers.
