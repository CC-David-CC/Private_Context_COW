# Model-free source checkpoint — 2026-10-14

FOCUSED-PASS, not full G1–G5 or neural-model qualification.

The combined CUDA/HIP page ownership work, native seeded fixtures, CPU oracle,
pressure timings and grouped backend are integrated on `feat/model-free-fleet`.
The latest routing fix inspects immutable checkpoint placement before assigning
a worker. A sibling belonging to `[0,1]` waits for that compatible worker;
unrelated work on GPU 2 can run instead. It does not replay to change topology.

Real r730 red/green affinity evidence is under
`/home/dflanag3/branching-g3/model-free-mixed-luna-20261014-01/`:
`worker-affinity-baseline-red.json` and `worker-affinity-fixed-green.json`.
The positive native test verifies A/B `[0,1]`, C `[2]`, byte integrity,
12 forward tokens per request, and exactly one reference revision advance.
The separate no-hold timing report is `worker-affinity-no-hold-timing-final.json`.

The exact combined sources were rechecked on llm-60, Python 3.12.3:

- 82 shared-pool/control tests PASS; failures/errors/skips all zero.
- G0 verifier: 51/51 PASS; failures/errors/skips all zero.
- Python compile check PASS. No GPU process used for these CPU checks.

Raw commands and logs:
`/home/dflanag3/branching-g3/model-free-routing-cpu-luna-20261014-02/`.
Coordinator LF SHA-256:
`470242b0d6c0e488032792a48b641b5834948cbc02cf310d974079eb59fe3ddb`.
Fixture backend Python LF SHA-256:
`62b11d78942288ad350f82bfd52eb67a1cd0b876fa1ae357af4db0a7579dcd67`.

Native component runs: llm-60 COW 17 checks; r730 local/split/mixed stage COW
58 checks; llm-79 HIP VMM 64 checks plus 16 native daemon smoke checks.
llm-79 Q4/Q8-sized pool fixtures passed tiny/medium/long. These are synthetic
capacities, not quantized-model inference measurements. a5500 is excluded by
the user. Full real-model serving, sanitizer completion and original gates
retain their separate outstanding statuses.

Portable package: local folder `context-cow-portable-a100-20261014`, separate
private source checkpoint and artifact backups. CUDA 12 fatbins embed sm61,
sm80 and PTX80. Python and ELF code stay in sealed memfds on the protected
target; data/results default to verified RAM-backed storage and return over
SSH. No model or target-side compiler/install is required.

First actual A100 failure: seeded COW failed before execution because the
initial binary dynamically required `libcudart.so.12`. The corrected portable
build uses `-cudart=static` without `-lcudart`. The same equations/assertions
then passed GPU 0 COW. The first failure remains in the returned ZIP.
GPU 0 also passed Q4/Q8-sized tiny/medium/long pool fixtures and 64MiB/1GiB
resident/stream pressure. GPU 5 checks are ongoing; `[5,6]` and `[0,5,6]`
remain unrun at this checkpoint. Final statuses belong in the portable matrix.

Source/code and raw artifacts are backed up privately; GitHub/upstream are
not publication destinations for this checkpoint. A Git push proves backup,
not qualification of the still-outstanding fleet pressure or neural gates.
