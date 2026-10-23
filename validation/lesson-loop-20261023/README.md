# Checked-lesson integration: CPU wiring proof

**CPU wiring PASS. Language learning is a separate experiment.** No candidate
was promoted and no serving model was replaced.

The test uses a generated two-layer hybrid GGUF with hidden width16 and a byte
vocabulary, built with the same native GGML training executable source as the
bounded A100 proof. It is not Wen and is not evidence of coding competence.
The controlled original mistake is injected, not labelled a spontaneous model
failure.

The harness executes real BW grants, frozen tool plans, isolated filesystem and
test processes, and canonical run records. A failed check, host correction and
successful verification occur inside one completed task. Lessons are authored
through the actual BW CLI, queued through both BW and the COW CLI, then trained
by the real native CPU process through the authenticated HTTP routes.

| Boundary | Observed result |
| --- | --- |
| One-command lesson author/train | A private teaching example is bound to failed/passing BW checks and produces a separate native LoRA candidate. |
| Later batch | Three pending lessons execute as batches of two and one; the default step count covers all selected examples. |
| Retry after pending queue empties | The original batch's frozen IDs and acceptance are recovered; no second native update occurs. |
| LoRA continuation | The batch warm-starts the completed single-example adapter. |
| Full weights | COW CLI requests a direct full update and another update from its completed full candidate; no LoRA is merged. |
| Save/reload | LoRA and full GGUF reload losses agree with completed-run losses within0.000001. |
| Original model | The generated fixture's hash remains unchanged. |

All32 LoRA tensors have finite gradients; initially16 are nonzero because B
starts at zero. Later all32 are nonzero. All28 full-mode parameter tensors have
finite, nonzero gradients. These counts describe the tiny fixture, not Wen's
32-layer decoder.

[cpu-results.json](cpu-results.json) records hashes, losses and exact scope.
Four native candidate runs occur, including full continuation; reconciliation
does not create a fifth run. No GPU is used for this proof.

The reusable BW `tests/lesson_native_cpu.py` harness also passes with the final
controller and client sources. It generates the fixture itself and accepts
explicit PCC checkout, CPU executable and fresh output paths. Its first launch
on an incomplete test snapshot failed before training because that snapshot
lacked the fixture generator; supplying the matching source resolved it. The
training-only HTTP fixture does not qualify production inference startup.

Additional tests cover changed operation bytes, changed token materializations,
wrong model, arbitrary client paths, partial batches, insufficient step budgets,
parent rank/mode, interruption without automatic replay, HTTP authentication,
forged acknowledgments and stale corrected workspaces. Interactive default
selection refuses an unfinished newest Work task rather than teaching from an
older one. The unchanged Linux G0 reference passes51 checks.

Earlier harness attempts are retained privately: the standalone CLI's closed
journal path menu initially rejected the new training endpoints; that was fixed.
The tiny model's default0.0001 update was below F32 loss resolution in some
cases, so the fixture explicitly uses0.01 for its LoRA exercise. No comparison
tolerance was loosened. Tests rerun with a fresh CLI journal avoid reusing
already committed operation IDs. A required CLI-source environment path and an
isolated terminal dependency were supplied before the affected regression test
was rerun. Skips are not counted as passes.

The final targeted regression checks pass: 38 PCC client/controller/checker
tests, 11 native admission tests and 35 BW lesson/client/terminal tests. The
native-admission module initially was absent from the remote source snapshot;
it passed after the matching file was supplied. These counts describe checked
boundaries, not language-model competence.

This does not qualify mid-step optimizer recovery, power-loss durability, useful
language learning, real Wen CPU training, every home GPU, general BW tasks or a
day of inference load. Read [the workflow and limits](../../docs/LEARNING-LOOP.md).

## Wen study

The separate A100 harness freezes1,000 checked signed-ceiling-division variants,
48 held-out cases,16 preservation tasks and16 blind final cases. It uses fresh
native greedy completions and an independent exact-fraction semantic checker.
Training, held-out and blind inputs are disjoint, and evaluation answers are not
admitted as lessons. Lower loss alone does not pass the learning gate.

The observed baseline is18/64 on training-pool probes,2/48 on held-out cases and
16/16 on preservation controls. The study compares1,32,256 and1,000-example
candidates. The [completed quality report](../wen-learning-quality-20261023/README.md)
records strong selection-pattern learning, preservation regressions and weak
blind transfer. No candidate qualified. This CPU proof remains a wiring result.
