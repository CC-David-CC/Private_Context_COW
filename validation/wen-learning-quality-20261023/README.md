# Wen: 1,000-variation native learning study

The CPU lesson wiring passes. This report separately measures whether native
LoRA updates improve generated answers. No candidate is automatically promoted.

**Final outcome: `NO_QUALIFIED_GAIN`.** The update learns a narrow prompt pattern,
but fails preservation and the blind wording-family test. The original model is
unchanged. All four training jobs are complete; their GPU leases are released.

## Design declared before evaluation

The task is integer ceiling division, including negative divisors and integers
too large for safe floating-point conversion. Training uses 1,000 distinct owned
prompt variations of this one narrow skill. Changes include function/parameter
names and compact comments. These are not 1,000 unrelated coding tasks.

An exact `Fraction`-based checker validates targets and generated expressions.
The evaluator permits a restricted arithmetic AST and accepts semantically
correct alternatives. It does not execute arbitrary model code or compare only
against a target string. Adversarial checker tests cover a wrong signed-divisor
shortcut, floating division, equivalent formulas and unsafe code.

There are 48 separate prompt variations for checkpoint selection, 16 unrelated
arithmetic preservation controls, and 16 blind final variations. Their prompt
families are separated before generation; none is admitted as a training lesson.
The selection set is reused across checkpoints, so the final blind comparison
is recorded separately. This is within-skill generalization, not evidence of
general coding or BW agent competence.

The 48 selection cases share one instruction wording and vary chiefly names.
The blind cases use a different instruction wording and two parameter-name
styles. Thus 48/48 is not success on 48 independent task families.

Fresh native greedy return-expression completions use no chat template or
grammar, stop at newline and allow at most 24 generated tokens. Each token is
evaluated with a fresh full-prefill oracle, with actual work/position checks.
NumPy decodes binary logits for argmax only. C++/GGML performs all training;
PyTorch is not used.

The original observed failure uses a shortcut correct for positive divisors but
wrong for some negative ones. The first failing training-pool probe is placed
first in the ordered teaching set. Its exact prompt/output and independently
checked correction are retained privately, separately from later mitigations.

## Updates and qualification policy

The study uses rank-8 LoRA, native SGD at 0.001, batch size up to 8 and ordinary
complete-graph backprop across two A100s. It does not use experimental section
gradient passing. Distinct chunks of 1, 31, 224 and 744 examples produce the
1-, 32-, 256- and 1,000-example checkpoints. Each chunk warm-starts the previous
completed adapter. Inputs occupy 20–21 tokens including their teaching answer;
the admitted limit is 32, with no truncation. This is one pass over the selected
examples, not 1,000 optimizer steps or an optimizer search.

The declared candidate gate requires all of:

- Repair the original captured failure.
- At least 90% correct on the 48 selection cases and repair at least 25% of the
  original selection errors.
- Preserve every baseline-passing unrelated control, with at least eight such
  controls available at baseline.

Among eligible checkpoints, choose most selection passes and then fewer
examples. If none is eligible, the final checkpoint is diagnostic only. After
selection, compare the original model and chosen candidate on the sealed blind
set: at least 90% correct and no fewer correct than the original are required.
Lower training loss alone cannot pass any answer-quality gate.

## Measured results

The baseline answers 18/64 training-pool probes, 2/48 selection cases and 16/16
preservation controls correctly.

| Cumulative examples | Training-pool probes | Selection correct | Preserved controls | Original failure repaired | Candidate gate |
| --- | --- | --- | --- | --- | --- |
| 1 | 18/64 | 2/48 | 16/16 | No | Fail |
| 32 | 19/64 | 2/48 | 16/16 | No | Fail |
| 256 | 26/64 | 6/48 | 16/16 | No | Fail |
| 1,000 | 63/64 | 48/48 | 11/16 | Yes | Fail: preservation |

The target skill improves substantially, but five previously passing unrelated
controls regress. This candidate cannot qualify under the declared rule, even
if it passes the final blind division cases. It remains diagnostic and unpromoted.

The blind comparison is **2/16 original, 3/16 candidate**. It fails the declared
90% requirement. The five preservation regressions are four signed floor-division
cases and one absolute-value case. These measurements support narrow learning
with poor transfer and loss of some previous behavior; they do not establish
the cause or an optimal optimizer, learning rate, curriculum or replay mixture.

All 1,000 selected examples are distinct and processed once across 126 native
updates. The first step has 498 finite adapter gradients, of which 249 are
nonzero because B starts at zero; later completed stages have all 498 nonzero.
The reported training losses are step-local comparisons, not final loss over
the whole dataset. None substitutes for the semantic answer checks above.

[results.json](results.json) contains the frozen policy, data/source/binary and
candidate fingerprints, update counts, measured outcomes and resource settlement.
The status means no candidate met all qualification requirements; it does not
mean that no measured behavior improved.

## Evaluation recovery

The initial process completed all four native updates and their selection/control
evaluations. An immediate transition between final evaluations failed the idle
GPU admission check. The original failure log and source are retained privately.
The harness now waits at most 30 seconds for the same free-memory and zero
utilization criteria while holding its account lease; thresholds are unchanged.

The explicit `--resume-final-evaluation` path validates frozen data hashes,
preserved semantic verdicts, checkpoint gates, the completed candidate hash and
the original model/oracle identities. It reuses a completed original-model blind
evaluation and runs the remaining candidate evaluation. This path cannot create
training jobs. Four training run directories remain after recovery, with no
additional update, changed lesson or relaxed quality gate.

## Resource and implementation scope

Gradient accumulation handles examples one at a time. A snapshot during the
1,000-example stage records about 17,034 MiB and 18,580 MiB for the owned native
process on its two training GPUs. This is an observation, not a measured peak.
Training budgets are 60/52 GiB with a 4 GiB admission reserve; a separate GPU
runs fresh evaluations only while needed. Other users' allocations are retained.

The exact native executable and original model are pinned by SHA-256. Dataset,
policy, candidate and source identities are recorded in the machine-readable
summary. Raw completions, teaching documents, full receipts, local paths and
candidate weights remain private. The active study uses the launch controller
version; a later operation-record checksum hardening is covered by the final
CPU wiring proof and does not change native training mathematics.

This does not qualify longer contexts, other datasets/optimizers, sustained
inference load, every home GPU, HIP training, real Wen CPU training, or an
always-improving assistant. Full-weight backprop has separate bounded mechanics
evidence; this answer-quality study trains LoRA only. Candidates stay separate
from existing COW contexts and normal BW connections.
