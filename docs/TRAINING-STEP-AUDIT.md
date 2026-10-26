# Inspect a bad first AdamW step without training again

PCC provides an optional owner-side, read-only audit for one fresh full AdamW
step with zero decay. It checks exact base/candidate/optimizer-state hashes,
compares every F32 weight, and reconstructs the recorded batch gradient from
the first moment. It performs no inference, backprop, job mutation or promotion.
The ordinary serving and training paths are unchanged.

Use Python 3.11 or newer with NumPy in the existing owner environment:

```sh
python -m tools.pcc_training_step_audit \
  --job-record /owner/service/operations/native-job.json \
  --base-model /owner/base.gguf \
  --output /owner/new-step-audit.json
```

The original registered job must be complete. The output must be new. The audit
verifies the job-record checksum and exact artifact bytes, rejects links,
checks complete tensor/state coverage and detects file changes during reading.
It is an offline owner tool, not a model-callable route accepting server paths.
It refuses LoRA, Muon, resumed or multi-step state and nonzero decay rather than
applying a first-step bound to the wrong optimizer. Three independent file
hashes run concurrently; tensor statistics use bounded chunks. No whole model
or optimizer state needs to fit in Python heap memory.

## What its numbers mean

At the first fresh AdamW step with zero decay, bias-corrected moments imply
`abs(delta_i) <= learning_rate`. The audit allows 0.1 percent plus two F32
spacings for rounding. A bound violation is useful evidence of a wrong update
size or mismatched artifacts; passing that bound is not evidence of learning.

The saved first moment gives `g = m / (1 - beta1)`. The audit checks its norm
against the original native gradient record and reports `g dot delta`.
A negative value means the *recorded* derivative predicts a descent direction.
It is not an independent finite-difference gradient test. Taylor's local
prediction `L(w + delta) ~= L(w) + g dot delta` need not hold for a large step.
Neither a negative dot product nor a passing audit accepts a candidate.

S3 owns lessons, behavioral panels and learning/forgetting decisions. BW owns
its task/client journal. PCC owns these weight/state identities and the native
optimization evidence. This tool adds no workflow ledger or alternative trainer.

## First real mixed-block observation

The [sanitized receipt](../validation/training-step-audit-20261026/native-first-step.json)
records the llm-60 MiniCPM block whose review score fell from 438 to 26/864:

| Observation | Measured result |
| --- | ---: |
| F32 parameter components checked | 2,516,756,480 |
| Changed components | 2,246,759,994 |
| Tensors / component-bound violations | 381 / 0 |
| First-moment vs native gradient-norm maximum relative difference | 1.09e-7 |
| Recorded gradient dot observed update | -4.48896 |
| Native practice loss before / after | 0.419726 / 1.212776 |
| Forward-only starting loss vs backward-graph starting loss difference | 2.60e-8 |

The earlier independent saved-candidate reload loss also matched within 6.98e-8.
These measurements rule out gross update-size failure and support forward-loss
and saved-loss consistency on the owned 28-example block. They do not prove the
recorded gradient is independently correct. The linear prediction would drive
loss below zero, far outside its usable local range. A forward-only local probe then measured two smaller fractions of the same
update on the 28 owned practice examples:

| Fraction of observed update | Mean practice loss |
| --- | ---: |
| 0 (original model) | 0.419726 |
| 0.01 | 0.379416 |
| 0.10 | 0.281242 |
| 1 (actual native candidate) | 1.212776 |

This supports overshooting on this batch: smaller fractions improve the same
objective that the full step worsened. It is not proof that the smaller files
improve held-out behavior, nor a general condemnation of AdamW. These are
interpolated diagnostic weight files, not new registered native training jobs.
No additional backprop or held-out question repetition occurred. The
[probe receipt](../validation/training-step-audit-20261026/local-step-probe.json)
records model/protocol hashes; temporary probe binaries can be retired after
report preservation. The actual named native candidate remains private.

The objective averages the selected-token loss within each example, then
averages examples. The block had 28 lessons and 310 supervised tokens; the code
lesson contributed 75 of those tokens. Those token counts are not equal to
relative gradient influence or a 75/310 objective weight. A small per-component
learning rate can still produce a large functional change across billions of
parameters. Step-size selection and behavior/retention gates remain separate.

The candidate remains a private, unaccepted diagnostic. Full joined assessment
is also incomplete because of a separate BW grammar-stream failure. See the
[S3 behavioral report](https://github.com/CC-David-CC/sss/blob/feat/learning-stack-20261026/docs/NATIVE-MIXED-BLOCK.md).
Seven distinct CPU controls passed, including independently calculated F32
rounding, corruption and unsupported-input refusals, no artifact mutation and
an uphill direction that cannot be mistaken for acceptance. Their
[receipt](../validation/training-step-audit-20261026/tests.json) records log hashes.
Only this optional Python audit and its tests/docs are added; no CUDA, HIP or
SYCL implementation or existing model/runtime profile was changed.
