# Registered LoRA AdamW through BW and COW

An owner can now register an explicit LoRA AdamW policy in PCC's existing
lesson/job service. Both BW and COW select it by its content hash. This uses
the same native backprop, immutable candidates and operation receipts as full
training; no separate trainer, task ledger or automatic model promotion is added.

The service path has real tiny CPU proofs and a separate actual MiniCPM A100
qualification through both clients. One update plus a resumed update produced
the same complete adapter and optimizer-state file hashes as two uninterrupted
updates. This qualifies that short integration case, not every model, GPU, long
training context or unattended learning loop. The running fixed-practice
comparison was not changed by this addition.

## Owner policy

Existing schema-1 policies remain **full-only**, with their original descriptor
and hash. Schema 2 adds explicit `mode` and `rank`. Save this descriptor as
`lora-fresh.json` and calculate its ID with PCC's existing identity function:

```json
{"schema":2,"title":"LoRA AdamW, fresh block","mode":"lora","rank":16,
 "optimizer":"adamw","learning_rate":0.0001,
 "optimizer_settings":{"schema":1,"weight_decay":0.0},
 "state_policy":"fresh","training_budget_seconds":null}
```

```bash
python -c "import json; from tools.pcc_optimizer_profiles import identity; print(identity(json.load(open('lora-fresh.json'))))"
```

Place that exact descriptor under its hash in `runner.optimizer_profiles` in
the owner's [training configuration](NATIVE-LEARNING-JOBS.md). To allow exact
continuation, register a second descriptor with a distinct title and
`state_policy: "resume"`; every recipe field must otherwise match. Its ID will
be different. Clients discover both with `lesson optimizers`.

The descriptor binds the optimizer, rate, mode, rank, settings and state policy.
Schema-2 full policies use `rank: null`. LoRA policies currently support AdamW,
rank 1 through 256 and explicit updates with a null training-phase time budget.
The existing ordinary-Llama qualification is at most 512 consumed training
tokens. Owner deadlines remain execution guards. These are supported ranges,
not a recommendation that every rank or rate improves a model.

## One checked example, then continuation

With an existing authenticated BW connection and `BW_COW_API_KEY`:

```bash
python -m branchwrite lesson optimizers
python -m branchwrite lesson submit checked-lesson.json \
  --operation-id example-01 --train-now --mode lora --rank 16 \
  --learning-rate 0.0001 --steps 2 --optimizer-profile-id FRESH_POLICY_SHA
python -m branchwrite lesson status example-01-train
python -m branchwrite lesson train LESSON_ID \
  --operation-id continue-01 --mode lora --rank 16 --batch-size 1 --steps 3 \
  --learning-rate 0.0001 --parent-job-id example-01-train \
  --expected-parent-sha256 COMPLETED_ADAPTER_SHA \
  --optimizer-profile-id RESUME_POLICY_SHA
```

`checked-lesson.json` must be an admitted, independently verified lesson bound
to the selected model contract. BW can author it from failed and corrected Work
checks with `lesson create`; a policy ID is not lesson approval.

For COW, use the same action and options after
`python -m tools.cow_cli --url YOUR_PCC_URL lesson`, with `COW_API_KEY`.
Its single-example action is `submit`; BW also supports `create --train-now`.
Explicit `--steps` on a registered single-example request is honored by both
clients. Omit the profile and the existing default LoRA path remains SGD.

`resume` retains the exact ordered lessons, learned adapter, optimizer recipe,
moments, global step and data cursor. PCC checks registered parent metadata
before queuing and checks again before execution. Its native executor checks
the actual adapter/state hashes, dataset and saved state header. No large model
or state file is hashed while holding the API's metadata lock.

For a **new lesson block**, select the **fresh** policy and a completed parent:

```bash
python -m branchwrite lesson train LESSON_A LESSON_B LESSON_C \
  --operation-id new-block-02 --mode lora --rank 16 --batch-size 2 --steps 2 \
  --learning-rate 0.0001 --parent-job-id continue-01 \
  --expected-parent-sha256 COMPLETED_ADAPTER_SHA \
  --optimizer-profile-id FRESH_POLICY_SHA
```

This keeps the learned hat but starts new moments and a new cursor. The two
updates consume two examples and then one, covering all three lessons. Changed
or reordered lessons cannot silently resume old moments. Registered adaptive
profiles remain separate from the legacy `--skill-guard` multi-round wrapper.

The parent must be a real completed service job. An offline study checkpoint
is not automatically imported into that service. Pin the **adapter** hash for
LoRA; its base GGUF hash alone does not identify the hat. Policy discovery and
the actual receipts distinguish these boundaries.

## Retry, measure, decide

Keep the original operation ID and request bytes after a timeout. Retry the
same command or inspect status. An accepted update is not repeated. `--pending`
selection stays frozen for that operation even if another lesson arrives.
Changing steps, policy, parent pin or selected lessons under an accepted ID is
refused. Never remove a pin or reset moments to work around an uncertain result.

S3 owns the independent learning/forgetting assessment and next-block proposal;
BW owns the task and verified correction; PCC owns native training and artifact
identity. A completed update is not an acceptance result. Attached evaluation,
merged-export equivalence and quantized-model acceptance are separate checks.
The existing full-weight `lesson preserve` command does not merge a LoRA hat.

## Reproduce the CPU proof

From S3's checkout, with NumPy and PCC's existing CPU native executable:

```bash
python -m sss_integration.lora_policy_cpu \
  --pcc-root /owned/PCC --bw-root /owned/BranchWrite \
  --executable /owned/build/pcc-native-train --output /owned/new-proof-directory
```

Both actual CLIs exercise HTTP authentication and native updates. For each
client, two single-example steps plus three resumed steps match five
uninterrupted steps **bit for bit**, including every optimizer moment. A fresh
three-lesson block matches an independent native control using the inherited
hat. The proof also covers partial batches, frozen pending selections, changed
operation refusal and incompatible resume refusal before a job is queued.
The base remains unchanged; no GPU, model promotion or model upload occurs.

[Native/client evidence](../validation/lora-policy-20261025-az/cli-integration.json)
and [source/regression receipts](../validation/lora-policy-20261025-az/checks.json)
record the exact qualification scope. The initial development fixture used an
out-of-vocabulary target and correctly failed native admission; that fixture
was corrected. It was an authoring error, not a measured model regression.

## Actual MiniCPM through both clients

A separate shared-A100 integration run used the original F16 MiniCPM5-2B base,
rank-16 AdamW LoRA at 1e-4 and S3's existing checked practice generator/renderer.
Its one selected example had 59 consumed tokens and three supervised answer/
terminator targets. This was owned synthetic practice, not a claimed spontaneous
model failure or copied development question.

BW submitted the first one-update job; COW continued its exact registered
adapter, moments and cursor. A separate COW job performed two updates from the
same original base. The final **whole adapter file hashes and whole optimizer
file hashes matched**, as did every inspected saved tensor/moment. The resumed
step/cursor were both two; all 590 reported gradient groups per update were
finite. Native training loss on this example moved 7.7261 → 2.4041 → 0.08077.
This is not a held-out learning or forgetting measurement.

Original client requests were retried without additional updates. A subsequent
attempt to rerun the whole probe was refused before service startup, leaving
all original operation hashes unchanged. The base stayed unchanged. The three
jobs reserved one shared A100 placement for 182.8 seconds (0.0508 allocation
GPU-hours), including model I/O and CPU checks; this is not a kernel-compute
benchmark. One initial placement attempt was refused before dispatch because
another owner held its lease; no native update was retried.

[Actual-model integration](../validation/registered-lora-model-20261025-ba/integration.json),
[source/recovery checks](../validation/registered-lora-model-20261025-ba/checks.json)
and [the repeatable owner-run probe](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/REGISTERED-LORA-MODEL-PROBE.md)
keep this qualification separate from the ongoing 68-update learning curves.
No inference harness, merged deployment, broad acceptance or automatic learning
coordinator is qualified merely by this result.

The [stricter finite-byte recheck](../validation/registered-lora-model-20261025-ba/byte-comparison.json) also passed on those unchanged saved artifacts. The frozen executed helper is retained alongside the receipt; no GPU updates were repeated for that comparison refinement.
