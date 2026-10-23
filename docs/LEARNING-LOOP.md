# Checked lessons and native training

Skill definitions, model tests and regression-aware practice are maintained in
the separate [Skills³ (S³)](https://github.com/CC-David-CC/sss) package. PCC keeps
the native runner, private lesson/job records and model/resource ownership.
Ordinary serving and unguarded updates do not import S³. Optional legacy study
entry points delegate to S³; install that package before using them.

An owner can add `skill_guard_config` to this guide's training configuration;
its tree/evaluator and bounded mixed-practice settings are described in
[the S³ integration guide](https://github.com/CC-David-CC/sss/blob/main/docs/INTEGRATION.md).
BW's `--skill` tags a lesson; `--skill-guard` explicitly requests that policy.
Its step budget bounds all native updates across rounds. Original operation
and policy identities remain frozen across retries. Guarded parent hats must
qualify under the same policy; changed trees require a deliberate new baseline.

BW can now write a lesson from a failed check and a verified correction. PCC can
train one lesson immediately or collect lessons for a later batch. A lesson is a
short teaching input and the corrected answer, with references to the original
failure and passing checks. It is separate from conversation history and cached
model context.

This is opt-in preview functionality. The complete CPU wiring test uses real
native weight updates with a generated tiny hybrid model. It proves the controls,
not the quality of a language model. Native Wen LoRA and full backprop already
have bounded two-A100 mechanics tests. Learning quality is measured separately;
more examples or a lower training loss do not automatically qualify a candidate.

## A single correction

After a Work task has failed a test, been corrected and passed its checks, a BW
terminal user can write and train a compact lesson:

```text
/lesson create --prompt "def double(n): return" --completion " 2*n" --title "Use multiplication" --operation-id double-lesson --train-now
```

This uses the latest Work task's existing ledger. Failure and correction can be
in the same task. BW requires a recorded failed check followed by successful
verification and refuses a corrected workspace that changed afterward. It does
not let the model approve its own training. The teaching text is host-selected;
reviewing recorded tests does not automatically prove every generalized lesson.

Use `--turn NUMBER` and optionally `--failed-turn NUMBER` to choose entries from
`/history`. Outside the interactive session, use explicit run directories:

```sh
branchwrite lesson create --corrected-state /private/bw-run \
  --prompt-file /private/lesson-input.txt --completion-file /private/correction.txt \
  --title "Checked correction" --operation-id correction-1 --train-now
```

If the original failure belongs to another run, also pass `--failed-state`.
Drafts are private files outside the project by default. `--output` chooses a
specific new draft path. `--learning-rate`, `--rank` and `--mode lora|full` select
update settings; the server still enforces its owner-configured limits. Defaults
use rank 4 LoRA and SGD at learning rate 0.0001. Those are starting settings, not
a promise that one example will noticeably change the model.

## Queue lessons and train a batch

Replace `--train-now` with `--queue` to save a checked lesson for later. An already
written lesson can also be submitted:

```sh
branchwrite lesson submit /private/lesson.json --operation-id saved-lesson
branchwrite lesson list
branchwrite lesson train --pending --operation-id batch-1 --batch-size 8
branchwrite lesson status batch-1
```

In the interactive terminal use `/lessons`, `/train --pending --operation-id
batch-1 --batch-size 8`, and `/training batch-1`.

`--pending` selects lessons never previously included in a job. Explicit lesson
IDs select exact examples; `--all` deliberately includes previously selected
ones. The selected list is frozen before submission. Repeating the same command
with the same operation ID reconciles that original selection, even if new
lessons arrive or the pending queue is now empty. Changing its request bytes is
rejected. A lost response never authorizes a second update.

By default, the CLI requests enough mini-batches to cover the selection once.
PCC refuses a step budget that would silently leave selected lessons untrained.
The final short batch receives equal per-example weights. Up to 4,096 distinct
lessons can be selected per job, within the server's step and token budgets.
Large selections may need a larger batch or an explicitly increased owner limit.
Failed/interrupted selections are retained for review rather than automatically
returned to the pending queue.

To continue a saved LoRA, add `--parent-job-id PREVIOUS_JOB` and retain its rank.
To update the model's own weights, use `--mode full`. A full-mode parent must be
a completed full candidate; no LoRA is merged. Each job selects a new ordered
lesson batch. This is continuation from completed weights, not automatic recovery
in the middle of an optimizer step.

## The standard COW CLI

Both clients use the same authenticated PCC endpoints and original-operation
receipts. The standalone client can submit existing checked lesson documents and
request either update mode:

```sh
export COW_API_KEY='your existing private server key'
python tools/cow_cli.py --url http://127.0.0.1:8099 lesson submit \
  /private/lesson.json --operation-id lesson-1 --train-now
python tools/cow_cli.py --url http://127.0.0.1:8099 lesson train \
  --pending --operation-id batch-2 --batch-size 8 --parent-job-id lesson-1-train
python tools/cow_cli.py --url http://127.0.0.1:8099 lesson status batch-2
```

Keep remote HTTP behind the existing SSH tunnel or use HTTPS. The service sees
an authenticated owner's verification attestation and hashes; it does not execute
client-provided test commands or read client filesystem paths. Model-generated
tool calls do not receive a new training or promotion capability.

## Server-owner setup

Training is disabled unless the owner supplies a private `training_config` path
in `server.json`, or starts `serve.cow_http` with `--training-config PATH`.
Authentication and text support are required. The training model must match the
serving model's tokenizer/source path. The current native implementation accepts
supported Wen/Qwen3.5 BF16/F32 GGUFs, not quantized Flash/COW serving weights.

The configuration has these fields:

```json
{
  "schema": 1,
  "directory": "/private/pcc-training",
  "max_tokens": 32,
  "max_steps": 128,
  "runner": {
    "model": "/absolute/supported-model.gguf",
    "base_sha256": "ACTUAL_64_CHARACTER_SHA256",
    "executable": "/absolute/private-native-build/pcc-native-train",
    "devices": ["CPU"],
    "device_budget_bytes": {"CPU": 1073741824},
    "reserve_bytes": 16777216,
    "gpu_uuids": []
  }
}
```

The 1 GiB example budget is for the generated tiny CPU wiring model, **not Wen**.
Build instructions and real-model memory observations are in the
[native training guide](../experimental/native_training/README.md). CPU admission
uses currently available RAM. CUDA configurations use logical `CUDA0`, `CUDA1`,
etc. with distinct physical UUIDs in the same order and explicit budgets for
each device. Clients cannot choose executable/model paths, devices or environment
overrides. Resource leases survive into the native child if its HTTP parent dies.

A training GPU cannot also be held by another COW server's account lease. Use
explicit spare GPUs, or CPU training alongside GPU inference. There is no
automatic inference eviction or cross-machine VRAM pool. More examples mainly
increase work; native gradient accumulation processes one example at a time.
All selected inputs must fit the configured token/memory bounds; no lesson is
silently truncated.

The current Wen trainer holds F32 master weights, so its base alone is about
32 GB before adapters, gradients and activations. This does not fit a 24 GB card.
The home hardware routes remain separately qualified: HIP training, P4 training
and real Wen CPU training have not been proved by the tiny CPU wiring test.

## Candidates, evaluation and using a new version

Jobs produce separate immutable GGUFs and content identities. The original model
is checked unchanged. Interrupted jobs are not replayed after a restart. All
candidates remain `promoted=false`; training does not change live conversations.

Before choosing a version, check the original repair, fresh variants excluded
from training, and unrelated preservation tasks. An owner can prepare a separate
dense PCC configuration with the candidate LoRA using the existing `--lora`
setup option, or with a full candidate as its model. Use its new connection for
new BW work. Existing contexts retain their original model/adapter identity.
Do not silently reuse their caches under changed weights.

The owner-run `python -m tools.pcc_learning_quality` harness generates up to
1,000 distinct signed-ceiling-division variations, validates targets against an
independent exact-fraction checker, and keeps held-out and preservation cases
outside training. Fresh native greedy completions are checked semantically;
equivalent correct expressions are accepted. It records a policy and data hashes
before baseline evaluation and compares declared checkpoints. This is a bounded
code-completion study, not general agent or day-long inference qualification.

The [completed 1,000-variation Wen study](../validation/wen-learning-quality-20261023/README.md)
shows why these checks matter: selection accuracy rises from 2/48 to 48/48, but
unrelated controls fall from 16/16 to 11/16 and differently worded blind cases
improve only from 2/16 to 3/16. The candidate is unqualified and unpromoted.
The plumbing works; an always-improving model has not been established.

The independent [S³ project](https://github.com/CC-David-CC/sss) now owns the
tree, arithmetic evaluator and mixed replay. Install the optional pinned package
with `python -m pip install -r requirements-skills.txt` in the server environment;
this does not enable training. Owner configuration and explicit `--skill-guard`
are still required. [Its integration guide](https://github.com/CC-David-CC/sss/blob/main/docs/INTEGRATION.md)
explains the frozen policy and bounded job configuration.

A completed five-skill A100 pilot used four rounds, 16 updates and 128 mixed
example slots. It preserved protected successes but stayed at 14/40 held-out
checks, so no candidate qualified or reached blind evaluation. See
[the full evidence](https://github.com/CC-David-CC/sss/blob/main/validation/README.md).
This is a real native update/evaluation loop, without demonstrated held-out gain.

If final evaluation is interrupted after every checkpoint completed, use the
same owner arguments and work directory with `--resume-final-evaluation`. It
validates saved evidence and evaluates only remaining frozen blind cases. It
cannot submit another training update or overwrite a completed study result.
