# S³ foundations, thinking and multi-turn lessons

S³ provides the curriculum, independent tests, teaching artifacts and reports.
BW executes tasks through its existing ledger and tool boundaries. PCC contains
modified Strata plus COW and owns inference, context, native training and candidates.

Start with the [S³ operator guide](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/START-HERE.md)
and [handoff](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/OPERATOR-HANDOFF.md).
The code supports bounded reference exploration now; live Wen qualification is
separate. No serving model or adapter is automatically changed.

## Inference thinking

The optional compatibility frontend now accepts an owner-selected
`--openai-thinking` setting, including through `cow_server_setup.py` and its
saved `openai_thinking` configuration. Default settings preserve the previous
nonthinking renderer. The thinking renderer uses a distinct cache identity.
A request explicitly specifying a conflicting thinking mode is refused.
Rendered input plus output reservation must fit the physical profile capacity.

When the owner enables private frontend tracing, records retain decoded text
before answer/tool parsing, its hash and the parsed fields. Raw traces stay
private. A parser error must not silently become a model skill failure.

## Multi-turn lesson schema 2

Existing schema-1 correction lessons remain supported. Schema 2 records exact
rendered spans with role, kind, verification and practice scope, plus archive,
checker evidence, tokenizer and renderer identities. The owner attests that the
examples were independently verified; a JSON boolean is not a new task checker.

Target policies are `verified_thinking_actions_answer` and `actions_answer`.
Inference thinking is independent of that selection. User/tool/context tokens
receive no target labels, but remain in the differentiable context. Unverified
assistant targets and evaluation material are rejected. Token boundaries must
align with the real tokenizer. Oversized examples fail instead of truncating.
The existing BW/PCC submit and single/batch controls accept these lessons.

## Opt-in long CPU mechanics

The ordinary `short-v1` native limit remains 8,192 tokens. The explicit
`tiny-cpu-long-v1` profile admits only a bounded tiny CPU hybrid fixture, up to
32,768 tokens and at most 256 selected target positions. Both native LoRA and
direct full-weight modes use ordinary complete-graph backprop, without PyTorch.
The section-adjoint experiment remains separate.

The long path assembles recurrent columns with balanced concatenation and uses
the ordinary concat backward rule in a private pinned GGML build. A single CPU
uses a direct graph allocator: the generic multi-device scheduler's worst-case
split reservation was excessive for this graph. Admission includes metadata,
activations, target-adjoint allowance, current available RAM and explicit headroom.
The normal short path retains its existing assembly and scheduler.

CPU fixtures exercised 16K and 32K updates in both modes and matching saved-
candidate reload loss. This does not qualify long Wen training, CUDA long
training, all-token 32K supervision, HIP/SYCL, or useful language learning.
Candidates remain separate; original model files are checked unchanged.

See the dated validation record and S³'s implementation gates for exact evidence,
remaining live-model checks and the operator's next authorized experiment.

CPU checkpoint: [dated validation](../validation/foundations-20261023/README.md).

Training admission follow-up: Owner-configured schema-2 lessons now use the tokenizer control-token mode for already-rendered chat spans. Schema-1 text keeps its previous plain-text tokenization. Seventeen lesson/segment checks passed, including the configured-service control-token distinction; native CPU guard integration passed.

## Read-only native training admission

The native executable accepts `--admit-training REQUEST.json`. It checks the same geometry, sequence policy and memory plan as `--train`, then reports `MEMORY_ADMISSION_PASS` or `MEMORY_ADMISSION_REFUSED` without loading weight tensors, updating weights or writing a candidate. A pass does not qualify graph allocation, derivatives or model quality. The normal training path still refuses insufficient memory before loading weights.

The 2026-10-23 Wen foundation pack passed exact rendering at 58–375 tokens. The current ordinary LoRA planner estimated 336.9 GiB for its longest lesson, so the single-A100 attempt was refused. Wen weights stayed unchanged. See the [S³ pilot evidence and limits](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/FOUNDATION-TRAINING-PILOT.md).

The configured launcher also preserves API keys beginning with a dash by binding their argument value explicitly. Ordinary launcher arguments remain unchanged.


## Dense GPU gradient checkpoint

The opt-in `dense-short-v1` profile now supports bounded complete-graph assembly
and explicit block placement. Tiny-model LoRA/full updates and reloads passed on
two A100s at 64 and 375 tokens. A separate real Wen 1.1 gradient-only check passed
at 178 tokens across four A100s, without updating weights or writing a candidate.
The longest full-pack lessons and an actual guarded learning campaign still need
separate admission and evidence. See the [native profile and limits](../experimental/native_training/README.md#bounded-dense-sequence-profile).


## Real candidate sessions for S³

`tools.pcc_foundation_session.DenseCandidateSession` supplies the host-owned
session expected by S³'s `FoundationGuardEvaluator`. Prepare a fresh baseline
server directory with the existing `cow_server_setup.py` after freezing the source.
Use a loopback OpenAI-compatible endpoint and one explicitly selected GPU.

```python
from tools.pcc_foundation_session import DenseCandidateSession
from sss.adapters.foundation_training import FoundationGuardEvaluator

session = DenseCandidateSession(server_directory, python=server_python,
    gpu_uuid=selected_uuid, deadline_seconds=900,
    maximum_activity=50, minimum_free_mib=32768, reserve_mib=8192)
evaluator = FoundationGuardEvaluator(tree, pack, session,
    base_sha256=base_sha256, runtime_id=session.runtime_id)
```

This adapter rebuilds each exact candidate profile with the existing PCC profile
builder and runs the existing COW server. Source, native build, geometry,
execution settings, thinking and output limits remain bound to the baseline.
Full candidates select their own weights; LoRA candidates select the immutable
base plus the exact adapter. Neither is automatically promoted.

Five spaced GPU samples must meet the declared utilization and memory limits.
PCC owns the actual GPU lease. The inherited training-worker lease stays open in
the server subprocess; a process deadline remains effective if the operator
loses its connection. Every session preserves raw frontend traces, exact prompt
token receipts, owned process/allocation identities and settlement. Errors raise
without retrying an evaluation request or inventing a model grade.

Eight CPU tests exercise resource refusal, runtime/candidate binding, loopback and
non-recursive configuration, and actual owned-process stop/reap. Live candidate
quality remains a separate measurement; this API alone does not establish it.

The initial live wrapper attempt stopped before inference because its isolated
process lacked the runtime import path. The failed operation was preserved with
zero evaluated cases and no training. A fresh-process regression test reproduces
that failure and passes with the runtime path initialized before frontend imports.

A second pre-inference attempt exposed interpreter-path handling: resolving a
virtual environment Python symlink selected the system Python and lost its
installed dependencies. The wrapper now preserves the configured interpreter
path. A real disposable-venv test reproduces the wrong-prefix failure and passes
with the fix. Both failed attempts contain zero evaluations and zero updates.
