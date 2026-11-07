# PCC model and operation support boundaries

Inspected 2026-11-07, from PCC main baseline
`624bb2a9d8e928223b0128e9f743b5a45a53c731` and the Qwen preservation branch.
This is a source/documentation audit, not a fresh rerun of historical campaigns.
No row means all formats, devices, context lengths or operations are qualified.

## Evidence vocabulary

- **Documented bounded result:** repository records an executed result; repeat
  its exact profile before claiming it for changed code or another device.
- **Implementation inspected:** a code path exists; this is not numerical proof.
- **Fixture result:** generated small models prove specific mechanisms, not the
  real large model's allocation, learning quality or general capability.
- **Resource probe:** admission/allocation/gradient observation, not a completed
  training update or served continuation unless explicitly recorded.
- **Not established:** the inspected evidence does not prove it. This does not
  prove no implementation exists elsewhere; inventory before assigning unavailable.
- **Missing PCC integration:** standalone code does not expose that PCC operation.

## Current operation matrix

| Model / route | Inference | LoRA inference | LoRA training | Full training | Optimizer resume | Strict checkpoint restore | Explicit reconstruction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8 Flash / qwen4exp / Strata | Bounded PCC Q4/Q8 serving and pressure results | Not established here | Not established here | Not established here | Not established here | Native COW path and bounded zero-prefix-work checks | Guarded path implemented for qualified normal-verifier profiles; map exact receipts before broad claims |
| Qwen3.5-9B, documented Q8_0/BF16 campaigns | Bounded text and image results on 4090 / FP16 KV | Attachment path exists; Qwen-specific numerical qualification not established here | Not established here | Not established here | Not established here | Documented full-logit comparisons and separate-process zero-work restores | Current explicit reconstruction guard excludes this adapter |
| Wen / Wen1.1 dense route | Bounded COW and client results | Rank-8 diagnostics/readback/restored-prefix checks | Narrow external PyTorch/PEFT pilot, followed by native evaluation; integrated native pathway incomplete | Not established here | Not established as general native support | Bounded hybrid/mixed results, exact model/profile required | Same current Qwen35-adapter guard limitation |
| MiniCPM5-2B dense-Llama | Explicit geometry-checked adapter | Real rank-16 A100 full-logit/restore checks | Real short-profile AdamW qualification; longer/higher-rank extensions have separate bounds | Real short full-model AdamW job and reload; no demonstrated gain over parent | Real short LoRA split/resume exact hashes; broader optimizer mechanisms on fixtures | Bounded attached-LoRA exact restores, zero forward work | Guarded dense-Llama consumed-selection route implemented; map receipts to advertised envelope |
| Qwen3.6-35B-A3B UD-Q4_K_XL | Standalone 128-token runs on GTX1060, RTX4090 and Blackwell | Missing/unqualified in PCC | Missing/unqualified in PCC | Missing/unqualified in PCC | Missing/unqualified in PCC | PCC adapter/state codec missing | PCC integration missing |

Branching is not implied by plain inference. Physical prefix sharing is not
implied by semantic branching or a whole-state copy. This table does not grant
G1–G5 certification to any whole model family.

## What the evidence actually establishes

### Flash / qwen4exp

[MODELS-AND-RESULTS.md](MODELS-AND-RESULTS.md) and
[TWO-COMPUTER-SETUP.md](TWO-COMPUTER-SETUP.md) describe the pinned default Q4 route
and bounded Q4/Q8 numerical pressure work. Lower quantizations, other hardware,
training and longer contexts do not inherit those results. The original ordinary
Strata server and PCC COW server are different entry points.

### Dense Qwen3.5 and Wen

[QWEN35-9B-BACKEND.md](QWEN35-9B-BACKEND.md) records 178 full-logit comparisons and
nine separate-process strict restores for each Qwen3.5-9B Q8_0 and original BF16
campaign. Its separate research mixed fixture has 38 comparisons and three
restores; Wen profiles do not inherit the 178/9 count. These use full native-state
copies; they do not establish physical GPU prefix
sharing or complete cumulative G1–G5 coverage. Profile and ordered-batch numerical
constraints matter.

In `experimental/branching_context/src/context_time_travel/real.py`, inspect
`restore_checkpoint` and `rebuild_materialization` separately. The latter's guard
currently permits qualifying dense-Llama consumed-selection or Strata normal
verifier profiles. Qwen35's `llama-forward-consumed-v1` path does not pass that
explicit reconstruction guard. This is a concrete current boundary to preserve
in capability reporting, and a future capability to extend deliberately.

[WEN-TRAINING.md](WEN-TRAINING.md) distinguishes preparation/import, native adapter
diagnostics and an actual narrow external training pilot: rank-8 final-layer MLP,
four AdamW updates on A100 with PyTorch/PEFT, then native COW evaluation on
Blackwell. That experiment did not establish general native Wen training or a
promoted improved adapter. Do not label external pilot training as native PCC
training support.

### MiniCPM inference and training

[DENSE-LORA-COW.md](DENSE-LORA-COW.md) records eight full-vocabulary frontiers,
two strict restores with zero forward work and bounded shared-root requests for
an attached rank-16 adapter. Attached and merged adapters have different identities.

[REGISTERED-LORA-ADAMW.md](REGISTERED-LORA-ADAMW.md) records an actual short-profile
MiniCPM A100 update plus resume matching uninterrupted two updates by complete
adapter and optimizer file hashes. Preserve this existing accomplishment.
[NATIVE-LEARNING-JOBS.md](NATIVE-LEARNING-JOBS.md) records a real short full-model
AdamW HTTP/CLI job, saved-weight reload and 864-case assessment. That establishes
bounded mechanics, not a learning gain over its immediate parent.

[DENSE-LLAMA-ACTION-TRAINING.md](DENSE-LLAMA-ACTION-TRAINING.md) adds generated-model
CPU/CUDA finite-difference, 8K execution and optimizer-continuation results.
[MINICPM-ACTION-MEMORY.md](MINICPM-ACTION-MEMORY.md) records a real 3,663-token
full-gradient probe with no weight update and longer-case resource refusals.
Later [SELECTED-HEAD-ACTION-TRAINING.md](SELECTED-HEAD-ACTION-TRAINING.md) and
[ATTENTION-RECOMPUTE-TRAINING.md](ATTENTION-RECOMPUTE-TRAINING.md) improve mechanisms;
they do not yet prove the complete real longer-lesson loop. Earlier refusal is
historical evidence, not proof that every newer implementation must fail.

[FACTORWISE-MUON-LORA.md](FACTORWISE-MUON-LORA.md) contains generated-model and
high-rank fixture results plus real persistent-admission probes. Admission is not
an executed full gradient/update. AdamW, Muon, factorwise variants, rank choices
and sequence profiles each need their own support declaration.

## Qwen3.6: why this work exists

[QWEN36-LOW-MEMORY-INTEGRATION.md](QWEN36-LOW-MEMORY-INTEGRATION.md) pins the model,
source patch, compiler conditions and recorded runs. This is a hybrid-attention
MoE model, distinct from dense Qwen3.5. The archived bounded expert cache makes
its weight access feasible on a small GPU; it does not save the model's complete
continuation state or its PCC controller.

Missing work, in dependency order:
1. Inventory exact math/state/format dependencies and choose an adapter route.
2. Supply constrained registration, loading and useful unsupported-operation errors.
3. Expose consumed frontier, complete attention/recurrent/convolution/position state
   as applicable, and next logits; establish fresh-history numerical correctness.
4. Integrate immutable histories/controllers, isolated branches and cancellation.
5. Implement complete checkpoint export/strict restore and explicit reconstruction
   with identity validation, process restart and truthful work accounting.
6. Qualify bounded expert/cache pressure and cleanup without changing the modeled
   function; qualify any physical sharing claim separately.
7. Prove the PCC service/API workflow and publish exact support boundaries.

LoRA/training are separate future operations for Qwen3.6, not implied by items 1–7.
Existing models' training must continue to meet their current qualified contracts.

The hybrid and all-resident smoke runs generated different text. This is an
unresolved equivalence question, not proof of either correctness or corruption.
Compare forced-input full logits and state against a pinned reference before
claiming placement transparency. Cross-device bit-identical text is not assumed.

## How this becomes a live support ledger

For every advertised operation record model/weight identity, format, backend/build,
device/toolkit/compiler, context and batch limits, adapter/optimizer constraints,
state codec, execution profile, prerequisites, evidence and known unsupported
combinations. Keep three fields separate: implementation status, historical
qualification and current-revision test status. Missing evidence is not PASS.

P00 in [AGENT-CONTROL-PLAN.md](AGENT-CONTROL-PLAN.md) maps these documentary claims
to runnable tests and archived receipts. It must record missing assets or devices
rather than manufacturing a full support matrix from this summary.
