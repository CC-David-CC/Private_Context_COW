# PCC native training bring-up

INCOMPLETE. This branch is implementation work, not a qualified trainer or a
claim of improved model behaviour. No serving model or adapter has been promoted.

The owner requests native training without PyTorch and without BW/interactive
CLI integration. Both LoRA updates and direct base-weight backpropagation are
required. Direct full-weight training is distinct from merging a LoRA into a base.

## Ordinary backpropagation is the default

The standard path builds the complete causal model graph, forms the masked
completion loss and invokes one end-to-end reverse pass. Multi-GPU tensor
placement must preserve that path; adding devices must never silently select a
section-boundary training experiment. All selected parameters must have checked
gradients. No unsupported operation may silently stop gradient propagation.

Passing adjoints between independently executed model sections is a separate
experimental, explicitly selected path. Its existence cannot substitute for
ordinary model backpropagation. Its API admission requires an experimental flag.

## Current files

- `native_train.cpp`: explicit GGML recurrent/attention/MLP graph primitives,
  small layer gradient gates, a complete small hybrid-model standard graph,
  LoRA parameters, direct weights and native SGD. The real-model training loader,
  checkpoint/restart path and A100 multi-GPU proof remain to be qualified.
- `build.py`: a private copy/build of the pinned GGML dependency, kept separate
  from every serving executable. A strided-noninplace-scale materialization
  patch is recorded explicitly. Its necessity and coverage are still under test.
- `tools/pcc_native_training.py`: offline admission and candidate-version helpers.
  Verified example provenance, explicit completion masks, batching weights,
  content identity and non-promotion are checked without importing a tensor
  training framework.
- `tools/test_native_training.py`: admission/version/default-backprop tests.

The initial compiler and backward-layout failures are retained in the private
operator run directory. A compile or a tiny loss reduction is not a gradient,
real-model, task-learning or multi-GPU pass.

## Acceptance sequence

1. Tiny recurrent and attention VJPs: directional finite differences through every
   parameter tensor. Fixed F32 perturbation 0.002; absolute error below
   0.0003 + 0.02 times the absolute numerical derivative.
2. Ordinary complete hybrid graph: check parameter directions in every stage,
   including embeddings and the head for full-weight mode. Fixed perturbation
   0.003; absolute error below 0.0004 + 0.03 times the numerical derivative.
3. Compare ordinary one-device and multi-device forward losses and gradients.
   Prove actual tensor allocation and backend execution on each selected GPU.
4. Validate the actual Wen graph/weights against the pinned native forward,
   keeping the execution/rounding schedule and any discrepancies explicit.
5. Real LoRA and full-weight native updates: finite gradients through the complete
   model, immutable original weights, explicit target scope and masked loss.
   Exercise single examples, accumulated batches and the partial final batch.
6. Save/reload and interrupted-step recovery: exact data/model/adapter/step
   identities, optimizer state, no silent repeat after an uncertain completion,
   new immutable candidate artifacts and old-cache incompatibility.
7. Repairs, independent held-out variants and preservation checks through PCC.
   Loss reduction alone never promotes a candidate. Do not claim an always
   improving adapter from a mechanics test or a failed task experiment.

## Capacity and ownership

Inspect actual A100 use before admission, identify selected physical devices,
retain per-device headroom and record capacities. Aggregate free VRAM is not one
automatic allocation. Shard weights/gradients/optimizer state explicitly and
measure transfers and memory. Leave other accounts and serving processes alone.

Only owned, verified training examples supply gradients. Held-out and protected
evaluation answers stay out of the training split and public artifacts. Candidate
model bytes, prompts and raw traces remain outside the source repository.
