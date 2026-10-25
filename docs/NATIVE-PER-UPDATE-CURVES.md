# Native mechanics for per-update learning curves

The owner requested quality measurements indexed by completed updates and
practice exposure, with no fixed training-phase time budget. The
[S3 protocol](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/validation/minicpm-native-learning-20261024/per-update-curve-protocol.json)
measures every update 0..68 for four full-update regimens and an AdamW LoRA
control. S3 owns the checked curriculum, frozen assessment and gain/loss plots.
PCC owns native updates, optimizer state, resource leases and GGUF artifacts.

## Timing without a training cutoff

Omit `training_budget_seconds` from a native request and set `record_timing=true`.
The explicit `steps` count determines the requested update count. Each step
reports gradient/coverage, optimizer, post-update batch-loss and total phase
seconds. Loading, saving, independent reload and held-out assessment are separate
costs. Existing requests without either option retain the prior receipt shape.
A lifecycle deadline is still enforced; it is not a quality stopping rule.

For a curve, request one update, preserve its completed checkpoint, assess it,
then resume the exact optimizer state and data cursor. Never restart moments
between points. Same examples, order and optimizer configuration are required.
An interrupted or unknown operation must be reconciled from its original
records rather than launched again with another ID. Tiny CPU and CUDA
three-update results match uninterrupted versus resumed weights and moments
exactly. This does not promise bit-identical real-model learning across hardware.

## Native AdamW LoRA

The ordinary-Llama short executor now accepts `optimizer="adamw"` with
`mode="lora"`. This is standard end-to-end native backpropagation, not PyTorch.
The existing adapter path trains A/B factors for all hidden linear matrices and
the prediction head; embeddings and norm/gain parameters remain frozen. Rank
and alpha match, so the adapter scale is 1. The MiniCPM rank 16 control has 295
target matrices/590 factor tensors. Full training has 381 parameter tensors.
The first update normally has zero A gradients because B begins at zero; that
is expected LoRA initialization behavior, not lost gradient coverage.

For continuation, the base model/path/hash remain original. Supply the exact
registered parent adapter and its SHA, optimizer-state path/SHA and example
cursor. Optimizer state binds the adapter SHA as the trained artifact. A full
update instead binds the full model SHA. Native and Python checks refuse the
wrong adapter, moments, dataset or cursor. Fresh moments for a changed lesson
block remain distinct from this exact-state curve continuation.

This is an experimental owner/native route. The registered service/CLI profile
selector still advertises full-weight jobs. Adaptive Muon LoRA is not implemented
and is not presented as one of the four full-update Muon regimens.

## Native F32 merge for a fair inference comparator

`pcc-native-train --merge-adapter OWNER_REQUEST.json` loads a pinned floating
base and adapter, computes each updated matrix with GGML F32 operations and
writes an exclusive full-weight GGUF with F32 tensor storage. It reuses the
existing native model loader, graph operations, memory admission and full-GGUF
writer. It performs no gradient update or live-model replacement.

The request supplies owner-configured model/adapter paths and hashes, output
path, rank, devices and memory reserves. Use `devices=["CPU"]` for the currently
qualified merge. Include the ordinary-Llama bounded sequence/memory profiles
and a checked example for the existing admission estimator. No network client
can select those resource paths through the training API.

The tiny fixture independently compares every merged weight against FP64
`W + B @ A`, including frozen tensors, and compares adapter versus merged
forward logits/loss. Maximum merged-weight error is 1.45e-8; the CPU-merge
logit difference is below 9e-8. All merged tensors store F32. Wrong input hashes
and an existing output are refused. These are small mechanics measurements.
The curve additionally gates its first actual-model LoRA point on a numerical
check before grading its merged GGUF. It will report that result separately.

The full native AdamW/Muon paths remain unchanged numerically in the tiny CPU
and CUDA compatibility checks. [Measured mechanics](../experimental/native_training/validation/native-curve-mechanics.json).
Repeat the source tests with `tools/test_lora_adamw_native.py` and
`tools/test_native_lora_merge.py` against an isolated native build. No model
files or protected evaluation answers are part of the public evidence.

The same native source also builds on the A100 host and passes the CPU LoRA
continuity/full-update compatibility and independent F32-merge checks there.
[CPU portability evidence](../experimental/native_training/validation/a100-curve-cpu-mechanics.json).
That check allocates no GPU and does not qualify A100 CUDA execution or allow
its results to be substituted into an existing frozen-runtime curve.

## Verified artifact retention

`tools/pcc_curve_retention.py` is an owner-only archive helper. S3 selects
milestones, the live parent and best-so-far identities; PCC verifies the private
owned archive, original copy receipt, immutable measured-point evidence and a
qualified later checkpoint before retiring superseded binaries. It operates
only on the exact model/state roles in this experiment's manifests. Scores,
response evidence, source and numerical evidence remain.

Retirement writes and fsyncs its original intent before unlink, then records
completion. Reconciliation follows that same intent. Named GGUF hardlinks are
retired with the corresponding model. Changed bytes, links or mismatched
receipts are refused before deletion. Six actual filesystem tests pass.

After-report cleanup additionally requires the completed report's SHA and
publication commit. It preserves the selected current model while retiring
superseded study checkpoints/states. The original base and other project
artifacts are outside this archive's authority. This is separate from COW
context/cache eviction and is not a remotely selectable filesystem API.
