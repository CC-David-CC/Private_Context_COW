# Bounded Wen precision checks, 2026-10-22

`results.json` contains full-vocabulary measurements for the exact original Wen
GGUF and a source-F32 Transformers import. No sampling or training took place.
Fixed prefix IDs begin `[759, 3796, 13256]` (`The quick brown`). The measured
source hash and pinned llama.cpp revision are in the receipt.

## Findings

1. Original native batched versus serial processing produced errors around 0.03
   to 0.05 at two/three tokens; PyTorch's equivalent difference was around 1e-5.
   Native BF16 MMF rounds F32 activations to BF16, unlike the single-vector path.
   A cuBLAS-only precision override plus fusion disable was an unchanged-output
   negative control: the custom kernel bypasses that setting.
2. Routing BF16/F16 products through FP32 reduced the two/three-token errors to
   about 0.0011/0.0020. FP32 attention/KV then reduced them to about 1e-5.
3. Extending the test exposed another failure: even the reference profile still
   differed by 0.00444/0.00637 at eight/sixteen tokens. The custom F32 MMF kernel
   executes explicit TF32 MMA. The final patch also routes F32 operands through
   FP32 cuBLAS; disabling NVIDIA TF32 then covers the BLAS path too.
4. Final reference results over 1,2,3,4,7,8,9,16,17,32,64 tokens have maximum
   absolute logit error 0.0000357628. All eleven top-token IDs match. Six additional
   text/code prompts also match top-token IDs, with worst error 0.0000200272.
   These are measured differences, not bit-exact cross-runtime equivalence.
5. Final patched build with options disabled matches all 127991 original logits
   bit-for-bit at 1,2,3,8,16,32 tokens. Both text and mixed adapters compile.
6. Both reference adapters pass three fresh-oracle suffix comparisons each after
   native state restore: exactly zero logit difference and zero prefix-forward
   tokens on restore. Mixed tests use text spans; no image encoder is qualified.

Receipts from intermediate implementations are retained as failed controls, not
represented as results of the final recipe. The final recipe SHA is recorded
separately. Native restore requires an identical operator schedule; it does not
claim batch-versus-serial bit equality.

## CPU controls and limits

Precision admission initially failed to import the new validator. Subsequent
adversarial FP32-attention controls failed four assertions: mismatched cache type,
Flash setting, missing native binding and unbound environment override. The
validator now rejects those configurations before native allocation.

`test_qwen35_math_profile`, `test_qwen35_lora_profile`, `test_qwen35_backend`,
`test_qwen35_mixed_binding`, `test_qwen35_greedy_dispatch` and
`test_weighted_prefix_scheduler`: 34 passed, no skips. G0: 51 passed, no skips.
The optional LoRA binding compiles and passes identity admission checks; no real
trained adapter or LoRA-update numerical effect has been qualified.

PyTorch source-F32 allocated about 32.9 GB at peak in the bounded probes. Concurrent
native and PyTorch processes occupied about 50 GB of GPU memory. These are forward
measurements, not backpropagation requirements or a context-length memory limit.
No long-context, BW task competence, training gain, A100/Ada/HIP numerical parity
or cumulative G1-G5 release certification is asserted here.

The configured FP32 HTTP server also passed two concurrent bounded generations,
two exact native oracle/restore comparisons, four tokenizer comparisons and
parent-head immutability. SIGTERM exited with code zero and left zero ephemeral
sessions. See `http.json`. Final launcher/vision/import CPU controls: 12 passed.

The completed cross-runtime report is `docs/WEN-CROSS-RUNTIME-NUMERICS.md`.
`layer-trace.json` records 154 semantic boundaries per initial prefix, an isolated
normalization replay, the diagnostic shared-native-norm experiment and a
head-order projection replay. The first norm becomes exact when its forward
kernel is shared, but the final model does not. No backward or training path was
promoted by that diagnostic.

The follow-up `activation-trace.json` shows that convolution is exact and the
first one-token recurrent-block divergence appears at SiLU. An identical-input
helper replay isolates native fast-math exponential/division choices. Removing
fast math yields exact PyTorch SiLU output on all 8192 values. Both observers
preserve logits. Independent downstream recurrence/gated-norm contributions,
remain unqualified by that isolated replay.

`strict-remeasurement.json` records the subsequent full-model strict diagnostic:
all 143 CUDA compile commands match except the intended math flags, after an
initial CMake optimization-flag mismatch was corrected. The eleven-prefix worst
error increases from 0.0000357628 to 0.0000677109; the six-prompt worst decreases
from 0.0000200272 to 0.0000162125. Both builds match all seventeen top-token
predictions. Strict text and mixed restore comparisons each pass three cases
with exactly zero error and zero prefix work. `strict-cuda-cmake.patch` records
the extra external-source change; this is not a production default change.

`lora-1k128.json` records one real LoRA update on a separate owned synthetic
coding fixture with 1024 prompt and 128 decode tokens. Adapter-delta RMS transfer
error is 0.3583% of the PyTorch update RMS for the regular reference build and
0.3488% for the strict diagnostic build. All 129 score-frontier top-token choices
and 128 independently generated tokens agree before and after. The update does
not change the greedy sequence on this fixture. Tensor readback, zero-adapter
controls and adapted-prefix restore checks pass exactly. Strict native runs and
each control use separate processes; initial same-process strict results are
superseded. No mixed-task BW training gain or adapter promotion is claimed.
