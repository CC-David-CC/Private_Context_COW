# Wen training and inference precision

The dense COW adapter has three explicit math profiles. Select one with
`tools/cow_server_setup.py --math-profile NAME`. These options apply to the pinned
dense llama.cpp adapter, not the separate Flash/MoE engine.

| Profile | Matrix products | Attention and KV |
| --- | --- | --- |
| `native` | Existing dispatch | Existing FP16 KV and Flash Attention |
| `f32-activations` | FP32 cuBLAS for BF16, F16 and F32 operands; TF32 disabled | FP16 KV and Flash Attention |
| `f32-reference` | Same FP32 products; fusion disabled | FP32 KV and non-Flash attention |

The original model file is never rewritten. BF16 weights remain BF16 on disk;
the reference multiplication path expands the operands needed by each operation.
TF32 uses FP32 storage and reduced multiplication precision. Removing TF32 alone
does not double storage. FP32 KV uses twice the bytes of FP16 KV for the same
number of entries. Neither choice implies twice the total process memory.

## Why the patch exists

At llama.cpp revision `972d2313bc0bf0a45f634f77d95c9fb03aeab12c`, the CUDA MMVF
and MMF dispatch can use different multiplication precision as batch size changes.
The BF16 MMF kernel converts F32 activations to BF16. Its F32 specialization can
also execute explicit TF32 MMA instructions. The cuBLAS precision environment
variable does not control those custom instructions.

The opt-in patch routes ordinary BF16/F16/F32 matrix products through the pinned
FP32 cuBLAS implementation. The profile also disables NVIDIA TF32. Default kernel
dispatch is unchanged when the option is off. This is a reference precision path,
not a throughput optimization or a guarantee of bit-identical PyTorch results.

Apply the checked recipe to the exact pin and rebuild its CUDA library:

```sh
python3 experimental/branching_context/scripts/prepare_wen_llama.py \
  --llama /path/to/pinned-llama.cpp --f32-activations
cmake --build /path/to/pinned-llama.cpp/build --target llama -j4
python3 experimental/branching_context/scripts/build_qwen35_mixed.py \
  --llama /path/to/pinned-llama.cpp --output /path/to/mixed-build --f32-activations
```

Configure serving with that build's library and the selected math profile.
`f32-reference` is the closest measured comparison to the source-F32 PyTorch
training representation. `f32-activations` retains FP16 KV as a separate measured
tradeoff. The patch recipe, native headers/libraries, environment and precision
settings enter the execution fingerprint. FP32 KV also changes the model/cache
contract. Old numerical checkpoints must not be relabelled for another profile.
New profiles require a separately rebuilt adapter; an old binary cannot claim
these settings. Ephemeral serving remains available with `--ephemeral`.

## Qualification boundary

See `validation/wen-precision-20261022/` for source/model identities, the failed
intermediate controls and the measured short-prefix results. Native checkpoint
restore must match a fresh native oracle exactly with the same operation schedule.
Cross-runtime agreement is measured separately, over every vocabulary logit.
Greedy selection does not force the underlying logits to be equal.

The CUDA reference path is measured on Blackwell. A100, Ada, HIP and other devices
need their own results. Existing Strata HIP support does not qualify this dense
Wen adapter: its explicit device selector currently requires CUDA. P4 also needs
a compatible older CUDA toolchain and cannot hold this BF16 model wholly in 8 GB.
No long-context, training-update or G1-G5 release qualification follows from these
short inference probes.

The layer trace, isolated norm replay, head-order control and LoRA acceptance
requirements are documented in [WEN-CROSS-RUNTIME-NUMERICS.md](WEN-CROSS-RUNTIME-NUMERICS.md).

The current CUDA build still compiles elementary functions with `-use_fast_math`,
including under `f32-reference`. This profile pins F32 products/attention/KV and
disables TF32; it does not promise strict IEEE arithmetic throughout the model.
An isolated strict SiLU control matches PyTorch exactly, while the native fast
helper does not. A complete strict serving build has not yet been qualified.
See the cross-runtime report's focused activation section and its compiler
factor controls before interpreting a dtype as an arithmetic guarantee.
