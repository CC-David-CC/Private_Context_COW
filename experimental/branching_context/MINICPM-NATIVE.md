# MiniCPM ordinary dense context integration

MiniCPM5-2B uses ordinary Llama full attention with grouped queries: 42 layers,
2048 hidden units, 16 query heads and two KV heads. This backend reuses the native
context/checkpoint and consumed-token machinery with a distinct architecture
contract and state magic. It does not enter the MoE/expert path or reinterpret
Wen's hybrid recurrent state as ordinary attention.

The measured model is the official F16 GGUF at Hugging Face revision
`2079a22f3beaa4e306449978533478fe0522f4b3`, SHA-256
`0ffba3682a853295566b98bc38c0ab755d6b2d91b994bc031ed727a08cfcdf25`.
The native engine pin is llama.cpp `972d2313bc0bf0a45f634f77d95c9fb03aeab12c`.

## Build and qualify

Use an owned checkout and build directory on the intended host. The scripts
reject a different engine revision, altered build bytes or mismatched geometry.
The following CPU path needs no GPU allocation:

```sh
python experimental/branching_context/scripts/build_llama_dense.py \
  --llama /absolute/llama.cpp --engine-build /absolute/new-cpu-engine \
  --output /absolute/new-dense-binding --build-engine --cpu-only
python experimental/branching_context/scripts/prepare_llama_dense_profile.py \
  --model /absolute/MiniCPM5-2B-F16.gguf --build /absolute/new-dense-binding \
  --output /absolute/cpu-profile.json --context 256 --batch 16 --threads 4
python experimental/branching_context/scripts/probe_llama_dense.py \
  --profile /absolute/cpu-profile.json --output /absolute/new-state-proof
python experimental/branching_context/scripts/probe_llama_dense_tokenizer.py \
  --profile /absolute/cpu-profile.json --output /absolute/new-tokenizer-proof
```

For CUDA, omit `--cpu-only`, supply an owned CUDA engine build and select explicit
physical GPUs with `--gpus` when preparing the profile. The state probe checks
fresh free memory with an 8 GiB reserve before loading. Allocation still needs
host ownership, leases and process settlement; a favorable memory reading alone
is not permission to disturb another workload. The tokenizer-only probe requires
a CPU profile, performs no forward passes, and compares actual native IDs.

Keep evidence output directories new. Failures retain their original operation
and result; do not overwrite a failed proof with a subsequent passing result.

## What the integration proves

Actual CPU and CUDA probes each passed 11 checks covering independent branches,
parent immutability, disk restore, mismatched-profile rejection, explicit eviction
reconstruction and rewind. Six matched-arrangement logit comparisons had zero
maximum error. Tolerances stayed fixed; fresh combined versus split batches can
differ even without COW, so that control is reported separately.

The corrected Python MiniCPM tokenizer matched 2,438 native token-ID comparisons
and all 130,560 vocabulary byte pieces. It applies native regex patterns
sequentially. A lossless byte round trip alone does not prove correct token IDs.
The GGUF-native path supports this splitter. Strata pack export rejects it
explicitly until the corresponding pack reader is qualified.

The text renderer supplies one root BOS and accepts both actual native end
tokens: model EOS `</s>` (1) and ChatML `im_end` (130073). Appending after either
preserves consumed history. New user frames do not reinsert the BOS or reinterpret
control-looking user text as protocol. Framing checks cover thinking enabled
and disabled. A separate BW file task passed independent source, output and
input-preservation checks through this native state path.

These are semantic checkpoint results. Whole checkpoint copies do not establish
physical prefix sharing or full G1–G5 certification. LoRA loading and full-gradient
training for MiniCPM remain unqualified. No model weights changed in this work.
[Measured evidence](evidence/minicpm-native-integration-20261024.json).
