# Real MiniCPM LoRA merge and serving diagnostics

The first native AdamW rank-16 LoRA update completed, but its all-position
adapter-versus-merged logit comparison failed the existing numerical gate.
Across 379 positions and a 130,560-token vocabulary, maximum error was
`0.0025386810302734375` and RMS error was `0.00007002313138727627`, against
predeclared `0.0002`/`0.00005` bounds. This failure remains a failure.

Two independent follow-ups help localize it:

1. The streaming CPU FP64 audit checks every saved merged weight against
   `W + alpha/rank * B@A`. All 381 tensors, 295 adapted matrices and
   2,516,756,480 values passed the existing `2e-7` weight bound. Maximum error
   was `5.9409964503487345e-8`; frozen tensor values were exact. This checks
   the saved merge, not equivalence of every inference operation.
2. The native training adapter and native serving adapter produced bit-exact
   last-logit vectors on three owned prefixes of 72, 354 and 379 tokens.
   Repeated serving calls were also bit-exact. No new training was performed.

The failed merge comparison's last position separately passed the original
bounds: maximum `0.00016832351684570312`, RMS `0.000036847836822262866`, same
top token. Passing that position does not waive the all-position failure.

These measurements support a correct saved matrix update and a working
direct-adapter serving path. FP32 evaluation order/rounding is a plausible
explanation for merged-versus-attached drift, not a complete causal diagnosis.
There is no model promotion or general generation-quality claim here.

Evidence:

- [Every-weight FP64 audit](../experimental/native_training/validation/real-lora-weight-audit-20261024.json)
- [Direct native adapter serving](../experimental/native_training/validation/direct-lora-serving-20261024.json)

Both were run on the actual MiniCPM candidate. The weight audit is CPU-only;
direct serving used a shared A100 and about 50.83 allocated GPU seconds.
Timing includes allocation work and contention and is not a clean throughput
benchmark. Models and raw protected traces remain private.

The reusable diagnostic is `python -m tools.test_real_lora_merge_weights`;
its CLI requires the base, adapter and merged paths plus their SHA-256 hashes.

## Reload, zero-adapter and storage controls

The saved adapter and merged model independently reproduced the original
27-example post-update training loss. The recorded loss was
`2.9028494358062744`; reloading the adapter gave `2.9028497448673956`, and the
merged F32 file gave `2.902850016399666`. Errors from the saved result were
`3.09e-7` and `5.81e-7`, within the unchanged `2e-4` reload-loss bound.
Every original artifact retained its SHA256 identity. This was read-only
forward evaluation, with no optimizer update, using 32.31 allocated GPU seconds.
An average loss can hide larger differences in individual vocabulary scores.

Three controlled native forwards used the same owned 379-token practice prefix:

| Pair | Maximum logit difference | RMS difference | Positions choosing a different top token |
| --- | ---: | ---: | ---: |
| Original F16 vs zero LoRA on original F16 | 0 | 0 | 0 / 379 |
| Original F16 vs its lossless F32 copy | 0 | 0 | 0 / 379 |
| Zero LoRA on original F16 vs lossless F32 | 0 | 0 | 0 / 379 |

The zero adapter was a separate diagnostic copy with all 590 F32 factor tensors
zeroed and verified. Its header/shapes were preserved; the trained adapter was
unchanged. The three forwards used the same native binary, GPU, inputs and
numerical settings. They consumed 45.34 allocated GPU seconds. The diagnostic
copy is not a trained candidate or a model selected for deployment.

This eliminates storage precision alone and the zero-adapter path as explanations
for this prefix. The observed discrepancy appears with the nonzero learned
update. Different FP32 operation order remains a plausible explanation, alongside
possible implementation differences that these controls do not exclude.

CPU analysis of the original saved nonzero adapter/merged logits also found
**zero different top-token choices across all 379 positions**, even though their
score vectors exceeded the original numerical bounds. These were teacher-forced
positions on one practice sequence. They do not establish equal free-running
generations, broad capability equivalence or equivalence on unseen inputs.

The all-position logit gate remains failed. Reload persistence, saved-weight
correctness, score-vector equivalence and behavioral quality answer different
questions. This diagnostic is not a quality rejection of all LoRA training and
does not replace the outstanding matched-practice LoRA learning curve.

Evidence:

- [Independent reload and batch-loss check](../experimental/native_training/validation/lora-reload-loss-w.json)
- [Zero-adapter and storage controls](../experimental/native_training/validation/lora-zero-dtype-x.json)
- [Original nonzero adapter/merge top-token comparison](../experimental/native_training/validation/lora-nonzero-argmax-x.json)

To repeat the controls, pin the original native executable and artifact hashes;
make a separate zero-factor GGUF using the existing GGUF tensor offsets;
verify all factor bytes are zero; and run native evaluate-only forwards on the
same practice input. Record all-position logits for each of the three conditions,
then compare maximum/RMS error and per-position argmax independently. Compare
the unmodified trained adapter and merge separately. Use new output paths,
retain original failures, and recheck source artifact hashes after execution.

## Direct attached-hat COW qualification

The separately bound native hat now matches eight independent fresh-oracle
frontiers exactly and preserves restored state and consumed COW request outputs.
See [the qualification and limits](DENSE-LORA-COW.md). This measures the attached
hat and does not change the earlier failed merged-weight equivalence result.
