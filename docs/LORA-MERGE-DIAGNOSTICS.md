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
