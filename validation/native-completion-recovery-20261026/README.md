# Native completion recovery: tiny CPU qualification

**Result: PASS for the bounded recovery mechanics described here.** This is
not a model-quality result.

## What was exercised

- Fixture: generated 2,744-byte, two-layer ordinary-Llama GGUF; embedding size
  4, vocabulary size 7, context 512.
- Update: full-weight AdamW, one step, one example, learning rate `0.0001`.
- Device and limit: CPU only, 4 GiB configured CPU budget; no CUDA, GPU, or
  PyTorch.
- Native training exit: `0`.
- Independent completion reload exit: `0`; completion gate passed.
- The service supervisor was deliberately stopped with exit code `73` after the
  native update and reload had succeeded but before the service committed job
  completion. This was controlled fault injection.
- After restart, the original job was inspected and explicitly adopted through
  BW, then its original learning round was reconciled. It reached
  `candidate_pending` without another update. Recovery dispatch guards recorded
  zero attempts. The candidate was not promoted.

## Reproduction pins

| Item | Pin |
| --- | --- |
| PCC recovery source | `4a536ab6e26a7d97f200452479a201b414ce55fb` |
| BW owner-driver source | `29078a1a914b828f4f3f8e82bf43ece75670a466` |
| Native trainer source SHA-256 | `9503130fc2627d55f07ea4e23eeb597fd4fdf905fc9e8df2e605f9d6738e3270` |
| Built native trainer SHA-256 | `773adb97dc19ab16d8be4ff4ade9f37b046a71da1f02d20face391056ba14593` |
| Generated base fixture SHA-256 | `9676647d4c4ed448d5f0119499d5498d8baaefb3d858a4560fd071309a99670b` |
| Saved candidate SHA-256 | `16331ccd17a8aec4f01a534d263161f666ff48d3b7407cbc428995e9184f97f2` |
| Completion reload gate SHA-256 | `3909df8bb4411960f61c510156d45e3be769f289eb6c8122e957838e1a6b249d` |

The saved candidate differs from the generated base fixture. The public receipt
omits machine paths, process IDs, service credentials, raw logs and private
request/evidence records.

## What this does not establish

The fixture is only large enough to exercise the native update, reload, job
interruption and operator recovery boundaries. It says nothing about whether
MiniCPM or Wen learned a skill, retained other skills, improved on held-out
questions, or works on a GPU. `candidate_pending` means the learning system is
waiting for its later candidate checks; it is not accepted or serving.
