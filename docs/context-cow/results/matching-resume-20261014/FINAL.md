# Private matching-batch restart results

Batched, execution-profile-bound durable resume PASS on llm-60. This is not cumulative G2 certification. The original G1–G5 requirements and cross-partition mismatch evidence remain unchanged.

Tiny (128/4/8), medium (1024/64/16), and target (8192/512/128) campaigns passed with 20, 36, and 12 full-logit comparisons respectively, all with zero error and frozen zero tolerances. Target comparisons sample prefix, completed suffix, and generated counts 1, 16, 64, 128 for each branch. Both target branches generated 128 tokens. Restart after four consumed output tokens matched the remaining continuation. Three process-separated target cold restores passed with zero saved-prefix forward work.

The four executed evidence methods MR-001 through MR-004 passed for the target: process separation, zero-work disk restore, independent matching reference, and branch/generated-boundary integrity. No failures or skips. Controller scope is deterministic greedy only. Nonzero RNG, grammar, pending decoder, crash injection, power-loss durability and broad numerical qualification remain outstanding.

## Lean speed measurement

Two branches per mode, 8192-token prefix, 512-token suffix, 16 generated tokens. All modes matched validated output tokens, final frontier hashes and controller records exactly. These timings exclude engine startup; request totals include prefix preparation, suffix, controller start and generation. Resident setup loaded the checkpoint before timing. Mandatory API integrity and immutable-history work remained enabled.

| Mode | Prefix preparation A/B (s) | Request total A/B (s) | Generation A/B (tokens/s) |
|---|---|---|---|
| Resident reuse | 0.067 / 0.075 | 16.664 / 15.950 | 1.074 / 1.104 |
| Disk restore | 0.794 / 0.732 | 17.445 / 16.494 | 1.070 / 1.114 |
| Fresh prefill | 26.204 / 26.026 | 43.544 / 41.947 | 1.015 / 1.104 |

Reuse avoids prefix computation and lowers request time, but API generation remains slow after optional audit work is removed. This does not establish the cause. Next work is profiling actual mandatory per-token checkpoint, integrity, history and native execution costs before optimizing. This is a 16-output speed fixture, not a 128-output performance claim.

The server was restored unchanged and returned HTTP 200. Machine-readable summaries contain backend, model, profile, process, timing and work identities. Q8 model, FP16 KV, RTX PRO 6000 Blackwell, native prefill operator only. No normal-decode qualification.

Code and summaries are committed locally and pushed only to private box-git. Raw target archive is backed up outside Git history; native state payloads remain retained on llm-60 and are audited by hash and byte count. No GitHub publication.
