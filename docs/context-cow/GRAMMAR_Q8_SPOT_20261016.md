# Q8 greedy grammar spot check

The first Q8 GPU campaign completed without a crash. Two fresh native references
and the COW candidate ran to completion, then all Python/native children exited
0 and were absent. This spot check uses the existing native-selector code;
the pending idle-expiry and Min-P changes were excluded.

| Measured median | Question A | Question B |
|---|---:|---:|
| Prepared prefix | 7,231 tokens | same prefix |
| Added question | 512 tokens | 488 tokens |
| Actual output | 32 tokens | 32 tokens |
| Generation | 82.11 tokens/s | 82.47 tokens/s |
| Whole branch workflow | 0.985 s | 0.979 s |
| Output/workflow seconds | 32.49 tokens/s | 32.70 tokens/s |
| First provisional consumed token | 0.460 s | 0.456 s |
| Median consumed-token interval | 11.85 ms | 11.58 ms |

One audit and one warmup pair preceded three measured pairs. Native startup
(81.239 s) and initial prefix creation (6.651 s) are separate from reused request
times. Timestamp callbacks and integrity checks remained enabled. Thirty-two
outputs amortize request overhead differently from the 128-output Q4 campaign;
these are not matched whole-request Q4/Q8 speed comparisons.

All ten COW requests matched their Q8 references exactly in generated tokens,
full controller hashes and consumed canonical histories. All 18 checked full
logit vectors matched byte for byte, maximum absolute error 0. The audit sampled
P, each suffix and consumed outputs 1/16/32; other runs checked the final vector.
Every request performed only suffix-plus-output forward work, one capture and
no saved-prefix recomputation or checkpoint load. The complete parent image's
before/after hashes matched. The outputs hit their cap with incomplete JSON;
this establishes no completed tool action or semantic review quality.

Qwen3.8-Flash-Next Q8_0, FP16 KV, GPU0 RTX PRO 6000 Blackwell, Prefill8192,
normal-verifier-decode and finite-mask legal greedy selection were pinned.
Execution profile: `9ad58f1fb7440fe06ef64de0cd4d8c99c8928f44f24ecb16dca8ff5a4fb81d49`.
Model contract: `fc6d0d643e4b9ae020964f8c4e5006c0da7199b19cce5285c573023dbe82ca46`.
Aggregate weights: `846b74d30a22552ad8cc2fabe5b17103049fe68ffdacf56daa457db8ecaa356f`.
Native executable: `2b58e8e872e46705805bf119bf7c7b382c85b63114b03f7f00f61f2452b225dc`.
The preflight reused 11 verified model-file hash receipts and hashed zero model
bytes in that process. Complete fingerprints, counters and per-run timings are
in [the qualification record](../../experimental/branching_context/evidence/GRAMMAR-Q8-SPOT-QUALIFICATION-2026-10-16.json).

Raw artifacts:
`llm-60:/home/dflanag3/spatial-artifacts/grammar-q8-spot-20261016-v1/`.
Private backup:
`box-git:/srv/data/git/private_context_cow.evidence/20261016-grammar-q8-spot-v1/evidence.tar`,
33,320,960 bytes, SHA-256
`8b96ad566f453d3db6fade3d044e10cca20fbead71acc077d9c4ddab4f174658`.
Independent streamed readback verified all 69 members/33,250,759 data bytes.
The source/runtime/numerical executable closure is retained in the preceding
verified private grammar archives. Nothing was published to GitHub.

Result: FOCUSED-PASS for this greedy matching-operator spot check. No observed
major breakage; no matched old/new Q8 native-only performance control was run.
Earlier 64K speeds used another profile and context, so they are historical
context rather than a regression threshold. This does not qualify long-context
Q8, Min-P, process restart, idle expiry, HTTP serving or cumulative G1-G5.
