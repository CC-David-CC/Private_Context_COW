# Per-generation settings: bounded executed results

Source: `462ab8158522d76776709f3fef6585578b72bb68`. Q4 and Q8 passed on
llm-60 GPU0, RTX PRO 6000 Blackwell, FP16 KV, capacity 98304. Native generated
tokens used the normal verifier operator; prefill chunks were 8192. Native
v12 ELF SHA-256: `8b96df82e47c7d1a367ec975ad3732b637856b34f0483ac521bd8149ca4ca681`.

The supported settings are temperature, Min-P, Top-K and Top-P, fixed for each
generation call. A partial continuation may change settings without resetting
RNG, output decoder or grammar progress. Unspecified CLI continuation fields
inherit their saved values. DRY is not implemented in this baseline.

Both models passed the 1024-prefix/64-suffix/16-output HTTP/CLI campaign:
different A/B branches, process exit, strict restore, continuation after four
outputs, settings changes, exact tokens/controllers and 38 exact full-logit
comparisons each. Restoring the prefix evaluated zero tokens. Q8 continuation
changed temperature alone and retained its saved filters. Q4 changed all four.
Both passed seven-policy resident sweeps and a separate settings-change restart.
The Q4 final sweep reused seven successful uncached reference stages by hash
from an earlier failed harness campaign; its COW/restart stages were rerun.

CPU: 477 tests, 10 focused observer tests and G0 51 tests passed; no final
failures, skips or expected failures. Four reference mutation-smoke outcomes
were detected; these are not real-GPU mutation qualification.

| Model | Temperature | Actual outputs | Generation TPS | Complete request seconds |
|---|---:|---:|---:|---:|
| Q4 | 0 | 512 | 105.97 | 5.129 |
| Q4 | 0.4 | 512 | 105.63 | 5.145 |
| Q4 | 1.0 | 512 | 105.24 | 5.164 |
| Q8 | 0 | 512 | 58.31 | 9.094 |
| Q8 | 0.4 | 512 | 70.29 | 7.610 |
| Q8 | 1.0 | 512 | 60.91 | 8.728 |

Speed rows are one trial each after a single disk restore, with a resident
question, one capture per call, no repeated disk load and no prefix replay.
They exclude engine startup and durable disk saving of each answer. Only the
leading eight tokens and frontier were compared with independent references;
the rest are performance smoke. This is short-context evidence, not a matched
quantization comparison or long-context throughput guarantee.

Earlier failed harness attempts remain recorded: CLI journal misuse, a
shadowed authentication key, a missing reference callback keyword, comparison
of an unreachable earlier frontier and a cleanup-variable shadow. None was
turned into a passing campaign or absorbed by a tolerance change. The native
Top-P denominator-order defect was fixed before the final pinned build.

Machine-readable qualification and complete fingerprints live in
`experimental/branching_context/evidence/GENERATION-SETTINGS-QUALIFICATION-2026-10-16.json`.
Raw evidence is private on llm-60 at
`/home/dflanag3/spatial-artifacts/sampler-serving-20261016-v1/`, with a private
box-git archive and separately verified readback receipt. Derived checkpoints
are excluded from that archive. An old inactive tmpfs cache was reclaimed by
the production pool collector: 30,086,880,672 payload bytes removed, canonical
history hashes unchanged, metadata and diagnostic evidence retained.

Cumulative G1-G5, cross-arrangement equivalence, long-context settings coverage,
and HIP/SYCL builds remain unqualified. The normal chat server remains off.
