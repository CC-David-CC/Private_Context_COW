# Q8 one-example Decimal repair — 2026-10-19

The original base-Q8 task passed9public and3heldout methods but failed2extra
precision methods. Its candidate also failed all96new seeded precision subcases.
User requested adding one example and warning,then another only if needed.

A fresh autonomous CLI49/HTTP/COW/GPU60 run used the same original broken
project,task,greedy profile,4096thinking admission and BranchWrite commit
9fe63fce3280a4f8708ab6d33e2f81ab2801c2f7. No adapter or MTP;FP16 KV.
One added domain example demonstrated integer coefficient/exponent alignment
and exact Decimal tuple construction. A warning explained default28digit
context rounding and forbade mutating caller precision. Existing general
tool-use examples remained;this was one additional domain example,not one
total example. The toy example itself was executed at precision3.

The immutable reference pack was supplied through existing CLI context_pack,
not by changing model output or tool decisions. Pack:
90379878ba9b30e2681b50fa2c9ce30001c255229c2443d322c4471fe00b6674
The example used different operands from protected probes. No hidden tests or
expected answers were included in prefill. The audit driver's config forwarding
was extended to pass the already-supported optional context_pack field.

Result:16test methods PASS,0failures/errors/skips:
9public,3heldout (64generated CSVcases),2unchanged precision probes,
2new methods (96seeded precision cases and caller-context preservation).
Seed737520. No second example was attempted or needed in this trajectory.
The model generated its own repair,converting Decimal coefficients to aligned
Python integers and constructing output Decimals without rounded addition.

Seven model calls,5234output tokens,366.246422069seconds complete positive CLI,
versus7calls,3045tokens,214.574366743seconds in the base-Q8 run.
This improvement in correctness cost additional work;it is not a speedup.
Combined example+warning does not isolate their individual causal effects.
One trajectory is not proof of a universal minimum example count or unlimited
decimal-resource support. Original G1–G5 qualification remains separate.

The148-file ledger remained byte-identical across offline audit. Server/native
exited0;scoped SSH key was revoked;original authorized_keys byte-exact. Chat OFF.
Native integrity counter reported0mismatches during the run.

Client:
llm-49:/home/dflanag3/spatial-artifacts/branchwrite-ubuntu-baseline-20261018-v1/agent-repair-20261019-q8-example1-v1
Model server:
llm-60:/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1/agent-repair-C02-q8-example1-v1
Preserved LoRA-class evidence and exact example/prefill (no training performed):
llm-60:/home/dflanag3/src/branchwrite-lora-examples/candidates/decimal-context-rounding-20261019/q8-one-example-mitigation

Verified private evidence archive labels under
box-git:/srv/data/git/private_context_cow.evidence/20261018-vc-rerun-v1/:
client-agent-repair-q8-one-example-20261019-v1,
service-agent-repair-q8-one-example-20261019-v1,
cache-agent-repair-q8-one-example-20261019-v1,
decimal-one-example-mitigation-20261019-v1.

Owner requests the same failure→one relevant example→fresh autonomous trial
method across the Lcatalog. L00–L02 have bounded older Q4 evidence;no claim of
current Q8 regression coverage. Next unfinished protected target isL03 scoped
config repair,thenL04 permissions,L05 report execution,L06 cancellation.
Infrastructure failures are repaired separately,not mislabeled model failures.
