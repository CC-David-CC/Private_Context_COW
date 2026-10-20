# Base Q8 autonomous coding comparison — 2026-10-19

Owner requested testing base Q8 before further adapter training. Training is
paused for that comparison; no model optimizer step has been performed. No MTP.

Actual client: BranchWrite CLI on llm-49 (r4090), commit
9fe63fce3280a4f8708ab6d33e2f81ab2801c2f7. It uses authenticated HTTP through a
scoped SSH forward to Context COW on llm-60. COW owns canonical model history,
controller state and native checkpoints; BranchWrite owns workflow/tool grants,
execution and the ledger. Strata executes the model. External audits check results
after the workflow has stopped; they do not supply model actions.

The same original CSV-repair task/source, grammar, examples, greedy settings,
4096-thinking admission and client commit were used as the Q4 C01-v2 reference.
KV is FP16 for both. This does not measure whether fewer examples or less grammar
would suffice, nor establish general Q8 superiority.

| Measurement | Q4 | Q8 |
|---|---:|---:|
| Model calls | 7 | 7 |
| Output tokens | 3558 | 3045 |
| Complete positive CLI seconds | 170.689286882 | 214.574366743 |
| Public test methods | 9 passed | 9 passed |
| Independent heldout methods | 3 passed | 3 passed |
| Generated heldout CSV cases | 64 | 64 |
| Subsequent extreme-precision methods | 2 failed | 2 failed |

Both generated Decimal addition without adjusting its default28digit context.
The two extra probes reproduce rounding in executed Python; they are not model
logit comparisons or evidence of KV numerical corruption. Both candidates made
overbroad claims about exact addition. Diagnostic inputs were not shown to the
model during the matched C01 task. This is one trajectory per configuration.

Q8 initially failed startup because the copied vision configuration pinned the
Q4 model path. That rejection is retained. A separate Q8 configuration binds the
actual Q8 first-shard hash while preserving encoder/projector checks. No integrity
check was removed. The corrected server started in90.095632291seconds.

Q8 model contract:
3fcd9cfc2e977dbf56522e710cb3903e03ab00ca2ed5d0daa3d9f840edf5a833
Execution profile:
66fe583a57231bede0c05c94e93e900fc1d7779450a2ee626f3d4ba23fa3fd89

Client access-denial probes produced zero forward tokens. Original workflow
ledger145files remained byte-identical across offline audit. Owned server/native
both exited0. Temporary credentials were removed, scoped key revoked, original
authorized_keys restored byte-for-byte. Chat server remains OFF.

Linux artifacts:
- Client llm-49:/home/dflanag3/spatial-artifacts/branchwrite-ubuntu-baseline-20261018-v1/agent-repair-20261019-q8-v1
- Server llm-60:/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1/agent-repair-C01-q8-v2
- LoRA candidate class llm-60:/home/dflanag3/src/branchwrite-lora-examples/candidates/decimal-context-rounding-20261019/q8-base-comparison

Verified private backups under
box-git:/srv/data/git/private_context_cow.evidence/20261018-vc-rerun-v1/:
client-agent-repair-q8-20261019-v1,
service-agent-repair-q8-startup-rejection-20261019-v1,
service-agent-repair-q8-20261019-v2.
All archive members were hash-verified. No GitHub publication.

Next: test known terminal/report-format failure classes on base Q8; independently
repair and broaden precision handling through the autonomous workflow. Keep
constraints and grants. Do not infer adapter need or remove examples from this
single comparison. Existing G1–G5 qualification remains separate and incomplete.
