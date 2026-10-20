# Real Q4 checkpoint-write failure harness

FOCUSED-PASS 2026-10-16: actual Q4 GPU-v5 completed the kernel EFBIG, one
explicit client error, same-operation publication-only retry, and eight-output
follow-up. Primary readback verifies payload hash, exact canonical token
ancestry, unchanged parent/source bytes, no native trace work during retry,
137 total forward tokens and zero prefill tokens; both processes exit 0.
This is not actual ENOSPC, a fresh logits comparison or cumulative G2.
Source now reuses the existing client `_fork` helper; GPU-v3 exposed the
incorrect assumption that fork, like save, returned a nested branch receipt.
GPU-v4 reached the real writer fault but rejected duplicate HTTP terminal
notifications; source and failure logs remain retained. No guard was weakened.

Current runner SHA `d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695`.
The unchanged 13 CPU checks pass again at error-guard/fork-cpu-v2.
Primary audit: publication-fault root/success-primary-audit.json, SHA
`1f2ad809585d5a18c17fff936b4bf0fcfac05012d2d699c49c624164199f8a4a`.
The detailed earlier checkpoints below describe their own source versions.

The next real probe also requires one explicit client error preserving the
original errno/message with cancelled=false. Silent EOF cannot qualify. A new
behavioral guard first fails RED, then passes all 13 CPU methods (no skips)
under `q4-publication-error-guard-20261016-v1/{red-v1,green-v1}` on llm-60.
Current runner SHA `35988ec5bab6cd849902d6314c2cf2a914cccf75fe6fe7022a3bfddf58f3b466`;
test SHA `9080026fae547bca561b51d618692b3e70410a05fdab2c6fb13fa791ab15c9d2`.
This strengthens the actual kernel-fault campaign; it does not qualify a GPU
write/retry by itself. The earlier generic stream guard retains its broader
EOF handling; the real writer-fault path now requires the stronger check too.

Update 2026-10-16: the serving-history correction allows the real history read,
but GPU-v2 then stops before warmup/injection because the harness inspects only
the main thread's Linux child list. The server creates its native child from
a worker thread. A real thread/subprocess regression reproduces the missing
child (RED-v3, one failure among 12 methods); aggregating all live task child
lists passes 12/12, no skips, command/wrapper 0/0. Transient exited threads are
ignored; other read/parse failures remain errors. PID, parent, executable and
start-time checks at the injection boundary remain unchanged.

Current runner SHA `d405934c4fa24d119389ce70824d117316abe26867f69eebee9c8741a559c4ee`;
test SHA `a9dbb0f0ed16961ad4edd25ae91c8f396f2ce8589b9871d5b205f40539e235df`.
Evidence: `llm-60:/home/dflanag3/spatial-artifacts/q4-publication-child-detection-20261015-v1`.
Retain red-v1 launcher refusal (precreated output directory) and red-v2's invalid
fixture (spawning thread exited before inspection); neither is the intended RED.
No GPU fault/retry pass has yet been obtained. The original harness record follows.

UNQUALIFIED GPU fault/retry path. Eleven evidence-validator CPU methods pass
on Linux, command/wrapper exits 0/0. These do not emulate or qualify GPU work.

Branch `work/luna-q4-publication-fault`, base `ddf57e7e`. Root owns the frozen
runner after Luna's review. Runner SHA-256
`061fb0a1ab18842c6e3b0caf77322159dcc0d283924f2dfda64ccefc7d763d56`;
test SHA-256 `fc0d0660e53d8b19d0aaf073c1e25f8ac1980c55782e127457e593026b04cbff`.
No production API, native engine or numerical profile changes.

The runner starts an owned loopback service, forks the selected branch, warms
its source, applies a 64 MiB soft RLIMIT_FSIZE only to the verified Python
service PID/start identity, and checks a real EFBIG from the checkpoint payload
writer. It requires unchanged old heads/history/payloads, no durable-success
event, a retryable operation, restored process limits, same-operation durable
publication without any new native trace work, then a successful new request.
The native child must retain its original file-size limit. EFBIG is not ENOSPC
and does not establish real disk-exhaustion or power-loss qualification.

Executed evidence root:
`llm-60:/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1`.
RED-v1/v2 fail before behavioral test loading; green-cpu-v1 has 5 passes and
2 fixture errors (omitted receipt vs None; default fault limit mismatch).
Corrected positive fixtures and new controls yield 11 passes, no skips, in
green-cpu-v2. Runtime/profile closure preflight passes. GPU-q4-v1 starts the
real Q4 engine but fails on the first canonical history read with HTTP 409:
two resume publications of the same canonical snapshot are considered
ambiguous by the serving presentation resolver. No fork/warmup/injection/retry
occurs. Primary audit confirms native work/file loads 0 and both process exits 0.
Preserve that failure; do not switch to another source to conceal it.

Next: integrate the independently tested branch/revision presentation fix,
stage a new immutable runtime with the same 17 profile-bound API files and
native/profile fingerprints, then rerun this exact real-GPU harness. Q4 only;
Q8 is deferred. Chat remains off. All commits/backups stay private box-git.
