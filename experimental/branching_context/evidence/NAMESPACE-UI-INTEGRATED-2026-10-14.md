# Canonical namespace HTTP and browser integration

FOCUSED PASS, not milestone completion or cumulative G1-G5 certification.

## Executed results

On llm-60, the integrated normal repository layout passed 202 control methods,
100 serving methods and G0 51 methods, with no failures, skips or expected
failures. The three adjacent NAMESPACE-UI JSON profiles contain executed IDs
and source fingerprints. CPU/filesystem and synthetic HTTP tests do not
qualify numerical GPU behavior.

The first combined control attempt had three fixture-builder path errors in
a flattened staging directory. Restoring the repository layout corrected
the harness; assertions and production code were unchanged. Failed output:
`/home/dflanag3/branching-g3/request-stop-20261014/namespace-ui-integrated-control.log`.
Successful logs use `namespace-ui-integrated-*-v2` in that directory.

Browser integration commit 368f8c83 routes native-mode chat through the
canonical COW HTTP endpoints rather than SQLite branch heads. Real HTTP
browser checks used synthetic inference, separately from model qualification.
Their scope and hashes are in `serve/COW-NATIVE-UI-2026-10-14.json`.
The math toggle and settings persistence were tested. The test-file SHA in
that profile was corrected from a malformed transcription against actual
remote sha256sum; JSON encoding was normalized. Test code was unchanged.

## Real Q4 workload and independent retained-data audit

The frozen Q4 namespace candidate predates the latest browser and publication
recovery changes. It prepared 8,835 input tokens (8,800 source-body tokens),
then independently appended 456/512 suffix tokens and generated 128 outputs
on each branch. Native capacity was 98,304, GPU=[0], FP16 KV, 8,192-token
prefill chunks, shared RAM budget 96 GiB and disk budget 128 GiB.

Received throughput was 64.898/65.156 TPS; TTFT 1.321/0.810 seconds, with
durable completion 4.965/4.556 seconds. These are single observations, not
repeated benchmark medians. Startup is separate. No 64K speed claim follows
from this fixture.

The offline auditor reread canonical histories/controllers, checksummed saved
payloads, execution schedules, source identities and independent cache-disabled
native reference output. Four full-vector comparisons (root/final for A/B)
were byte-identical, maximum absolute error zero. Both complete greedy
controllers and token sequences matched. Actual forward-token deltas were
584/640: suffix plus output only, without prefix replay. It did not compare
every generated frontier. See NAMESPACE-Q4-8K-AUDIT and MEASUREMENTS JSON.

The auditor's four comparator controls reject truncated, changed and nonfinite
vectors and accept exact finite vectors. Red/green logs are retained under
`/home/dflanag3/branching-g3/request-stop-20261014/namespace-audit-*`.

Raw GPU case:
`/home/dflanag3/spatial-artifacts/gpu-http-q4-namespace-luna-20261014/gpu-8k-512-128-v1/result`.
Raw new Q4 data is now also backed up directly to private box-git in a
2,685,798,400-byte tar, matching SHA-256 on both hosts and passing archive
listing. See NAMESPACE-Q4-BACKUP-2026-10-14.json for exact scope and exclusions.

## Current next step

Freeze this integrated source and validate Q8 with independent references,
then exercise the actual browser against real GPU inference. Keep ordinary
chat off and all new code/evidence private. Native graph, remaining chat
controls, broader controllers, repeated long-context performance and original
gate qualification remain outstanding. The active goal is not complete.
