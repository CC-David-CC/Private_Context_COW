# Approved eviction behavior and CLI notice

User explicitly approved the current policy: a missing optional checkpoint does
not prevent metadata fork/retry. A later continuation may recompute retained
history, and must tell the user. Corrupted checkpoints still fail integrity
validation. This correction changes the obsolete implementation-test expectation;
it does not change registered gate tests, mathematics or numerical tolerances.

The CLI prints once, on stderr:

> Notice: saved prefix is unavailable; recomputing it from conversation history. This operation is still running.

The fork/retry action itself performs no model work. The notice appears when a
continuation actually starts reconstruction. Ordinary resident/disk restoration
stays quiet, and model text remains on stdout. Replaying a pending CLI operation
uses the same handling. A reconstruction-failed event stops processing before
any later answer or durable-success receipt; untrusted error text is not echoed.

Primary reviewed all production/test changes. All original corrupted-payload
rejection assertions remain. New missing-state assertions check namespace,
idempotency, exact branch target and zero model work for metadata actions.
Socket regressions cover notices, malformed progress, ordinary restoration,
operation retry, and failure followed by an otherwise successful completion.

llm-60 CPU/socket results:38/38 focused tests and G051/51, no skips. Final
unchanged failure-event regression is RED against the preceding CLI and GREEN
against the final CLI. Historical obsolete-test failure and setup failures are
retained. This ticket ran no model/GPU work. Existing combined Q4 numerical
evidence belongs to source97bab1a2, before this CLI/test-only overlay.

Private archive/readback:880640B,185 members/683088 data bytes,
SHA-256 `0f9e8896ee4dc24f7eb85496c16a296992e99c8531be5a8806e2e9bbb6c998b8`.
Destination:
`box-git:/srv/data/git/private_context_cow.evidence/20261016-eviction-recomputation-notice-v1/evidence.tar`.
No extraction, no GitHub publication. Primary uses the established manually
audited Git exception because the installed helper pins an unrelated repo.
Next: integrate into combined candidate, run the union of affected CPU controls
and G0, then update the main private handoff. Original gate status is unchanged.
