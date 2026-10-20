# Sparse native cancellation checkpoint

Qualification: FOCUSED-PASS, not cumulative G1-G5 certification.

Controller worktree: `C:\Users\dflanag3\Documents\fleet\context-cow-sparse-cancel-luna`.
Branch: `work/luna-sparse-cancel`; base: `d22633eb7644c460c3fcdfd7c0c46bc3af045f38`.
Root took ownership after Luna stopped. Production API and native engine bytes
were not changed by this ticket. Explicit runtime roots and pinned source
closures prevent accidental imports from historical API checkouts.

The new runner copies an existing isolated Store, verifies the executed Q4/Q8
8K/512/128 references, cancels after three native-consumed outputs and requires
no snapshot, durable receipt, reference move or canonical history mutation.
It then completes A and B independently, requiring exact IDs, full final logits,
tested greedy controllers and suffix-plus-output native work. Source bytes and
sibling ancestry are checked. All operations use the production coordinator.

Executed on llm-60 GPU0 with the sparse native executable SHA
`4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451`:

- SPARSE-CANCEL-001/002/003 PASS separately on Q4 and Q8; no skips/xfails.
- Two 128-output recoveries per model, four final full-vector pairs, max error 0.
- Cancellation evaluates 512 suffix + 3 outputs; each recovery evaluates 512 +
  128, with no prefix recomputation or serialized prefix reload in those requests.
- Both Python launchers and native workers exit 0. Q4 campaign 29.84 seconds;
  Q8 103.46 seconds. These include startup/audit and are not serving benchmarks.
- Final focused CPU suite: 10 PASS, zero failures/errors/skips. Both validator
  bypass mutations detected. G0: 51 PASS on unchanged logical/reference source.

The final runner SHA is
`c95807de21e0b2acd7db2b8650c9b7d14a2b68142135bd6a2cf86ffae3e5ec68`.
The final test SHA is
`ae107c573993e30de72f535098e3d2821585ca0f5a34a604c5f6abcf296c8500`.
GPU runs used source-v3; source-v4 changes only CPU fixtures, with identical
runner bytes. Positive controls exposed an invalid negative-test baseline
schedule; the corrected fixture retains all assertions and strengthens coverage.

All earlier RED, preflight and GPU harness failures are retained. In particular,
Q4-v1 rejected the explicitly declared metadata helper after native startup and
before cancellation. Its numeric native exit was not retained and is UNKNOWN.
The corrected provenance guard checks that helper's exact declared path and
native-build SHA separately from staged API roots; it does not permit arbitrary
external imports. No tolerances or registered gate assertions were changed.

Remote raw evidence:
`llm-60:/home/dflanag3/spatial-artifacts/sparse-cancel-20261015-v1`.
Machine-readable qualification and primary audits are adjacent to this handoff.
The compact private backup is indexed by
`SPARSE-CANCEL-PRIVATE-BACKUP-2026-10-15.json`; it excludes full checkpoint states.

Limitations: the oracle is reused, previously executed independent uncached
matching-operator inference, verified against this case/profile and raw vectors.
Cancelled outputs are checked as IDs/receipts, not full frontier vectors. Only
final recovery vectors are compared. No crash, disk-full, sustained pressure,
general sampler or registered cumulative-gate qualification is added here.

Next: exercise the existing actual CLI/service fork, ask, continue, retry and
process-restart workflow on this sparse backend, Q4 then Q8. Retain canonical
stdout/controller/work assertions; separately report numerical oracle scope.
Chat stays off. All pushes are to private box-git only.
