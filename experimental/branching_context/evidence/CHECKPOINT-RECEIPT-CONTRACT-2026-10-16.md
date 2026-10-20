# Checkpoint receipt contract reconciliation (2026-10-16)

This change updates two prepared-presentation tests that previously hid
`checkpoint_receipt` metadata by dropping all `Store.move` keyword arguments.
The replacement spies forward every keyword to the real Store and assert the
exact checkpoint descriptor. Production publication and checkpoint formats
were not changed.

The new `tests/implementation/test_checkpoint_receipt_contract.py` checks that
the atomic branch receipt contains the exact snapshot ID, payload SHA-256,
model contract ID, and execution profile ID; the descriptor digest and move
event agree; and a reopened Store returns the same receipt. It also checks
that missing or altered descriptor/event metadata fails Store validation,
legacy committed operations preserve their original request digest, and a
logical `Store.move` remains valid without checkpoint metadata. All payloads
are synthetic CPU test bytes; this is not GPU qualification.

Evidence is retained outside this source tree at
`/home/dflanag3/spatial-artifacts/checkpoint-receipt-contract-20261016-v1/`:

- `red-mutant-v1`: the intentional `COW_STRIP_CHECKPOINT_RECEIPT=1` test-only
  Store.move mutant returned a durable branch move without metadata. The
  contract check failed as intended with `NotFound: Operation has no
  checkpoint receipt: save`; command/wrapper exits were 1/1.
- `legacy-baseline-v1`: the unmodified base `test_prepared_presentation`
  produced the two known assertion failures because both spies discarded
  kwargs; the rejected-presentation, corrupt-checkpoint, and root-receipt
  controls passed. Command/wrapper exits were 1/1.
- `green-focused-v2`: the new receipt contract, updated presentation tests,
  prepared-checkpoint publication tests, and durable-restore tests passed
  28/28, with 0 errors, skips, or failures; command/wrapper exits were 0/0.
- `g0-v1`: G0 passed 51/51, with 0 errors, skips, or failures; command/wrapper
  exits were 0/0.

The exact remote runner was
`/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh`
(SHA-256 `7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537`),
using `/home/dflanag3/src/Strata/.venv/bin/python -B`. No Windows Python,
model process, GPU job, or production source mutation was used.
