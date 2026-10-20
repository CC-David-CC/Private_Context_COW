# Execution-profile-bound resume: accepted experimental contract

The user accepts batch partition dependence as a limitation of this prototype.
For identical token history, native execution profile and ordered batch/request
boundaries, restoring a fully consumed checkpoint must reproduce uninterrupted
continuation exactly. Changing boundaries selects a different execution schedule;
cross-schedule numerical or semantic equivalence is not promised.

The token histories 128+4+1 and 128+5 can be identical while their execution
histories differ. Their observed 1.4495954513549805 logit difference remains
recorded evidence. It is outside this restricted contract's equivalence claim.
It does not establish that the resulting answers have the same meaning.

This contract is separate from the original mathematical specification and
registered G1-G5 acceptance tests. Those equations, assertions and numerical
tolerances remain unchanged. A restricted milestone PASS cannot qualify them.

## Required behavior and validation

1. Bind the checkpoint to the actual source/build/model, numerical formats,
   positional conventions, consumed count, controller and ordered execution
   schedule. Record successful native batch boundaries, not merely Python calls.
2. Keep immutable history/controller as canonical state. Schedule metadata is an
   execution receipt, not a second history store. Retain exact token identities.
3. Reject incompatible restore profiles and unsupported operations explicitly.
   Do not silently replay saved inputs or reshape accepted work.
4. Compare a process-separated restored continuation against an independently
   initialized, uninterrupted native run with the identical successful batches.
   That reference must not import the tested checkpoint or share mutable state.
5. Require exact saved/restored frontiers, continued logits/tokens and controller
   records at the measured boundaries, and zero saved-prefix forward work on
   restore. Both forced A/B continuations must preserve the stored prefix.
6. A checkpoint may precede different supported future batches on A and B. Each
   future is compared with its own uninterrupted execution schedule; branches
   must not inherit each other's suffixes. Unsupported schedule changes are
   rejected, rather than labelled exact continuations.
7. Test tiny first, then 1K and 8K/512/up-to-128. Record all failures and early
   stopping. Nonzero RNG/grammar/decoder and crash tests remain separate required
   evidence before claiming those capabilities.

## Current evidence boundary

The matching-schedule batches-only campaign passed at 128, 1024 and 8192 prefix
tokens. The 8K fixture consumed two different 512-token suffixes and generated
128 tokens on each branch. Twelve sampled frontiers matched with zero error;
three cold restores evaluated zero saved-prefix tokens. A consumed-generated
checkpoint also resumed exactly. Scope is deterministic greedy continuation.
See results/matching-resume-20261014/FINAL.md for executed evidence and limits.

The existing cross-partition failing test remains a limitation/regression
witness. Normal decode stays separate research. Original G1–G5 remain
unqualified. Lean API generation measured approximately one token/second even
without optional audit work; performance investigation is outstanding.
All work remains private on box-git.
