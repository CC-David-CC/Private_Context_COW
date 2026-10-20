# Primary review: root presentation query

CPU FOCUSED-PASS; not integrated and not GPU/performance qualified.
Reviewed all three production changes and the new 11-method regression module.
The Store query retains namespace authorization, one history transaction and
full validation for selected root descriptors. The full catalog is unchanged.
Rewind descriptors targeting the root are hash-checked but their unrelated
future source is not loaded by this capability query.

Executed on llm-60: focused 11/11, serving/Store 84/84, G0 51/51, no failures
or skips. The preceding implementation fails the added rewind regression.
Raw command/wrapper receipts and final source hashes were independently checked
by the backup builder. No Python or model tests ran on Windows.

Verified private archive:
`box-git:/srv/data/git/private_context_cow.evidence/20261016-root-presentation-query-controller-v1/evidence.tar`
9,932,800 bytes, SHA-256
`2cae90ff96d857d85df4319043221345274ee54b161ed6f3b6428bf7464ca232`.
Independent streaming readback verified 831 members / 9,091,287 data bytes;
no files were extracted. Model/native state is outside this CPU-only ticket.

Git checkpoint uses the established manual exception: the installed coordination
helper pins a different Spatial repository. Primary verifies this exact worktree,
branch, base40ac335d, David identity, explicit staged paths, whitespace check,
private destination and remote SHA. No GitHub publication is authorized.

Next: combine with retention0313ad6f and publication ownership5272acd8 in a new
integration worktree, rerun affected CPU tests, then qualify a fresh GPU profile.
Original cumulative G1-G5 status is unchanged. The separate known baseline
missing-checkpoint expectation failure remains unresolved.
