# Q4 generation after incremental execution-receipt validation

The real CLI streams about92 tokens per second instead of12. On llm-60,
the same image-context question and186 generated tokens take3.72s end to end,
previously17.00s. The186 tokens include thinking, final output and end-of-turn.
This is a small-context Q4 result:1,024 requested source-code tokens plus chat
controls/image rows make1,393 model cells; each question adds25 model cells.
It does not establish Q8 or long-context performance.

## Measured medians

One warm-up and three measurements per path, same native/model bytes,
CPU/GPU configuration, input tokens and native batch/operator boundaries.
Changed Python source has its own profile identity. Original baseline has24
complete measurements before an unrelated text-root harness reporting failure.

| Path | Before, whole CLI | After, whole CLI | Speedup | After, stream TPS |
|---|---:|---:|---:|---:|
| Resident GPU | 17.002s | 3.718s | 4.57x | 92.16 |
| RAM source | 17.369s | 4.029s | 4.31x | 91.85 |
| Disk source | 17.622s | 4.272s | 4.13x | 91.92 |

Stream TPS measures consumed token arrivals. Whole CLI includes startup of the
client, request setup, suffix ingestion and durable publication. Resident whole
CLI throughput is50.03 TPS. Mean token interval fell83.59ms to10.85ms; resident
first-token wait is essentially unchanged (221.47ms to222.43ms). Last token to
durable acknowledgment still costs about1.38s. Streamed output is not a durable
checkpoint before that acknowledgment.

Native decode computation remains about1.406s for186 outputs. The measured
saving comes from host validation: repeating all earlier receipt checks after
each token took13.6035s in a three-repeat CPU replay. Checking only each exact
new extension with full entry/publication audits takes0.3055s:44.53x faster,
97.75% less validation time. Complete resident request time is4.57x faster;
that44.53x number is not the whole-engine speedup.

The private owner-local cursor keeps the already-validated immutable prefix
and detached receipts. It rejects changed ancestry, wrong inputs/positions,
batch/operator errors and foreign-thread access; failures cannot publish a
snapshot. Existing hashes, full load/save validation and native integrity checks
remain enabled. Resident requests have one capture and one restore, rather
than per-token snapshots. Restored questions evaluate only25 suffix+186 output
tokens. RAM imports154,522,572 native-file bytes through tmpfs; disk adds
155,515,852 logical pool-read bytes. Filesystem cache warmth is uncontrolled;
these are not measured physical NVMe reads.

## Correctness and actual failures

The new profile's tiny process-separated campaign passes102 full248,320-float
comparisons, seven complete controller comparisons and every generated frontier.
Three prefix restores and a generated-four restart evaluate zero prefix tokens;
parent bytes remain unchanged. All18 stage Python/native PIDs are absent.

All36 CLI timing requests completed, including cold encoder, embedding reuse,
resident/plain CLI/RAM/disk, text control and return to the image branch. The24
paired baseline measurements have identical schedules, generated tokens and
full final vectors, zero error. Separate checkpoint-free raw image and text
references reproduce prefix/final vectors exactly, evaluating complete history.
Focused CPU regression101/101 and G051/51 pass, no skips.

The original whole timing runner remains FAIL. Its final text-only reference
sent a zero-image manifest to an image-only operator, which correctly rejected
it. Cleanup then masked the rejection with BrokenPipeError. The harness now
uses plain batched feed for zero images, matching the actual adapter; cleanup
retains the primary failure. Nine focused tests pass. Only the missing fresh
text-reference stage was rerun, in a new native process, and passed. The
combined independent audit records that recovery explicitly. The entire corrected
runner has not rerun, so this report does not rename its original failure PASS.

No current-profile medium/8K every-frontier requalification, Q8 campaign,
complete supported vision matrix, full VC09 or original G1-G5 qualification is
claimed. The thinking new-image suffix fix remains a separate CPU-only branch.

## Private locations

Checkout: `/home/dflanag3/src/context-cow-vision-active-20261017` on llm-60.
Core commit: `c2af920f`; task branch: `work/vision-owned-schedule-20261017`.
Raw evidence lives under `/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1`.
The JSON beside this report contains complete fingerprints, individual timings,
audit/backup receipts and explicit outstanding work. No Windows artifacts,
GitHub publication or Strata upstream push was made. Chat stays OFF; Q8 deferred.

Verified private archive: `box-git:/srv/data/git/private_context_cow.evidence/20261017-vision-owned-schedule-latency-v2/evidence.tar.gz`

104,272,700 bytes; SHA-256 `b6a8e29cb4518baf6f0945700ccad0d5953c734eedd01cb4ee57eeb59ea424e0`.

After receiver verification, 614,345,360 bytes of optional numerical payloads were retired. Canonical records, original images, embeddings and checked vectors remain.
