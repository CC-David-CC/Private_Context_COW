# Retain closed COW evaluations without filling their fast workspace

PCC provides an owner-operated maintenance command for completed COW evaluation
stores. It complements the existing model/optimizer checkpoint retention.
S³ keeps its scores and response evidence; BW keeps its workflow authority.
This command performs filesystem maintenance and never runs a model, trains,
changes a question or starts another evaluation.

In the shared A100 campaign, the ordinary disk still had about 2 TB free while
the roughly 1-TiB shared RAM filesystem lacked unreserved working headroom.
Completed COW stores had accumulated after their point evidence was archived.
Two verified cleanup passes reclaimed 172,450,432,845 bytes (160.6 GiB) from
248 closed stores. Models, optimizer states, scores and raw case responses were
preserved. Existing experiment controllers continued from their own positions.

An [independent archive-content audit](../validation/closed-store-retention-20261026/a100-archive-audit.json)
then streamed all 248 archived stores and confirmed that their contents match
the source hashes recorded before retirement. This audit made no model calls
and retired no additional artifacts.

## One pass, or finite maintenance alongside existing workers

Run from the installed PCC checkout:

```sh
python -m tools.pcc_closed_store_retention \
  --policy /data/study/retention-policy.json \
  --output /data/study/maintenance-01 \
  --maximum-passes 1
```

For a finite campaign, choose explicit pass and interval limits, for example
`--maximum-passes 1440 --interval-seconds 60`. It uses the same maintenance lease
on each pass and writes a new pass receipt. No system service or automatic
restart is installed. A refusal writes `NEEDS_REVIEW` and retains the original
attempt. Inspect it before starting another output directory.

Example owner policy:

```json
{
  "schema": 1,
  "workspace": "/dev/shm/owned-study",
  "groups": [{
    "points": "/dev/shm/owned-study/full",
    "evidence": "/data/study/full-evidence",
    "arms": ["adamw-full"]
  }],
  "keep_latest": 3,
  "maximum_stores_per_pass": 8,
  "maintenance_lock": "/data/study/private-control/cleanup.lock",
  "completion_points": ["/dev/shm/owned-study/full/adamw-full/update-068"]
}
```

The workspace, individual point and archive directories remain private and
owned. Grouping folders may have inherited readable modes; they must be real
owned directories. The lock's parent is private. The optional completion list
lets a campaign include historical groups while stopping maintenance when all
requested final scoring records are present. It does not certify that every
training controller or archive-copy thread has exited.

## Proof required before retirement

The command retains the latest configured points in each arm and considers only
`development/cow-store` beneath older points. It checks original complete result,
settlement and score records, then verifies the original evaluator process has
exited. A live original evaluator is skipped. The source records must match the
preserved point manifest.

It checks the archive's checksum and streams its actual COW-store members to
verify each member's bytes against the preserved inventory. Source files must
match those same bytes. An intent is fsynced before retiring literal store
files; a completion receipt follows. A partial retirement can reconcile that
same intent. Changed archives, missing members, changed source files, linked
paths and different intents refuse retirement.

The initial supported store contains owned JSON/lock files, with an 8-GiB
per-store archive bound. Numerical checkpoint payloads remain a separate COW
pool concern. Restore an archived store before an audit or recovery operation
that requires its original canonical files. Raw per-case response JSON outside
the store, evaluation scores, model weights and optimizer states stay in place.

The [qualification receipt](../validation/closed-store-retention-20261026/qualification.json)
records CPU filesystem/process/archive tests, including a matching outer checksum
that cannot hide incorrect archived bytes, real live/retired process checks,
interrupted retirement, recent-point retention and the finite CLI.
