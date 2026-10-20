# Q4 bounded-filesystem ENOSPC handoff

Latest: real Q4 checkpoint-volume ENOSPC and retry are FOCUSED-PASS, independently
byte-audited. See ENOSPC-Q4-PRIMARY and ENOSPC-Q4-QUALIFICATION-2026-10-16.json.
Current driver SHA is `2a68537ec177f273fdaa5ce19838a2c6dd29285c8de3ae27fb770d003901da52`;
final CPU tests SHA is `e48495b95a4edb85962e2f6b72d10870ce1b7017f27836090eb2406a2bc7c8bb`.
Final CPU controls pass 16/16, G0 51/51. Reversed receipt/release ordering is
detected by an executed mutant. No production/native/profile source changed.
The following initial-stage pins and setup notes are historical; current
source and result details are in the later repair/result sections.

## Initial preflight and failed launches (historical)
This handoff separates harness qualification from the real filesystem fault run.

The test driver and CPU guard tests are at `scripts/run_q4_enospc.py` and `tests/implementation/test_q4_enospc.py`. Their SHA-256 values are `1f48c4bb737dda54cd642a829be3f875fbd089aeaec1e353285321f5e9ffa48c` and `75991c91aee3fc95d9364295944b58a626aa86d0936ca77d2cf8dc33355a02dd`. The remote clean staged copy is `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/preflight-green-v6/source`; the base helper `run_q4_publication_fault.py` is pinned at `d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695`.

CPU guard suite: 12 passed, 0 skipped, command/wrapper exits `0/0`. Receipt directory: `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/preflight-green-v6-run`. It checks driver status handling and input pin validation; it does not emulate or qualify kernel ENOSPC.

Private ext4 mount preflight v2 completed with command/wrapper exits `0/0`, emitted `PREPARED_NOT_RUN`, and cleaned up the exact mount and loop device. Receipt: `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/private-mount-preflight-v2/terminal-receipt.json`. Disposable image: `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/mount-preflight-v1/bounded-ext4.img`, inode `5252779`, UUID `f068766f-7ce6-4584-9063-2f80024a0e09`. The v2 preflight image hash was `f66b8fa52434c6a9737a3130a534938555d30c4f27db3d0e2be3834654c5c8ea`; a subsequent failed launch attempt changed filesystem metadata, and root recorded the newer hash as `582af27adbc96f58cc07a6cca507bd794b343fb42cd27964979904187f342329`.

Failed launch attempt 1 is a setup failure, not an ENOSPC result: it exited `1/1` before model startup because the probe result parent directory did not exist. Root recorded that the private mount wrapper cleaned up successfully (`0/0`, mount removed and loop detached). Preserve this attempt and its receipt. Root is preparing a new v4 launcher with the result parent created as the ordinary service user and a unique receipt directory. The v3 launcher is frozen and must not be edited in place.

The fault boundary is deliberately limited to the disposable Store `checkpoints` volume. Canonical history, references, and metadata remain outside that mounted volume; this cannot be described as full-Store-volume ENOSPC or hardware disk failure/power-loss qualification.

Pinned source inputs for the eventual run:

- Q4 profile: `/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/profiles-v2/q4/profile.json`, SHA-256 `f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c`.
- Native executable SHA-256: `4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451`.
- Expected source snapshot: `808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85`.
- Source manifest SHA-256: `f97f7fcb8834d0dd3e69e6a3dc705e131a1fe2b62bf62f5506766b2f86f3e177`.
- Source payload: `423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810.state`, 327,985,908 bytes.

The v2 setup failure alone establishes none of the filesystem fault/retry claims; only the separate v3 audit below records the scoped checkpoint-volume fault and recovery result.

## Root-owned Q4 GPU result and launcher pins

Root's v3 run subsequently reached the intended kernel ENOSPC fault and same-operation retry; independent audit is `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/primary-audit-gpu-q4-enospc-v3.json`, SHA-256 `84f52fa326cbf120794a52cd5838f63ef87650cf4900114549340d1013cdc449`, status `FOCUSED-PASS`. It records one errno-28 error event, 128 fault outputs, 8 following outputs, native and HTTP exits 0, 7,999 saved prefix tokens, zero prefill tokens, 136 forward tokens, same-operation retry with unchanged native work, unchanged source checkpoint bytes, and no live PIDs. The primary audit explicitly limits scope: only the checkpoint volume was full, no fresh numerical oracle was run, and this is not cumulative G2 or power-loss qualification.

Root-owned launch sources (kept outside this worker's checkout) are:

- v4: `/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v4/launch-q4-enospc.sh`, SHA-256 `52004399477d76afcd8076cee6bdc6d3b59b484bf5a2f0023cd2cde00e446d3a`; `private-mount-run.sh`, SHA-256 `e1a6e5e02adf93f1ac50d77452178034b9b2f83689b816834feca4b74b3b7637`.
- v5: `/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v5/launch-q4-enospc.sh`, SHA-256 `e700c283d73825ce268a1679ca667a58d4161236be288d7d89a75a03017c0045`; `private-mount-run.sh`, SHA-256 `0ea9f737c5076006006d978312a7b4d5c018fa57fffc067a88690337aff8287a`.

The v2 setup failure and v3 successful run are separate records. The v3 result does not retroactively qualify the failed v2 attempt.

## Additional orchestration regressions

The filler success test now inspects the result file at the moment `RELEASE` is sent, so reversed persistence/release order fails. A temporary reversed-order driver mutant failed that regression with `filler receipt must exist before RELEASE`; mutant command/wrapper exits were 1/1 in `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/order-mutant-v2-run/`. Mutant driver SHA is `5f71af53b5be0bf24ea62898d2ff58761974f6e12b82504c031eff013f7d7ba0`; test SHA is `e48495b95a4edb85962e2f6b72d10870ce1b7017f27836090eb2406a2bc7c8bb`. A persistence `OSError(EIO)` control requires `ABORT` and no filler-success receipt. Final CPU GREEN run `repair-v4-green-run` passed 16/16, 0 skipped, command/wrapper 0/0; stderr SHA `3284dfffad13444b11f2f264113e7329c635d922c8b50aaf741fb22061bd422c`. Frozen driver SHA remains `2a68537ec177f273fdaa5ce19838a2c6dd29285c8de3ae27fb770d003901da52`; updated tests SHA is `e48495b95a4edb85962e2f6b72d10870ce1b7017f27836090eb2406a2bc7c8bb`.

## Repair checkpoint

The first actual harness launch (`gpu-q4-enospc-v2`) reached the real writer gate for operation `op-q4-enospc-621ff1fb93104769bd3b6359e9744b03`, with a 331,328,756-byte target payload, then failed in the harness before releasing the writer: `NameError: name '_atomic_json' is not defined`. The exact stderr is `/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v4/gpu-command-v2/stderr.log`; progress and gate evidence are under `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/gpu-q4-enospc-v2/probe-result/`. No `result.json` or writer-error record was produced. The private-mount receipt is `/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/private-mount-run-v4/terminal-receipt.json` (probe/cleanup exits 1/0; mount removed and loop detached). The original input source payload still hashes to its pinned value `423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810`.

Repair extracts the gate/fill/persist/release callback and calls the already-pinned helper as `_base._atomic_json` for the filler receipt and final result. CPU RED (`repair-v1-red-run`) ran 13 tests: the 12 previous controls passed and the new writer-gate orchestration test failed at the expected persistence/release assertion. CPU GREEN (`repair-v3-green-run`) ran 15 tests, all passed, 0 skipped; command/wrapper exits were 0/0. New tests exercise successful persistence-before-release, filler-error abort without a success receipt, and rejection of unqualified `_atomic_json` references. GREEN staged driver/test hashes are `2a68537ec177f273fdaa5ce19838a2c6dd29285c8de3ae27fb770d003901da52` and `9d34aeec38eed53beb2a9c0760bd2ddf53dbe3df6baa1ca192bd4a2ae34e95b9`.

These checks qualify only the harness CPU orchestration. The actual filesystem ENOSPC/retry run remains unqualified and must use a newly reviewed launcher stage; the failed v2 attempt remains preserved without retrying it in place.
