# Q4 checkpoint-volume ENOSPC reproducer

These are exact pinned launch and audit sources for the completed 2026-10-16
llm-60 run. They contain absolute artifact paths and intentionally refuse a
changed image, profile, native executable or reused output directory. They are
not a portable installer. Do not rerun them against an existing run directory.

The actual driver and CPU controls are at:

- `scripts/run_q4_enospc.py` under experimental/branching_context.
- `tests/implementation/test_q4_enospc.py` in that same subtree.

`launch-q4-enospc.sh` holds the GPU lock outside sudo. Root is used only for the
private mount namespace and bounded loop attachment; Python and inference run
as uid 1000. `private-mount-run.sh` mounts only the disposable Store checkpoint
directory. Canonical history stays on the host filesystem. The 1.5 GiB image
cannot fill the 2 TB host disk, and the launch guard preserves 16 GiB plus any
remaining image allocation. The image had largely allocated its maximum extent on this retry.

`start-root-gpu-v3.sh` pins all source/runner launch hashes and invokes the
external receipt-producing runner. The source driver SHA is
`2a68537ec177f273fdaa5ce19838a2c6dd29285c8de3ae27fb770d003901da52`.
The launch scripts SHA-256 values are `e700c283d73825ce268a1679ca667a58d4161236be288d7d89a75a03017c0045`
and `0ea9f737c5076006006d978312a7b4d5c018fa57fffc067a88690337aff8287a`.

The auditor reads the unmounted image through a separately attached read-only
loop/mount, verifies source/target/following payloads and canonical histories,
and removes that attachment. Its output is recorded in
`evidence/ENOSPC-Q4-PRIMARY-2026-10-16.json`. The image hash is unchanged by this
read-only audit. The export helper retains small checksum-verified manifests
using debugfs in read-only mode, without copying numerical state.

Measured recovery: 7,999 saved prefix tokens; 128 consumed outputs awaiting
publication; one errno 28 error; no durable acknowledgment on failure; successful
retry of the same payload with no native work; then eight more outputs.
Both HTTP/native exit 0, all PIDs absent, mount and loop removed. Complete
payload hashes, controller records and trace counters pass independent audit.
No fresh numerical comparison or throughput qualification is claimed here.
This does not test exhaustion of the canonical-history volume or power loss.

After private backup and independent readback, the completed disposable image
was evicted, freeing 1,564,143,616 allocated bytes. Its full hash/audit records,
exported manifests and 73 canonical metadata files remain. The original source
checkpoint is retained separately. These pinned scripts now refuse the absent
image; reproducing the experiment requires preparing a new disposable fixture
and new run paths, not replaying the historical launch verbatim.
