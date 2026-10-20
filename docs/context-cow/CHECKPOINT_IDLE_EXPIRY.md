# Idle expiry for managed RAM and disk checkpoint copies

Set `PoolConfig(checkpoint_idle_seconds=1800)` or the COW service option
`--checkpoint-idle-seconds 1800` for this testing/building phase. Production can
choose 7200 or 10800 seconds. Omitting it leaves idle expiry disabled. The chat
server stayed off; no new server configuration was deployed.

Successful RAM admission/get/pin refreshes monotonic last-use time; final pin
release starts a fresh idle interval after a long request. Disk touch persists
last-use time in derived manifest mtime, without changing content/hash or
canonical history. Enabled peer pools/restarts observe the same timestamp.
Clock rollback retains future-stamped copies conservatively. A successful
resident request refreshes its existing disk source before releasing its read
lease, without rehashing payload bytes on every hot request.

The coordinator sweeps managed RAM/disk pools, never the GPU backend. For the
1800-second setting it checks once per minute. Copies become eligible strictly
after 30 minutes and normally disappear by the next sweep. Active pins/read
leases, publication reservations, shared payload aliases and dirty state keep
their protections. Existing budget LRU can evict eligible copies sooner.
Immutable histories and branch references remain; later use can reload another
copy or recompute with the existing notice. Expiry does not delete a chat.

`metrics()['idle_expiry']` reports seconds, sweeps, removed counts, last error and
timer state. Errors remain visible until a successful sweep. `close()` joins the
timer. Model weights, filesystem page cache and the separately bounded native
GPU-image cache are outside this timer's ownership.

## Executed evidence

Five wiring tests first failed (three assertions, two errors). After the fix,
all 20 pool/coordinator expiry tests passed. Combined CPU:363/363; G0:51/51;
no failures/errors/skips/xfails, frozen source unchanged. Actual threads/files
exercise automatic idle cleanup without another request, retained history,
timer I/O error/recovery, close/join and successful resident-source touches.
Deterministic clocks cover cutoff, aliases, rollback, persisted peer/restart
recency and lease/reservation protection. Earlier pool regressions passed71/71.
These scheduler tests use a synthetic model; no GPU is represented as mocked
passing hardware. Real-GPU expiry/recompute/restart on this combined version is
UNRUN. Original cumulativeG1-G5 remains unqualified.

[Qualification, test IDs and hashes](../../experimental/branching_context/evidence/IDLE-EXPIRY-QUALIFICATION-2026-10-16.json).
Raw root: `llm-60:/home/dflanag3/spatial-artifacts/pool-idle-integration-20261016-v1/`.
Private archive: `box-git:/srv/data/git/private_context_cow.evidence/20261016-pool-idle-integration-v1/evidence.tar`,
27,002,880 bytes, SHA-256 `e62efbdacb97f5c23fa1375a4ac6db319bd8c95d3e2d482901e9ecf0a7dbc8a1`.
Streamed readback verified1,912 members/25,067,340 data bytes,0 extracted.

## Separate one-time cleanup

Inventory found677 `.state` files/276,941,789,234 bytes (about258GiB). Authorized
production-pool GC removed265,317,737,620 bytes of old checkpoint/manifest files
from165 inactive dated Stores. Canonical before/after hashes matched. Manifests
and declared payload identities were recorded; payload bytes were not rehashed
for deletion or claimed backed up. Current-day checkpoint exemplars remained.

Another8 unpinned/unrun build binaries/objects were removed after stat/hash and
open-process checks, freeing283,038,000 bytes. Source/build receipts and current
Q4/Q8 profile closure remain. No >=1MiB logs were found in the four reviewed
grammar roots; older archives outside that review were left alone. Latest free
space was about230GiB. Model weights were preserved.
