# Park inactive computational contexts in RAM

`BranchingModel.park_checkpoint(snapshot_id)` replaces an optional GPU checkpoint
with immutable bytes from the existing native serializer. It preserves consumed
count, next logits, input identity and execution receipts. Canonical history,
controller records and movable branch references do not change. This operation
does not acknowledge a durable save; use the existing checkpoint/CAS publication
for that guarantee.

The same bounded `CheckpointCache` admits GPU images and serialized RAM bytes.
There is no second history store or new native checkpoint file format. Its quota
counts referenced checkpoint extents; it is not a bound on total model memory,
all Python allocations or the operating system's filesystem cache. Oversized
admission and serialization failure retain the prior hot cache entry. Native
forward counters must remain unchanged through parking.

The private Strata backend now stages numerical transport files on `/dev/shm`
only when `/proc/mounts` verifies that exact directory is a tmpfs mount. Other
platforms retain the ordinary private temporary directory. The inspected native
serializer already supports buffered tmpfs writes and atomic publication there.
Transient transport is separate from the canonical store's durable checkpoint
directory. Tmpfs disappears on reboot; it does not establish disk durability.

## The real-GPU isolation experiment

`scripts/host_context_switch.py` runs three distinct synthetic token contexts:

1. Prepare each prefix, save a durable root, produce an uninterrupted answer,
   and park that prefix in RAM.
2. Move this test store's durable checkpoint directory out of the lookup path.
3. Drain queued native snapshot releases with the existing lifecycle API. Require
   zero retained native snapshot handles; keep all three Python checkpoints as
   immutable bytes. The active engine workspace and model weights still exist.
4. Switch among contexts in order 0, 1, 2, 0, 2, 1. Require measured host-to-device
   restoration and exactly suffix/output forward work, without prefix replay.
   Compare complete histories/controllers, tokens and final logits to uninterrupted
   execution. Audit a separate completed-suffix boundary.
5. Require unchanged reserialized prefix bytes and unchanged offline disk files.
6. Restore disk access, evict RAM checkpoints, and prove strict zero-forward-work
   disk fallback. Terminate the native worker before independent fresh references.
7. Compare prefix, suffix and final frontiers with those references using the
   same native schedule, with zero error. This is sampled numerical evidence,
   not every-token G1 qualification.

Native `loaded_file_bytes` remains meaningful: it counts the serialized bytes
the loader reads, including tmpfs transport. It is not a measurement of physical
storage-device reads. Host upload bytes come from completed native copy counters.

## Executed evidence

The five new CPU tests passed; the full focused suite passed 58 methods, G0 passed
51/51, and mutation smoke detected all four reference mutants. These CPU fixtures
do not substitute for real-GPU proof.

The first Q4 GPU pilot stopped because one finalized snapshot's native release
was still queued. Reading counters does not drain that queue. The failed run and
its original harness are retained. The corrected harness explicitly drains
releases before retaining the same zero-native-handle assertion; it does not
weaken numerical assertions or tolerances.

All four corrected Q4/Q8 tiny and 8K jobs passed on llm-60. Each tested three
independent contexts, six RAM-only switches, disk fallback and three independent
fresh references. The worker restored the original server configuration and
authenticated HTTP returned 200.

| Model and fixture | Actual outputs for contexts 0 / 1 / 2 | Parked native handles | RAM payload bytes |
|---|---|---:|---:|
| Q4 128 / 4 / up to 8 | 8 / 1 / 1 | 0 | 364,240,872 |
| Q4 8K / 512 / up to 128 | 128 / 128 / 3 | 0 | 996,039,144 |
| Q8 128 / 4 / up to 8 | 8 / 2 / 1 | 0 | 364,240,872 |
| Q8 8K / 512 / up to 128 | 2 / 128 / 2 | 0 | 996,039,144 |

EOS stopped some synthetic continuations early. Each switch reproduced its own
uninterrupted continuation; short outputs are not expanded into long-output
performance claims. All four used verified tmpfs staging and zero prefix-forward
work during disk fallback. Source/build/model fingerprints are pinned in the host-v1 profiles;
harness v2 differs only by the explicit release drain. Profile source files stay
frozen. No original G1–G5 qualification is implied.

Results, including the initial failure, are under
`docs/context-cow/results/native-resident-20261014/`. Raw state/vector data remains
on llm-60 under `host-context-v1-20261014` and `host-context-v2-20261014`.

Further work: large real-code RAM contexts, measured GPU/RAM admission, long
client-arrival repetitions, direct cost attribution and ordinary serving
integration. Host serialization is a truthful copy fallback; no physical sharing
between these independent RAM payloads is claimed.
