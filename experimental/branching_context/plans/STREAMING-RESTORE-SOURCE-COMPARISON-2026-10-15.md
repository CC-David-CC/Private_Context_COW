# Streaming restore source comparison — 2026-10-15

## Scope and source identity

This is a source-only comparison of the requested commits:

| Label | Commit | Parent | Commit subject |
|---|---|---|---|
| PR #1269 snapshot | `d258f18920daf7c85a1c5fe407bf907d1fb9639a` | `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253` | `serve: restore a session file without holding its K/V in RAM` |
| PR #2510 snapshot | `c514173ea3a790e1527b2f4e5c960184c10af3ea` | `07613f4b6731bfd509b4bc604024666f2f65c048` | `serve: two-pass streaming session restore (read pass validates, apply pass copies 16 MiB at a time)` |

These are sibling commits, not an ancestor/descendant pair. The large raw tree difference includes unrelated work in their different parent histories, so this review compares each target with its own parent and then inspects the relevant files in both target trees. The missing PR #2510 commit object was fetched by its exact SHA into the local Git object database; no branch or remote ref was updated. No build, test, GPU run, server run, or runtime benchmark was performed for this review.

## Findings

Both trees implement the same central restore design: parse and validate the session file in a first pass while keeping K/V bytes on disk; retain bounded metadata and K/V part headers; then reopen it and stream blocks into the existing authoritative K/V pools. The file remains v1. The streaming reader uses a 16 MiB transfer block, checks file identity/configuration and payload integrity, validates K/V part extents, and computes a bounded read-memory estimate. The apply pass rechecks layer and part sizes before sending that part to the sink and verifies the payload trailer/checksum at completion. The test additions in both commits cover bounded block sizes, no materialized K/V image, byte equality, mutation/corruption/truncation, model identity, and rejection of invalid extents before sink writes.

The PR #2510 restore caller preserves the safety boundary. `src/program/generate.cpp` reads and validates metadata, validates it against the current session/model, ensures capacity, binds target pools, and synchronizes before applying. Its sink writes each bounded block to the target. If apply fails before any write, it can refuse and keep the outgoing session; after the first write, it emits a fatal restore error and exits rather than decoding from partial state. On success, each target `finish()` refreshes its resident/streaming window before the running checkpoint is restored and the session is marked live. The target binding in `src/core/conversation_snapshot.cpp` points at the existing validated pools; it does not create a second full K/V image.

The direct restore-copy change visible between the requested trees is in `src/core/conversation_snapshot.cpp`: PR #1269 uses synchronous `cudaMemcpy(..., cudaMemcpyDefault)`, while PR #2510 routes through `conversation_detail::copy_nonblocking` in the new `src/core/conversation_copy.hpp`. That helper maintains one lazily-created nonblocking CUDA stream per device per thread, enqueues `cudaMemcpyAsync`, then synchronizes that stream before returning. This avoids relying on legacy/default-stream ordering, while preserving the caller's synchronous-completion contract. It is not asynchronous end-to-end: each block still waits for its stream copy. The helper also calls `progress_beat()` and keeps streams until process exit.

PR #2510's commit message additionally describes pinned-checkpoint selection and Metal decode-window checkpoint support. These require careful attribution: in the PR #2510 tree, `session_deepest_checkpoint()` chooses the deepest pinned checkpoint when any is pinned, otherwise the deepest checkpoint; however, that selection logic is already present in the PR #2510 commit's parent, so it is not a change introduced by exact commit `c514173`. The tree-level Metal dispatch is in `src/core/conversation_state.cpp`, under `STRATA_GPU_METAL`, and calls `metal_engine::window_checkpoint_save/restore`; that code is also inherited from the PR #2510 parent. The exact PR #2510 commit changes no Metal source file. A port that needs these behaviors must include their dependency history and validate the backend path separately rather than treating the streaming-restore commit alone as proof of Metal support.

## Checkpoint pin caveat

`ConversationCheckpoint::pinned` in `include/strata/core/conversation_cache.hpp` is documented as a runtime-only mark, not stored in the session file. `put_checkpoint()` serializes IDs, image keys, state vectors, and `used`; `get_checkpoint()` reads those same fields, not `pinned`. The save selector can therefore choose and persist the pinned shared-prefix checkpoint, but the persisted checkpoint does not retain its pin bit. The generator re-establishes a pin when a later request specifies `pin=N` and creates that matching checkpoint. The source inspected here does not prove that an immediate post-restore sibling request gets the same no-park/pinned-retention optimization before that re-pin occurs. Treat saved pinned-checkpoint selection as a save-point choice, not a durable pin-metadata guarantee; add a restart-then-sibling test if porting the optimization.

## File replacement and partial-write boundary

The two passes open/read the path separately and bind the apply pass to the first pass's K/V shape. This catches ordinary content mutation through the second-pass checksum, but a replacement with a different self-consistent same-shape file between passes is not cryptographically bound to the bytes of the first pass. The source comments state the expected no-writer contract. Also, checksum failure can be discovered only after some streamed blocks were written; the caller's fail-stop behavior is essential. A port should preserve exclusive/immutable file ownership during restore, or add stable file identity/first-pass digest binding, and must never continue inference after a post-write failure.

## Smallest source bundle to port

For a CUDA-like native COW implementation, the minimum functional slice is:

1. The streaming file API and parser in `include/strata/core/conversation_file.hpp` and `src/core/conversation_file.cpp`, including limits, identity checks, the two-pass checksum behavior, and the existing v1 encoding.
2. Validated target-pool binding and residency refresh from `include/strata/core/conversation_snapshot.hpp` and `src/core/conversation_snapshot.cpp`; this is the part that keeps restore writes inside the existing authoritative pools.
3. The caller-side ordering/fail-stop logic in `src/program/generate.cpp` (metadata validation and capacity first, synchronization before pool writes, fatal exit after partial mutation).
4. The private-copy stream helper `src/core/conversation_copy.hpp` and its progress hook, if the target backend has the same legacy/default-stream capture hazard. Otherwise implement a backend-specific copy primitive with an explicit completion and ordering contract.
5. The streaming cases in `src/core/conversation_file_test.cpp`, plus target-pool/residency tests from `src/core/conversation_snapshot_test.cpp`.

The deepest-pinned selection policy is a separate optional behavior. If ported, define whether the pin must survive restart; the current v1 serialized checkpoint does not encode it. Metal support is also separate: include the Metal window save/restore integration and test it on Metal only if that is an actual target. Do not pull in unrelated changes from the large parent-history diff.

## Port test delta and evidence limits

Retain the current file-format compatibility tests and add a native COW integration test that restores into pre-existing canonical pool allocations, compares the entire authoritative payload, and verifies `finish()` refreshed the same resident ring/window that decoding will read. Add fault tests for file replacement between passes, mutation after at least one block, and post-write failure causing process/session fail-stop. Add restart-plus-sibling coverage for pin behavior. Exercise concurrent graph-capture/decode and restore-copy ordering on each supported GPU backend; for Metal, test the window checkpoint path independently.

The examined source and tests support the implementation claims above, but this comparison did not reproduce the commit message's 3.05 GiB / 3.4–3.5 s measurement, 99.99% reuse claim, peak-RAM claim on hardware, numerical continuation, or any CUDA/HIP/Metal runtime behavior. Those remain unverified here.
