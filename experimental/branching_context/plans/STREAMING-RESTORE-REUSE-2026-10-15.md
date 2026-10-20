# Streaming cold-restore reuse plan

Current execution status (2026-10-15): the selected native streaming loader is
implemented and built in the combined v3 CUDA candidate. CPU parser checks,
model-free GPU restoration, partial-apply failure injection, and bounded Q4
tiny/1K/8K matching-profile restart checks passed. See
`../evidence/STATUS.md`, `../evidence/STREAMED-LOAD-APPLY-FAILURE-2026-10-15.json`
and `../evidence/COMBINED-Q4-8K-2026-10-15.json` for executed scope and limitations.
The broader snapshot suite retains a pre-existing pointer-assertion failure;
HIP/SYCL remain unrun. Python still reads the outer payload in full: this is
native streaming, not an end-to-end bounded-memory loader. Current v3 Q8
qualification and corrected matched performance measurements are outstanding.
The source-review text below records the original plan, not current execution.

## Source and ancestry proof

The inspected Context COW checkout was `ccd50ae565121d2723d7461397f7e05dc499f963`. Its `include/strata/core/conversation_file.hpp` declares `session_file_read(...)`, but no `session_file_read_streamed(...)` or `session_file_apply_streamed(...)`; `include/strata/core/conversation_snapshot.hpp` has no `conversation_snapshot_validate_meta(...)`. `rg` over the checkout's `include` and `src` also found none of those symbols. The inspected worker checkout at `4d8f150a17926d7374e7b6d1d1817bf6a7f52f4d` has the same absence. The PR commit is not in either HEAD's ancestry.

The proposed source is PR1269 commit `d258f18920daf7c85a1c5fe407bf907d1fb9639a`, parent `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`. The parent is an ancestor of `ccd50ae5` (merge-base equals the parent); the PR commit itself is not. `740fa3392dea42c08c8baec1ea8f1bb2ab08820c` is a separate local object on another parent, not current source. Do not describe either object as already integrated.

The smallest useful selective port from d258 is its six core files: `include/strata/core/conversation_file.hpp`, `src/core/conversation_file.cpp`, `src/core/conversation_file_test.cpp`, `include/strata/core/conversation_snapshot.hpp`, `src/core/conversation_snapshot.cpp`, and `src/core/conversation_state.cpp`. The PR's `src/program/generate.cpp` change is the ordinary session-restore caller; `docs/DETAILS.md` documents that user-facing path. Neither is required for the COW `BranchSession::load` call site. Treat those as separate optional integrations, not dependencies to merge with the COW change.

PR1480 commit `dd146f495534f950847ba8dfff93d9ec9d9712d2` and merge `8cfd2518` add serving disk-cache policy: `ConversationSpillCache`, sidecars, cache selection/eviction, and `--conversation-cache-disk-only`. That policy is not needed to stream one COW cold restore. The later `fix/1529-retain-disk-prefix` serving-cache correction is also outside this loader scope. Keep the existing native session-file format and outer COW manifest unchanged.

## COW integration shape

Replace the whole-image read inside `BranchSession::load` with the two-pass read/apply interface from PR1269. First, establish the existing file identity and runtime read limits, call the streamed validation pass, then check exact expected token history, unsupported images/checkpoints/cvec, geometry, layer layout, and K/V headers before device writes. Allocate and validate the caller's logits frontier before mutating the live session.

Apply only after preparing a fresh private live page lease. `BranchSession::reset` clears and regrows its `BranchPages` mappings; the loader must never sink file bytes into a frozen `BranchSessionImage` or any shared page. Bind each streamed K/V layer/part/offset to the currently rebound private QSA pools (using the existing conversation K/V target helper if its ownership and bounds fit, otherwise a small COW-specific adapter). Restore the running metadata, finish the required device/resident-ring synchronization, and publish `ids`, logits, and ready state only after the whole second pass and final integrity checks succeed.

Failure rules: corrupt identity, malformed bounds, bad metadata, truncation, or a read-pass rejection must leave device state untouched. A file change detected after applying begins, sink/device error, or incomplete transfer must leave the session failed and prohibit inference or publication from it. A frozen sibling image must remain byte-identical. Keep the outer checkpoint's source-history and frontier hashes as the authoritative binding; the native file does not contain logits or controller state.

## Test-first validation

Start with CPU session-file tests comparing streamed and full reads, rejecting corrupt/truncated files before sink writes, and mutating a same-size file between validation and apply to prove post-write fail-stop behavior. Add COW cold-load tests for exact positions, token history, frontier, zero extra forward work, and no ready state after any partial apply failure. Freeze a sibling snapshot before load and verify its page digests/ownership and full bytes afterward.

Then run independent model-backed cold-load/resume/fork oracle vectors at Q4 lengths 128, 1K, and 8K, followed by the pinned Q8 cases. Compare the next-token result and frontier against an independently initialized matching-operator/schedule oracle, audit complete bytes and page ownership, check zero unexpected prefix recomputation, and verify native session-file bytes/format remain unchanged. Keep the source commit and model/execution profile pins fixed across arms.

The reviewed PR1269 source is CUDA-oriented (`generate.cpp`, QSA/CUDA copy helpers). Before claiming portability, inspect the final selected build closure and compile/test each supported backend that the changed core files reach. HIP and SYCL integration/build/runtime are currently **UNRUN**; no backend is qualified by this plan. No implementation, build, test, model load, remote job, or GPU job was performed for this review.
