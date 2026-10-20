# Consumed-boundary cancellation prerequisite

Added optional transient `cancelled` callable to coordinator submit and the
native greedy request wrapper. It is not canonical controller/input state and
does not enter checkpoint identities. Before execution or at a consumed-token
boundary, a true value raises CancelledError. Exiting the existing native
request context sends its cancel/drain protocol before lease release. An
unfinished request never reaches canonical snapshot publication or branch CAS.
The old `cancel(handle)` still removes queued work only. Once a complete
snapshot is handed to durable publication, that publication finishes normally.

Coordinator futures/state report active abandonment as cancelled, with a
separate counter, rather than a backend failure. Failure counters/source lease
cleanup are updated before completion wakes clients. Existing default paths,
test assertions, reference equations, tolerances and integrity checks remain.

Executed on llm-60 CPU, Python 3.12.3:

- Initial new model/coordinator tests RED: unexpected `cancelled` argument.
- Reporting RED: two tests found `failed` instead of `cancelled`.
- Focused integrated suite: 41/41, no failures/errors/skips.
- Full control runner: 106/106, no failures/errors/skips/xfails. Executed IDs
  and source hashes: `REQUEST-STOP-CONTROL-2026-10-14.json`.
- G0: 51/51, evidence `20261014T210021156515Z-G0.json`.
- Reference mutant smoke: all four deliberate wrong outcomes rejected.

An initial ad hoc scratch G0 attempt failed with 50 tests because supplemental
test modules were placed at the G0 discovery root and original logical/numerical
modules were not copied. Restored the repository's directory structure and
original modules unchanged; both scratch and final packaged G0 passed 51/51.
The failed evidence remains at `request-stop-20261014/evidence/runs/20261014T205536490635Z-G0.json`.

Frozen candidate:
`/home/dflanag3/branching-g3/request-stop-20261014/api-v1`.
These are synthetic control/protocol tests; they do not qualify actual GPU
cancellation. That check remains outstanding, as do exact rendered-prefix
validation and ordinary serving integration. No web service was started.

The Q8 64K pool campaign uses this candidate with no cancellation callback,
so it tests ordinary continuation behavior of the new source. It does not
test active cancellation. Its native model/binaries are still pinned digest-v1.

## Full-size real GPU cancellation campaign

`scripts/gpu_request_stop.py` exercises an isolated copy of the completed Q8
64K source-context store with 96K capacity, 96 GiB RAM and 128 GiB disk budgets.
Only immutable `.state` payloads are hard-linked; canonical records and branch
references are copied. The original campaign is not used for mutations.

The test requests cancellation after three consumed outputs, checks that no
canonical snapshot or branch movement was published, then runs a fresh request
on the same retained prefix. It compares 32 outputs and the final full frontier
with the campaign's independently executed matching-schedule reference and
checks zero prefix work, source payload integrity and serialized GPU ownership.

First attempt, `request-stop-20261014/gpu-full-v1`, ended FAIL after 91.85 s:
the harness accessed `_protocol_failed` directly, but the backend creates that
flag only on failure. The worker exception masked cancellation reporting.
Both Python/native processes exited. This is harness evidence, not a backend
numerical failure. The failed record and log are retained unchanged.

Corrected inspection uses `getattr(..., False)`. A new run, `gpu-full-v2`, was
launched in tmux `cow-gpu-stop-v2`, retaining the same native/backend source,
model, profile and original independent reference. Result is pending; no GPU
cancellation PASS is claimed yet. Chat server remains off; no public push.

### Terminal result: bounded real GPU PASS

Corrected `gpu-full-v2` completed in 100.72 s including 72.20 s startup.
GPU-STOP-001 and GPU-STOP-002 passed: cancellation after three consumed outputs
drained the native protocol, left the committed branch/canonical objects
unchanged, and did not poison the pipe. The following request returned 32 exact
reference tokens and an identical full final frontier (maximum error 0).
Requests evaluated only the 8,336-token suffix plus actual outputs: 8,339 and
8,368 forward tokens, with no saved 65,536-token prefix work. Parent payload
hash remained unchanged. One cancellation, one completion, zero failures or
failed publications; no outstanding pins/reservations and at most one active
request on GPU 0. Native exit 0; no inference process remained.

Counters/check IDs/identities are in `REQUEST-STOP-GPU-2026-10-14.json`;
build/model fingerprints are bound by its matching execution/model identities
to `SOURCE-CONTEXT-Q8-64K-2026-10-14.json`. Scope: one deterministic cancellation
boundary, not arbitrary cancellation races/crashes/controllers. Serving
integration and original cumulative certification remain outstanding.
