# Canonical browser transport

FOCUSED-PASS. With its text workspace enabled, the isolated COW HTTP server
serves the existing page/assets and native capability discovery from one origin.
Authenticated `/cow/workspace/chats`, `/branches?chat=...` and
`/history?branch=...` project the canonical Store, including controller status,
durable incomplete assistants, real snapshot boundaries and saved points.
There are no ordinary `/workspace/*` mutation endpoints or SQLite heads here.
Assets and `/health` are public so a browser can load the page and enter its API
key. History and all COW operations retain authentication and same-origin checks.
Decoded/resolved asset paths must remain within `serve/web`.

`/cow/fork` accepts explicit source branch plus stable operation identity;
namespace-bound rewind uses the coordinator's typed, checkpoint-validated move.
The no-option legacy fork path is retained. Namespace creation does not create
a numerical head or fabricate a saved assistant. The existing `--text` startup
constructs this projection; the ordinary chat server is not started.

Executed on llm-60/Python 3.12.3:

- Six new real-socket methods use the actual Store/coordinator/controller and
  synthetic model-state fixture, not a GPU backend.
- Full serving suite: 92 methods PASS, zero failures/errors/skips/xfails.
  `WORKSPACE-HTTP-SERVING-2026-10-14.json` records executed IDs and source hashes.
- G0: 51/51 PASS, `WORKSPACE-HTTP-G0-2026-10-14.json`.
- Initial RED: five new methods errored because `workspace` was not supported.
  The runner also discovered sixteen imported fixture tests; imports were changed
  to module references to avoid duplicate discovery. No assertions were removed.
- First implementation run: four of five passed; one assertion incorrectly
  expected Store.branches() to return a list rather than its tuple contract.
  Replaced that representation assumption with an empty baseline plus exact
  before/after equality. This retains the no-mutation obligation.
- Fork/rewind RED: new method failed on unsupported source fields (HTTP 400).
  Unchanged method passed after the adapter and route were connected.

Raw logs and executed JSONs are under
`/home/dflanag3/branching-g3/request-stop-20261014/`:
`workspace-http-red.log`, `workspace-http-green-v1.log`,
`workspace-http-actions-red.log`, `workspace-http-actions-green.log` and
`workspace-http-actions-g0.log`. CPU checkout is `serving-v8` there.
Reproduce using `PYTHONPATH=.:src:/home/dflanag3/issue-2026-20261012/deps`
with `python3 serve/verify_cow_serving.py --output RESULT.json`.

Remaining: browser-native transport wiring/interaction checks, explicit fork
provenance for ambiguous same-state annotations, operation-publication recovery
UI, graph projection and complete-route Q4/Q8 GPU qualification. No new numerical,
memory, latency or throughput measurement is supplied by this slice. It does not
qualify the original G1-G5 gates. All code/evidence remain private.
