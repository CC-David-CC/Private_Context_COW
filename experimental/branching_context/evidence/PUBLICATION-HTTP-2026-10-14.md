# Explicit publication recovery

FOCUSED-PASS; CPU/filesystem/socket controls with synthetic computation only.

The coordinator exposes process-scoped operation status and an idempotent
publication-only retry. Authenticated `GET /cow/operation?operation=...` reports
the existing request state, whether it was durably acknowledged and whether its
retained failed save can be retried. It returns a receipt only after the durable
future succeeds. `POST /cow/publication/retry` with `operation_id` retries the
existing writer/CAS phase or waits for that same phase; it never admits another
model request. Repeating it after success returns the same receipt.

Namespace-scoped HTTP history/branch projections now include transient pending
request status, separately from durable controller state. No history/checkpoint
format changes, second head store, decoder changes or numerical fallback.
The new branch association on Request is transient admission metadata.

The retry waits until the original failure notification completes before
replacing its failed future. A test exposes that race using a blocked failure
notification: premature status is not retryable, premature retry returns409,
the old branch stays unchanged, then a later retry succeeds without forward work.

Executed on llm-60/Python3.12.3 in `request-stop-20261014/serving-v8`:

- Initial two new HTTP methods failed on missing routes, retained in
  `publication-http-red.log`.
- Initial implementation passed95 serving and192 core methods.
- New race regression failed on prematurely advertised retryability,
  `publication-http-race-red.log`; unchanged assertions passed after guarding
  the original future's completed failure boundary.
- Final serving96/96, core192/192 and G0 51/51 PASS, no failures/errors/skips/
  expected failures. PUBLICATION-HTTP-{SERVING,CONTROL,G0}-2026-10-14.json
  records IDs, imported-source hashes and exact result scope.
- Raw final logs: `publication-http-green-v2.log`,
  `publication-control-green-v2.log`, `publication-http-g0.log` in the parent
  remote run directory. `PYTHONPATH=.:src:/home/dflanag3/issue-2026-20261012/deps`;
  run `serve/verify_cow_serving.py`, `scripts/verify_pool_control.py` with output
  paths, then `scripts/verify.py --gate G0`, using python3 on that Linux host.

Limits: operation status and retained failed-publication retry apply to the
current coordinator process. After restart an unknown operation returns404;
this endpoint does not certify an absent checkpoint or repeat computation.
Existing signed-request/committed-receipt recovery remains a separate path.
Live pending flags are observational and may change after a read. The numerical
backend, real-GPU publication fault campaign and browser save-retry interaction
are not qualified by these tests. Original G1-G5 status is unchanged. Ordinary
chat remains off; private Git only.
