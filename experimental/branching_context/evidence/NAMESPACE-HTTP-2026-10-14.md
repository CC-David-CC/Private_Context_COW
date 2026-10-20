# Namespace-bound HTTP controls

FOCUSED-PASS, not GPU or milestone qualification.

Authenticated routes now expose canonical namespace creation/listing and
presentation catalogs, plus explicit code-context preparation, prepared-prompt
answers and first questions. Namespace reads do not submit inference. Typed
presentation descriptors remain server-generated. Streamed progress remains
distinct from the durable completion event. Preparation creates no assistant
answer. Existing prompt and conversation cache settings are unchanged.

Executed on llm-60 using Python 3.12.3 in
`/home/dflanag3/branching-g3/request-stop-20261014/serving-v8`:

- Supplemental serving suite: 72 methods PASS, zero failures/errors/skips or
  expected failures. Five new methods use actual HTTP sockets with synthetic
  computation; executed IDs and imported-source hashes are in
  `NAMESPACE-HTTP-SERVING-2026-10-14.json`.
- G0: 51 methods PASS; `NAMESPACE-HTTP-G0-2026-10-14.json` retains the result.
- Initial new-route run: five methods, eight failing subtest/test comparisons
  because the routes returned 404. Raw log: `namespace-http-red.log` in the
  parent remote run directory.
- An additional negative run exposed a synthetic preparation fixture that
  incorrectly reported an output token. One of five methods failed. Corrected
  the fixture to return an empty consumed-output list, preserving the assertion;
  `namespace-http-empty-root-red.log` retains that failure. This was not a
  native-model defect.

Reproduce with `PYTHONPATH=.:src:/home/dflanag3/issue-2026-20261012/deps`
and `python3 serve/verify_cow_serving.py --output RESULT.json`, then
`python3 scripts/verify.py --gate G0` on that Linux host.

Limitations: complete namespace-to-native routes have not yet received the new
Q4/Q8 GPU qualification. Browser workspace integration and typed branch-action
HTTP routes remain outstanding. Preparation has no request cancellation
callback yet. These results contain no numerical error, work, memory or speed
measurement and do not qualify G1-G5. Ordinary chat remains off.

Private Git checkpoint uses reviewed explicit Git commands: the installed
Spatial helper pins a different project's origin. No remotes are changed to
satisfy that helper. User authorization requires this project's private box-git
destination; no GitHub publication is permitted.
