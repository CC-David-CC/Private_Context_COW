# Prepared prompt, then initial answer

The presentation validator incorrectly required additional prompt tokens for
an initial answer even when the prepared root already contained the first user
prompt. The new deterministic test executed on llm-60 before the change:
10 tests, one error, SSH exit 1. The error was `InvalidOperation: A new answer
must consume its prompt suffix`. The original red log remains at
`/home/dflanag3/branching-g3/request-stop-20261014/presentation-prepared-answer-red.log`.

The implementation now allows zero new prompt tokens for an answer from an
idle prepared root. It still validates the actual consumed output span,
operation/controller scope, source revision and root. New user turns still
require a nonempty prompt suffix and a stopped previous turn. No registered
test, numerical tolerance or reference equation changed.

Executed Linux CPU/filesystem regression results: 136 pool/control tests,
46 serving controls and G0's 51 tests passed; no failures, skips or xfails.
The three adjacent machine-readable profiles record source hashes and scope.
One preliminary `--help` invocation used an incorrect absolute serving-script
path and exited 1 before running tests; the corrected serving command executed
the 46 tests successfully. No new GPU evidence is claimed for this validator.

The complete storage slice is backed up at private commit `3ad036c4`.
Luna's isolated branch is implementing versioned namespace associations and
atomic presentation receipts. Coordinator input/retry binding and the existing
chat interface remain outstanding. The native-speed goal remains active.
