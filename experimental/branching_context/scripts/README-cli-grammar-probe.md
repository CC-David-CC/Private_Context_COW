# CLI GBNF workflow probe

`run_cli_grammar_probe.py` drives a small READ/TEST grammar workflow through
`cow_cli.py`, the HTTP server, the native worker, and canonical Store receipts.
The server receives a registry at startup; CLI requests name a registry state
and never provide grammar source or a local registry path.

The probe checks a one-token READ answer followed by continuation to EOS, an
active-state switch rejection before native work, an independent TEST branch,
persisted controller state, durable output spans, and service/worker shutdown.
It reports `CLI_GRAMMAR_FLOW_PASS_NUMERICAL_AUDIT_PENDING` on success. It is a
workflow check, not an independent numerical qualification.

The companion `test_cli_grammar_probe.py` runs real `cow_cli.py` subprocesses
against the deterministic CPU `FlowModel` HTTP fixture. It does not start the
production native executable or qualify GPU behavior.
