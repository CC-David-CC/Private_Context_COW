# CLI workflow probe

`run_cli_workflow_probe.py` drives the user-facing `cow_cli.py` process against
the qualified COW HTTP service. The bounded workflow creates a namespace and
prepared root, forks two candidates, generates one answer on each, evaluates
and inspects both immutable reports, selects and CAS-adopts the winner, checks
the winner and retained loser histories, then continues from the adopted
controller.

The runner writes a checkpointed JSON report after each CLI action, redacts the
service key recursively from the saved report (including nested stop receipts,
commands, and failure strings), hashes stdout/stderr, records
source/profile/native pins and requires numeric service/native shutdown
receipts. Redaction applies to a detached copy so the live stop receipt and
shutdown checks remain unchanged. Store access inside the runner is read-only
and used to bind CLI receipts to canonical snapshots, lineage, presentation
output spans, branch revisions, and evaluation reports.

This increment is a workflow check, not an independent numerical comparison.
The result keeps `numerical_oracle` pending. Retry API, unchanged replay,
changed-setting retry, and post-restart replay remain explicitly pending until
the retry CLI contract is frozen and added to the workflow.

For the CPU subprocess test, run the new unittest module against the frozen G4
`green-v7` candidate source tree (not an integrated or qualified source). It
uses `CowPresentationFlowTests.FlowModel` and labels the backend synthetic. The
production GPU runner that invokes `_Campaign.execute` has not been run. Retry,
restart, and independent numerical qualification remain pending.
