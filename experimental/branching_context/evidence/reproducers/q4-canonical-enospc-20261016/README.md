# Canonical branch-reference ENOSPC: executed reproducer sources

The final harness and its 20 CPU controls are in the ordinary scripts/tests
directories, integrated at `3cd7a14e`. This folder preserves the root launcher,
filler-only probe, read-only export and independent audit sources used on
llm-60. No model, checkpoint state or fault image is included here.

The successful runner used `dispatch-v3.sh run final-green-v6` with driver SHA
`59ea6586862da15b93b28dbee5de19cdcdadfaa3aea1c7b77be42ac71aa3c0fe`
and pre-attach image SHA
`565609b604849c1fd37d0a7f9935a3f28b603c3dde54ab7acce75490b4436e06`.
It wrote `gpu-q4-v2/campaign`, `launch-v4/run-command-v1`, and
`mount-run-v3/terminal-receipt.json` under the recorded artifact root.
The fault volume is 64 MiB; deployment pools remain 96/128 GiB.

These are pinned, one-attempt scripts, not a clean-machine installer. They
refuse existing output/receipt paths and mismatched image identities. Do not
run them unchanged expecting another attempt. Prepare a fresh isolated image,
review the source Store/payload/profile pins and assign new output names.
No producer or model should run during the model-free filler control.

`audit_canonical_enospc_failed_v1_20261016.py` validates the aborted first GPU
attempt. The first successful-run audit source
`audit_q4_canonical_enospc_20261016.py` stopped on an API/stored-head shape
comparison: the API adds a branch name. The corrected
`audit_q4_canonical_enospc_v2_20261016.py` checks that name separately and all
three stored head fields, then completes all byte/work/controller audits.
Both sources are retained; the first audit is not presented as a pass.

Read CANONICAL-ENOSPC-Q4-QUALIFICATION and CANONICAL-ENOSPC-Q4-PRIMARY for scope
and actual measurements. No fresh numerical oracle or power-loss test ran.

After both private compact archives passed receiver and independent readback,
`evict_completed_canonical_fixture_20261016.py` removed the exact audited image
and two child checkpoints/manifests. It retained hash-verified canonical metadata
and the original source payload. These compact archives cannot restore the
evicted numerical copies; canonical histories remain reconstructable.
