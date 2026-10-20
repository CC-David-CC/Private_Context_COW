# Sampler test-double metadata correction

FOCUSED-PASS on llm-60's CPU, based on private parent `635a321a`.

Two older token-adapter test stores returned a controller without the snapshot
schema. The adapter now reads canonical input history through its schema-aware
reader. Both unchanged baseline suites therefore stopped at `KeyError: schema`
before testing their sampler-setting assertions: 46 passes and one error each.

Each corrected fixture now declares schema 1 for its existing text lineage.
Every assertion remains unchanged. No production code, registered gate test,
reference equation, tolerance or backend check was changed. The collector
verified that these two fixture files are the only changed code files against
the retained baseline; the other 1,803 source files are byte-identical.

Both complete suites together passed 94/94 methods with no failures, errors or
skips. G0 passed 51/51. This is a CPU fixture correction; it does not qualify
image samplers or any additional GPU configuration.

Exact method IDs, commands and hashes are in the adjacent JSON report. Retained
baseline failures and corrected logs are in the verified private archive:

- Destination: `box-git:/srv/data/git/private_context_cow.evidence/20261017-cow-vision-test-metadata-v1/evidence.tar.gz`
- Bytes: 108,808; members: 33.
- SHA256: `4be4c60aa52bd316bb217f1b6a77375f6560ad2be2f38dfbf5edd977d8de631e`.

The isolated branch is `fix/vision-test-store-metadata`. Integration into the
vision feature branch and a combined-source regression run remain separate.
