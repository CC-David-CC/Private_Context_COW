# P4 paired-arm archive staging helper

This helper is a reusable building block for the forthcoming current-profile
disk/resident pair. It is not a run launcher and has not read a run directory,
opened a checkpoint payload, contacted box-git, or removed a file. Keep the
existing Q4 archives and three original Q4 roots protected.

The helper is `scripts/p4_evidence_bundle.py`. The caller supplies archive
member prefixes and source paths. It rejects duplicate/noncanonical member
paths and symlinks, hashes a closed source list, creates the same PAX tar
stream shape used by the existing verified receiver, and rehashes each file
while streaming so a post-manifest change stops the stream. `stream_archive`
accepts a caller-owned binary sink: a future runner can pass `ssh` stdin to the
existing receiver without creating a local tar spool. The helper does not
start SSH, check a receiver pin, or publish an archive.

After a real arm reaches terminal PASS and root supplies its confirmed output
and provenance paths, a small arm-specific caller should:

1. Add the complete arm output, inner/outer command receipts, frozen runner,
   profile, input manifest, source-closure receipt, and compact numeric command
   receipts using `collect_archive_inputs`. Use distinct member prefixes such
   as `p4-disk-v1/output` and `p4-disk-v1/provenance`; do not invent paths before
   the actual run and root path confirmation exist.
2. Call `build_manifest(rows, scope, metadata)`, retaining the exact returned
   manifest bytes and source rows outside the protected run tree.
3. Run a separately reviewed streaming command against the installed
   `receive_verified_archive_p4q4_20261015.py` receiver, pass its binary stdin
   to `stream_archive`, and preserve its raw receipt. Require the receiver's
   PASS receipt to bind the target and exact generated manifest before mapping
   states. The output target must be new/no-clobber and each arm needs a unique
   run directory and archive path.
4. Read `files` from the exact manifest that the receiver accepted and call
   `make_state_map(output_root, member_prefix, manifest["files"])`. The map
   discovers the actual direct `store/checkpoints/*.state` count, hashes each
   current source, validates its content-addressed filename, and matches exact
   member path, byte length, and digest. Persist the map separately and bind it
   to the receiver receipt and manifest digests. `verify_state_map` recomputes
   it before any separately reviewed reclaim step.

Example calls for the eventual arm-specific caller (paths intentionally left
as values supplied from the completed run, not prefilled assumptions):

```python
rows = collect_archive_inputs(
    [(member_prefix + "/output", output_root),
     (member_prefix + "/inner-receipt", inner_receipt_dir),
     (member_prefix + "/outer-receipt", outer_receipt_dir)],
    [(member_prefix + "/provenance/profile.json", profile_path),
     (member_prefix + "/provenance/source-closure.json", source_closure_path)],
)
records, manifest_bytes = build_manifest(rows, bundle_scope, metadata)
stream_receipt = stream_archive(rows, records, manifest_bytes, ssh_process.stdin)
state_map = make_state_map(output_root, member_prefix, accepted_manifest["files"])
```

Do not assume nine states, a particular payload size, or those illustrative
member names will match the new profile. The next action is to wait for the
root-confirmed arm paths and terminal receipts, then create only the narrow
arm-specific caller/config from those facts. No archive transfer, reclaim, or
second-arm start is authorized by this helper or plan alone.

CPU fixture tests run on llm-60 with:

```bash
python3 -m py_compile scripts/p4_evidence_bundle.py scripts/test_p4_evidence_bundle.py
python3 scripts/test_p4_evidence_bundle.py
```

The tests use tiny temporary files and an in-memory tar stream; they do not
exercise SSH, the remote receiver, a live run, model files, or GPU behavior.
