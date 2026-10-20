"""Verify new payload bytes against an existing archive manifest; never delete."""
import hashlib
import json
import os
from pathlib import Path
import stat


art = Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
output = art / 'p4-disk-v1/output'
mapping_path = Path('/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-map-audit-20261015-01/map.json')
manifest_path = Path('/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-20261015-03/manifest.json')
assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == '91ec8f54db1fc53fb1983dfc6722c86a898edbde79483f992fe86ff5ee462291'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == '7f74320abfe33b2aa2e89b176c357b5544ac0dc2b9770e558076b7cbbd9c728d'
mapping = json.loads(mapping_path.read_text())
manifest = json.loads(manifest_path.read_text())
expected = {row['payload_sha256']: row['bytes'] for row in mapping['large_file_inventory']['state_files']}
members = {row['path']: row for row in manifest['files']}
paths = sorted((output / 'store/checkpoints').glob('*.state'))
assert len(paths) == len(expected) == 9
assert {p.stem for p in paths} == set(expected)
result_path = output / 'result.json'
assert hashlib.sha256(result_path.read_bytes()).hexdigest() == 'a10cdeb23d3abea39b82d894b9e4190c71e8409aa2720d28f417605e5b3b0b03'
result = json.loads(result_path.read_text())
assert result['status'] == 'PASS' and result['native_exit_codes'] == [0]
rows = []
for path in paths:
    assert path.resolve().parent == output.resolve() / 'store/checkpoints'
    initial = path.lstat()
    assert stat.S_ISREG(initial.st_mode) and initial.st_nlink == 1
    assert initial.st_uid == os.geteuid() and stat.S_IMODE(initial.st_mode) == 0o600
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        size = 0
        while block := os.read(fd, 8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode, st.st_nlink)
    assert identity(initial) == identity(before) == identity(after)
    assert digest.hexdigest() == path.stem and size == expected[path.stem]
    member = 'lifecycle-run/output/resident/store/checkpoints/' + path.name
    assert members[member]['sha256'] == path.stem and members[member]['size'] == size
    rows.append(dict(relative_path=path.relative_to(output).as_posix(), size=size,
        sha256=path.stem, archive_member=member, device=before.st_dev,
        inode=before.st_ino, mtime_ns=before.st_mtime_ns, mode='0600', nlink=1))
assert sum(row['size'] for row in rows) == 17789976604
proof = dict(schema='p4-existing-archive-payload-proof-v1',
    status='PASS_CURRENT_BYTES_MATCH_EXISTING_ARCHIVE_MANIFEST',
    scope='Read-only current-byte audit; archive-file rehash and new metadata backup required separately before reclaim',
    output_root=str(output), result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    archive=dict(target=mapping['archive']['target'], sha256=mapping['archive']['archive_sha256'],
        bytes=mapping['archive']['archive_bytes'],
        receiver_receipt_path=mapping['archive']['receiver_receipt_path'],
        receiver_receipt_sha256=mapping['archive']['receiver_receipt_sha256'],
        manifest_raw_sha256=mapping['archive']['manifest_raw_sha256'],
        canonical_manifest_sha256=mapping['archive']['receiver_canonical_manifest_sha256'],
        member_listing_sha256=mapping['archive']['member_listing_sha256']),
    candidate_count=len(rows), candidate_bytes=sum(row['size'] for row in rows),
    candidates=rows, files_deleted=0)
with (art / 'p4-disk-existing-payload-proof-v1.json').open('x') as stream:
    json.dump(proof, stream, indent=2)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(dict(status=proof['status'], files=len(rows), bytes=proof['candidate_bytes'], files_deleted=0)))
