"""Verify and privately back up root-query CPU evidence and owned overlays."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/root-presentation-query-20261016-v1')
review = root / 'controller-review-v1'
qualified = json.loads((review / 'ROOT-PRESENTATION-QUERY-2026-10-16.json').read_bytes())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for label in ('green_focused', 'serving_regressions', 'g0'):
    row = qualified['red_green'][label]
    assert row['failed'] == row['skipped'] == 0
    assert row['passed'] == {'green_focused': 11, 'serving_regressions': 84, 'g0': 51}[label]
    for name in ('command.exit', 'wrapper.exit'):
        assert (Path(row['run']) / name).read_text().strip() == '0'
    for name, expected in qualified['change']['files'].items():
        assert sha(Path(row['stage']) / name) == expected, (label, name)
red = qualified['red_green']['red']
assert all((Path(red['run']) / name).read_text().strip() == '1'
           for name in ('command.exit', 'wrapper.exit'))
assert 'root lookup must skip' in (Path(red['run']) / 'stderr.log').read_text()
files = {}
owned = set(qualified['change']['files'])
for path in sorted(root.rglob('*')):
    if not path.is_file():
        continue
    assert not path.is_symlink(), str(path)
    relative = path.relative_to(root)
    if relative.parts[0] == 'controller-backup-v1':
        continue
    if path.suffix in ('.tar', '.pyc', '.state', '.gguf', '.pack') or '__pycache__' in relative.parts:
        continue
    if relative.parts[0] == 'checkouts':
        tail = '/'.join(relative.parts[2:])
        if tail not in owned:
            continue
    data = path.read_bytes()
    assert len(data) < 16 * 1024**2, str(path)
    files[str(relative)] = data
members = [dict(path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
           for name, data in sorted(files.items())]
manifest = dict(schema='cow-root-query-private-backup-v1', members=members,
    member_count=len(members), total_member_bytes=sum(row['size_bytes'] for row in members),
    base_commit=qualified['repository']['base_commit'],
    scope='CPU11/84/G0-51, failing rewind regression, setup failures, runner scripts and owned source overlays',
    exclusions=['Unchanged source closure available from private Git base', 'No model/GPU work in this ticket'])
assert manifest['total_member_bytes'] < 64 * 1024**2
output = root / 'controller-backup-v1'
output.mkdir(mode=0o700)
manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
archive = output / 'evidence.tar'
with tarfile.open(archive, 'x') as stream:
    for name, data in list(sorted(files.items())) + [('member-manifest.json', manifest_raw)]:
        info = tarfile.TarInfo(name)
        info.mode = 0o600
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))
archive.chmod(0o600)
digest = sha(archive)
destination = '/srv/data/git/private_context_cow.evidence/20261016-root-presentation-query-controller-v1'
subprocess.run(['ssh', 'box-git', 'test ! -e ' + destination], check=True)
subprocess.run(['ssh', 'box-git', 'mkdir -m 700 ' + destination], check=True)
subprocess.run(['scp', str(archive), 'box-git:' + destination + '/evidence.tar.part'], check=True)
assert subprocess.check_output(['ssh', 'box-git', 'sha256sum ' + destination + '/evidence.tar.part'], text=True).split()[0] == digest
subprocess.run(['ssh', 'box-git', 'chmod 600 ' + destination + '/evidence.tar.part'], check=True)
subprocess.run(['ssh', 'box-git', 'mv ' + destination + '/evidence.tar.part ' + destination + '/evidence.tar'], check=True)
receipt = dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',
    archive=destination + '/evidence.tar', archive_bytes=archive.stat().st_size,
    archive_sha256=digest, manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    member_count=len(members), total_member_bytes=manifest['total_member_bytes'])
with (output / 'transfer.json').open('x') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps(receipt, sort_keys=True))
