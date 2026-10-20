"""Verify and compactly archive the approved eviction/CLI-notice CPU ticket."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/eviction-recomputation-notice-20261016-v1')
qualified = json.loads((root / 'controller-review-v1/EVICTION-RECOMPUTATION-NOTICE-2026-10-16.json').read_bytes())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for row in qualified['red_green']:
    for name, key in (('command.exit', 'command_exit'), ('wrapper.exit', 'wrapper_exit')):
        assert int((Path(row['run']) / name).read_text()) == row[key]
    if row.get('stderr_sha256'):
        assert sha(Path(row['run']) / 'stderr.log') == row['stderr_sha256']
green = next(row for row in qualified['red_green'] if row['name'] == 'focused_green_final')
assert green['tests_passed'] == 38 and green['tests_failed'] == 0
for name, expected in qualified['changed_files'].items():
    assert sha(Path(qualified['remote_environment']['final_staged_source']) / name) == expected
g0 = qualified['g0']
assert g0['tests_passed'] == 51 and g0['tests_failed'] == 0
assert sha(Path(g0['evidence_json'])) == g0['evidence_sha256']
assert all((Path(g0['run']) / name).read_text().strip() == '0' for name in ('command.exit', 'wrapper.exit'))
files = {}
for path in sorted(root.rglob('*')):
    if not path.is_file():
        continue
    rel = path.relative_to(root)
    if path.is_symlink():
        continue
    if rel.parts[0] in ('controller-backup-v1', 'controller-backup-command-v1', 'controller-readback-command-v1'):
        continue
    if path.suffix in ('.tar', '.pyc', '.state', '.gguf', '.pack') or '__pycache__' in rel.parts:
        continue
    if rel.parts[0] == 'checkouts' and '/'.join(rel.parts[2:]) not in qualified['changed_files']:
        continue
    data = path.read_bytes()
    assert len(data) < 16*1024**2, str(path)
    files[str(rel)] = data
members = [dict(path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
           for name, data in sorted(files.items())]
manifest = dict(schema='cow-eviction-notice-backup-v1', members=members,
    member_count=len(members), total_member_bytes=sum(row['size_bytes'] for row in members),
    base_commit=qualified['worktree']['base_commit'],
    scope='Approved missing-checkpoint test correction, CLI notice RED/GREEN and final failure-event regression, CPU38/G0-51, setup failures',
    exclusions=['Unchanged source closure available from private Git base', 'No GPU/model work in this ticket'])
assert manifest['total_member_bytes'] < 64*1024**2
output = root / 'controller-backup-v1'
output.mkdir(mode=0o700)
raw = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
archive = output / 'evidence.tar'
with tarfile.open(archive, 'x') as stream:
    for name, data in list(sorted(files.items())) + [('member-manifest.json', raw)]:
        info = tarfile.TarInfo(name)
        info.mode, info.size = 0o600, len(data)
        stream.addfile(info, io.BytesIO(data))
archive.chmod(0o600)
digest = sha(archive)
destination = '/srv/data/git/private_context_cow.evidence/20261016-eviction-recomputation-notice-v1'
subprocess.run(['ssh', 'box-git', 'test ! -e '+destination], check=True)
subprocess.run(['ssh', 'box-git', 'mkdir -m 700 '+destination], check=True)
subprocess.run(['scp', str(archive), 'box-git:'+destination+'/evidence.tar.part'], check=True)
assert subprocess.check_output(['ssh', 'box-git', 'sha256sum '+destination+'/evidence.tar.part'], text=True).split()[0] == digest
subprocess.run(['ssh', 'box-git', 'chmod 600 '+destination+'/evidence.tar.part'], check=True)
subprocess.run(['ssh', 'box-git', 'mv '+destination+'/evidence.tar.part '+destination+'/evidence.tar'], check=True)
receipt = dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING', archive=destination+'/evidence.tar',
    archive_bytes=archive.stat().st_size, archive_sha256=digest, manifest_sha256=hashlib.sha256(raw).hexdigest(),
    member_count=len(members), total_member_bytes=manifest['total_member_bytes'])
with (output / 'transfer.json').open('x') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps(receipt, sort_keys=True))
