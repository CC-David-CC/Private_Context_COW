"""Back up terminal combined CPU233/G0-51 receipts and exact tested overlays."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1')
cpu = json.loads((root / 'cpu-v2.json').read_bytes())
assert cpu['status'] == 'PASS' and cpu['tests'] == 233 and cpu['source_unchanged']
assert not any(cpu[key] for key in ('failures', 'errors', 'skips', 'xfails'))
for name in ('cpu-command-v2', 'g0-command-v2'):
    assert all((root / name / field).read_text().strip() == '0' for field in ('command.exit', 'wrapper.exit'))
files = {}
for name in ('cpu-command-v2', 'g0-command-v2', 'g0-evidence-v2'):
    for path in (root / name).rglob('*'):
        if path.is_file():
            assert not path.is_symlink()
            files[str(path.relative_to(root))] = path.read_bytes()
for name in ('cpu-v2.json', 'run_native_combined_cpu_v2.py', 'launch_native_combined_cpu_v2.sh'):
    files[name] = (root / name).read_bytes()
for name in ('tools/cow_cli.py', 'tools/test_cow_cli_recomputation_notice.py',
             'experimental/branching_context/tests/implementation/test_pool_branch_actions.py'):
    data = (root / 'source-v2' / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == cpu['source_manifest'][name]
    files['tested-overlays/' + name] = data
members = [dict(path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
           for name, data in sorted(files.items())]
manifest = dict(schema='cow-native-combined-final-cpu-backup-v1', members=members,
    member_count=len(members), total_member_bytes=sum(row['size_bytes'] for row in members),
    source_commit=cpu['source_commit'], scope='CPU233 and G0-51, exact unchanged source manifest, raw terminal logs/receipts and tested CLI/policy overlays')
raw = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
output = root / 'final-cpu-backup-v1'
output.mkdir(mode=0o700)
archive = output / 'evidence.tar'
with tarfile.open(archive, 'x') as stream:
    for name, data in list(sorted(files.items())) + [('member-manifest.json', raw)]:
        info = tarfile.TarInfo(name)
        info.mode, info.size = 0o600, len(data)
        stream.addfile(info, io.BytesIO(data))
archive.chmod(0o600)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
destination = '/srv/data/git/private_context_cow.evidence/20261016-native-client-combined-cpu-final-v1'
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
