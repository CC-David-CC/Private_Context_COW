"""Stream private evidence (including RAM reference arrays) without a local tar."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1')
reference = Path('/dev/shm/context-cow-native-combined-reference-20261016-v1')
def read(path):
    return json.loads(path.read_bytes())
for run in ('cpu-command-v1', 'g0-command-v1', 'gpu-smoke-command-v1',
            'gpu-smoke-reference-command-v1', 'gpu-campaign-command-v1',
            'gpu-campaign-reference-command-v1', 'root-query-measurement-command-v2'):
    assert all((root / run / name).read_text().strip() == '0'
               for name in ('command.exit', 'wrapper.exit')), run
assert read(reference / 'result.json')['status'] == 'PASS'
assert read(reference / 'result.json')['numerical_max_error'] == 0
assert read(root / 'gpu-campaign-v1/result.json')['status'] == 'PASS_PIPELINE_NUMERICS_PENDING'
owned = {'experimental/branching_context/src/context_time_travel/coordinator.py',
    'experimental/branching_context/src/context_time_travel/store.py', 'serve/cow_adapter.py',
    'serve/cow_text_adapter.py',
    'experimental/branching_context/tests/implementation/test_root_presentation_query.py',
    'experimental/branching_context/tests/implementation/test_publication_handoff_ownership.py'}
files = {}
for parent, prefix in ((root, 'campaign'), (reference, 'ram-fresh-references')):
    for path in sorted(parent.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(parent)
        if path.is_symlink():
            assert str(rel) == 'source-v1/experimental/branching_context/evidence/runs'
            continue
        if path.suffix in ('.state', '.tar', '.pyc', '.gguf', '.pack') or '__pycache__' in rel.parts:
            continue
        if rel.parts[0] in ('stream-backup-v1', 'stream-backup-command-v1', 'stream-readback-command-v1'):
            continue
        if rel.parts[0] == 'source-v1' and '/'.join(rel.parts[1:]) not in owned:
            continue
        limit = 256 * 1024**2 if parent == reference and path.suffix == '.f32' else 16 * 1024**2
        assert path.stat().st_size < limit, str(path)
        files[prefix + '/' + str(rel)] = path
assert sum(path.stat().st_size for path in files.values()) < 4 * 1024**3
destination = '/srv/data/git/private_context_cow.evidence/20261016-native-client-combined-v1'
subprocess.run(['ssh', 'box-git', 'test ! -e ' + destination], check=True)
subprocess.run(['ssh', 'box-git', 'mkdir -m 700 ' + destination], check=True)
output = root / 'stream-backup-v1'
output.mkdir(mode=0o700)
child = subprocess.Popen(['ssh', 'box-git', 'umask 077; set -C; cat > ' + destination + '/evidence.tar.part'], stdin=subprocess.PIPE)
class Writer:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0
    def write(self, data):
        written = self.stream.write(data)
        assert written == len(data)
        self.digest.update(data)
        self.bytes += written
        return written
    def flush(self):
        self.stream.flush()
writer = Writer(child.stdin)
members = []
try:
    with tarfile.open(fileobj=writer, mode='w|') as archive:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.mode, info.size = 0o600, len(data)
            archive.addfile(info, io.BytesIO(data))
            members.append(dict(path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(schema='cow-native-combined-stream-backup-v1', members=members,
            member_count=len(members), total_member_bytes=sum(row['size_bytes'] for row in members),
            source_commit='97bab1a24ee53df036a3ac5bd083e18ec7f5cc71',
            scope='Combined CPU200/G0-51, explicit baseline failure, real Q4 smoke and three-tree campaign, full independent fresh-reference arrays',
            exclusions=['Native checkpoint state payloads; their actual byte hashes remain in campaign fingerprints',
                'Models and packs; complete identities retained', 'Unchanged CPU source closure retained in private Git'])
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        info = tarfile.TarInfo('member-manifest.json')
        info.mode, info.size = 0o600, len(raw)
        archive.addfile(info, io.BytesIO(raw))
    writer.flush()
    child.stdin.close()
    assert child.wait(timeout=60) == 0
    digest = writer.digest.hexdigest()
    assert subprocess.check_output(['ssh', 'box-git', 'sha256sum ' + destination + '/evidence.tar.part'], text=True).split()[0] == digest
    subprocess.run(['ssh', 'box-git', 'mv ' + destination + '/evidence.tar.part ' + destination + '/evidence.tar'], check=True)
    receipt = dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',
        archive=destination + '/evidence.tar', archive_bytes=writer.bytes, archive_sha256=digest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(), member_count=len(members),
        total_member_bytes=manifest['total_member_bytes'], local_tar_created=False,
        reference_storage='RAM filesystem during audit; private box-git archive after execution')
    with (output / 'transfer.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, sort_keys=True))
finally:
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=20)
