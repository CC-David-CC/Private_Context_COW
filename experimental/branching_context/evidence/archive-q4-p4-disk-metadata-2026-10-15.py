"""Archive terminal disk-run metadata; reference nine verified archived payloads."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from p4_evidence_bundle import build_manifest, collect_archive_inputs, stream_archive

ART = Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
WORK = ART / 'disk-meta-archive-v1'
OUTPUT = ART / 'p4-disk-v1/output'
API = Path('/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red')
SOURCE = Path('/dev/shm/context-cow-digest-q4-64k-source-20261015-v1')
TARGET = '/srv/data/git/private_context_cow.evidence/20261015-checkpoint-digest-q4-p4-disk/disk.tar'
RECEIVER = '/srv/data/git/private_context_cow.evidence/receive_verified_archive_p4q4_20261015.py'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'box-git']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def prerequisites():
    pins = {
        ART/'p4-disk-existing-payload-proof-v1.json': 'f996c02cb99b8ec8fd633328435e86f528dfc2e17bee6021f271d188bbdd0db3',
        ART/'p4-disk-existing-archive-recheck-v1.json': '57aef756bdcb10ee480693a3a945a45f458ae4ec2445b543942216f541993a4e',
        OUTPUT/'result.json': 'a10cdeb23d3abea39b82d894b9e4190c71e8409aa2720d28f417605e5b3b0b03',
        WORK/'p4_evidence_bundle.py': '5438a97aa384a036c2d184d2b2e61bc2824c76cdb72e306dcf4bf84536e5449d',
    }
    for path, expected in pins.items():
        assert sha(path) == expected, path
    result = json.loads((OUTPUT/'result.json').read_text())
    assert result['status'] == 'PASS' and result['native_exit_codes'] == [0]
    for name in ('command.exit', 'wrapper.exit'):
        assert (ART/'p4-disk-command-v1'/name).read_text().strip() == '0'
    proof = json.loads((ART/'p4-disk-existing-payload-proof-v1.json').read_text())
    assert proof['candidate_count'] == 9 and proof['candidate_bytes'] == 17789976604
    excluded = {'disk/output/' + row['relative_path'] for row in proof['candidates']}
    assert len(excluded) == 9
    current = {'disk/output/' + p.relative_to(OUTPUT).as_posix()
               for p in (OUTPUT/'store/checkpoints').glob('*.state')}
    assert current == excluded
    for row in proof['candidates']:
        st = (OUTPUT/row['relative_path']).lstat()
        assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_nlink) == (
            row['device'], row['inode'], row['size'], row['mtime_ns'], row['nlink'])
    receiver = subprocess.check_output(SSH + ['sha256sum', RECEIVER], text=True)
    assert receiver.split()[0] == 'a3012c68b9010412a84b309b539c608fe18710d962c02cd8800ef68efcf0156e'
    subprocess.run(SSH + ['test', '!', '-e', TARGET], check=True)
    assert shutil.disk_usage(ART).free >= 10 * 1024**3
    return proof, excluded


def preflight():
    proof, excluded = prerequisites()
    profile = json.loads((SOURCE/'profile.json').read_text())
    files = [
        ('provenance/payload-reference.json', ART/'p4-disk-existing-payload-proof-v1.json'),
        ('provenance/archive-recheck.json', ART/'p4-disk-existing-archive-recheck-v1.json'),
        ('provenance/profile.json', SOURCE/'profile.json'),
        ('provenance/fixture.json', SOURCE/'fixture.json'),
        ('provenance/source-result.json', SOURCE/'result.json'),
        ('provenance/source-audit.json', ART/'source64k-independent-audit-v1.json'),
        ('provenance/p4_return_latency.py', API/'scripts-p4-v1/p4_return_latency.py'),
        ('provenance/p4-start.sh', ART/'p4-disk-start-v1.sh'),
        ('provenance/archive-caller.py', Path(__file__)),
        ('provenance/p4_evidence_bundle.py', WORK/'p4_evidence_bundle.py'),
        ('provenance/native-build.json', Path('/home/dflanag3/branching-g3/native-request-probe-20261014/binaries-digest-v1/build.json')),
    ]
    for path, expected in profile['native_build']['sha256'].items():
        p = Path(path)
        if p.parent == API/'src/context_time_travel':
            assert sha(p) == expected
            files.append(('provenance/api/' + p.name, p))
    assert len([name for name, _ in files if name.startswith('provenance/api/')]) == 17
    rows = collect_archive_inputs([('disk/output', OUTPUT),
        ('disk/command', ART/'p4-disk-command-v1')], files)
    assert {name for name, _ in rows if name in excluded} == excluded
    rows = [(name, path) for name, path in rows if name not in excluded]
    records, manifest = build_manifest(rows,
        'New terminal Q4 disk timing metadata and reconstructable Store; nine payloads reside in the explicitly referenced verified archive; NOT standalone.',
        dict(external_payload_archive=proof['archive'], external_payload_count=9,
             external_payload_bytes=17789976604,
             source_full_backup='Pending; tmpfs source retained, no cleanup authorized'))
    with (WORK/'MANIFEST.json').open('xb') as stream:
        stream.write(manifest); stream.flush(); os.fsync(stream.fileno())
    receipt = dict(status='PREFLIGHT_PASS_NO_STREAM', target=TARGET,
        rows=[dict(name=name, source=str(path)) for name, path in rows],
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        members=len(records), input_bytes=sum(row['size'] for row in records),
        excluded_payload_count=9, excluded_payload_bytes=17789976604,
        caller_sha256=sha(Path(__file__)))
    publish(WORK/'preflight.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'rows'}))


def stream_reviewed(expected_manifest):
    prerequisites()
    pre = json.loads((WORK/'preflight.json').read_text())
    manifest = (WORK/'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == pre['manifest_sha256'] == expected_manifest
    assert sha(Path(__file__)) == pre['caller_sha256']
    rows = [(row['name'], Path(row['source'])) for row in pre['rows']]
    records = json.loads(manifest)['files']
    process = subprocess.Popen(SSH + ['python3', RECEIVER, '--target', TARGET],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    try:
        sent = stream_archive(rows, records, manifest, process.stdin)
        process.stdin.close(); process.stdin = None
        stdout, stderr = process.communicate()
    except BaseException:
        if process.stdin is not None:
            process.stdin.close(); process.stdin = None
        process.communicate()
        raise
    (WORK/'receiver.stdout').write_bytes(stdout)
    (WORK/'receiver.stderr').write_bytes(stderr)
    assert process.returncode == 0, (process.returncode, stderr)
    receipt = json.loads(stdout)
    assert receipt['status'] == 'PASS' and receipt['target'] == TARGET
    assert receipt['archive_sha256'] == receipt['final_sha256'] == sent['sha256']
    assert receipt['archive_bytes'] == receipt['final_bytes'] == sent['bytes']
    assert receipt['member_count'] == len(records)
    publish(WORK/'result.json', dict(status='PASS_METADATA_ARCHIVE_EXTERNAL_PAYLOADS',
        target=TARGET, stream=sent, receiver=receipt,
        manifest_raw_sha256=expected_manifest,
        receiver_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        files_deleted=0, standalone=False))
    print(json.dumps(dict(status='PASS_METADATA_ARCHIVE_EXTERNAL_PAYLOADS', target=TARGET, **sent)))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'stream-reviewed'))
    parser.add_argument('--expected-manifest')
    args = parser.parse_args()
    with (WORK/'archive.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            preflight()
        else:
            assert args.expected_manifest
            stream_reviewed(args.expected_manifest)
