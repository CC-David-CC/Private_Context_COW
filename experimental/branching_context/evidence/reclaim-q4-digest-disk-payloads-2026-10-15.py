"""Reclaim exactly nine archived derived payloads; retain all run metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ART = Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
OUTPUT = ART / 'p4-disk-v1/output'
PROOF = ART / 'p4-disk-existing-payload-proof-v1.json'
PROOF_SHA = 'f996c02cb99b8ec8fd633328435e86f528dfc2e17bee6021f271d188bbdd0db3'
ARCHIVES = {
    '/srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution/lifecycle-attribution-20261015.tar':
        'd01af0087b43f4c8f3558ae64d49377d8f927c0ec31420035134644cd347684f',
    '/srv/data/git/private_context_cow.evidence/20261015-checkpoint-digest-q4-p4-disk/disk.tar':
        '0faff80456ccf29b59797b70773a1028024b2f908b75741baf9839690849d961',
    '/srv/data/git/private_context_cow.evidence/20261015-checkpoint-digest-q4-p4-disk/disk.tar.receipt.json':
        'fb4635eb4862870206af3a5074fd8840e45a350845d34de49b4268a46388b4e9',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
            stat.S_IMODE(st.st_mode), st.st_nlink)


def metadata_inventory(excluded):
    return {str(p.relative_to(OUTPUT)): sha(p)
            for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p not in excluded}


def publish(path, record):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--receipt-dir', type=Path, required=True)
    args = parser.parse_args()
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    require(sha(PROOF) == PROOF_SHA, 'payload proof changed')
    proof = json.loads(PROOF.read_text())
    require(proof['output_root'] == str(OUTPUT), 'wrong output root')
    require(len(proof['candidates']) == 9, 'wrong payload count')
    checkpoint_dir = OUTPUT / 'store/checkpoints'
    for p in (ART, OUTPUT, OUTPUT / 'store', checkpoint_dir):
        require(p.resolve() == p and not p.is_symlink(), 'symlinked target root')
    paths = [OUTPUT / row['relative_path'] for row in proof['candidates']]
    require(len(set(paths)) == 9 and set(paths) == set(checkpoint_dir.glob('*.state')),
            'candidate set differs from exact nine checkpoint files')
    for p in paths:
        require(p.parent == checkpoint_dir and p.resolve() == p, 'target escapes checkpoint directory')
    for fd in Path('/proc').glob('[0-9]*/fd/*'):
        try:
            target = os.readlink(fd)
        except (FileNotFoundError, PermissionError):
            continue
        require(target not in {str(p) for p in paths}, 'candidate is open by a process')
    backup = subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', 'box-git', 'sha256sum', *ARCHIVES], text=True)
    actual = dict((line.split(None, 1)[1].strip(), line.split()[0]) for line in backup.splitlines())
    require(actual == ARCHIVES, 'private archive bytes changed')
    publish(args.receipt_dir / 'private-archive-recheck.json', actual)
    before = metadata_inventory(set(paths))
    verified = []
    for p, row in zip(paths, proof['candidates']):
        expected = (row['device'], row['inode'], row['size'], row['mtime_ns'],
                    int(row['mode'], 8), row['nlink'])
        require(identity(p.lstat()) == expected, 'payload identity changed')
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            require(identity(os.fstat(stream.fileno())) == expected, 'opened payload differs')
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            require(identity(os.fstat(stream.fileno())) == expected, 'payload changed while hashing')
        require(digest == row['sha256'] == p.stem, 'payload bytes differ from archived proof')
        verified.append(dict(path=str(p), bytes=row['size'], sha256=digest, identity=list(expected)))
    require(sum(row['bytes'] for row in verified) == 17789976604, 'wrong byte total')
    record = dict(status='VERIFIED_NOT_DELETED', files=verified, deleted=[],
                  metadata_before=before, free_before=os.statvfs(ART).f_bavail * os.statvfs(ART).f_frsize)
    receipt = args.receipt_dir / 'result.json'
    publish(receipt, record)
    if args.apply:
        record['status'] = 'RECLAIMING'
        publish(receipt, record)
        for row in verified:
            p = Path(row['path'])
            require(list(identity(p.lstat())) == row['identity'], 'payload changed before unlink')
            p.unlink()
            record['deleted'].append(row['path'])
            publish(receipt, record)
        fd = os.open(checkpoint_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        require(metadata_inventory(set(paths)) == before, 'run metadata changed')
        record.update(status='PASS_EXACT_NINE_DERIVED_PAYLOADS_RECLAIMED',
                      reclaimed_bytes=17789976604,
                      free_after=os.statvfs(ART).f_bavail * os.statvfs(ART).f_frsize)
        publish(receipt, record)
    print(json.dumps({key: record[key] for key in ('status', 'deleted')}))


if __name__ == '__main__':
    main()
