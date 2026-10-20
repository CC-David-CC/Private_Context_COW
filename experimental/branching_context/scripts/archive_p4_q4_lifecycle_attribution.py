#!/usr/bin/env python3
"""Stream a terminal P4 Q4 lifecycle-attribution run to the private archive receiver.

The default mode validates inputs and target availability without sending bytes.
--stream-reviewed sends the exact frozen run tree plus source/input provenance;
it does not alter or delete any source files and creates no local tar spool.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import BinaryIO

RUN = Path('/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-20261015-v1')
P4_SRC = Path('/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context')
DIAG_SCRIPT = Path('/home/dflanag3/branching-g3/checkpoint-lifecycle-diagnostics-luna-20261015-v9/experimental/branching_context/scripts/checkpoint_lifecycle_diagnostics.py')
NATIVE_SOURCE = Path('/home/dflanag3/branching-g3/native-request-probe-20261014/source-digest-v1')
NATIVE_BUILD_LINK = NATIVE_SOURCE / 'build'
NATIVE_BUILD_LINK_TARGET = '/home/dflanag3/branching-g3/durable-resume-20261013/source/build'
NATIVE_BUILD_LINK_RESOLVED = '/home/dflanag3/branching-g3/context-cow-public-20261013/build'
NATIVE_PROGRAM_LINK = NATIVE_SOURCE / 'src/program'
NATIVE_PROGRAM_LINK_TARGET = '/home/dflanag3/branching-g3/durable-resume-20261013/source/src/program'
NATIVE_PROGRAM_LINK_RESOLVED = NATIVE_PROGRAM_LINK_TARGET
NATIVE_BUILD_SCRIPT = Path('/home/dflanag3/branching-g3/native-request-probe-20261014/api-digest-v1/scripts/build_digest_reuse_probe.py')
NATIVE_DRIVER = Path('/home/dflanag3/branching-g3/native-request-probe-20261014/binaries-digest-v1/branch_driver.cpp')
NATIVE_BUILD_RECEIPT = Path('/home/dflanag3/branching-g3/native-request-probe-20261014/binaries-digest-v1/build.json')
SENDER = Path('/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-20261015-03/archive_p4_q4_lifecycle_attribution.py')
START_SCRIPT = Path('/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-20261015-03/start.sh')
RUN_COMMAND = Path('/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh')
Q4 = Path('/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure')
ARCHIVE = '/srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution/lifecycle-attribution-20261015.tar'
RECEIVER = '/srv/data/git/private_context_cow.evidence/receive_verified_archive_p4q4_20261015.py'
RECEIVER_SHA256 = 'a3012c68b9010412a84b309b539c608fe18710d962c02cd8800ef68efcf0156e'
RESULT_SHA256 = 'fa1f7b856f1808c8f8a109e248d5a5191f4fb0ab0b213a1ef4547c407976345c'
RUN_COMMAND_SHA256 = '7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537'
START_SHA256 = '053b9325061c5e3aa462d0f60757e0315caba7f37daaac67519d61ef87aa336a'
MANIFEST_OUT = Path('/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-20261015-03/manifest.json')
MIN_FREE_BYTES = 10 * 1024**3

Q4_INPUTS = {
    'fixture.json': 'c661084b390720c1cbb1ca2465734cd5a099d366f2811704b4661251f5ca1903',
    'fresh-tree-0.stdout': '47aa637d0bfc996406b04d8e00744e50f35501ee5e98f975a50207cec5af1d6c',
    'fresh-tree-2.stdout': '947e692e37a056be63ac8ae24a6101a376b4f67239b295b2f2b05c794e2ce0a1',
    'fresh-tree-4.stdout': '7280b2fe26543464122c546b6d07f9e1541adfeefd5512fe09943fe7df99cf98',
    'prepare.json': '784978eb9e290b1ba02c2bbdd852e495189619b0ddb7811f17559d339522fc5a',
    'profile.json': 'f3c69ad908786548cdd325f33af03a623553134d9406483e7622edd62135ea4b',
    'result.json': '40965f747ebda3fc87eb61cd274c96e2847d46dba65d346eef99e6d25c3d1cfa',
}


def digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open('rb', buffering=0) as f:
        before = os.fstat(f.fileno())
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            h.update(block)
        after = os.fstat(f.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or size != after.st_size:
        raise RuntimeError(f'input changed while hashing: {path}')
    return size, h.hexdigest()


def pin(path: Path, expected: str) -> None:
    size, actual = digest(path)
    if actual != expected:
        raise RuntimeError(f'hash mismatch: {path}: {actual} != {expected}')
    print(f'PIN {path} bytes={size} sha256={actual}', flush=True)


def add_tree(rows: list[tuple[str, Path]], prefix: str, root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f'required non-symlink source tree missing: {root}')
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            if path == NATIVE_BUILD_LINK:
                target = os.readlink(path)
                resolved = str(path.resolve(strict=True))
                if target != NATIVE_BUILD_LINK_TARGET or resolved != NATIVE_BUILD_LINK_RESOLVED:
                    raise RuntimeError(f'native build link target changed: {target} -> {resolved}')
                continue
            if path == NATIVE_PROGRAM_LINK:
                target = os.readlink(path)
                resolved = path.resolve(strict=True)
                if target != NATIVE_PROGRAM_LINK_TARGET or str(resolved) != NATIVE_PROGRAM_LINK_RESOLVED or not resolved.is_dir():
                    raise RuntimeError(f'native program link target changed: {target} -> {resolved}')
                logical_prefix = f'{prefix}/{path.relative_to(root).as_posix()}'
                for source in sorted(resolved.rglob('*')):
                    if source.is_symlink():
                        raise RuntimeError(f'symlink not permitted in resolved native program source: {source}')
                    if '__pycache__' in source.parts or source.suffix == '.pyc':
                        continue
                    if source.is_file():
                        rows.append((f'{logical_prefix}/{source.relative_to(resolved).as_posix()}', source))
                continue
            raise RuntimeError(f'symlink not permitted in archive input: {path}')
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_file():
            rows.append((f'{prefix}/{path.relative_to(root).as_posix()}', path))


def build_entries() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    add_tree(rows, 'lifecycle-run', RUN)
    for folder in ('src', 'scripts', 'tests'):
        add_tree(rows, f'source/p4-branching-context/{folder}', P4_SRC / folder)
    add_tree(rows, 'source/native-source-digest-v1', NATIVE_SOURCE)
    for name, path in (
        ('source/checkpoint_lifecycle_diagnostics.py', DIAG_SCRIPT),
        ('source/build_digest_reuse_probe.py', NATIVE_BUILD_SCRIPT),
        ('source/branch_driver.cpp', NATIVE_DRIVER),
        ('source/native-build.json', NATIVE_BUILD_RECEIPT),
        ('source/archive_p4_q4_lifecycle_attribution.py', SENDER),
        ('source/start.sh', START_SCRIPT),
        ('source/run-command.sh', RUN_COMMAND),
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'required regular source file missing: {path}')
        rows.append((name, path))
    result = json.loads((RUN / 'output/resident/result.json').read_text())
    recorded = result.get('input_sha256')
    if not isinstance(recorded, dict):
        raise RuntimeError('lifecycle result has no input hash map')
    for name, expected in Q4_INPUTS.items():
        path = Q4 / name
        if recorded.get(str(path)) != expected:
            raise RuntimeError(f'lifecycle result input pin differs: {path}')
        rows.append((f'references/q4/{name}', path))
    pin(START_SCRIPT, START_SHA256)
    pin(RUN_COMMAND, RUN_COMMAND_SHA256)
    rows.sort(key=lambda row: row[0])
    names = [name for name, _ in rows]
    if len(names) != len(set(names)):
        raise RuntimeError('duplicate archive member path')
    return rows


class HashingWriter:
    def __init__(self, f: BinaryIO):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def write(self, b: bytes) -> int:
        n = self.f.write(b)
        if n != len(b):
            raise OSError('short SSH receiver write')
        self.h.update(b)
        self.n += n
        return n

    def flush(self) -> None:
        self.f.flush()


class HashingReader:
    def __init__(self, f: BinaryIO, member: str):
        self.f, self.h, self.n, self.member = f, hashlib.sha256(), 0, member
        self.next_report = 1024**3

    def read(self, size: int = -1) -> bytes:
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        while self.n >= self.next_report:
            print(f'MEMBER_PROGRESS path={self.member} bytes={self.n}', flush=True)
            self.next_report += 1024**3
        return b


def require_terminal() -> dict:
    inner = RUN / 'command-receipts/resident'
    if (inner / 'terminal-status.txt').read_text().strip() != 'COMPLETE':
        raise RuntimeError('lifecycle command receipt is not COMPLETE')
    if (inner / 'command.exit').read_text().strip() != '0' or (inner / 'wrapper.exit').read_text().strip() != '0':
        raise RuntimeError('lifecycle command/wrapper exits are not numeric 0/0')
    result = json.loads((RUN / 'output/resident/result.json').read_text())
    if result.get('status') != 'PASS' or result.get('arm') != 'resident' or result.get('measured_requests') != 9:
        raise RuntimeError('lifecycle result status or scope differs')
    if result.get('native_exit_codes') != [0] or result.get('native_exit_observed') is not True:
        raise RuntimeError('lifecycle native terminal status is not successful')
    if result.get('source_checkpoints_unchanged') is not True:
        raise RuntimeError('lifecycle result does not prove source checkpoints unchanged')
    execution = result.get('execution_identity', {})
    if execution.get('GPU') != [0] or execution.get('capacity') != 98304 or execution.get('prefill_chunk') != 8192:
        raise RuntimeError('lifecycle execution profile differs from the pinned P4 Q4 setup')
    diag = json.loads((RUN / 'diagnostics/resident.json').read_text())
    if diag.get('measurements', {}).get('dropped_events') != 0:
        raise RuntimeError('diagnostics dropped lifecycle events')
    pin(RUN / 'output/resident/result.json', RESULT_SHA256)
    return result


def probe_target() -> None:
    command = (
        'test ! -e /srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution '
        '&& test ! -e ' + ARCHIVE + ' && test ! -e ' + ARCHIVE + '.receipt.json '
        '&& sha256sum ' + RECEIVER
    )
    probe = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'box-git', command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    fields = probe.stdout.decode('ascii', errors='replace').split()
    if probe.returncode or not fields or fields[0] != RECEIVER_SHA256:
        raise RuntimeError(f'private receiver/target preflight failed: {probe.stderr.decode(errors="replace")}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream-reviewed', action='store_true')
    args = parser.parse_args()
    result = require_terminal()
    rows = build_entries()
    records = []
    for name, path in rows:
        size, sha = digest(path)
        records.append({'path': name, 'size': size, 'sha256': sha})
    manifest = {
        'schema': 1,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'bundle_scope': 'Raw terminal Q4 P4 lifecycle-attribution run, executed source snapshots and exact Q4 fixture/reference inputs; not a standalone rerun bundle.',
        'result_sha256': RESULT_SHA256,
        'profile_id': result.get('profile_identity', {}).get('execution_profile_id'),
        'model_contract_id': result.get('profile_identity', {}).get('model_contract_id'),
        'excluded_source_links': [{
            'path': 'source/native-source-digest-v1/build',
            'target': NATIVE_BUILD_LINK_TARGET,
            'resolved_target': NATIVE_BUILD_LINK_RESOLVED,
            'reason': 'External generated build tree; exact native include/src closure and executed build receipt/driver/generator are archived separately.',
        }],
        'resolved_source_links': [{
            'path': 'source/native-source-digest-v1/src/program',
            'target': NATIVE_PROGRAM_LINK_TARGET,
            'resolved_target': NATIVE_PROGRAM_LINK_RESOLVED,
            'archive_mapping': 'Resolved regular files are archived below the symlink logical path; nested symlinks are refused.',
        }],
        'files': records,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()
    target_pre = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'box-git',
         'test ! -e /srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution '
         '&& test ! -e ' + ARCHIVE + ' && test ! -e ' + ARCHIVE + '.receipt.json'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if target_pre.returncode:
        raise RuntimeError('archive target is not absent or cannot be checked')
    receiver = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'box-git', 'sha256sum', RECEIVER],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    recv_fields = receiver.stdout.decode('ascii', errors='replace').split()
    if receiver.returncode or not recv_fields or recv_fields[0] != RECEIVER_SHA256:
        raise RuntimeError('private archive receiver SHA mismatch')
    free = shutil_disk_free(Path('/home/dflanag3/spatial-artifacts'))
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f'llm60 free-space floor failed: {free} < {MIN_FREE_BYTES}')
    if MANIFEST_OUT.exists():
        raise RuntimeError(f'manifest output already exists: {MANIFEST_OUT}')
    total = sum(r['size'] for r in records)
    print(f'PREFLIGHT_ONLY members={len(records)+1} input_bytes={total} archive={ARCHIVE} llm60_free={free} manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}', flush=True)
    if not args.stream_reviewed:
        print('NO_TRANSFER: preflight only; use --stream-reviewed for the reviewed archive operation', flush=True)
        return 0

    mfd = os.open(MANIFEST_OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(manifest_bytes)
        while view:
            wrote = os.write(mfd, view)
            if wrote <= 0:
                raise OSError('short manifest evidence write')
            view = view[wrote:]
        os.fsync(mfd)
    finally:
        os.close(mfd)

    ssh = subprocess.Popen(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'box-git',
         'python3', RECEIVER, '--target', ARCHIVE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    assert ssh.stdin is not None
    writer = HashingWriter(ssh.stdin)
    try:
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for index, (name, path) in enumerate(rows, 1):
                before = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f'nonregular archive input: {path}')
                raw = path.open('rb', buffering=0)
                opened = os.fstat(raw.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
                ):
                    raw.close()
                    raise RuntimeError(f'input changed before transfer: {path}')
                record = records[index - 1]
                print(f'MEMBER_START index={index}/{len(rows)+1} path={name} bytes={record["size"]}', flush=True)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.uid, info.gid, info.mtime = record['size'], 0o600, 0, 0, 0
                info.uname = info.gname = ''
                hashed = HashingReader(raw, name)
                try:
                    tar.addfile(info, hashed)
                finally:
                    raw.close()
                if hashed.n != record['size'] or hashed.h.hexdigest() != record['sha256']:
                    raise RuntimeError(f'input changed while streaming: {name}')
                after = path.stat(follow_symlinks=False)
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
                ):
                    raise RuntimeError(f'input changed during transfer: {path}')
                if index % 20 == 0:
                    print(f'STREAM members={index}/{len(rows)+1} source_bytes={writer.n}', flush=True)
            info = tarfile.TarInfo('MANIFEST.json')
            info.size, info.mode, info.uid, info.gid, info.mtime = len(manifest_bytes), 0o600, 0, 0, 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(manifest_bytes))
        ssh.stdin.close()
        ssh.wait()
        out = ssh.stdout.read() if ssh.stdout else b''
        err = ssh.stderr.read() if ssh.stderr else b''
        if ssh.returncode:
            raise RuntimeError(f'receiver failed {ssh.returncode}: {err.decode(errors="replace")}')
        print(f'SOURCE_STREAM bytes={writer.n} sha256={writer.h.hexdigest()}', flush=True)
        print(out.decode('utf-8', errors='replace'), end='', flush=True)
        return 0
    finally:
        if ssh.poll() is None:
            ssh.kill()
            ssh.wait()
        if ssh.stdin and not ssh.stdin.closed:
            ssh.stdin.close()


def shutil_disk_free(path: Path) -> int:
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ARCHIVE_FAILED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
