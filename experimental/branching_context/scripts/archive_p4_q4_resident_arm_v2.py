#!/usr/bin/env python3
"""Prepare or stream the frozen P4 Q4 resident-return archive.

Default behavior is read-only preflight. Transfer requires --stream-reviewed.
No local tar spool is created; the receiver publishes only after validating all
streamed member bytes. This archive is evidence, not a standalone rerun bundle.
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

RUN = Path('/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01')
ART = Path('/home/dflanag3/spatial-artifacts')
SOURCE_CHECK = ART / 'p4-q4-resident-source-check-20261015-01'
INNER = RUN / 'command-receipts/resident'
OUTER_OK = ART / 'p4-q4-resident-start-wrapper-v3-20261015-01'
OUTER_FAIL1 = ART / 'p4-q4-resident-start-wrapper-20261015-01'
OUTER_FAIL2 = ART / 'p4-q4-resident-start-wrapper-v2-20261015-01'
PREFLIGHT = ART / 'p4-q4-resident-preflight-v3-20261015-01'
Q4_DISK_RESULT = RUN / 'disk-parent/output/result.json'
ARCHIVE = '/srv/data/git/private_context_cow.evidence/20261015-p4-q4-resident-return/p4-q4-resident-return-20261015.tar'
RECEIVER = '/srv/data/git/private_context_cow.evidence/receive_verified_archive_p4q4_20261015.py'
RECEIVER_SHA256 = 'a3012c68b9010412a84b309b539c608fe18710d962c02cd8800ef68efcf0156e'
RESULT_SHA256 = '0c160914ed28406bd2cfee935726630e1cae974c0c8f5c4bb83f2e9baee2446e'
DISK_RESULT_SHA256 = '76438074db2af7117f79764de759cb71363d08185bf24353225783ccc5ccd9f0'
SOURCE_MANIFEST_SHA256 = '5414ce7de30f5e2cc6c51133b3e07dcee4672253dca67ee6fdf9b6b13a31d500'
SOURCE_CLOSURE_SHA256 = 'fc8b5d43d31349d517850e41515604d79e5dc032b836776e5768d5c9c3dddadb'
START_V3_SHA256 = 'ee4dc60c44d9250da11ac7e46274d0c81057a22cfa948f923d1cc2a0e9c033d6'
PREFLIGHT_SCRIPT_SHA256 = '55363ed521ce91e7d97a6c92cc9cc1a60e672553863ee1b8f8f584a2791c9fba'
PIN_VERIFY_SHA256 = '646becefff5af8a3c46e7ab5332718b0a5ff69b7f036832ff5794a0aa469beed'
PIN_TEST_SHA256 = '4307eb630dc6d71da857fee30cc9fb3c104ef3f3a5a0ec9a41b5deea3fa39289'
DISK_ARCHIVE = '/srv/data/git/private_context_cow.evidence/20261015-p4-q4-disk-return/p4-q4-disk-return-20261015.tar'
DISK_ARCHIVE_SHA256 = '51e0015403e6772545ffd90e8b2b813c9899102f61c6318643505501bb2edc78'
Q4_ARCHIVE = '/srv/data/git/private_context_cow.evidence/20261015-p5-q4-full96/p5-q4-full96-20261015.tar'
Q4_ARCHIVE_SHA256 = '5e7339ce9d751afe64e5bf166c5c3f0ea6b7cc62255cacca7021ee0f8771d0ce'
Q4_ARCHIVE_BYTES = 41838049280
PROFILE_ID = '9b23df373eb42687a2eca77fb99a22409ada0ab4c3edac732da2674bd23d3763'


def digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256(); size = 0
    with path.open('rb', buffering=0) as f:
        before = os.fstat(f.fileno())
        while block := f.read(8 * 1024 * 1024):
            size += len(block); h.update(block)
        after = os.fstat(f.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != after.st_size:
        raise RuntimeError(f'input changed while hashing: {path}')
    return size, h.hexdigest()


def pin(path: Path, expected: str) -> None:
    size, got = digest(path)
    if got != expected:
        raise RuntimeError(f'hash mismatch {path}: {got} != {expected}')
    print(f'PIN {path} size={size} sha256={got}', flush=True)


def add_tree(rows: list[tuple[str, Path]], prefix: str, root: Path) -> None:
    if not root.is_dir():
        raise RuntimeError(f'missing archive tree: {root}')
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise RuntimeError(f'symlink rejected: {p}')
        if p.is_file():
            rows.append((f'{prefix}/{p.relative_to(root).as_posix()}', p))


def receipt_ok(path: Path, command: int, wrapper: int) -> None:
    if (path / 'terminal-status.txt').read_text().strip() != 'COMPLETE':
        raise RuntimeError(f'incomplete receipt: {path}')
    if (path / 'command.exit').read_text().strip() != str(command) or (path / 'wrapper.exit').read_text().strip() != str(wrapper):
        raise RuntimeError(f'unexpected numeric exits: {path}')


def build_entries() -> tuple[list[tuple[str, Path]], bytes]:
    rows: list[tuple[str, Path]] = []
    add_tree(rows, 'resident-arm/output', RUN / 'resident-parent/output')
    add_tree(rows, 'resident-arm/inner-command-receipt', INNER)
    add_tree(rows, 'resident-arm/outer-command-receipt', OUTER_OK)
    add_tree(rows, 'resident-arm/failed-v1-outer-receipt', OUTER_FAIL1)
    add_tree(rows, 'resident-arm/failed-v2-outer-receipt', OUTER_FAIL2)
    add_tree(rows, 'resident-arm/preflight-only-command-receipt', PREFLIGHT)
    # Keep source-closure provenance and reviewed launch artifacts, excluding
    # generated __pycache__ and unrelated temporary files.
    src_files = {
        'input/manifest.json': SOURCE_CHECK / 'input/manifest.json',
        'input/verify.py': SOURCE_CHECK / 'input/verify.py',
        'input/p4-q4-resident-start-v2-2026-10-15.sh': SOURCE_CHECK / 'input/p4-q4-resident-start-v2-2026-10-15.sh',
        'input/p4-q4-resident-start-v3-2026-10-15.sh': SOURCE_CHECK / 'input/p4-q4-resident-start-v3-2026-10-15.sh',
        'input/p4-q4-resident-preflight-v3-2026-10-15.sh': SOURCE_CHECK / 'input/p4-q4-resident-preflight-v3-2026-10-15.sh',
        'input/p4-q4-resident-verify-build-pins-v2.py': SOURCE_CHECK / 'input/p4-q4-resident-verify-build-pins-v2.py',
        'input/test_p4_q4_resident_build_pins_v2.py': SOURCE_CHECK / 'input/test_p4_q4_resident_build_pins_v2.py',
        'receipt/source-closure.json': SOURCE_CHECK / 'receipt/source-closure.json',
    }
    for rel, path in src_files.items():
        if not path.is_file() or path.is_symlink(): raise RuntimeError(f'missing regular source evidence: {path}')
        rows.append((f'resident-arm/source-closure/{rel}', path))
    add_tree(rows, 'resident-arm/source-check-receipts', SOURCE_CHECK / 'run')
    add_tree(rows, 'resident-arm/pin-control-test-receipts', SOURCE_CHECK / 'cpu-controls-run')
    rows.append(('resident-arm/reference-disk-arm/result.json', Q4_DISK_RESULT))
    rows.sort(key=lambda r: r[0])
    names = [name for name, _ in rows]
    if len(names) != len(set(names)): raise RuntimeError('duplicate archive member name')

    result = json.loads((RUN / 'resident-parent/output/result.json').read_text())
    if result.get('status') != 'PASS' or result.get('arm') != 'resident' or result.get('measured_requests') != 9: raise RuntimeError('resident result status/scope mismatch')
    if result.get('native_exit_codes') != [0] or result.get('source_checkpoints_unchanged') is not True: raise RuntimeError('resident native/checkpoint invariant failed')
    latency = result.get('latency_summary', {})
    if latency.get('actual_output_tokens') != [128] * 9 or latency.get('eos_completed_requests') != 0: raise RuntimeError('resident bounded-output evidence mismatch')
    if result.get('profile_identity', {}).get('execution_profile_id') != PROFILE_ID: raise RuntimeError('resident profile mismatch')
    pin(RUN / 'resident-parent/output/result.json', RESULT_SHA256)
    pin(Q4_DISK_RESULT, DISK_RESULT_SHA256)
    pin(SOURCE_CHECK / 'input/manifest.json', SOURCE_MANIFEST_SHA256)
    pin(SOURCE_CHECK / 'receipt/source-closure.json', SOURCE_CLOSURE_SHA256)
    pin(SOURCE_CHECK / 'input/p4-q4-resident-start-v3-2026-10-15.sh', START_V3_SHA256)
    pin(SOURCE_CHECK / 'input/p4-q4-resident-preflight-v3-2026-10-15.sh', PREFLIGHT_SCRIPT_SHA256)
    pin(SOURCE_CHECK / 'input/p4-q4-resident-verify-build-pins-v2.py', PIN_VERIFY_SHA256)
    pin(SOURCE_CHECK / 'input/test_p4_q4_resident_build_pins_v2.py', PIN_TEST_SHA256)
    receipt_ok(INNER, 0, 0); receipt_ok(OUTER_OK, 0, 0); receipt_ok(PREFLIGHT, 0, 0)
    receipt_ok(OUTER_FAIL1, 1, 1); receipt_ok(OUTER_FAIL2, 1, 1)
    closure = json.loads((SOURCE_CHECK / 'receipt/source-closure.json').read_text())
    if closure.get('status') != 'PASS' or closure.get('expected_file_count') != 16 or closure.get('candidate_file_count') != 16 or closure.get('mismatch_count') != 0: raise RuntimeError('source closure does not prove 16/16 exact source match')
    if 'Ran 3 tests' not in (SOURCE_CHECK / 'cpu-controls-run/stdout.log').read_text(errors='replace') and 'Ran 3 tests' not in (SOURCE_CHECK / 'cpu-controls-run/stderr.log').read_text(errors='replace'):
        raise RuntimeError('pin verifier CPU receipt does not show three executed tests')
    refs = {
        'disk_return_archive': {'path': DISK_ARCHIVE, 'sha256': DISK_ARCHIVE_SHA256},
        'q4_campaign_archive': {'path': Q4_ARCHIVE, 'sha256': Q4_ARCHIVE_SHA256, 'bytes': Q4_ARCHIVE_BYTES},
        'source_commit': '24b4fbbf85deec2b3c4beb8cc8af389c730ea32c',
        'source_manifest_sha256': SOURCE_MANIFEST_SHA256,
        'source_closure_receipt_sha256': SOURCE_CLOSURE_SHA256,
        'bundle_scope': 'Resident-return evidence with unchanged build/model/Q4 campaign closure referenced by prior verified private archives; not a standalone rerun bundle.',
    }
    refs_bytes = (json.dumps(refs, sort_keys=True, separators=(',', ':')) + '\n').encode()
    # Generated member is content-addressed by this stream's MANIFEST and does
    # not depend on mutable external evidence.
    return rows, refs_bytes


class HashingWriter:
    def __init__(self, f: BinaryIO): self.f, self.h, self.n = f, hashlib.sha256(), 0
    def write(self, b: bytes) -> int:
        n = self.f.write(b)
        if n != len(b): raise OSError('short SSH receiver write')
        self.h.update(b); self.n += n; return n
    def flush(self) -> None: self.f.flush()


class HashingReader:
    def __init__(self, f: BinaryIO): self.f, self.h, self.n = f, hashlib.sha256(), 0
    def read(self, size: int = -1) -> bytes:
        b = self.f.read(size); self.h.update(b); self.n += len(b); return b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream-reviewed', action='store_true', help='stream only after separate parent review')
    args = parser.parse_args()
    rows, refs_bytes = build_entries()
    records = []
    for name, path in rows:
        size, sha = digest(path); records.append({'path': name, 'size': size, 'sha256': sha})
    records.append({'path': 'resident-arm/verified-archive-references.json', 'size': len(refs_bytes), 'sha256': hashlib.sha256(refs_bytes).hexdigest()})
    manifest = {'schema': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'bundle_scope': 'P4 Q4 resident-return evidence; not standalone rerun bundle', 'unchanged_native_model_closure': 'Referenced by verified disk and Q4 campaign archives; no model or binary payload duplicated here.', 'files': records}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()
    target_probe = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','box-git', 'test ! -e /srv/data/git/private_context_cow.evidence/20261015-p4-q4-resident-return && test ! -e '+ARCHIVE+' && test ! -e '+ARCHIVE+'.receipt.json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if target_probe.returncode != 0: raise RuntimeError('archive target is not absent or cannot be checked')
    receiver = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','box-git','sha256sum',RECEIVER], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    fields = receiver.stdout.decode('ascii',errors='replace').split()
    if receiver.returncode or not fields or fields[0] != RECEIVER_SHA256: raise RuntimeError('private archive receiver SHA mismatch')
    total = sum(x['size'] for x in records)
    print(f'PREFLIGHT_ONLY members={len(records)} payload_bytes={total} tar_target={ARCHIVE} receiver_sha256={RECEIVER_SHA256}', flush=True)
    if not args.stream_reviewed:
        print('NO_TRANSFER: rerun with --stream-reviewed only after parent approval', flush=True)
        return 0
    ssh = subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','box-git','python3',RECEIVER,'--target',ARCHIVE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert ssh.stdin is not None
    writer = HashingWriter(ssh.stdin)
    try:
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            all_rows = rows + [('resident-arm/verified-archive-references.json', None)]
            for idx, (name, path) in enumerate(all_rows, 1):
                if path is None:
                    raw = io.BytesIO(refs_bytes); before = os.fstat(raw.fileno()) if False else None
                else:
                    st = path.stat(follow_symlinks=False)
                    if not stat.S_ISREG(st.st_mode): raise RuntimeError(f'nonregular archive input {path}')
                    raw = path.open('rb', buffering=0)
                    opened = os.fstat(raw.fileno())
                    if (opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns)!=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns): raise RuntimeError(f'input changed before read {path}')
                info = tarfile.TarInfo(name); info.size = records[idx-1]['size']; info.mode=0o600; info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
                hashed=HashingReader(raw); tar.addfile(info,hashed)
                if path is not None: raw.close()
                if hashed.n != records[idx-1]['size'] or hashed.h.hexdigest()!=records[idx-1]['sha256']: raise RuntimeError(f'input changed while archiving {name}')
                if path is not None:
                    after=path.stat(follow_symlinks=False)
                    if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns): raise RuntimeError(f'input changed during archive {path}')
                if idx % 50 == 0: print(f'STREAM members={idx}/{len(all_rows)} source_bytes={writer.n}',flush=True)
            info=tarfile.TarInfo('MANIFEST.json'); info.size=len(manifest_bytes); info.mode=0o600; info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0; tar.addfile(info,io.BytesIO(manifest_bytes))
        ssh.stdin.close(); ssh.wait(); out=ssh.stdout.read() if ssh.stdout else b''; err=ssh.stderr.read() if ssh.stderr else b''
        if ssh.returncode: raise RuntimeError(f'receiver failed {ssh.returncode}: {err.decode(errors="replace")}')
        print(f'SOURCE_STREAM bytes={writer.n} sha256={writer.h.hexdigest()}',flush=True); print(out.decode('utf-8',errors='replace'),end='',flush=True)
        return 0
    finally:
        if ssh.poll() is None: ssh.kill(); ssh.wait()
        if ssh.stdin and not ssh.stdin.closed: ssh.stdin.close()

if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'ARCHIVE_FAILED: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True); raise SystemExit(2)