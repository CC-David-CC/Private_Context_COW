"""Bounded Q4 canonical branch-reference ENOSPC fault/retry harness.

CPU controls validate the evidence guards and receipt protocol. The real mode
requires a separately mounted, root-prepared private ext4 fixture and is only
launched by the designated controller after review.
"""
from __future__ import annotations

import argparse
import errno
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid
from urllib.parse import quote


class CanonicalEnospcError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def persist_receipt_before_release(receipt_path: Path, receipt: dict, release, *,
                                   canonical_st_dev: int) -> None:
    """Persist and verify the off-volume filler receipt before releasing bytes."""
    if Path(receipt_path).parent.stat().st_dev == canonical_st_dev:
        raise CanonicalEnospcError('filler receipt must be stored outside the bounded canonical volume')
    _atomic_json(receipt_path, receipt)
    try:
        saved = json.loads(receipt_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise CanonicalEnospcError(f'filler receipt could not be read back: {exc}') from exc
    if saved != receipt:
        raise CanonicalEnospcError('filler receipt differs after durable publication')
    release()


def validate_run_directory(run_dir: Path, canonical_mount: Path) -> None:
    """Require a fresh run directory whose parent is outside the bounded Store."""
    run_dir = Path(run_dir).absolute()
    canonical_mount = Path(canonical_mount).absolute()
    try:
        run_dir.relative_to(canonical_mount)
    except ValueError:
        pass
    else:
        raise CanonicalEnospcError('result directory must be outside the bounded canonical Store volume')
    if not run_dir.parent.is_dir():
        raise CanonicalEnospcError('result directory parent must already exist before service startup')
    if run_dir.exists() or run_dir.is_symlink():
        raise CanonicalEnospcError('result directory must be new and must not be a symlink')


def invoke_probe(args, probe_runner=None) -> tuple[dict, int]:
    """Run injected orchestration and derive process status from its result."""
    runner = run_probe if probe_runner is None else probe_runner
    try:
        result = runner(args)
    except BaseException as exc:
        return dict(status='FAIL', error=f'{type(exc).__name__}: {exc}'), 1
    status = result.get('status') if type(result) is dict else None
    return result, exit_code_for_status(status)


def exit_code_for_status(status: str) -> int:
    """Only an explicitly completed PASS result may return success."""
    return 0 if status == 'PASS' else 1


def validate_mount_layout(receipt: dict, *, image_sha256: str, image_uuid: str,
                          image_inode: int, canonical_mount: str,
                          checkpoint_directory: str) -> None:
    """Require canonical metadata and checkpoints to live on their pinned volumes."""
    if type(receipt) is not dict:
        raise CanonicalEnospcError('mount receipt is malformed')
    canonical = receipt.get('canonical') or {}
    checkpoint = receipt.get('checkpoint') or {}
    image = receipt.get('image') or {}
    if (canonical.get('mountpoint') != canonical_mount
            or checkpoint.get('mountpoint') != checkpoint_directory
            or canonical.get('filesystem') != 'ext4'
            or canonical.get('device') != image.get('loop_device')
            or not isinstance(canonical.get('st_dev'), int)
            or not isinstance(checkpoint.get('st_dev'), int)
            or canonical.get('st_dev') == checkpoint.get('st_dev')
            or image.get('sha256') != image_sha256
            or image.get('uuid') != image_uuid
            or image.get('inode') != image_inode
            or receipt.get('mount_namespace_verified') is not True
            or not isinstance(receipt.get('current_mount_namespace'), str)
            or not isinstance(receipt.get('parent_mount_namespace'), str)
            or re.fullmatch(r'mnt:\[[0-9]+\]', receipt['current_mount_namespace']) is None
            or re.fullmatch(r'mnt:\[[0-9]+\]', receipt['parent_mount_namespace']) is None
            or receipt.get('current_mount_namespace') == receipt.get('parent_mount_namespace')):
        raise CanonicalEnospcError('canonical Store and checkpoint volumes are not the exact isolated mounts')


def validate_reference_fault(observation: dict, *, operation_id: str, branch: str,
                              reference_path: str, old_reference_sha256: str,
                              target_snapshot: str) -> dict:
    """Require a real ENOSPC raised by the selected atomic branch-ref writer."""
    if type(observation) is not dict:
        raise CanonicalEnospcError('branch-reference writer observation is missing')
    message = observation.get('message')
    if (observation.get('operation_id') != operation_id
            or observation.get('branch') != branch
            or observation.get('branch_write') is not True
            or observation.get('reference_path') != reference_path
            or observation.get('old_reference_sha256') != old_reference_sha256
            or observation.get('reference_sha256_after') != old_reference_sha256
            or observation.get('exception_type') != 'OSError'
            or type(observation.get('errno')) is not int
            or observation.get('errno') != errno.ENOSPC
            or not isinstance(message, str)
            or '[Errno 28]' not in message
            or os.strerror(errno.ENOSPC) not in message
            or observation.get('stage_files_after') != []
            or observation.get('target_snapshot') != target_snapshot
            or type(observation.get('serialized_reference_bytes')) is not int
            or observation.get('serialized_reference_bytes') <= 0):
        raise CanonicalEnospcError('Observed failure was not kernel ENOSPC at the exact target branch-reference writer')
    return dict(operation_id=operation_id, branch=branch, target_snapshot=target_snapshot,
                serialized_reference_bytes=observation['serialized_reference_bytes'],
                old_reference_sha256=old_reference_sha256)


def validate_error_stream(events: list[dict], *, expected_errno: int = errno.ENOSPC) -> None:
    if type(events) is not list or any(type(row) is not dict for row in events):
        raise CanonicalEnospcError('fault event stream is malformed')
    terminal = [row for row in events if row.get('event') in ('error', 'failure')]
    if (len(terminal) != 1
            or any(row.get('event') in ('complete', 'durable') for row in events)
            or any(row.get('durable') is True or 'receipt' in row for row in events)
            or terminal[0].get('event') != 'error'
            or terminal[0].get('cancelled') is not False
            or not isinstance(terminal[0].get('error'), str)
            or f'[Errno {expected_errno}]' not in terminal[0]['error']
            or os.strerror(expected_errno) not in terminal[0]['error']):
        raise CanonicalEnospcError('faulted request must emit one original-errno error and no durable/complete acknowledgement')


def validate_retryable_failure(status: dict, old_head_before: dict, old_head_after: dict,
                               *, operation_id: str, branch: str, parent_snapshot: str) -> None:
    if (type(status) is not dict or status.get('state') != 'failed'
            or status.get('operation_id') != operation_id or status.get('branch') != branch
            or status.get('durable') is not False or status.get('publication_retryable') is not True
            or 'receipt' in status):
        raise CanonicalEnospcError('failed operation status is missing')
    if (type(old_head_before) is not dict or old_head_before.get('name') != branch
            or old_head_before.get('tip') != parent_snapshot or old_head_before != old_head_after):
        raise CanonicalEnospcError('failed branch publication changed the old branch head')


def validate_retry(response: dict, *, operation_id: str, target_snapshot: str,
                   branch: str, revision_before: int, moves_before: list, moves_after: list,
                   operation_status: dict, work_before: bytes, work_after: bytes) -> None:
    receipt = response.get('receipt') if type(response) is dict else None
    branch_record = receipt.get('branch') if type(receipt) is dict else None
    if (type(response) is not dict or response.get('event') != 'complete'
            or type(receipt) is not dict or receipt.get('snapshot') != target_snapshot
            or type(branch_record) is not dict or branch_record.get('name') != branch
            or branch_record.get('revision') != revision_before + 1
            or branch_record.get('tip') != target_snapshot):
        raise CanonicalEnospcError('same-operation retry did not complete')
    if type(moves_before) is not list or moves_after != moves_before + [operation_id]:
        raise CanonicalEnospcError('retry did not move exactly once')
    if (type(operation_status) is not dict or operation_status.get('operation_id') != operation_id
            or operation_status.get('branch') != branch or operation_status.get('state') != 'durable'
            or operation_status.get('durable') is not True or operation_status.get('receipt') != receipt):
        raise CanonicalEnospcError('retry operation status is not bound to its durable receipt')
    if type(work_before) is not bytes or work_after != work_before:
        raise CanonicalEnospcError('same-operation retry performed additional native work')


def validate_target_checkpoint(checkpoint: dict, *, target_snapshot: str) -> None:
    sha = checkpoint.get('payload_sha256') if type(checkpoint) is dict else None
    if (type(checkpoint) is not dict or checkpoint.get('snapshot') != target_snapshot
            or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)
            or type(checkpoint.get('payload_bytes')) is not int or checkpoint['payload_bytes'] <= 0
            or type(checkpoint.get('manifest_sha256')) is not str
            or not re.fullmatch('[0-9a-f]{64}', checkpoint['manifest_sha256'])):
        raise CanonicalEnospcError('prepared target checkpoint is absent or corrupt')


def validate_filler_receipt(receipt: dict, *, canonical_mount: str,
                            max_fill_bytes: int, canonical_st_dev: int | None = None) -> None:
    fill_path = Path(str(receipt.get('fill_path', ''))) if type(receipt) is dict else Path('.')
    if (type(receipt) is not dict or type(receipt.get('errno')) is not int
            or receipt.get('errno') != errno.ENOSPC
            or receipt.get('fill_path') is None
            or fill_path.parent.resolve() != Path(canonical_mount).resolve()
            or type(receipt.get('filled_bytes')) is not int or receipt['filled_bytes'] <= 0
            or receipt['filled_bytes'] > max_fill_bytes
            or type(receipt.get('allocated_bytes')) is not int or receipt['allocated_bytes'] <= 0
            or receipt['allocated_bytes'] > max_fill_bytes + 4096
            or receipt.get('available_bytes_after') != 0
            or receipt.get('filesystem_write_error') != '[Errno 28] No space left on device'
            or (canonical_st_dev is not None and receipt.get('canonical_st_dev') != canonical_st_dev)):
        raise CanonicalEnospcError('filler did not establish bounded-volume kernel ENOSPC')


def _mount_row(path: Path) -> dict:
    completed = subprocess.run(['findmnt', '-n', '-o', 'TARGET,SOURCE,FSTYPE', '-T', str(path)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    if completed.returncode != 0:
        raise CanonicalEnospcError(f'findmnt failed for {path}: {completed.stderr.strip()}')
    fields = completed.stdout.strip().split()
    if len(fields) != 3:
        raise CanonicalEnospcError(f'findmnt returned malformed topology for {path}')
    target, source, filesystem = fields
    return dict(target=target, source=source, filesystem=filesystem,
                st_dev=Path(path).stat().st_dev)


def live_mount_receipt(args) -> dict:
    canonical = _mount_row(args.canonical_mount)
    checkpoint = _mount_row(args.checkpoint_directory)
    namespace = os.readlink('/proc/self/ns/mnt')
    parent_ns = args.parent_mount_namespace
    if parent_ns.isdigit():
        parent_ns = f'mnt:[{parent_ns}]'
    source = canonical['source']
    loop_name = Path(source).name
    if not re.fullmatch(r'loop[0-9]+', loop_name):
        raise CanonicalEnospcError('canonical Store volume is not mounted from a loop device')
    backing_file = Path('/sys/class/block') / loop_name / 'loop' / 'backing_file'
    try:
        backing = Path('/' + backing_file.read_text(encoding='utf-8').strip().lstrip('/'))
        backing = backing.resolve(strict=True)
        image_stat = args.loop_image.stat()
    except OSError as exc:
        raise CanonicalEnospcError(f'cannot inspect exact loop image identity: {exc}') from exc
    if backing != args.loop_image.resolve(strict=True) or image_stat.st_ino != args.expected_image_inode:
        raise CanonicalEnospcError('loop device backing file/inode differs from the approved private image')
    uuid_result = subprocess.run(['blkid', '-p', '-s', 'UUID', '-o', 'value', '--', str(args.loop_image)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    if uuid_result.returncode != 0 or uuid_result.stdout.strip() != args.expected_image_uuid:
        raise CanonicalEnospcError('mounted ext4 UUID differs from the approved private image')
    receipt = dict(mount_namespace_verified=True, parent_mount_namespace=parent_ns,
        current_mount_namespace=namespace,
        image=dict(path=str(args.loop_image), inode=image_stat.st_ino,
                   sha256=_sha256(args.loop_image), uuid=uuid_result.stdout.strip(), loop_device=source),
        canonical=dict(mountpoint=canonical['target'], source=canonical['source'],
                       device=canonical['source'], filesystem=canonical['filesystem'], st_dev=canonical['st_dev']),
        checkpoint=dict(mountpoint=checkpoint['target'], source=checkpoint['source'],
                        device=checkpoint['source'], filesystem=checkpoint['filesystem'], st_dev=checkpoint['st_dev']))
    validate_mount_layout(receipt, image_sha256=receipt['image']['sha256'],
        image_uuid=args.expected_image_uuid, image_inode=args.expected_image_inode,
        canonical_mount=str(args.canonical_mount), checkpoint_directory=str(args.checkpoint_directory))
    if canonical['target'] != str(args.canonical_mount) or checkpoint['target'] != str(args.checkpoint_directory):
        raise CanonicalEnospcError('actual mount targets differ from exact CLI paths')
    if canonical['filesystem'] != 'ext4' or receipt['current_mount_namespace'] == parent_ns:
        raise CanonicalEnospcError('private canonical ext4 mount namespace is not active')
    return receipt


def _branch_reference_path(store: Path, branch: str) -> Path:
    refs = json.loads((store / 'refs.json').read_text(encoding='utf-8'))
    directory = refs.get('directory')
    if not isinstance(directory, str) or not re.fullmatch(r'references(?:-[0-9a-f]{64})?', directory):
        raise CanonicalEnospcError('Store refs.json names an invalid reference directory')
    path = store / directory / f'{branch}.json'
    if path.is_symlink() or not path.is_file():
        raise CanonicalEnospcError('target branch reference is missing or not a regular file')
    return path


def _fill_to_enospc(mount: Path, max_bytes: int, nonce: str) -> dict:
    path = mount / f'.q4-canonical-enospc-{nonce}.filler'
    if path.exists() or path.is_symlink():
        raise CanonicalEnospcError('filler path already exists')
    block = bytes(range(256)) * 4096
    total = 0
    observed = None
    fd = None
    identity = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        identity = os.fstat(fd)
        chunk_size = len(block)
        previous_available = None
        while True:
            remaining = max_bytes - total
            if remaining <= 0:
                raise CanonicalEnospcError('filler reached its configured byte cap before filesystem ENOSPC')
            count, write_error = _write_until_error(fd, block[:min(chunk_size, remaining)])
            total += count
            if write_error is None:
                continue
            if write_error.errno != errno.ENOSPC:
                raise write_error
            observed = write_error
            # ENOSPC on a large write can leave smaller allocatable extents. Flush
            # delayed allocation, then consume those extents without exceeding the cap.
            try:
                os.fsync(fd)
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
            stats = os.statvfs(mount)
            available = stats.f_bavail * stats.f_frsize
            if available == 0:
                break
            if available < 0:
                raise CanonicalEnospcError('filesystem reported negative available space')
            if chunk_size > 4096:
                chunk_size = 4096
            elif chunk_size > 1:
                chunk_size = 1
            elif previous_available == available:
                raise CanonicalEnospcError(
                    f'1-byte write returned ENOSPC with {available} available bytes remaining')
            previous_available = available
        os.close(fd)
        fd = None
        if observed is None or observed.errno != errno.ENOSPC:
            raise CanonicalEnospcError(f'bounded fill did not reach kernel ENOSPC: {observed!r}')
        dir_fd = os.open(mount, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        stats = os.statvfs(mount)
        available = stats.f_bavail * stats.f_frsize
        if available != 0:
            raise CanonicalEnospcError(f'filesystem still has {available} available bytes after ENOSPC')
        after = path.stat(follow_symlinks=False)
        if (after.st_dev, after.st_ino) != (identity.st_dev, identity.st_ino):
            raise CanonicalEnospcError('filler file identity changed during the write')
        return dict(fill_path=str(path), filled_bytes=total,
            allocated_bytes=after.st_blocks * 512, errno=observed.errno,
            filesystem_write_error=str(observed), available_bytes_after=available,
            st_dev=after.st_dev, inode=after.st_ino)
    except BaseException as primary:
        cleanup_errors = []
        if fd is not None:
            try:
                os.close(fd)
            except BaseException as exc:
                cleanup_errors.append(f'close: {type(exc).__name__}: {exc}')
        if identity is not None:
            _cleanup_owned_filler_path(path, mount, identity, cleanup_errors)
        if cleanup_errors and hasattr(primary, 'add_note'):
            primary.add_note('owned filler cleanup failures: ' + '; '.join(cleanup_errors))
        raise


def _cleanup_owned_filler_path(path: Path, mount: Path, identity, errors: list[str]) -> None:
    """Remove only the file created by this invocation, including on early write/fsync errors."""
    try:
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            errors.append('filler path identity changed; refusing cleanup')
            return
        path.unlink()
        dir_fd = os.open(mount, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except FileNotFoundError:
        pass
    except BaseException as exc:
        errors.append(f'unlink/fsync: {type(exc).__name__}: {exc}')


def _write_until_error(fd: int, data: bytes) -> tuple[int, OSError | None]:
    """Consume short writes until all bytes are accepted or the kernel errors."""
    view = memoryview(data)
    total = 0
    while view:
        try:
            written = os.write(fd, view)
        except OSError as exc:
            return total, exc
        if written <= 0:
            raise OSError(errno.EIO, 'zero-length filler write')
        total += written
        view = view[written:]
    return total, None


def _stop_service_if_process_exists(service, result: dict, receipt_key: str) -> bool:
    """Stop a child even when start() raised after Popen but before readiness."""
    if getattr(service, 'process', None) is None:
        return False
    try:
        result[receipt_key] = service.stop()
        return True
    except BaseException as exc:
        result[receipt_key + '_error'] = f'{type(exc).__name__}: {exc}'
        return True


def _require_request_waiting_at_release(future) -> None:
    if future.done():
        raise CanonicalEnospcError('request escaped the writer gate before controlled release')


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if type(row) is not dict:
            raise CanonicalEnospcError('control event log contains a non-object')
        values.append(row)
    return values


def _wait_gate(event_path: Path, future, timeout: float, progress) -> dict:
    deadline = time.monotonic() + timeout
    last_heartbeat = 0.0
    while time.monotonic() < deadline:
        rows = _events(event_path)
        found = [row for row in rows if row.get('kind') == 'branch-reference-gate']
        if found:
            if len(found) != 1:
                raise CanonicalEnospcError('writer emitted duplicate branch-reference gates')
            return found[0]
        if future.done():
            raise CanonicalEnospcError('generation ended before the selected branch-reference writer gate')
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            progress.write_progress()
            progress.check()
            last_heartbeat = now
        time.sleep(.05)
    raise TimeoutError('generation did not reach the target branch-reference writer before deadline')


def _append_jsonl(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError('short JSONL event write')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


class EventRecorder:
    def __init__(self, path: Path):
        self.path = path

    def event(self, operation_id, event):
        _append_jsonl(self.path, dict(operation_id=operation_id, event=event))


def _safe_http_events(events: list[dict]) -> list[dict]:
    """Retain IDs, snapshots, and counters while omitting generated prose."""
    result = []
    for row in events:
        item = dict(row)
        item.pop('text', None)
        answer = item.get('answer')
        if type(answer) is dict:
            answer = dict(answer)
            answer.pop('text', None)
            item['answer'] = answer
        result.append(item)
    return result


def _reference_wrapper_source(control_dir: Path, target_branch: str, target_operation: str,
                              store: Path) -> str:
    """Build a child-only observer around Store's unchanged branch ref writer."""
    config = json.dumps(dict(control_dir=str(control_dir), branch=target_branch,
                              operation=target_operation,
                              store=str(store)), ensure_ascii=False)
    return f'''import hashlib,json,os,sys,threading,time,traceback
from pathlib import Path
cfg=json.loads({config!r})
control=Path(cfg["control_dir"])
branch_name=cfg["branch"]
target_operation=cfg["operation"]
store_root=Path(cfg["store"])
local=threading.local()
def append(value):
    raw=(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\\n").encode()
    fd=os.open(control/"writer-events.jsonl",os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        view=memoryview(raw)
        while view:
            n=os.write(fd,view)
            if n<=0: raise OSError("short writer-event append")
            view=view[n:]
        os.fsync(fd)
    finally: os.close(fd)
from context_time_travel.store import Store,_json_bytes
from context_time_travel.coordinator import ContextCoordinator
original_ref=Store._publish_reference_json
original_coordinator=ContextCoordinator._publish
def coordinator_publish(self,model,job,prepared,key,pin,lease):
    local.operation=job.request.operation
    local.branch=job.branch
    try: return original_coordinator(self,model,job,prepared,key,pin,lease)
    finally:
        local.operation=None
        local.branch=None
def observed_ref(self,path,value,*,branch=False):
    path=Path(path)
    is_target=(branch is True and getattr(local,"branch",None)==branch_name
               and getattr(local,"operation",None)==target_operation
               and not (control/"gate-consumed.json").exists()
               and path.resolve()==(self.references/(branch_name+".json")).resolve())
    if not is_target: return original_ref(self,path,value,branch=branch)
    op=getattr(local,"operation",None)
    raw=_json_bytes(value)
    old=path.read_bytes()
    prior=json.loads(old)
    target=(value.get("head") or {{}}).get("tip")
    append({{"kind":"branch-reference-gate","operation_id":op,"branch":branch_name,
        "branch_write":True,"reference_path":str(path.resolve()),
        "old_reference_sha256":hashlib.sha256(old).hexdigest(),
        "serialized_reference_bytes":len(raw),"target_snapshot":target,
        "old_head":prior.get("head"),"new_head":value.get("head")}})
    deadline=time.monotonic()+900
    release=control/"release.json"
    abort=control/"abort.json"
    while time.monotonic()<deadline and not release.exists() and not abort.exists(): time.sleep(.025)
    if abort.exists(): raise RuntimeError("parent aborted branch-reference ENOSPC gate")
    if not release.exists(): raise TimeoutError("parent did not release branch-reference gate")
    consumed_fd=os.open(control/"gate-consumed.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try: os.fsync(consumed_fd)
    finally: os.close(consumed_fd)
    try:
        result=original_ref(self,path,value,branch=branch)
        append({{"kind":"branch-reference-published","operation_id":op,"branch":branch_name,
            "reference_path":str(path.resolve()),"reference_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}})
        return result
    except OSError as exc:
        try: stages=sorted(p.name for p in path.parent.iterdir() if p.name.startswith(".stage-"))
        except OSError: stages=["<directory-read-failed>"]
        try: new_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError: new_sha=None
        append({{"kind":"branch-reference-write-error","operation_id":op,"branch":branch_name,
            "branch_write":True,"reference_path":str(path.resolve()),
            "old_reference_sha256":hashlib.sha256(old).hexdigest(),"reference_sha256_after":new_sha,
            "serialized_reference_bytes":len(raw),"target_snapshot":target,
            "exception_type":type(exc).__name__,"errno":exc.errno,"message":str(exc),
            "stage_files_after":stages,"old_head":prior.get("head"),"new_head":value.get("head")}})
        raise
Store._publish_reference_json=observed_ref
ContextCoordinator._publish=coordinator_publish
sys.argv=["serve.cow_http",*sys.argv[1:]]
from serve.cow_http import main
main()
'''


def _write_wrapper(path: Path, control_dir: Path, target_branch: str,
                   target_operation: str, store: Path) -> None:
    content = _reference_wrapper_source(control_dir, target_branch, target_operation, store)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _pinned_path(value, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink():
        raise CanonicalEnospcError(f'{label} path must not be a symlink')
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise CanonicalEnospcError(f'{label} path ancestors must not redirect through symlinks')
    return resolved


def _source_head(store: Path, branch: str) -> dict:
    refs = json.loads((store / 'refs.json').read_text(encoding='utf-8'))
    directory = refs.get('directory')
    if not isinstance(directory, str) or not re.fullmatch(r'references(?:-[0-9a-f]{64})?', directory):
        raise CanonicalEnospcError('source Store has invalid reference directory metadata')
    branch_path = store / directory / f'{branch}.json'
    if branch_path.is_symlink() or not branch_path.is_file():
        raise CanonicalEnospcError(f'source branch reference is missing: {branch}')
    branch_record = json.loads(branch_path.read_text(encoding='utf-8'))
    head = branch_record.get('head')
    if (type(head) is not dict or head.get('revision') is None
            or not isinstance(head.get('tip'), str)):
        raise CanonicalEnospcError('source branch head is malformed')
    return dict(path=branch_path, sha256=_sha256(branch_path), head=head,
                reference_directory=directory)


def _metadata_fingerprint(store: Path) -> dict:
    rows = {}
    for path in sorted(store.rglob('*')):
        relative = path.relative_to(store)
        if relative.parts and relative.parts[0] == 'checkpoints':
            continue
        if path.is_symlink():
            raise CanonicalEnospcError(f'canonical metadata contains symlink {relative}')
        if path.is_file():
            rows[relative.as_posix()] = dict(bytes=path.stat().st_size, sha256=_sha256(path))
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return dict(files=rows, sha256=hashlib.sha256(encoded).hexdigest())


def _checkpoint_details(store: Path, snapshot: str, fault_helpers) -> dict:
    files = fault_helpers._snapshot_file_fingerprint(store, snapshot)
    manifest_name = f'{snapshot}.json'
    manifest_path = store / 'checkpoints' / manifest_name
    envelope = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest = envelope.get('manifest') if type(envelope) is dict else None
    payload_sha = manifest.get('payload_sha256') if type(manifest) is dict else None
    payload_name = f'{payload_sha}.state' if isinstance(payload_sha, str) else None
    payload_entry = files.get(payload_name) if payload_name else None
    manifest_entry = files.get(manifest_name)
    if not payload_entry or not manifest_entry:
        raise CanonicalEnospcError('checkpoint files did not pass the existing strict fingerprint helper')
    return dict(snapshot=snapshot, payload_sha256=payload_sha,
        payload_bytes=payload_entry['bytes'], manifest_sha256=manifest_entry['sha256'], files=files)


def preflight(args) -> dict:
    """Validate all pins and live mount identity without starting a model/service."""
    if not sys.platform.startswith('linux'):
        raise CanonicalEnospcError('real canonical ENOSPC run requires Linux')
    if (args.ram_gib != 96 or args.disk_gib != 128 or args.max_tokens != 16
            or not 1 <= args.max_fill_mib <= 64):
        raise CanonicalEnospcError('Q4 canonical ENOSPC test is pinned to 96/128 GiB pools, 16 outputs, and <=64 MiB fill')
    run_dir = Path(args.run_dir).expanduser().absolute()
    canonical_mount = _pinned_path(args.canonical_mount, 'Canonical mount')
    store = _pinned_path(args.store, 'Store')
    checkpoint_directory = _pinned_path(args.checkpoint_directory, 'Checkpoint directory')
    original_store = _pinned_path(args.original_store, 'Original source Store')
    runtime_root = _pinned_path(args.runtime, 'Runtime root')
    api_src = _pinned_path(args.api_src, 'API source')
    profile_path = _pinned_path(args.profile, 'Profile')
    loop_image = _pinned_path(args.loop_image, 'Loop image')
    if store != canonical_mount or checkpoint_directory != store / 'checkpoints':
        raise CanonicalEnospcError('Store and checkpoint mount CLI paths must be the exact mounted layout')
    if original_store == store or original_store.is_relative_to(store):
        raise CanonicalEnospcError('original Store must remain outside the disposable mounted Store')
    if _sha256(profile_path) != args.expected_profile_sha256:
        raise CanonicalEnospcError('profile SHA-256 differs from the approved execution profile')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    execution = profile.get('execution')
    if type(execution) is not dict:
        raise CanonicalEnospcError('execution profile is malformed')
    native_binary = Path(execution['sut_command'][0]).resolve(strict=True)
    if _sha256(native_binary) != args.expected_native_sha256:
        raise CanonicalEnospcError('native executable SHA-256 differs from the selected Q4 binary')
    if args.expected_image_inode != loop_image.stat().st_ino:
        raise CanonicalEnospcError('private ext4 image inode differs from the prepared fixture')
    mount = live_mount_receipt(args)
    if store / 'refs.json' != canonical_mount / 'refs.json' or not (store / 'refs.json').is_file():
        raise CanonicalEnospcError('mounted canonical Store metadata is absent')
    if not checkpoint_directory.is_dir():
        raise CanonicalEnospcError('separate external checkpoint volume is not mounted')
    validate_run_directory(run_dir, canonical_mount)

    script_dir = runtime_root / 'experimental' / 'branching_context' / 'scripts'
    if not script_dir.is_dir():
        raise CanonicalEnospcError('runtime lacks the existing client recovery helpers')
    sys.path[:0] = [str(script_dir), str(runtime_root), str(api_src)]
    import run_q4_publication_fault as fault_helpers
    helper_profile = json.loads(profile_path.read_text(encoding='utf-8'))
    closure = fault_helpers._runtime_closure(helper_profile, runtime_root)
    fault_helpers._ensure_q4_profile(profile)
    source = _source_head(store, args.branch)
    if source['head']['tip'] != args.expected_source_snapshot:
        raise CanonicalEnospcError('mounted source branch does not point to the pinned original snapshot')
    source_object = fault_helpers._snapshot_record_fingerprint(store, args.expected_source_snapshot)
    source_cp_original = _checkpoint_details(original_store, args.expected_source_snapshot, fault_helpers)
    source_cp_mounted = _checkpoint_details(store, args.expected_source_snapshot, fault_helpers)
    if source_cp_original['files'] != source_cp_mounted['files']:
        raise CanonicalEnospcError('mounted source checkpoint bytes differ from the untouched original Store')
    return dict(profile_path=str(profile_path), profile_sha256=_sha256(profile_path),
        profile_id=profile.get('execution_profile_id'), native_binary=str(native_binary),
        native_sha256=_sha256(native_binary), runtime_root=str(runtime_root), api_src=str(api_src),
        runtime_source_closure=closure, canonical_mount=mount, run_directory=str(run_dir),
        original_store=str(original_store), original_store_metadata=_metadata_fingerprint(original_store),
        mounted_store_metadata=_metadata_fingerprint(store), branch=args.branch,
        source_head=source['head'], source_reference_sha256=source['sha256'],
        source_snapshot_record_sha256=source_object, source_checkpoint=source_cp_mounted,
        target_checkpoint_directory=str(checkpoint_directory), pool_ram_gib=args.ram_gib,
        pool_disk_gib=args.disk_gib, fault_max_fill_bytes=args.max_fill_mib * 1024**2,
        max_tokens=args.max_tokens)


def _history_sha(fault_helpers, history: dict) -> str:
    return fault_helpers._canonical_history_sha(history)


def _request_events(client, route, body):
    try:
        events, disconnected = client.post_stream(route, body)
        if disconnected:
            raise CanonicalEnospcError('request stream disconnected before terminal event')
        return dict(events=events)
    except BaseException as exc:
        return dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())


def _terminal_status(fault_helpers, client, operation_id, branch, timeout):
    return fault_helpers._operation_status(client, operation_id, branch,
        time.monotonic() + timeout)


def _remove_owned_filler(receipt: dict, canonical_mount: Path) -> None:
    path = Path(receipt['fill_path'])
    if path.parent.resolve(strict=True) != canonical_mount.resolve(strict=True):
        raise CanonicalEnospcError('refusing to remove filler outside the exact canonical mount root')
    stat = path.stat(follow_symlinks=False)
    if (stat.st_dev != receipt.get('st_dev') or stat.st_ino != receipt.get('inode')
            or path.is_symlink() or not path.is_file()):
        raise CanonicalEnospcError('filler identity changed before controlled capacity release')
    path.unlink()
    fd = os.open(canonical_mount, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_probe(args) -> dict:
    """Run one real branch-reference ENOSPC and same-operation retry."""
    if not sys.platform.startswith('linux'):
        raise CanonicalEnospcError('real canonical ENOSPC probe requires Linux')
    run_dir = Path(args.run_dir).expanduser().absolute()
    pins = preflight(args)
    run_dir.mkdir(mode=0o700)
    os.chmod(run_dir, 0o700)
    control = run_dir / 'control'
    control.mkdir(mode=0o700)
    result_path = run_dir / 'result.json'
    gate_path = control / 'writer-events.jsonl'
    gate_path.touch(mode=0o600, exist_ok=False)
    coordinator_events_path = run_dir / 'coordinator-events.jsonl'
    coordinator_events_path.touch(mode=0o600, exist_ok=False)
    event_recorder = EventRecorder(coordinator_events_path)
    native_trace = run_dir / 'native-trace.jsonl'
    native_trace.touch(mode=0o600, exist_ok=False)
    wrapper = run_dir / 'server_wrapper.py'
    nonce = uuid.uuid4().hex
    target_branch = 'Q4NoSpace' + nonce[:16]
    operation_id = 'op-q4-canonical-enospc-' + nonce
    _write_wrapper(wrapper, control, target_branch, operation_id, Path(args.store))
    runtime = Path(args.runtime).resolve(strict=True)
    api_src = Path(args.api_src).resolve(strict=True)
    script_dir = runtime / 'experimental' / 'branching_context' / 'scripts'
    if not script_dir.is_dir():
        raise CanonicalEnospcError('pinned runtime lacks the reviewed recovery helpers')
    sys.path[:0] = [str(script_dir), str(runtime), str(api_src)]
    import run_q4_publication_fault as fault_helpers
    from run_cow_client_recovery import HttpClient, ProgressTracker, ServiceProcess, _ready_url, _fork

    profile = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    command = [sys.executable, str(wrapper), '--profile', str(args.profile),
        '--store', str(args.store), '--host', '127.0.0.1', '--port', '0', '--text',
        '--ram-gib', str(args.ram_gib), '--disk-gib', str(args.disk_gib),
        '--pool-trace-output', str(run_dir / 'pool-trace.json')]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(runtime), str(api_src)))
    env['CTT_NATIVE_TRACE'] = str(native_trace)
    progress = ProgressTracker(run_dir, 'q4-canonical-reference-enospc')
    service = ServiceProcess(command, cwd=runtime, env=env,
        stdout_path=run_dir / 'server.stdout.log', stderr_path=run_dir / 'server.stderr.log',
        timeout=args.timeout, progress=progress)
    result = dict(schema='q4-canonical-reference-enospc-v1', status='RUNNING',
        scope='canonical Store reference-volume ENOSPC; checkpoint payload volume remains separate',
        pins=pins, runner_sha256=_sha256(Path(__file__).resolve()), wrapper=str(wrapper),
        wrapper_sha256=_sha256(wrapper), operation_id=operation_id, target_branch=target_branch)
    filler = None
    client = None
    source_head = source_history = source_cp = None
    target_ref_path = None
    service_started = False
    service_stop_attempted = False
    try:
        progress.start()
        progress.phase('starting_service')
        ready = service.start()
        service_started = True
        result['server_pid'] = service.process.pid
        result['ready'] = ready
        client = HttpClient(_ready_url(ready), timeout=args.timeout, progress=progress)
        code, health = client.get('/health')
        if code != 200 or health.get('service') != 'context-cow':
            raise CanonicalEnospcError(f'owned service health check failed: HTTP {code}')
        if ready.get('execution_profile_id') != profile.get('execution_profile_id'):
            raise CanonicalEnospcError('ready service profile differs from pinned Q4 profile')
        native_pids = fault_helpers._children(service.process.pid)
        if len(native_pids) != 1:
            raise CanonicalEnospcError(f'expected one native child after service startup, got {native_pids}')
        native_pid = native_pids[0]
        native_info = fault_helpers._read_proc(native_pid)
        native_binary = Path(profile['execution']['sut_command'][0]).resolve(strict=True)
        if (native_info.get('parent_pid') != service.process.pid
                or Path(native_info.get('exe', '')).resolve() != native_binary):
            raise CanonicalEnospcError('native child identity differs from the pinned Q4 executable')
        progress.native_started(native_pid)
        result['native_process_before'] = native_info

        source_head, source_history = fault_helpers._parse_head_and_history(client, args.branch)
        source_snapshot = source_head['tip']
        source_cp = fault_helpers._snapshot_file_fingerprint(Path(args.store), source_snapshot)
        source_object_sha = fault_helpers._snapshot_record_fingerprint(Path(args.store), source_snapshot)
        result['source_before'] = dict(head=source_head, history_sha256=_history_sha(fault_helpers, source_history),
            snapshot_record_sha256=source_object_sha, checkpoint_files=source_cp)

        progress.phase('forking_isolated_branch')
        fork = _fork(client, args.branch, source_snapshot, target_branch, 'op-fork-' + nonce)
        target_head, target_history = fault_helpers._parse_head_and_history(client, target_branch)
        if fork.get('tip') != source_snapshot or target_head.get('tip') != source_snapshot:
            raise CanonicalEnospcError('isolated target branch did not start at the selected source snapshot')
        target_ref_path = _branch_reference_path(Path(args.store), target_branch)
        old_ref_bytes = target_ref_path.read_bytes()
        old_ref_sha = hashlib.sha256(old_ref_bytes).hexdigest()
        old_moves = [row.get('operation_id') for row in target_history.get('moves', [])]
        route, body = fault_helpers._generation_body(target_history, target_head, target_branch,
            operation_id, args.max_tokens, 'Continue this branch to exercise durable publication.')
        result['fault_request'] = dict(route=route, operation_id=operation_id,
            parent_head=target_head, parent_history_sha256=_history_sha(fault_helpers, target_history),
            old_reference_path=str(target_ref_path), old_reference_sha256=old_ref_sha,
            parent_checkpoint_files=source_cp)

        progress.phase('request_waiting_at_exact_branch_reference_writer')
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='q4-enospc-http')
        future = executor.submit(_request_events, client, route, body)
        try:
            gate = _wait_gate(gate_path, future, args.timeout, progress)
            if (gate.get('operation_id') != operation_id or gate.get('branch') != target_branch
                    or gate.get('old_reference_sha256') != old_ref_sha
                    or gate.get('reference_path') != str(target_ref_path.resolve())):
                raise CanonicalEnospcError('writer gate is not bound to the exact selected branch operation/path')
            target_snapshot = gate.get('target_snapshot')
            if not isinstance(target_snapshot, str) or target_snapshot == source_snapshot:
                raise CanonicalEnospcError('writer gate did not identify a distinct prepared target snapshot')
            target_cp = _checkpoint_details(Path(args.store), target_snapshot, fault_helpers)
            target_object_sha = fault_helpers._snapshot_record_fingerprint(Path(args.store), target_snapshot)
            validate_target_checkpoint(dict(snapshot=target_snapshot,
                payload_sha256=target_cp['payload_sha256'], payload_bytes=target_cp['payload_bytes'],
                manifest_sha256=target_cp['manifest_sha256']), target_snapshot=target_snapshot)
            progress.phase('filling_bounded_volume_to_real_enospc')
            filler = _fill_to_enospc(Path(args.canonical_mount), args.max_fill_mib * 1024**2, nonce)
            filler.update(schema='q4-canonical-enospc-filler-v1', target_snapshot=target_snapshot,
                operation_id=operation_id, canonical_st_dev=Path(args.canonical_mount).stat().st_dev)
            validate_filler_receipt(filler, canonical_mount=str(args.canonical_mount),
                max_fill_bytes=args.max_fill_mib * 1024**2,
                canonical_st_dev=Path(args.canonical_mount).stat().st_dev)
            receipt_path = run_dir / 'filler-receipt.json'
            release_path = control / 'release.json'
            def release():
                _require_request_waiting_at_release(future)
                _atomic_json(release_path, dict(operation_id=operation_id, at_unix=time.time()))
            persist_receipt_before_release(receipt_path, filler, release,
                canonical_st_dev=Path(args.canonical_mount).stat().st_dev)
            request_result = future.result(timeout=args.timeout)
        except BaseException:
            _atomic_json(control / 'abort.json', dict(reason='parent failed before completing the gated request',
                at_unix=time.time()))
            service_stop_attempted = _stop_service_if_process_exists(
                service, result, 'service_stop_after_gate_abort')
            if service_stop_attempted:
                service_started = False
            try:
                future.result(timeout=10)
            except BaseException:
                pass
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if 'error' in request_result:
            raise CanonicalEnospcError(request_result['error'])
        events = request_result.get('events')
        validate_error_stream(events)
        event_summary = _safe_http_events(events)
        failed_status = _terminal_status(fault_helpers, client, operation_id, target_branch, args.timeout)
        after_head, after_history = fault_helpers._parse_head_and_history(client, target_branch)
        validate_retryable_failure(failed_status, target_head, after_head,
            operation_id=operation_id, branch=target_branch, parent_snapshot=source_snapshot)
        if _history_sha(fault_helpers, after_history) != _history_sha(fault_helpers, target_history):
            raise CanonicalEnospcError('failed branch-reference write changed canonical target history')
        if target_ref_path.read_bytes() != old_ref_bytes:
            raise CanonicalEnospcError('failed branch-reference write changed the previous reference bytes')
        writer_rows = _events(gate_path)
        failures = [row for row in writer_rows if row.get('kind') == 'branch-reference-write-error'
                    and row.get('operation_id') == operation_id]
        if len(failures) != 1:
            raise CanonicalEnospcError(f'expected one exact branch-ref writer error observation, got {len(failures)}')
        fault_observation = failures[0]
        checked_target_cp = _checkpoint_details(Path(args.store), target_snapshot, fault_helpers)
        if checked_target_cp != target_cp:
            raise CanonicalEnospcError('prepared checkpoint changed after reference publication failure')
        if fault_helpers._snapshot_record_fingerprint(Path(args.store), target_snapshot) != target_object_sha:
            raise CanonicalEnospcError('prepared snapshot object changed after reference publication failure')
        source_after_fault, source_history_after_fault = fault_helpers._parse_head_and_history(client, args.branch)
        source_object_after_fault = fault_helpers._snapshot_record_fingerprint(Path(args.store), source_snapshot)
        source_cp_after_fault = fault_helpers._snapshot_file_fingerprint(Path(args.store), source_snapshot)
        if (source_after_fault != source_head
                or _history_sha(fault_helpers, source_history_after_fault) != _history_sha(fault_helpers, source_history)
                or source_object_after_fault != source_object_sha or source_cp_after_fault != source_cp):
            raise CanonicalEnospcError('original source branch or checkpoint changed during the fault')
        validate_reference_fault(fault_observation, operation_id=operation_id,
            branch=target_branch, reference_path=str(target_ref_path.resolve()),
            old_reference_sha256=old_ref_sha, target_snapshot=target_snapshot)
        result['fault'] = dict(events=event_summary, operation_status=failed_status,
            writer=fault_observation, target_checkpoint=target_cp,
            target_snapshot_record_sha256=target_object_sha,
            target_checkpoint_validated_after_failure=True,
            source_checkpoint_unchanged_after_failure=True,
            old_reference_unchanged=True,
            target_history_unchanged=True)

        progress.phase('releasing_owned_filler')
        _remove_owned_filler(filler, Path(args.canonical_mount))
        filler = None
        fault_helpers._wait_http_idle(client, args.timeout, event_recorder)
        trace_before_retry = native_trace.read_bytes()
        retry_code, retry_response = client.post_json('/cow/publication/retry', dict(operation_id=operation_id))
        if retry_code != 200:
            raise CanonicalEnospcError(f'same-operation retry returned HTTP {retry_code}: {retry_response}')
        retry_head, retry_history = fault_helpers._parse_head_and_history(client, target_branch)
        operation_after_retry = _terminal_status(fault_helpers, client, operation_id, target_branch, args.timeout)
        fault_helpers._wait_http_idle(client, args.timeout, event_recorder)
        moves_after = [row.get('operation_id') for row in retry_history.get('moves', [])]
        trace_after_retry = native_trace.read_bytes()
        validate_retry(retry_response, operation_id=operation_id, target_snapshot=target_snapshot,
            branch=target_branch, revision_before=target_head['revision'], moves_before=old_moves,
            moves_after=moves_after, operation_status=operation_after_retry,
            work_before=trace_before_retry, work_after=trace_after_retry)
        if _checkpoint_details(Path(args.store), target_snapshot, fault_helpers) != target_cp:
            raise CanonicalEnospcError('same-operation retry changed the already-prepared target checkpoint')
        result['retry'] = dict(http_status=retry_code, response=retry_response,
            operation_status=operation_after_retry, native_trace_unchanged=True,
            native_trace_sha256=hashlib.sha256(trace_after_retry).hexdigest())

        following_head, following_history = fault_helpers._parse_head_and_history(client, target_branch)
        following_id = 'op-q4-canonical-follow-' + nonce
        following_route, following_body = fault_helpers._generation_body(following_history,
            following_head, target_branch, following_id, 8, 'Verify the recovered branch still accepts work.')
        progress.phase('following_request_after_retry')
        following_events, disconnected = client.post_stream(following_route, following_body)
        if disconnected:
            raise CanonicalEnospcError('following request disconnected')
        terminal = fault_helpers._terminal(following_events)
        if terminal.get('event') != 'complete':
            raise CanonicalEnospcError(f'following request did not complete: {terminal}')
        fault_helpers.validate_followup(terminal, target_branch, following_head['revision'])
        fault_helpers._wait_http_idle(client, args.timeout, event_recorder)
        if (fault_helpers._snapshot_record_fingerprint(Path(args.store), source_snapshot) != source_object_sha
                or fault_helpers._snapshot_file_fingerprint(Path(args.store), source_snapshot) != source_cp):
            raise CanonicalEnospcError('original source checkpoint changed during retry/follow-up')
        result['following_request'] = dict(route=following_route,
            events=_safe_http_events(following_events), terminal=_safe_http_events([terminal])[0])
        source_after, source_history_after = fault_helpers._parse_head_and_history(client, args.branch)
        if (source_after != source_head or _history_sha(fault_helpers, source_history_after) != _history_sha(fault_helpers, source_history)):
            raise CanonicalEnospcError('original source branch/history changed')
        result['source_after'] = dict(head=source_after, unchanged=True)
        stop_receipt = service.stop()
        service_started = False
        service_stop_attempted = True
        from run_cow_client_recovery import _trace_worker
        result['service_stop'] = stop_receipt
        result['native_shutdown'] = _trace_worker(run_dir / 'pool-trace.json', service.process.pid)
        if stop_receipt.get('exit_code') != 0 or not stop_receipt.get('graceful'):
            raise CanonicalEnospcError('owned service/native worker did not exit cleanly')
        if result['native_shutdown'].get('native_pid') != native_pid:
            raise CanonicalEnospcError('native worker PID changed during the owned service lifetime')
        result['status'] = 'PASS'
        progress.finish('PASS')
        progress.close()
        _atomic_json(result_path, result)
        return result
    except BaseException as exc:
        try:
            _atomic_json(control / 'abort.json', dict(reason=f'{type(exc).__name__}: {exc}', at_unix=time.time()))
        except BaseException:
            pass
        result['status'] = 'FAIL'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        try:
            progress.finish('FAIL', result['error'])
        except BaseException:
            pass
        if filler is not None:
            try:
                _remove_owned_filler(filler, Path(args.canonical_mount))
                result['filler_cleanup'] = True
            except BaseException as cleanup_error:
                result['filler_cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
        if not service_stop_attempted:
            service_stop_attempted = _stop_service_if_process_exists(service, result, 'service_stop')
        try:
            progress.close()
        except BaseException as progress_error:
            result['progress_close_error'] = f'{type(progress_error).__name__}: {progress_error}'
        _atomic_json(result_path, result)
        return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--api-src', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--original-store', type=Path, required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--canonical-mount', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    parser.add_argument('--loop-image', type=Path, required=True)
    parser.add_argument('--parent-mount-namespace', required=True)
    parser.add_argument('--expected-profile-sha256', required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--expected-source-snapshot', required=True)
    parser.add_argument('--expected-image-uuid', required=True)
    parser.add_argument('--expected-image-inode', type=int, required=True)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128)
    parser.add_argument('--max-fill-mib', type=int, default=48)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--preflight-only', action='store_true')
    return parser


def main(argv=None, *, probe_runner=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens != 16 or args.ram_gib != 96 or args.disk_gib != 128:
        print('Q4 canonical ENOSPC probe is pinned to 16 outputs and 96/128 GiB pools', file=sys.stderr)
        return 2
    if args.preflight_only:
        try:
            receipt = preflight(args)
        except BaseException as exc:
            print(f'PREFLIGHT_FAIL {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
        print(json.dumps(dict(status='PREPARED_NOT_RUN', preflight=receipt),
                         sort_keys=True, ensure_ascii=False), flush=True)
        return 0
    result, code = invoke_probe(args, probe_runner=probe_runner)
    run_dir = Path(args.run_dir).expanduser().absolute()
    control = run_dir / 'control'
    result_file = run_dir / 'result.json'
    # A setup exception after this runner created its private control directory
    # still gets a durable failure record. Never claim ownership of a preexisting
    # run directory that preflight rejected.
    if (code != 0 and run_dir.is_dir() and not run_dir.is_symlink()
            and control.is_dir() and not control.is_symlink() and not result_file.exists()):
        _atomic_json(result_file, dict(schema='q4-canonical-reference-enospc-v1',
            status='FAIL', error=result.get('error', 'probe failed before result publication')))
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
