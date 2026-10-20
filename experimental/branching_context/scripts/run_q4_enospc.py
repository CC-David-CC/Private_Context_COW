"""Real Q4 COW publication/retry probe using a private ext4 ENOSPC fault.

The runner expects a private mount namespace where only the disposable Store's
``checkpoints`` directory is mounted from a bounded loop ext4 image. CPU tests
cover evidence/mount guards only; this script's real mode is a GPU integration
test and is not executed by this assignment.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid

import run_q4_publication_fault as _base


class EnospcProbeError(RuntimeError):
    """The observed filesystem fault/recovery missed its required contract."""


def _write_signal(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise EnospcProbeError('Could not persist writer gate signal')
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def validate_enospc_writer_observation(row, operation_id: str, branch: str,
                                      *, checkpoint_directory: Path | None = None) -> dict:
    """Accept only ENOSPC from the real atomic state-payload writer."""
    if type(row) is not dict:
        raise EnospcProbeError('Checkpoint writer observation is missing')
    message = row.get('message')
    if (row.get('operation_id') != operation_id or row.get('branch') != branch
            or row.get('destination_kind') != 'checkpoint-payload-state'
            or row.get('exception_type') != 'OSError'
            or type(row.get('errno')) is not int or row['errno'] != errno.ENOSPC
            or not isinstance(message, str) or f'[Errno {errno.ENOSPC}]' not in message
            or os.strerror(errno.ENOSPC) not in message
            or not isinstance(row.get('snapshot'), str) or not row['snapshot']
            or not isinstance(row.get('payload_name'), str)
            or not row['payload_name'].endswith('.state')
            or not isinstance(row.get('destination_directory'), str)
            or type(row.get('requested_payload_bytes')) is not int
            or row['requested_payload_bytes'] <= 0
            or row.get('published_payload_exists') is not False
            or row.get('published_manifest_exists') is not False
            or row.get('staging_files_after') != []):
        raise EnospcProbeError('Observed failure was not ENOSPC during atomic checkpoint-payload publication')
    if checkpoint_directory is not None:
        destination = Path(row['destination_directory'])
        if destination.is_symlink() or destination.resolve(strict=True) != Path(checkpoint_directory).resolve(strict=True):
            raise EnospcProbeError('Failed state-payload writer escaped the mounted Store checkpoint directory')
    return dict(snapshot=row['snapshot'], payload_name=row['payload_name'],
                errno=row['errno'], requested_payload_bytes=row['requested_payload_bytes'])


def validate_enospc_stream(events) -> None:
    """Require one explicit original ENOSPC error and no durable acknowledgement."""
    try:
        _base.validate_fault_stream(events)
    except _base.PublicationFaultError as exc:
        raise EnospcProbeError(str(exc)) from exc
    if any(row.get('event') == 'durable' or row.get('durable') is True or 'receipt' in row
           for row in events):
        raise EnospcProbeError('Filesystem-full request emitted a durable acknowledgement or receipt')
    errors = [row for row in events if row.get('event') == 'error']
    if len(errors) != 1:
        raise EnospcProbeError('Filesystem-full request did not emit exactly one explicit error event')
    row = errors[0]
    message = row.get('error')
    if (row.get('cancelled') is not False or not isinstance(message, str)
            or f'[Errno {errno.ENOSPC}]' not in message
            or os.strerror(errno.ENOSPC) not in message):
        raise EnospcProbeError('Client event did not preserve original ENOSPC with cancelled=false')


def validate_ext4_mount(mount: Path, image: Path, *, parent_mount_namespace: str | None = None,
                        min_capacity: int = 512 * 1024**2,
                        max_capacity: int = 1536 * 1024**2) -> dict:
    """Prove the checkpoint directory is the expected bounded loop ext4 mount."""
    mount = Path(mount)
    image = Path(image)
    if mount.is_symlink() or image.is_symlink():
        raise EnospcProbeError('Mountpoint and loop image must not be symlinks')
    resolved_mount = mount.resolve(strict=True)
    resolved_image = image.resolve(strict=True)
    if not resolved_mount.is_dir() or not resolved_image.is_file():
        raise EnospcProbeError('Mountpoint/image type is invalid')
    try:
        row = json.loads(subprocess.check_output(
            ['findmnt', '--json', '--target', str(resolved_mount)], text=True
        ))['filesystems'][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise EnospcProbeError(f'Cannot verify the mounted checkpoint filesystem: {exc}') from exc
    if row.get('fstype') != 'ext4' or not str(row.get('source', '')).startswith('/dev/loop'):
        raise EnospcProbeError('Refusing a checkpoint mount other than private loop-backed ext4')
    try:
        devices = json.loads(subprocess.check_output(
            ['losetup', '--json', '--output', 'NAME,BACK-FILE,MAJ:MIN'], text=True
        ))['loopdevices']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EnospcProbeError(f'Cannot verify loop backing file: {exc}') from exc
    backing = [entry for entry in devices
               if entry.get('name') == row['source']
               and Path(str(entry.get('back-file', ''))).resolve() == resolved_image]
    if len(backing) != 1:
        raise EnospcProbeError('Mounted loop device is not backed by the selected private image')
    stats = os.statvfs(resolved_mount)
    capacity = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    image_bytes = resolved_image.stat().st_size
    if not min_capacity <= capacity <= max_capacity or image_bytes > max_capacity:
        raise EnospcProbeError(f'Private ext4 capacity is outside the bounded range: {capacity}')
    if image_bytes < capacity or capacity - image_bytes > 16 * 1024**2:
        raise EnospcProbeError('Filesystem capacity does not match its bounded backing image')
    namespace = os.readlink('/proc/self/ns/mnt')
    init_namespace = (os.readlink('/proc/1/ns/mnt') if parent_mount_namespace is None
                      else parent_mount_namespace)
    if (not isinstance(namespace, str) or not namespace.startswith('mnt:[') or not namespace.endswith(']')
            or not isinstance(init_namespace, str) or not init_namespace.startswith('mnt:[')
            or not init_namespace.endswith(']')):
        raise EnospcProbeError('Cannot establish both current and parent mount namespace identities')
    if namespace == init_namespace:
        raise EnospcProbeError('Runner is not inside a private mount namespace')
    return dict(mountpoint=str(resolved_mount), image=str(resolved_image), image_bytes=image_bytes,
                source=row['source'], fstype=row['fstype'], capacity_bytes=capacity,
                available_bytes=available, mount_namespace=namespace,
                init_mount_namespace=init_namespace,
                loop_major_minor=backing[0].get('maj:min'))


def validate_filler_result(result: dict, *, capacity_bytes: int, max_filler_bytes: int) -> None:
    if (type(result) is not dict or type(result.get('errno')) is not int
            or result.get('errno') != errno.ENOSPC
            or type(result.get('attempted_bytes')) is not int
            or type(result.get('staged_bytes')) is not int
            or not 0 <= result['staged_bytes'] <= result['attempted_bytes'] <= max_filler_bytes
            or type(result.get('available_before')) is not int
            or type(result.get('available_after')) is not int
            or result['available_before'] > capacity_bytes
            or not 0 <= result.get('available_after', -1) <= 8 * 1024**2):
        raise EnospcProbeError('Private filesystem filler did not prove bounded kernel ENOSPC')


def _writer_wrapper_source(writer_events: Path, gate_path: Path, release_path: Path) -> str:
    """Observe and pause at the prepared-payload writer without replacing it."""
    return r'''import json, os, signal, sys, threading, time
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
from context_time_travel.real import BranchingModel
from context_time_travel.coordinator import ContextCoordinator
local = threading.local()
original_publish = BranchingModel.publish_prepared_checkpoint
original_write = BranchingModel._publish_checkpoint_parts
original_coordinator_publish = ContextCoordinator._publish
events_path, gate_path, release_path = sys.argv[1:4]
server_args = sys.argv[4:]
def append_event(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
def observed_publish(self, prepared):
    local.snapshot = prepared.snapshot
    try:
        return original_publish(self, prepared)
    finally:
        local.snapshot = None
def observed_write(self, destination, parts):
    parts = tuple(parts)
    destination = os.fspath(destination)
    if destination.endswith(".state") and not os.path.exists(gate_path):
        append_event(gate_path, {"event":"writer_gate", "operation_id":getattr(local,"operation_id",None),
            "branch":getattr(local,"branch",None), "snapshot":getattr(local,"snapshot",None),
            "destination":destination, "destination_directory":os.path.dirname(destination),
            "requested_payload_bytes":sum(len(part) for part in parts),
            "pid":os.getpid()})
        deadline = time.monotonic() + 180.0
        while not os.path.exists(release_path):
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded ENOSPC filler did not release the payload writer")
            time.sleep(.02)
        with open(release_path, "r", encoding="ascii") as signal_file:
            release = signal_file.read().strip()
        if release != "RELEASE":
            raise RuntimeError("bounded ENOSPC filler aborted before the real writer")
    try:
        return original_write(self, destination, parts)
    except OSError as error:
        if destination.endswith(".state"):
            directory = os.path.dirname(destination)
            stages = sorted(name for name in os.listdir(directory) if name.startswith(".stage-"))
            snapshot = getattr(local, "snapshot", None)
            append_event(events_path, {"operation_id":getattr(local,"operation_id",None),
                "branch":getattr(local,"branch",None), "snapshot":snapshot,
                "destination_kind":"checkpoint-payload-state", "payload_name":os.path.basename(destination),
                "destination_directory":os.path.dirname(destination),
                "requested_payload_bytes":sum(len(part) for part in parts),
                "exception_type":type(error).__name__, "errno":error.errno,
                "message":str(error), "stage_name":None, "staging_files_after":stages,
                "published_payload_exists":os.path.exists(destination),
                "published_manifest_exists":bool(snapshot and os.path.exists(os.path.join(directory, snapshot + ".json")))})
        raise
BranchingModel.publish_prepared_checkpoint = observed_publish
BranchingModel._publish_checkpoint_parts = observed_write
def observed_coordinator_publish(self, model, job, prepared, key, pin, lease):
    local.operation_id = job.request.operation
    local.branch = job.branch
    try:
        return original_coordinator_publish(self, model, job, prepared, key, pin, lease)
    finally:
        local.operation_id = None
        local.branch = None
ContextCoordinator._publish = observed_coordinator_publish
sys.argv = ["serve.cow_http", *server_args]
from serve.cow_http import main
main()
'''


def _fill_private_mount(mount: Path, *, max_bytes: int, keep_free_bytes: int = 4 * 1024**2) -> dict:
    """Force a real ENOSPC on the private mount, then free metadata headroom."""
    mount = Path(mount)
    mount_stat = os.stat(mount)
    before = os.statvfs(mount)
    available = before.f_bavail * before.f_frsize
    capacity = before.f_blocks * before.f_frsize
    if not 0 < available <= max_bytes <= capacity:
        raise EnospcProbeError(f'Private filesystem fill bounds are invalid: available={available}, capacity={capacity}')
    filler = mount / '.q4-enospc-filler'
    if filler.exists() or filler.is_symlink():
        raise EnospcProbeError('Private filesystem already contains the reserved filler path')
    block = hashlib.shake_256(b'q4-cow-private-ext4-enospc-filler-v1').digest(1024**2)
    attempted = staged = 0
    failure = None
    try:
        with filler.open('xb', buffering=0) as stream:
            try:
                while attempted < max_bytes:
                    attempted += len(block)
                    view = memoryview(block)
                    while view:
                        written = stream.write(view)
                        if not written:
                            raise EnospcProbeError('Private filler write made no progress')
                        staged += written
                        view = view[written:]
                    os.fsync(stream.fileno())
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                failure = dict(errno=exc.errno, message=str(exc))
            if failure is None:
                raise EnospcProbeError('Bounded filler reached its byte guard before kernel ENOSPC')
            # Keep a little directory/journal headroom so the real payload writer
            # can create its stage before the much larger payload write fails.
            retained = max(0, staged - keep_free_bytes)
            os.ftruncate(stream.fileno(), retained)
            os.fsync(stream.fileno())
    except BaseException:
        if filler.exists() and not filler.is_symlink():
            filler.unlink()
            directory_fd = os.open(mount, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        raise
    after = os.statvfs(mount)
    result = dict(mount_device=str(mount_stat.st_dev), attempted_bytes=attempted,
        staged_bytes=staged, errno=failure['errno'], message=failure['message'],
        available_before=available, available_after=after.f_bavail * after.f_frsize,
        filesystem_capacity_bytes=capacity, filler_path=str(filler),
        filler_retained_bytes=filler.stat().st_size)
    validate_filler_result(result, capacity_bytes=capacity, max_filler_bytes=max_bytes)
    return result


def _write_server_wrapper(run_dir: Path, writer_events: Path, gate_path: Path,
                          release_path: Path) -> Path:
    wrapper = run_dir / 'server_wrapper.py'
    with wrapper.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(_writer_wrapper_source(writer_events, gate_path, release_path))
        stream.flush()
        os.fsync(stream.fileno())
    wrapper.chmod(0o600)
    return wrapper


def _read_gate(path: Path, deadline: float) -> dict:
    while time.monotonic() < deadline:
        try:
            if path.is_file() and not path.is_symlink():
                row = json.loads(path.read_text(encoding='utf-8'))
                if type(row) is dict and row.get('event') == 'writer_gate':
                    return row
        except (OSError, ValueError):
            pass
        time.sleep(.05)
    raise EnospcProbeError('Prepared checkpoint writer did not reach the ENOSPC gate before deadline')


def _remove_owned_filler(fill_result: dict) -> dict:
    path = Path(fill_result['filler_path'])
    if path.is_symlink() or not path.is_file():
        raise EnospcProbeError('Owned ENOSPC filler is missing or redirected before cleanup')
    stat = path.stat()
    if stat.st_nlink != 1 or stat.st_size != fill_result.get('filler_retained_bytes'):
        raise EnospcProbeError('Owned filler inode/length changed before cleanup')
    before = os.statvfs(path.parent)
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    after = os.statvfs(path.parent)
    return dict(unlinked=True, inode=stat.st_ino, bytes=stat.st_size,
                available_before=before.f_bavail * before.f_frsize,
                available_after=after.f_bavail * after.f_frsize)


def _fill_after_writer_gate(*, gate_path: Path, release_path: Path, filler_path: Path,
                            mount: Path, operation_id: str, branch: str,
                            source_snapshot: str, timeout: float, max_fill_bytes: int,
                            fill_result_holder: dict, fill_error) -> None:
    """Fill the bounded filesystem only after the real prepared writer pauses."""
    try:
        gate = _read_gate(gate_path, time.monotonic() + timeout)
        if (gate.get('operation_id') != operation_id or gate.get('branch') != branch
                or gate.get('snapshot') in (None, source_snapshot)
                or gate.get('requested_payload_bytes', 0) < 300 * 1024**2):
            raise EnospcProbeError('Writer gate does not identify the expected prepared target payload')
        fill_result_holder['gate'] = gate
        fill_result_holder['fill'] = _fill_private_mount(mount, max_bytes=max_fill_bytes)
        # Keep persistence before RELEASE: the caller must not proceed unless
        # the evidence describing the exact filler is durable.
        _base._atomic_json(filler_path, fill_result_holder['fill'])
        _write_signal(release_path, b'RELEASE\n')
    except BaseException as exc:
        fill_error.put(exc)
        _write_signal(release_path, b'ABORT\n')


def run_probe(args) -> dict:
    """Run one real HTTP publication fault against the exact mounted checkpoint FS."""
    if not sys.platform.startswith('linux'):
        raise EnospcProbeError('Real ENOSPC runner requires Linux')
    run_dir = Path(args.run_dir).expanduser().absolute()
    if run_dir.exists() or run_dir.is_symlink():
        raise EnospcProbeError('Run directory must be new and not a symlink')
    if (run_dir.parent.resolve(strict=True) / run_dir.name) != run_dir:
        raise EnospcProbeError('Run directory parent path must not redirect through symlinks')
    profile_path = _base._pinned_path(args.profile, 'Profile')
    store = _base._pinned_path(args.store, 'Disposable Store')
    original_store = _base._pinned_path(args.original_store, 'Original source Store')
    if (store == original_store or store.is_relative_to(original_store)
            or original_store.is_relative_to(store)):
        raise EnospcProbeError('Refusing to run against the original source Store')
    runtime_root = _base._pinned_path(args.runtime_root, 'Runtime root')
    api_src = _base._pinned_path(args.api_src, 'API source')
    api_scripts = _base._pinned_path(args.api_scripts, 'API scripts')
    image = _base._pinned_path(args.loop_image, 'Loop image')
    mount = _base._pinned_path(args.checkpoint_mount, 'Checkpoint mount')
    if (store / 'checkpoints').is_symlink() or mount != (store / 'checkpoints').resolve(strict=True):
        raise EnospcProbeError('Only the copied Store checkpoint directory may be mounted')
    resolved_run_dir = run_dir.parent.resolve(strict=True) / run_dir.name
    if resolved_run_dir.is_relative_to(mount) or resolved_run_dir.is_relative_to(store):
        raise EnospcProbeError('Run output must stay outside the Store and capacity-limited mount')
    if not (store / 'refs.json').is_file() or store.is_symlink():
        raise EnospcProbeError('A pre-existing disposable Store copy is required')
    mount_receipt = validate_ext4_mount(
        mount, image, parent_mount_namespace=args.parent_mount_namespace)
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    _base._ensure_q4_profile(profile)
    runtime_closure = _base._runtime_closure(profile, runtime_root)
    native_binary = Path(profile['execution']['sut_command'][0]).resolve(strict=True)
    native_sha = _base._sha256(native_binary)
    if native_sha != args.expected_native_sha256:
        raise EnospcProbeError('Native executable hash differs from the explicitly selected build')
    profile_sha = _base._sha256(profile_path)
    if profile_sha != args.expected_profile_sha256:
        raise EnospcProbeError('Q4 execution-profile hash differs from the selected profile')

    run_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    output_path = run_dir / 'result.json'

    script_dir = runtime_root / 'experimental' / 'branching_context' / 'scripts'
    if not script_dir.is_dir():
        raise EnospcProbeError('Pinned runtime lacks existing client-recovery helpers')
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(runtime_root))
    sys.path.insert(0, str(api_src))
    sys.path.insert(0, str(api_scripts))
    from run_cow_client_recovery import (HttpClient, ProgressTracker, ServiceProcess,
        _ready_url, _trace_worker, _fork)

    writer_events = run_dir / 'writer-errors.jsonl'
    writer_events.touch(mode=0o600, exist_ok=False)
    gate_path, release_path = run_dir / 'writer-gate.json', run_dir / 'writer-release'
    filler_path = run_dir / 'filler-result.json'
    progress = ProgressTracker(run_dir, 'q4-publication-enospc-retry')
    progress.start()
    wrapper = _write_server_wrapper(run_dir, writer_events, gate_path, release_path)
    pool_trace = run_dir / 'pool-trace.json'
    native_trace = run_dir / 'native-trace.jsonl'
    native_trace.touch(mode=0o600, exist_ok=False)
    command = [sys.executable, str(wrapper), str(writer_events), str(gate_path), str(release_path),
        '--profile', str(profile_path), '--store', str(store), '--host', '127.0.0.1', '--port', '0',
        '--text', '--ram-gib', str(args.ram_gib), '--disk-gib', str(args.disk_gib),
        '--pool-trace-output', str(pool_trace)]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(runtime_root), str(api_src), str(api_scripts)))
    env['CTT_NATIVE_TRACE'] = str(native_trace)
    service = ServiceProcess(command, cwd=runtime_root, env=env,
        stdout_path=run_dir / 'server.stdout.log', stderr_path=run_dir / 'server.stderr.log',
        timeout=args.timeout, progress=progress)
    result = dict(schema='q4-publication-enospc-v1', status='RUNNING',
        scope='real Q4 COW HTTP service; bounded private loop ext4; kernel ENOSPC; same-operation retry',
        fault_fs_role='only copied Store checkpoints directory; test-only bounded filesystem',
        test_filesystem=dict(capacity_bytes=mount_receipt['capacity_bytes'], bounded=True,
            deployment_profile_unchanged=True, pool_disk_budget_gib=args.disk_gib),
        profile_path=str(profile_path), profile_sha256=profile_sha,
        native_binary=str(native_binary), native_sha256=native_sha,
        runtime_root=str(runtime_root), runtime_source_closure=runtime_closure,
        api_src=str(api_src), api_scripts=str(api_scripts), store_path=str(store),
        checkpoint_mount=mount_receipt, loop_image_sha256=_base._sha256(image),
        branch=args.branch, run_directory=str(run_dir), runner_sha256=_base._sha256(Path(__file__).resolve()),
        wrapper_path=str(wrapper), wrapper_sha256=_base._sha256(wrapper), steps=[])
    server_pid = native_pid = None
    fill_thread = None
    fill_error = queue.Queue()
    fill_result_holder = {}
    try:
        progress.phase('start_owned_q4_http_service')
        ready = service.start()
        server_pid = service.process.pid
        progress.server_started(server_pid)
        result['server_pid'] = server_pid
        result['ready'] = ready
        client = HttpClient(_ready_url(ready), timeout=args.timeout, progress=progress)
        status, health = client.get('/health')
        if status != 200 or health.get('service') != 'context-cow':
            raise EnospcProbeError('Owned loopback process is not the COW HTTP service')
        if ready.get('execution_profile_id') != profile.get('execution_profile_id'):
            raise EnospcProbeError('Ready service profile differs from the pinned Q4 profile')
        source_head, source_history = _base._head_history(client, args.branch)
        source_snapshot = source_head.get('tip')
        if source_snapshot != args.expected_source_snapshot:
            raise EnospcProbeError('Disposable Store branch tip differs from the pinned source snapshot')
        source_record_sha = _base._snapshot_record_fingerprint(store, source_snapshot)
        source_checkpoint = _base._snapshot_file_fingerprint(store, source_snapshot)
        source_payloads = [name for name in source_checkpoint if name.endswith('.state')]
        if len(source_payloads) != 1:
            raise EnospcProbeError('Source snapshot does not resolve to exactly one checkpoint payload')
        source_history_sha = _base._canonical_history_sha(source_history)
        result['source_before'] = dict(head=source_head, history_sha256=source_history_sha,
            snapshot_record_sha256=source_record_sha, checkpoint_files=source_checkpoint)

        children = _base._children(server_pid)
        if len(children) != 1:
            raise EnospcProbeError(f'Expected one native child after service readiness, got {children}')
        native_pid = children[0]
        native_info = _base._read_proc(native_pid)
        server_info = _base._read_proc(server_pid)
        if native_info.get('parent_pid') != server_pid or Path(native_info['exe']).resolve() != native_binary:
            raise EnospcProbeError('Native worker identity differs from pinned executable')
        result['server_identity'] = server_info
        result['native_identity'] = native_info

        nonce = uuid.uuid4().hex
        target_branch = 'Q4ENOSPC' + nonce[:16]
        fork_op = 'op-enospc-fork-' + nonce
        progress.phase('fork_isolated_q4_branch')
        _fork(client, args.branch, source_snapshot, target_branch, fork_op)
        branch_head, branch_history = _base._head_history(client, target_branch)
        source_head_target, source_history_target = _base._head_history(client, args.branch)
        if source_head_target != source_head or _base._canonical_history_sha(source_history_target) != source_history_sha:
            raise EnospcProbeError('Isolated branch fork changed original source branch/history')
        route, body = _base._generation_body(branch_history, branch_head, target_branch,
            'op-q4-enospc-' + nonce, args.max_tokens, 'Continue the prepared Q4 ENOSPC test turn.')
        result.update(target_branch=target_branch, fault_operation_id=body['operation_id'], fault_route=route,
            fault_parent=dict(head=branch_head, history_sha256=_base._canonical_history_sha(branch_history),
                source_snapshot=source_snapshot))

        fill_thread = threading.Thread(target=_fill_after_writer_gate, name='private-ext4-enospc-filler',
            kwargs=dict(gate_path=gate_path, release_path=release_path, filler_path=filler_path,
                mount=mount, operation_id=body['operation_id'], branch=target_branch,
                source_snapshot=source_snapshot, timeout=args.timeout,
                max_fill_bytes=args.max_fill_mib * 1024**2,
                fill_result_holder=fill_result_holder, fill_error=fill_error), daemon=True)
        fill_thread.start()
        progress.phase('request_until_kernel_enospc_at_prepared_payload_writer')
        events, disconnected = client.post_stream(route, body)
        fill_thread.join(timeout=args.timeout)
        if fill_thread.is_alive():
            raise EnospcProbeError('Private filesystem filler did not finish within the request deadline')
        if not fill_error.empty():
            raise EnospcProbeError(f'ENOSPC gate/filler failed: {fill_error.get()}')
        if disconnected:
            raise EnospcProbeError('HTTP service disconnected during actual ENOSPC publication failure')
        validate_enospc_stream(events)
        writer_rows = [json.loads(line) for line in writer_events.read_text(encoding='utf-8').splitlines() if line]
        writer_faults = [row for row in writer_rows if row.get('operation_id') == body['operation_id']
                         and row.get('branch') == target_branch]
        if len(writer_faults) != 1:
            raise EnospcProbeError(f'Expected one real payload-writer ENOSPC row, got {len(writer_faults)}')
        writer = validate_enospc_writer_observation(writer_faults[0], body['operation_id'], target_branch,
                                                    checkpoint_directory=mount)
        if writer['snapshot'] == source_snapshot or writer['payload_name'] == source_payloads[0]:
            raise EnospcProbeError('Prepared target aliases the immutable source checkpoint')
        filler = fill_result_holder['fill']
        validate_filler_result(filler, capacity_bytes=mount_receipt['capacity_bytes'],
                               max_filler_bytes=args.max_fill_mib * 1024**2)
        fault_status = _base._operation_status(client, body['operation_id'], target_branch,
                                               time.monotonic() + args.timeout)
        head_after_fault, history_after_fault = _base._head_history(client, target_branch)
        _base.validate_faulted_operation(fault_status, branch_head, head_after_fault,
            operation_id=body['operation_id'], branch=target_branch, parent_snapshot=branch_head['tip'])
        if _base._canonical_history_sha(history_after_fault) != _base._canonical_history_sha(branch_history):
            raise EnospcProbeError('Failed publication changed canonical branch history')
        if (_base._snapshot_record_fingerprint(store, source_snapshot) != source_record_sha
                or _base._snapshot_file_fingerprint(store, source_snapshot) != source_checkpoint):
            raise EnospcProbeError('Faulted operation changed the immutable original source checkpoint')
        checkpoints = store / 'checkpoints'
        target_manifest = checkpoints / f"{writer['snapshot']}.json"
        target_payload = checkpoints / writer['payload_name']
        if target_manifest.exists() or target_payload.exists() or list(checkpoints.glob('.stage-*')):
            raise EnospcProbeError('ENOSPC left a target manifest/payload or writer stage behind')
        if Path(filler['filler_path']).stat().st_dev != os.stat(mount).st_dev:
            raise EnospcProbeError('Filler escaped the verified private ext4 filesystem')
        cleanup_filler = _remove_owned_filler(filler)
        result['fault'] = dict(stream=events, operation_status=fault_status,
            writer=writer_faults[0], filler=filler, filler_cleanup=cleanup_filler,
            target_manifest_absent=True, target_payload_absent=True, stage_files_absent=True,
            branch_head_unchanged=True, source_checkpoint_unchanged=True)

        _base._wait_http_idle(client, args.timeout, progress)
        trace_before_retry = native_trace.read_bytes()
        retry_code, retry_response = client.post_json('/cow/publication/retry',
            dict(operation_id=body['operation_id']))
        if retry_code != 200:
            raise EnospcProbeError(f'Same-operation publication retry returned HTTP {retry_code}')
        retry_head, retry_history = _base._head_history(client, target_branch)
        status_after_retry = _base._operation_status(client, body['operation_id'], target_branch,
                                                    time.monotonic() + args.timeout)
        _base._wait_http_idle(client, args.timeout, progress)
        if (status_after_retry.get('durable') is not True
                or status_after_retry.get('publication_retryable') is not False
                or status_after_retry.get('receipt') != retry_response.get('receipt')):
            raise EnospcProbeError('Publication retry lacks its exact durable status/receipt')
        trace_after_retry = native_trace.read_bytes()
        retry_check = _base.validate_retry_publication(retry_response, body['operation_id'], target_branch,
            writer['snapshot'], branch_head, retry_head,
            moves_before=[row.get('operation_id') for row in branch_history.get('moves', [])],
            moves_after=[row.get('operation_id') for row in retry_history.get('moves', [])],
            operation_status=status_after_retry,
            work_before=trace_before_retry, work_after=trace_after_retry)
        if _base._snapshot_file_fingerprint(store, writer['snapshot']) is None:
            raise EnospcProbeError('Retried payload does not validate from the private ext4 filesystem')
        result['retry'] = dict(http_status=retry_code, response=retry_response,
            operation_status=status_after_retry, validation=retry_check,
            native_work_unchanged=trace_before_retry == trace_after_retry,
            native_trace_sha256_before=hashlib.sha256(trace_before_retry).hexdigest(),
            native_trace_sha256_after=hashlib.sha256(trace_after_retry).hexdigest())

        head_before_follow, history_before_follow = _base._head_history(client, target_branch)
        follow_route, follow_body = _base._generation_body(history_before_follow, head_before_follow,
            target_branch, 'op-q4-enospc-follow-' + nonce, min(args.max_tokens, 8),
            'Confirm the branch is usable after ENOSPC publication recovery.')
        follow_events, disconnected = client.post_stream(follow_route, follow_body)
        if disconnected:
            raise EnospcProbeError('Following request disconnected after the retry')
        follow_terminal = _base._terminal(follow_events)
        follow_result = _base.validate_followup(follow_terminal, target_branch, head_before_follow['revision'])
        result['following_request'] = dict(route=follow_route, operation_id=follow_body['operation_id'],
            events=follow_events, validation=follow_result)
        if (_base._snapshot_file_fingerprint(store, source_snapshot) != source_checkpoint
                or _base._snapshot_record_fingerprint(store, source_snapshot) != source_record_sha):
            raise EnospcProbeError('Following request changed the original source checkpoint')
        result['status'] = 'PASS'
        source_after, source_history_after = _base._head_history(client, args.branch)
        if (source_after != source_head
                or _base._canonical_history_sha(source_history_after) != source_history_sha):
            raise EnospcProbeError('Probe changed the original source branch/head/history')
        result['source_after'] = dict(head=source_after, unchanged=True)
        stop_receipt = service.stop()
        trace_record = _trace_worker(pool_trace, server_pid)
        result['service_stop'] = stop_receipt
        result['native_shutdown'] = trace_record
        if stop_receipt.get('exit_code') != 0 or not stop_receipt.get('graceful'):
            raise EnospcProbeError('Owned HTTP/native service did not close gracefully')
        progress.finish('PASS')
        _base._atomic_json(output_path, result)
        return result
    except BaseException as exc:
        result['status'] = 'FAIL'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        if fill_thread is not None and fill_thread.is_alive():
            try:
                _write_signal(release_path, b'ABORT\n')
            except FileExistsError:
                pass
            fill_thread.join(timeout=5)
        if fill_result_holder.get('fill'):
            filler = Path(fill_result_holder['fill']['filler_path'])
            if filler.exists() and not filler.is_symlink():
                try:
                    result.setdefault('cleanup', {})['filler'] = _remove_owned_filler(
                        fill_result_holder['fill'])
                except BaseException as cleanup_error:
                    result.setdefault('cleanup_errors', []).append(
                        f'{type(cleanup_error).__name__}: {cleanup_error}')
        if service.process is not None:
            try:
                result['service_stop'] = service.stop()
            except BaseException as stop_error:
                result['service_stop_error'] = f'{type(stop_error).__name__}: {stop_error}'
        try:
            progress.finish('FAIL', result['error'])
        except BaseException:
            pass
        _base._atomic_json(output_path, result)
        raise
    finally:
        if fill_thread is not None and fill_thread.is_alive():
            # The daemon thread owns only the bounded filesystem filler and its
            # release signal; ask it to stop, then observe this exact thread.
            try:
                _write_signal(release_path, b'ABORT\n')
            except FileExistsError:
                pass
            fill_thread.join(timeout=5)
        if service.process is not None and service.process.poll() is None:
            try:
                service.stop()
            except BaseException:
                pass
        try:
            progress.close()
        except BaseException:
            pass


def validate_preflight_inputs(args) -> dict:
    """Validate pinned runtime/profile and the copied source checkpoint without starting a service."""
    profile_path = _base._pinned_path(args.profile, 'Profile')
    store = _base._pinned_path(args.store, 'Disposable Store')
    original_store = _base._pinned_path(args.original_store, 'Original source Store')
    runtime_root = _base._pinned_path(args.runtime_root, 'Runtime root')
    api_src = _base._pinned_path(args.api_src, 'API source')
    api_scripts = _base._pinned_path(args.api_scripts, 'API scripts')
    image = _base._pinned_path(args.loop_image, 'Loop image')
    mount = _base._pinned_path(args.checkpoint_mount, 'Checkpoint mount')
    if (store == original_store or store.is_relative_to(original_store)
            or original_store.is_relative_to(store)):
        raise EnospcProbeError('Refusing to preflight against the original source Store')
    if store.is_symlink() or (store / 'checkpoints').is_symlink():
        raise EnospcProbeError('Disposable Store/checkpoints may not be symlinks')
    if mount != (store / 'checkpoints').resolve(strict=True):
        raise EnospcProbeError('Only the copied Store checkpoint directory may be mounted')
    if not (store / 'refs.json').is_file():
        raise EnospcProbeError('Disposable Store metadata is incomplete')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    _base._ensure_q4_profile(profile)
    profile_sha = _base._sha256(profile_path)
    if profile_sha != args.expected_profile_sha256:
        raise EnospcProbeError('Q4 execution-profile hash differs from the selected profile')
    runtime_closure = _base._runtime_closure(profile, runtime_root)
    native_binary = Path(profile['execution']['sut_command'][0]).resolve(strict=True)
    native_sha = _base._sha256(native_binary)
    if native_sha != args.expected_native_sha256:
        raise EnospcProbeError('Native executable hash differs from the explicitly selected build')
    checkpoint_record = _base._snapshot_record_fingerprint(store, args.expected_source_snapshot)
    checkpoint_files = _base._snapshot_file_fingerprint(store, args.expected_source_snapshot)
    if not api_src.is_dir() or not api_scripts.is_dir():
        raise EnospcProbeError('Pinned API source/scripts directories are missing')
    return dict(profile_path=str(profile_path), profile_sha256=profile_sha,
        native_binary=str(native_binary), native_sha256=native_sha,
        runtime_root=str(runtime_root), runtime_source_closure=runtime_closure,
        api_src=str(api_src), api_scripts=str(api_scripts), store_path=str(store),
        original_store_path=str(original_store), source_snapshot=args.expected_source_snapshot,
        source_snapshot_record_sha256=checkpoint_record,
        source_checkpoint_files=checkpoint_files, loop_image=str(image), checkpoint_mount=str(mount))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--api-src', type=Path, required=True)
    parser.add_argument('--api-scripts', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True,
                        help='Disposable copied Store; never original source Store')
    parser.add_argument('--original-store', type=Path, required=True,
                        help='Original source Store path; read-only identity check only')
    parser.add_argument('--branch', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--loop-image', type=Path, required=True)
    parser.add_argument('--checkpoint-mount', type=Path, required=True)
    parser.add_argument('--parent-mount-namespace', required=True,
                        help='Mount namespace ID captured before entering the private mount namespace')
    parser.add_argument('--expected-profile-sha256', required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--expected-source-snapshot', required=True)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128,
                        help='Test-budget override only; not a deployment profile')
    parser.add_argument('--max-fill-mib', type=int, default=1400)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--preflight-only', action='store_true',
                        help='Validate inputs/mount without starting the service or making a request')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_tokens <= 128:
        raise SystemExit('max-tokens must be between 1 and 128 for this bounded Q4 probe')
    if args.ram_gib != 96 or args.disk_gib != 128:
        raise SystemExit('Pinned test-budget values are 96 GiB RAM and 128 GiB disk')
    if not 1 <= args.max_fill_mib <= 1400:
        raise SystemExit('Private filler is capped at 1400 MiB')
    try:
        store = Path(args.store).resolve(strict=True)
        original_store = Path(args.original_store).resolve(strict=True)
        mount = Path(args.checkpoint_mount).resolve(strict=True)
        if (store == original_store or store.is_relative_to(original_store)
                or original_store.is_relative_to(store)):
            raise EnospcProbeError('Refusing to run against the original source Store')
        if (store / 'checkpoints').is_symlink() or mount != (store / 'checkpoints').resolve(strict=True):
            raise EnospcProbeError('Only the disposable Store checkpoints directory may be mounted')
        preflight = validate_preflight_inputs(args)
        mount_info = validate_ext4_mount(
            mount, args.loop_image, parent_mount_namespace=args.parent_mount_namespace)
        if args.preflight_only:
            print(json.dumps({'status':'PREPARED_NOT_RUN',
                'scope':'input/profile/source-checkpoint and mount preflight only; no service, COW request, or filesystem fill',
                'inputs':preflight, 'mount':mount_info},
                sort_keys=True), flush=True)
            return 0
        result = run_probe(args)
        print(json.dumps({'status':result['status'], 'result_path':str(Path(args.run_dir) / 'result.json'),
                          'native_sha256':result.get('native_sha256')}, sort_keys=True), flush=True)
        if result.get('status') != 'PASS':
            return 1
    except Exception as exc:
        print(f'q4-enospc: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
