"""Real Q4 COW HTTP checkpoint-write EFBIG and same-operation retry probe.

The runner does not change production source or profile settings. It starts an
owned loopback service with an in-process observer around the real checkpoint
writer, warms an isolated branch while unlimited, then applies RLIMIT_FSIZE
only to that exact HTTP process. It restores the limit before asking the
coordinator to retry publication of the already-prepared bytes.

CPU tests validate evidence guards only. This runner's real mode is GPU work
and must be explicitly reviewed and scheduled before execution.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import queue
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from urllib.parse import quote, urlsplit


class PublicationFaultError(RuntimeError):
    """The observed kernel fault/recovery did not satisfy the exact contract."""


def _pinned_path(value, label: str) -> Path:
    raw = Path(value).expanduser().absolute()
    if raw.is_symlink():
        raise PublicationFaultError(f'{label} path must not be a symlink')
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise PublicationFaultError(f'{label} path ancestors must not redirect through symlinks')
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_efbig_writer_observation(observation, operation_id: str, branch: str,
                                     *, limit_bytes: int = 64 * 1024**2) -> dict:
    """Accept only EFBIG from the real prepared checkpoint payload writer."""
    if type(observation) is not dict:
        raise PublicationFaultError('Checkpoint writer observation is missing')
    if (observation.get('operation_id') != operation_id or observation.get('branch') != branch
            or observation.get('destination_kind') != 'checkpoint-payload-state'
            or observation.get('exception_type') != 'OSError'
            or type(observation.get('errno')) is not int or observation['errno'] != errno.EFBIG
            or not isinstance(observation.get('message'), str)
            or 'File too large' not in observation['message']
            or not isinstance(observation.get('snapshot'), str) or not observation['snapshot']
            or not isinstance(observation.get('payload_name'), str)
            or not observation['payload_name'].endswith('.state')
            or type(observation.get('requested_payload_bytes')) is not int
            or type(limit_bytes) is not int or limit_bytes <= 0
            or observation['requested_payload_bytes'] <= limit_bytes
            or observation.get('published_payload_exists') is not False
            or observation.get('published_manifest_exists') is not False
            or observation.get('staging_files_after') != []):
        raise PublicationFaultError('Observed failure was not EFBIG during atomic checkpoint-payload publication')
    return dict(snapshot=observation['snapshot'], payload_name=observation['payload_name'],
                errno=observation['errno'], requested_payload_bytes=observation['requested_payload_bytes'])


def validate_fault_stream(events) -> None:
    """Reject false durable acknowledgements; the real handler may close on OSError."""
    if type(events) is not list or any(type(row) is not dict for row in events):
        raise PublicationFaultError('Fault request stream is malformed')
    if any(row.get('event') == 'complete' for row in events):
        raise PublicationFaultError('Faulted HTTP request emitted a false durable complete event')
    terminal = [row for row in events if row.get('event') in ('error', 'failure')]
    if len(terminal) > 1:
        raise PublicationFaultError('Fault request stream has duplicate terminal errors')
    if terminal and terminal[0].get('event') not in ('error', 'failure'):
        raise PublicationFaultError('Fault request stream has an invalid terminal event')


def validate_faulted_operation(status, head_before, head_after, *, operation_id: str,
                               branch: str, parent_snapshot: str) -> dict:
    """Require the operation to remain retryable without moving its source head."""
    if (type(status) is not dict or status.get('operation_id') != operation_id
            or status.get('branch') != branch or status.get('scope') != 'coordinator-process'
            or status.get('state') != 'failed' or status.get('durable') is not False
            or status.get('publication_retryable') is not True or 'receipt' in status):
        raise PublicationFaultError('Faulted operation is not a retryable, non-durable coordinator failure')
    if (type(head_before) is not dict or type(head_after) is not dict
            or head_before.get('name') != branch or head_before.get('tip') != parent_snapshot
            or head_after != head_before):
        raise PublicationFaultError('Failed publication moved or changed its source branch head')
    return dict(parent_snapshot=parent_snapshot, revision=head_before.get('revision'))


def validate_reported_write_failure(events, *, expected_errno: int) -> None:
    """A controlled backend write fault must be visible to its connected client."""
    validate_fault_stream(events)
    errors = [row for row in events if row.get('event') == 'error']
    if len(errors) != 1:
        raise PublicationFaultError('Write failure did not produce one explicit client error')
    error = errors[0]
    message = error.get('error')
    if (error.get('cancelled') is not False or not isinstance(message, str)
            or f'[Errno {expected_errno}]' not in message
            or os.strerror(expected_errno) not in message):
        raise PublicationFaultError('Client error did not preserve the original backend write failure')


def validate_retry_publication(response, operation_id: str, branch: str, target_snapshot: str,
                               head_before: dict, head_after: dict, *, moves_before, moves_after,
                               operation_status=None,
                               work_before=None, work_after=None) -> dict:
    """Check one same-ID publication-only retry and its exact single CAS."""
    receipt = response.get('receipt') if type(response) is dict else None
    result_branch = receipt.get('branch') if type(receipt) is dict else None
    if (type(response) is not dict or response.get('event') != 'complete'
            or type(receipt) is not dict or receipt.get('snapshot') != target_snapshot
            or type(result_branch) is not dict or result_branch.get('name') != branch
            or type(result_branch.get('revision')) is not int
            or type(head_before.get('revision')) is not int
            or result_branch.get('revision') != head_before['revision'] + 1
            or result_branch.get('tip') != target_snapshot or head_after != result_branch):
        raise PublicationFaultError('Retry receipt does not bind the prepared snapshot to one branch advance')
    if (type(moves_before) is not list or type(moves_after) is not list
            or moves_after != moves_before + [operation_id]):
        raise PublicationFaultError('Retry did not create exactly one move under the original operation ID')
    if (type(operation_status) is not dict or operation_status.get('operation_id') != operation_id
            or operation_status.get('branch') != branch or operation_status.get('state') != 'durable'
            or operation_status.get('durable') is not True
            or operation_status.get('publication_retryable') is not False
            or operation_status.get('receipt') != receipt):
        raise PublicationFaultError('Durable status is not bound to the original operation/retry receipt')
    if (work_before is None) != (work_after is None):
        raise PublicationFaultError('Native work comparison is incomplete')
    if work_before is not None and work_after != work_before:
        raise PublicationFaultError('Publication-only retry changed native work counters/trace')
    return dict(snapshot=target_snapshot, revision=result_branch['revision'])


def validate_limit_isolation(receipt, *, expected_server_pid: int, expected_native_pid: int,
                             limit_bytes: int = 64 * 1024**2) -> None:
    """Prove the soft limit targeted the same server PID and never its child."""
    if type(receipt) is not dict:
        raise PublicationFaultError('RLIMIT evidence is missing')
    server = receipt.get('server')
    native = receipt.get('native')
    applied = receipt.get('applied')
    restored = receipt.get('restored')
    if any(type(row) is not dict for row in (server, native, applied, restored)):
        raise PublicationFaultError('RLIMIT evidence is incomplete')
    original = (server.get('soft_fsize'), server.get('hard_fsize'))
    hard_limit = original[1]
    if hard_limit != resource.RLIM_INFINITY and limit_bytes > hard_limit:
        raise PublicationFaultError('Requested soft file-size limit exceeds the unchanged hard limit')
    if (server.get('pid') != expected_server_pid or native.get('pid') != expected_native_pid
            or native.get('parent_pid') != expected_server_pid
            or applied.get('pid') != expected_server_pid
            or restored.get('pid') != expected_server_pid
            or server.get('start_time') != applied.get('start_time')
            or server.get('start_time') != restored.get('start_time')
            or server.get('sigxfsz_ignored') is not True
            or applied.get('soft_fsize') != limit_bytes
            or applied.get('hard_fsize') != original[1]
            or (native.get('soft_fsize'), native.get('hard_fsize')) != original
            or (restored.get('soft_fsize'), restored.get('hard_fsize')) != original):
        raise PublicationFaultError('RLIMIT_FSIZE was not isolated to and restored on the owned HTTP process')


def validate_idle_metrics(metrics: dict) -> None:
    """Require a status snapshot with no active worker or queued request."""
    if type(metrics) is not dict or type(metrics.get('GPU')) is not list:
        raise PublicationFaultError('Coordinator idle metrics are malformed')
    active = metrics.get('active_per_gpu')
    if type(active) is not dict or not active:
        raise PublicationFaultError('Coordinator idle metrics lack per-GPU activity')
    expected = {str(gpu) for gpu in metrics['GPU']}
    if set(active) != expected or any(type(value) is not int or value != 0 for value in active.values()):
        raise PublicationFaultError('Coordinator still has active or mismatched GPU workers')
    queued = metrics.get('queued')
    if type(queued) is not int or queued != 0:
        raise PublicationFaultError('Coordinator still has queued requests')


def validate_source_checkpoint_unchanged(before: dict, after: dict, *, phase: str) -> None:
    if type(before) is not dict or type(after) is not dict or before != after:
        raise PublicationFaultError(f'Original source checkpoint changed {phase}')


def validate_applied_limit(expected: tuple[int, int], observed: tuple[int, int]) -> None:
    if type(expected) is not tuple or type(observed) is not tuple or observed != expected:
        raise PublicationFaultError('HTTP process file-size limit changed before restoration')


def _wait_http_idle(client, timeout: float, progress) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, last = client.get('/cow/status')
        if status != 200:
            raise PublicationFaultError(f'Coordinator status returned HTTP {status}')
        try:
            validate_idle_metrics(last)
        except PublicationFaultError:
            time.sleep(.1)
            continue
        progress.event('coordinator-idle', {'queued': last['queued'],
            'active_per_gpu': last['active_per_gpu']})
        return last
    raise PublicationFaultError(f'Coordinator did not become idle before deadline: {last}')


def validate_followup(response, expected_branch: str, expected_revision: int) -> dict:
    """Ensure the same service can durably serve a request after publication retry."""
    receipt = response.get('receipt') if type(response) is dict else None
    branch = receipt.get('branch') if type(receipt) is dict else None
    answer = response.get('answer') if type(response) is dict else None
    tokens = answer.get('tokens') if type(answer) is dict else None
    if (type(response) is not dict or response.get('event') != 'complete'
            or type(receipt) is not dict or type(branch) is not dict
            or branch.get('name') != expected_branch or type(branch.get('revision')) is not int
            or branch['revision'] != expected_revision + 1
            or branch.get('tip') != receipt.get('snapshot')
            or not isinstance(tokens, list) or not tokens
            or any(type(token) is not int for token in tokens)):
        raise PublicationFaultError('Following request did not complete durably after publication recovery')
    return dict(snapshot=receipt['snapshot'], revision=branch['revision'], token_count=len(tokens))


def _read_proc(pid: int) -> dict:
    proc = Path('/proc') / str(pid)
    raw_stat = (proc / 'stat').read_text(encoding='ascii')
    end = raw_stat.rfind(')')
    if end < 0:
        raise PublicationFaultError('Cannot parse owned process identity')
    fields = raw_stat[end + 2:].split()
    if len(fields) < 20:
        raise PublicationFaultError('Owned process stat record is incomplete')
    status = (proc / 'status').read_text(encoding='ascii')
    sig_ign_line = next((line for line in status.splitlines() if line.startswith('SigIgn:')), None)
    ignored = int(sig_ign_line.split()[1], 16) if sig_ign_line else 0
    return dict(pid=pid, parent_pid=int(fields[1]), start_time=fields[19],
                exe=os.readlink(proc / 'exe'),
                argv=[value.decode('utf-8', 'replace') for value in (proc / 'cmdline').read_bytes().split(b'\0') if value],
                sigxfsz_ignored=bool(ignored & (1 << (signal.SIGXFSZ - 1))))


def _children(pid: int) -> list[int]:
    """Return direct children attached to any live task of this process."""
    process = Path('/proc') / str(pid)
    task_root = process / 'task'
    try:
        task_paths = list(task_root.iterdir())
    except OSError as exc:
        raise PublicationFaultError(f'Cannot enumerate owned process tasks: {exc}') from exc
    found = set()
    for task in task_paths:
        child_file = task / 'children'
        try:
            content = child_file.read_text(encoding='ascii')
        except FileNotFoundError as exc:
            # A thread may exit between task enumeration and reading its file.
            if not task.exists():
                continue
            raise PublicationFaultError(f'Cannot read child list for live task {task.name}') from exc
        except OSError as exc:
            raise PublicationFaultError(f'Cannot read child list for task {task.name}: {exc}') from exc
        try:
            values = [int(value) for value in content.split()]
        except ValueError as exc:
            raise PublicationFaultError(f'Invalid child PID in task {task.name}') from exc
        if any(value <= 0 for value in values):
            raise PublicationFaultError(f'Nonpositive child PID in task {task.name}')
        found.update(values)
    if not process.is_dir():
        raise PublicationFaultError(f'Owned process {pid} exited during child enumeration')
    return sorted(found)


def _process_fsize(pid: int) -> tuple[int, int]:
    soft, hard = resource.prlimit(pid, resource.RLIMIT_FSIZE)
    return int(soft), int(hard)


def _head_history(client, branch: str) -> tuple[dict, dict]:
    status, head = client.get('/cow/head?branch=' + quote(branch, safe=''))
    if status != 200 or type(head) is not dict:
        raise PublicationFaultError(f'Cannot read branch head (HTTP {status})')
    status, history = client.get('/cow/workspace/history?branch=' + quote(branch, safe=''))
    if status != 200 or type(history) is not dict:
        raise PublicationFaultError(f'Cannot read canonical branch history (HTTP {status})')
    return head, history


def _canonical_history_sha(history: dict) -> str:
    """Hash durable projection while excluding transient request status."""
    if type(history) is not dict:
        raise PublicationFaultError('Canonical history projection is malformed')
    durable = {key: value for key, value in history.items() if key != 'pending_request'}
    return hashlib.sha256(json.dumps(durable, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _terminal(events: list[dict]) -> dict:
    complete = [event for event in events if event.get('event') == 'complete']
    if complete:
        if len(complete) != 1:
            raise PublicationFaultError('HTTP stream contained duplicate complete events')
        return complete[0]
    failures = [event for event in events if event.get('event') in ('error', 'failure')]
    if not failures:
        raise PublicationFaultError('HTTP stream ended without a complete or failure event')
    return failures[-1]


def _operation_status(client, operation_id: str, branch: str, deadline: float) -> dict:
    while time.monotonic() < deadline:
        status, value = client.get('/cow/operation?operation=' + quote(operation_id, safe=''))
        if status == 200 and type(value) is dict:
            if value.get('operation_id') != operation_id or value.get('branch') != branch:
                raise PublicationFaultError('Operation status is not bound to the requested operation/branch')
            if value.get('state') in ('durable', 'failed', 'cancelled'):
                return value
        elif status not in (404, 409):
            raise PublicationFaultError(f'Operation status returned HTTP {status}')
        time.sleep(.1)
    raise PublicationFaultError(f'Operation {operation_id} did not reach a terminal state')


def _native_trace_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise PublicationFaultError('Native trace contains malformed JSON') from exc
        if not isinstance(item, dict):
            raise PublicationFaultError('Native trace row is not an object')
        rows.append(item)
    return rows


def _snapshot_file_fingerprint(store: Path, snapshot: str) -> dict:
    checkpoints = store / 'checkpoints'
    manifest = checkpoints / f'{snapshot}.json'
    if manifest.is_symlink() or not manifest.is_file():
        raise PublicationFaultError(f'Expected checkpoint manifest missing for snapshot {snapshot}')
    envelope = json.loads(manifest.read_text(encoding='utf-8'))
    body = envelope.get('manifest') if type(envelope) is dict else None
    payload_sha = body.get('payload_sha256') if type(body) is dict else None
    if not isinstance(payload_sha, str) or len(payload_sha) != 64:
        raise PublicationFaultError('Checkpoint manifest lacks payload SHA-256')
    payload = checkpoints / f'{payload_sha}.state'
    if payload.is_symlink() or not payload.is_file() or _sha256(payload) != payload_sha:
        raise PublicationFaultError('Checkpoint payload missing or does not match manifest identity')
    return {manifest.name: dict(bytes=manifest.stat().st_size, sha256=_sha256(manifest)),
            payload.name: dict(bytes=payload.stat().st_size, sha256=payload_sha)}


def _snapshot_record_fingerprint(store: Path, snapshot: str) -> str:
    # Store snapshot object IDs are content-addressed. Locate exactly one direct
    # object file by the requested lowercase SHA and hash the bytes.
    if len(snapshot) != 64 or any(char not in '0123456789abcdef' for char in snapshot):
        raise PublicationFaultError('Snapshot ID is not a content identifier')
    candidate = store / 'objects' / (snapshot + '.json')
    if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != snapshot:
        raise PublicationFaultError('Canonical snapshot object is absent or corrupt')
    return snapshot


def _runtime_closure(profile: dict, runtime_root: Path) -> dict:
    """Bind runtime files to the profile's selected source closure by bytes."""
    closure = profile.get('native_build', {}).get('selected_runtime_source_closure')
    files = closure.get('files_sha256') if type(closure) is dict else None
    if type(files) is not dict or not files:
        raise PublicationFaultError('Profile lacks a selected runtime source closure')
    verified = {}
    for declared, expected in files.items():
        parts = PurePosixPath(str(declared).replace('\\', '/')).parts
        candidates = []
        for marker in ('serve', 'experimental', 'tools'):
            if marker in parts:
                suffix = Path(*parts[parts.index(marker):])
                candidate = runtime_root / suffix
                if candidate.is_file() and not candidate.is_symlink():
                    candidates.append(candidate)
        unique = {item.resolve(): item for item in candidates}
        matches = [item for item in unique.values() if _sha256(item) == expected]
        if len(matches) != 1:
            raise PublicationFaultError(f'Runtime source closure mismatch or ambiguity for {declared}')
        verified[str(declared)] = dict(path=str(matches[0].resolve()), sha256=_sha256(matches[0]))
    return dict(source_commit=closure.get('source_commit'), files=verified,
                files_sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def _wrapper_source(writer_events: Path) -> str:
    """Build a child bootstrap that observes, but does not replace, the writer."""
    return r'''import json, os, signal, sys, threading, traceback
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
from context_time_travel.real import BranchingModel
from context_time_travel.coordinator import ContextCoordinator
local = threading.local()
original_publish = BranchingModel.publish_prepared_checkpoint
original_write = BranchingModel._publish_checkpoint_parts
original_coordinator_publish = ContextCoordinator._publish
event_path = sys.argv[1]
server_args = sys.argv[2:]
def append_event(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
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
    try:
        return original_write(self, destination, parts)
    except OSError as error:
        destination = os.fspath(destination)
        if destination.endswith(".state"):
            directory = os.path.dirname(destination)
            stages = sorted(name for name in os.listdir(directory) if name.startswith(".stage-"))
            snapshot = getattr(local, "snapshot", None)
            append_event({"operation_id": getattr(local, "operation_id", None),
                "branch": getattr(local, "branch", None), "snapshot": snapshot,
                "destination_kind": "checkpoint-payload-state", "payload_name": os.path.basename(destination),
                "requested_payload_bytes": sum(len(part) for part in parts),
                "exception_type": type(error).__name__, "errno": error.errno,
                "message": str(error), "stage_name": None, "staging_files_after": stages,
                "published_payload_exists": os.path.exists(destination),
                "published_manifest_exists": bool(snapshot and os.path.exists(os.path.join(directory, snapshot + ".json")))})
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


def _write_server_wrapper(run_dir: Path, writer_events: Path) -> Path:
    wrapper = run_dir / 'server_wrapper.py'
    content = _wrapper_source(writer_events)
    with wrapper.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    wrapper.chmod(0o600)
    return wrapper


def _parse_head_and_history(client, branch):
    from run_cow_client_recovery import _head_history, _require_ok
    head, history = _head_history(client, branch)
    if type(head.get('revision')) is not int or not isinstance(head.get('tip'), str):
        raise PublicationFaultError('Selected branch head has invalid revision/tip')
    if history.get('controller_status') not in ('active', 'stopped'):
        raise PublicationFaultError(f"Selected branch controller state is unsupported: {history.get('controller_status')}")
    return head, history


def _client_request(client, route, body, *, progress, timeout):
    events, disconnected = client.post_stream(route, body)
    if disconnected:
        raise PublicationFaultError('Unexpected HTTP disconnect during publication-fault request')
    progress.event(body['operation_id'], {'event': 'stream_eof', 'route': route,
                                          'event_count': len(events)})
    if time.monotonic() > timeout:
        raise PublicationFaultError('HTTP request exceeded its deadline')
    return events


def _complete_receipt(events, *, branch, expected_revision):
    terminal = [row for row in events if row.get('event') == 'complete']
    if len(terminal) != 1:
        raise PublicationFaultError('Expected exactly one complete receipt')
    receipt_event = terminal[0]
    receipt = receipt_event.get('receipt')
    answer = receipt_event.get('answer')
    branch_row = receipt.get('branch') if type(receipt) is dict else None
    if (type(answer) is not dict or type(answer.get('tokens')) is not list or not answer['tokens']
            or type(receipt) is not dict or type(branch_row) is not dict
            or branch_row.get('name') != branch or type(branch_row.get('revision')) is not int
            or branch_row.get('revision') != expected_revision + 1
            or branch_row.get('tip') != receipt.get('snapshot')
            or answer.get('snapshot') != receipt.get('snapshot')):
        raise PublicationFaultError('Warm/follow-up completion receipt is malformed or stale')
    return receipt_event


def _generation_body(history, head, branch, operation, max_tokens, content):
    controller = history.get('controller_status')
    if controller == 'active':
        return '/cow/text/resume', dict(branch=branch, revision=head['revision'],
            operation_id=operation, max_tokens=max_tokens)
    if controller == 'stopped':
        return '/cow/text/append-user', dict(branch=branch, revision=head['revision'],
            operation_id=operation, content=content, max_tokens=max_tokens)
    raise PublicationFaultError('Cannot choose request route for unsupported controller state')


def _proc_identity_matches(pid, start_time):
    try:
        return _read_proc(pid)['start_time'] == start_time
    except (FileNotFoundError, ProcessLookupError):
        return False


def _ensure_q4_profile(profile: dict) -> None:
    execution = profile.get('execution')
    if (type(execution) is not dict or execution.get('GPU') != [0]
            or execution.get('prefill_chunk') != 8192
            or execution.get('capacity') != 98304):
        raise PublicationFaultError('Profile is not the pinned single-GPU Q4 98304/8192 FP16 configuration')


def run_probe(args) -> dict:
    if not sys.platform.startswith('linux') or not hasattr(resource, 'prlimit'):
        raise PublicationFaultError('Real publication-fault runner requires Linux resource.prlimit')
    run_dir = Path(args.run_dir).expanduser().absolute()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    output_path = run_dir / 'result.json'
    progress_path = run_dir / 'progress.json'
    profile_path = _pinned_path(args.profile, 'Profile')
    store = _pinned_path(args.store, 'Store')
    runtime_root = _pinned_path(args.runtime_root, 'Runtime root')
    api_src = _pinned_path(args.api_src, 'API source')
    api_scripts = _pinned_path(args.api_scripts, 'API scripts')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    _ensure_q4_profile(profile)
    if store.is_symlink() or not (store / 'refs.json').is_file() or not (store / 'checkpoints').is_dir():
        raise PublicationFaultError('Store must be a pre-existing private copy of a completed Q4 CLI Store')
    runtime = _runtime_closure(profile, runtime_root)
    native_binary = Path(profile['execution']['sut_command'][0]).resolve(strict=True)
    native_sha = _sha256(native_binary)
    if native_sha != args.expected_native_sha256:
        raise PublicationFaultError('Native executable hash differs from the explicitly selected build')
    if _sha256(profile_path) != args.expected_profile_sha256:
        raise PublicationFaultError('Serving profile hash differs from the explicitly selected profile')

    script_dir = runtime_root / 'experimental' / 'branching_context' / 'scripts'
    if not script_dir.is_dir():
        raise PublicationFaultError('Pinned runtime lacks the existing client-recovery process helpers')
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(runtime_root))
    sys.path.insert(0, str(api_src))
    sys.path.insert(0, str(api_scripts))
    from run_cow_client_recovery import HttpClient, ProgressTracker, ServiceProcess, _ready_url, _trace_worker, _fork

    writer_events = run_dir / 'writer-errors.jsonl'
    writer_events.touch(mode=0o600, exist_ok=False)
    progress = ProgressTracker(run_dir, 'q4-publication-efbig-retry')
    progress.start()
    wrapper = _write_server_wrapper(run_dir, writer_events)
    trace_path = run_dir / 'pool-trace.json'
    native_trace = run_dir / 'native-trace.jsonl'
    native_trace.touch(mode=0o600, exist_ok=False)
    command = [sys.executable, str(wrapper), str(writer_events), '--profile', str(profile_path),
               '--store', str(store), '--host', '127.0.0.1', '--port', '0', '--text',
               '--ram-gib', str(args.ram_gib), '--disk-gib', str(args.disk_gib),
               '--pool-trace-output', str(trace_path)]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(runtime_root), str(api_src), str(api_scripts)))
    env['CTT_NATIVE_TRACE'] = str(native_trace)
    service = ServiceProcess(command, cwd=runtime_root, env=env,
        stdout_path=run_dir / 'server.stdout.log', stderr_path=run_dir / 'server.stderr.log',
        timeout=args.timeout, progress=progress)
    result = dict(schema='q4-publication-efbig-retry-v1', status='RUNNING',
        scope='real Q4 COW HTTP service; real kernel RLIMIT_FSIZE/EFBIG; no disk exhaustion',
        profile_path=str(profile_path), profile_sha256=_sha256(profile_path),
        native_binary=str(native_binary), native_sha256=native_sha,
        runtime_root=str(runtime_root), api_src=str(api_src), api_scripts=str(api_scripts),
        runtime_source_closure=runtime, store_path=str(store), branch=args.branch,
        run_directory=str(run_dir), runner_sha256=_sha256(Path(__file__).resolve()),
        writer_wrapper=str(wrapper), writer_wrapper_sha256=_sha256(wrapper),
        fsize_limit_bytes=args.fsize_mib * 1024**2, steps=[])
    server_pid = native_pid = None
    original_server_limit = None
    original_native_limit = None
    server_start = None
    fault_error = None
    try:
        progress.phase('start_real_q4_http_service')
        ready = service.start()
        server_pid = service.process.pid
        progress.server_started(server_pid)
        result['server_pid'] = server_pid
        result['server_ready'] = ready
        client = HttpClient(_ready_url(ready), timeout=args.timeout, progress=progress)
        status, health = client.get('/health')
        if status != 200 or health.get('service') != 'context-cow':
            raise PublicationFaultError('The owned loopback service is not the COW HTTP service')
        if ready.get('execution_profile_id') != profile.get('execution_profile_id'):
            raise PublicationFaultError('Ready service execution profile differs from pinned profile')
        if not isinstance(args.branch, str) or not args.branch:
            raise PublicationFaultError('A selected existing branch is required')
        source_head, source_history = _parse_head_and_history(client, args.branch)
        if source_history.get('controller_status') not in ('active', 'stopped'):
            raise PublicationFaultError('Selected CLI branch has no supported controller state')
        source_snapshot = source_head['tip']
        source_record_sha = _snapshot_record_fingerprint(store, source_snapshot)
        source_checkpoint = _snapshot_file_fingerprint(store, source_snapshot)
        result['source_before'] = dict(head=source_head, history_sha256=_canonical_history_sha(source_history),
            snapshot_record_sha256=source_record_sha, checkpoint_files=source_checkpoint)

        child_pids = _children(server_pid)
        if len(child_pids) != 1:
            raise PublicationFaultError(f'Expected one native child after service readiness, got {child_pids}')
        native_pid = child_pids[0]
        native_info = _read_proc(native_pid)
        if native_info.get('parent_pid') != server_pid or Path(native_info['exe']).resolve() != native_binary:
            raise PublicationFaultError('Native child identity does not match selected profile executable')
        progress.native_started(native_pid)
        server_info = _read_proc(server_pid)
        server_start = server_info['start_time']
        original_server_limit = _process_fsize(server_pid)
        original_native_limit = _process_fsize(native_pid)
        if (original_server_limit[1] != resource.RLIM_INFINITY
                and args.fsize_mib * 1024**2 > original_server_limit[1]):
            raise PublicationFaultError('Configured test limit exceeds the HTTP process hard file-size limit')
        result['process_before_fault'] = dict(server={**server_info, 'soft_fsize': original_server_limit[0],
            'hard_fsize': original_server_limit[1]}, native={**native_info,
            'soft_fsize': original_native_limit[0], 'hard_fsize': original_native_limit[1]})

        # Fork the already acknowledged source so warming and the fault never
        # mutate the caller's original CLI branch.
        nonce = uuid.uuid4().hex
        target_branch = 'Q4Fault' + nonce[:16]
        fork_operation = 'op-fork-' + nonce
        progress.phase('fork_isolated_warm_branch')
        # Reuse the client helper exercised by the actual recovery campaign.
        # The fork route returns a BranchHead directly, unlike save receipts.
        fork_head = _fork(client, args.branch, source_snapshot, target_branch, fork_operation)

        # Unlimited warm-up loads source bytes into the native worker, ensuring
        # the later constrained request cannot fail during Python cold import.
        warm_head, warm_history = _parse_head_and_history(client, target_branch)
        warm_route, warm_body = _generation_body(warm_history, warm_head, target_branch,
            'op-warm-' + nonce, 1, 'Warm this branch before the publication fault test.')
        progress.phase('warm_source_materialization_unlimited')
        warm_events = client.post_stream(warm_route, warm_body)[0]
        warm_complete = _complete_receipt(warm_events, branch=target_branch,
                                          expected_revision=warm_head['revision'])
        warm_snapshot = warm_complete['receipt']['snapshot']
        warm_head, warm_history = _parse_head_and_history(client, target_branch)
        if warm_head.get('tip') != warm_snapshot:
            raise PublicationFaultError('Warm-up durable receipt is not the selected branch head')
        warm_checkpoint = _snapshot_file_fingerprint(store, warm_snapshot)
        warm_record_sha = _snapshot_record_fingerprint(store, warm_snapshot)
        validate_source_checkpoint_unchanged(source_checkpoint,
            _snapshot_file_fingerprint(store, source_snapshot), phase='after warm-up')
        source_head_after_warm, source_history_after_warm = _parse_head_and_history(client, args.branch)
        if (source_head_after_warm != source_head
                or _canonical_history_sha(source_history_after_warm) != _canonical_history_sha(source_history)):
            raise PublicationFaultError('Warm-up changed the original CLI source branch/history')
        result['warmup'] = dict(operation_id=warm_body['operation_id'], route=warm_route,
            event_summary=[row.get('event') for row in warm_events], branch_head=warm_head,
            controller_status=warm_history.get('controller_status'), snapshot_record_sha256=warm_record_sha,
            checkpoint_files=warm_checkpoint)

        # Pick a real native child now and verify its pre-limit identity. It is
        # the same direct child that handled the warm-up; a respawn invalidates
        # the isolation claim.
        if (_children(server_pid) != [native_pid] or not _proc_identity_matches(server_pid, server_start)
                or not _proc_identity_matches(native_pid, native_info['start_time'])):
            raise PublicationFaultError('HTTP/native process identity changed after warm-up')
        server_info = _read_proc(server_pid)
        native_info = _read_proc(native_pid)
        if not server_info['sigxfsz_ignored']:
            raise PublicationFaultError('Test-only HTTP process did not ignore SIGXFSZ')
        limit_before = _process_fsize(server_pid)
        native_limit_before = _process_fsize(native_pid)
        if limit_before != original_server_limit or native_limit_before != original_native_limit:
            raise PublicationFaultError('Process file-size limit changed before controlled fault')

        fault_source_head, fault_source_history = _parse_head_and_history(client, target_branch)
        fault_source_snapshot = fault_source_head['tip']
        fault_source_record_sha = _snapshot_record_fingerprint(store, fault_source_snapshot)
        fault_source_checkpoint = _snapshot_file_fingerprint(store, fault_source_snapshot)
        fault_source_history_sha = _canonical_history_sha(fault_source_history)
        _wait_http_idle(client, args.timeout, progress)
        route, body = _generation_body(fault_source_history, fault_source_head, target_branch,
            'op-q4-fault-' + nonce, args.max_tokens, 'Continue the prepared Q4 test turn.')
        result['fault_operation_id'] = body['operation_id']
        result['fault_route'] = route
        result['fault_parent'] = dict(head=fault_source_head, history_sha256=fault_source_history_sha,
            snapshot_record_sha256=fault_source_record_sha, checkpoint_files=fault_source_checkpoint)
        trace_before_fault = native_trace.read_bytes()
        applied_limit = (args.fsize_mib * 1024**2, original_server_limit[1])
        if (not _proc_identity_matches(server_pid, server_start)
                or not _proc_identity_matches(native_pid, native_info['start_time'])
                or _children(server_pid) != [native_pid]):
            raise PublicationFaultError('Owned HTTP/native process identity changed before applying RLIMIT_FSIZE')
        resource.prlimit(server_pid, resource.RLIMIT_FSIZE, applied_limit)
        actual_applied = _process_fsize(server_pid)
        native_after_limit = _process_fsize(native_pid)
        applied_evidence = dict(server={**_read_proc(server_pid), 'soft_fsize': actual_applied[0],
                                        'hard_fsize': actual_applied[1]},
                               native={**_read_proc(native_pid), 'soft_fsize': native_after_limit[0],
                                       'hard_fsize': native_after_limit[1]})
        if actual_applied != applied_limit or native_after_limit != original_native_limit:
            raise PublicationFaultError('RLIMIT_FSIZE was not isolated to the exact server process')
        result['rlimit_applied'] = applied_evidence

        try:
            progress.phase('constrained_request_until_real_payload_efbig')
            events = client.post_stream(route, body)[0]
            validate_fault_stream(events)
            validate_reported_write_failure(events, expected_errno=errno.EFBIG)
            result['fault_stream'] = [{key: value for key, value in row.items() if key != 'text'} for row in events]
            operation = _operation_status(client, body['operation_id'], target_branch,
                                          time.monotonic() + args.timeout)
            after_fault_head, after_fault_history = _parse_head_and_history(client, target_branch)
            validate_faulted_operation(operation, fault_source_head, after_fault_head,
                operation_id=body['operation_id'], branch=target_branch, parent_snapshot=fault_source_snapshot)
            if _canonical_history_sha(after_fault_history) != fault_source_history_sha:
                raise PublicationFaultError('Failed request changed canonical branch history')
            if _snapshot_record_fingerprint(store, fault_source_snapshot) != fault_source_record_sha:
                raise PublicationFaultError('Failed request changed its canonical parent snapshot bytes')
            if _snapshot_file_fingerprint(store, fault_source_snapshot) != fault_source_checkpoint:
                raise PublicationFaultError('Failed request changed its durable parent checkpoint bytes')
            if _snapshot_record_fingerprint(store, source_snapshot) != source_record_sha:
                raise PublicationFaultError('Failed request changed the original CLI root snapshot bytes')
            validate_source_checkpoint_unchanged(source_checkpoint,
                _snapshot_file_fingerprint(store, source_snapshot), phase='after fault')
            writer_rows = [json.loads(line) for line in writer_events.read_text(encoding='utf-8').splitlines() if line]
            exact = [row for row in writer_rows if row.get('operation_id') == body['operation_id']
                     and row.get('branch') == target_branch]
            if len(exact) != 1:
                raise PublicationFaultError(f'Expected one writer-side EFBIG observation, got {len(exact)}')
            writer = validate_efbig_writer_observation(exact[0], body['operation_id'], target_branch,
                limit_bytes=args.fsize_mib * 1024**2)
            if exact[0]['snapshot'] == fault_source_snapshot:
                raise PublicationFaultError('Prepared failure snapshot aliases the source snapshot')
            failed_target_checkpoint_dir = store / 'checkpoints'
            if ((failed_target_checkpoint_dir / f"{writer['snapshot']}.json").exists()
                    or (failed_target_checkpoint_dir / exact[0]['payload_name']).exists()
                    or any(failed_target_checkpoint_dir.glob('.stage-*'))):
                raise PublicationFaultError('Failed payload publication left a target, manifest, or stage file')
            progress.phase('fault_classified_and_parent_unchanged', completed='fault_classified')
            _wait_http_idle(client, args.timeout, progress)
            trace_after_fault = native_trace.read_bytes()
            if trace_after_fault == trace_before_fault:
                raise PublicationFaultError('Faulted request produced no native trace work after the warm-up baseline')
            native_before_retry = trace_after_fault
            server_current = _process_fsize(server_pid)
            validate_applied_limit(applied_limit, server_current)
            result['limit_before_restore'] = dict(pid=server_pid, start_time=server_start,
                soft_fsize=server_current[0], hard_fsize=server_current[1])
            if not _proc_identity_matches(server_pid, server_start):
                raise PublicationFaultError('HTTP PID changed before restoring file-size limit')
            resource.prlimit(server_pid, resource.RLIMIT_FSIZE, original_server_limit)
            restored = _process_fsize(server_pid)
            native_restored = _process_fsize(native_pid)
            limit_receipt = dict(
                server=dict(pid=server_pid, start_time=server_start,
                    soft_fsize=original_server_limit[0], hard_fsize=original_server_limit[1],
                    sigxfsz_ignored=server_info['sigxfsz_ignored']),
                native=dict(pid=native_pid, parent_pid=server_pid,
                    start_time=native_info['start_time'], soft_fsize=original_native_limit[0],
                    hard_fsize=original_native_limit[1]),
                applied=dict(pid=server_pid,start_time=server_start,soft_fsize=actual_applied[0],
                    hard_fsize=actual_applied[1]),
                restored=dict(pid=server_pid,start_time=server_start,soft_fsize=restored[0],
                    hard_fsize=restored[1]))
            validate_limit_isolation(limit_receipt, expected_server_pid=server_pid,
                expected_native_pid=native_pid, limit_bytes=args.fsize_mib * 1024**2)
            if restored != original_server_limit or native_restored != original_native_limit:
                raise PublicationFaultError('Original RLIMIT_FSIZE was not restored before retry')
            result['rlimit_isolation'] = limit_receipt
            progress.phase('restored_limit_and_retry_same_operation')
            retry_code, retry_response = client.post_json('/cow/publication/retry',
                dict(operation_id=body['operation_id']))
            if retry_code != 200:
                raise PublicationFaultError(f'Same-operation publication retry returned HTTP {retry_code}: {retry_response}')
            retry_head, retry_history = _parse_head_and_history(client, target_branch)
            operation_after_retry = _operation_status(client, body['operation_id'], target_branch,
                                                       time.monotonic() + args.timeout)
            _wait_http_idle(client, args.timeout, progress)
            if (operation_after_retry.get('durable') is not True
                    or operation_after_retry.get('publication_retryable') is not False
                    or operation_after_retry.get('receipt') != retry_response.get('receipt')):
                raise PublicationFaultError('Operation status does not expose the one durable retry receipt')
            moves_before = list(fault_source_history.get('moves', []))
            moves_after = [row.get('operation_id') for row in retry_history.get('moves', [])]
            receipt = retry_response.get('receipt')
            target_snapshot = writer['snapshot']
            native_after_retry = native_trace.read_bytes()
            retry_result = validate_retry_publication(retry_response, body['operation_id'], target_branch,
                target_snapshot, fault_source_head, retry_head,
                moves_before=[row.get('operation_id') for row in moves_before], moves_after=moves_after,
                operation_status=operation_after_retry,
                work_before=native_before_retry, work_after=native_after_retry)
            if not isinstance(receipt, dict) or receipt.get('snapshot') != target_snapshot:
                raise PublicationFaultError('Retry receipt identifies a different prepared snapshot')
            if _snapshot_file_fingerprint(store, target_snapshot) is None:
                raise PublicationFaultError('Retried target checkpoint did not validate from disk')
            validate_source_checkpoint_unchanged(source_checkpoint,
                _snapshot_file_fingerprint(store, source_snapshot), phase='after retry')
            progress.phase('same_operation_retry_durable', completed='same_operation_retry_durable')
            result['retry'] = dict(http_status=retry_code, response=retry_response,
                operation_status=operation_after_retry, validation=retry_result,
                native_trace_sha256_before=hashlib.sha256(native_before_retry).hexdigest(),
                native_trace_sha256_after=hashlib.sha256(native_after_retry).hexdigest(),
                native_work_unchanged=True)

            # The following request is a separate operation and must still be
            # accepted after the retained publication is completed.
            following_head, following_history = _parse_head_and_history(client, target_branch)
            following_operation = 'op-follow-' + nonce
            following_route, following_body = _generation_body(following_history, following_head,
                target_branch, following_operation, min(args.max_tokens, 8),
                'Confirm the recovered branch remains usable.')
            progress.phase('following_request_after_retry')
            following_events = client.post_stream(following_route, following_body)[0]
            following_terminal = _terminal(following_events)
            if following_terminal.get('event') != 'complete':
                raise PublicationFaultError('Following request did not complete after publication retry')
            following_result = validate_followup(following_terminal, target_branch, following_head['revision'])
            progress.phase('following_request_complete', completed='following_request_complete')
            validate_source_checkpoint_unchanged(source_checkpoint,
                _snapshot_file_fingerprint(store, source_snapshot), phase='after follow-up')
            result['following_request'] = dict(operation_id=following_operation, route=following_route,
                validation=following_result)
        except BaseException as exc:
            fault_error = exc
            raise
        finally:
            # Never let the service retain the injected limit after the fault
            # phase, including when HTTP classification or validation fails.
            if server_pid is not None and server_start is not None and _proc_identity_matches(server_pid, server_start):
                current = _process_fsize(server_pid)
                if current != original_server_limit:
                    resource.prlimit(server_pid, resource.RLIMIT_FSIZE, original_server_limit)
                    if _process_fsize(server_pid) != original_server_limit:
                        raise PublicationFaultError('Failed to restore owned HTTP process file-size limit')

        # The untouched source branch must still reference the original head.
        source_after, source_history_after = _parse_head_and_history(client, args.branch)
        if (source_after != source_head
                or _canonical_history_sha(source_history_after) != _canonical_history_sha(source_history)):
            raise PublicationFaultError('The caller\'s original source branch changed during the probe')
        result['source_after'] = dict(head=source_after, unchanged=True)
        progress.phase('stop_owned_service')
        stop_receipt = service.stop()
        result['service_stop'] = stop_receipt
        trace_record = _trace_worker(run_dir / 'pool-trace.json', server_pid)
        result['native_shutdown'] = trace_record
        if stop_receipt.get('exit_code') != 0 or not stop_receipt.get('graceful'):
            raise PublicationFaultError('Owned HTTP/native service did not close cleanly')
        result['status'] = 'PASS'
        progress.finish('PASS')
        _atomic_json(output_path, result)
        return result
    except BaseException as exc:
        result['status'] = 'FAIL'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        if service.process is not None:
            try:
                result['service_stop'] = service.stop()
            except BaseException as stop_error:
                result['service_stop_error'] = f'{type(stop_error).__name__}: {stop_error}'
        progress.finish('FAIL', result['error'])
        _atomic_json(output_path, result)
        raise
    finally:
        if (server_pid is not None and server_start is not None and original_server_limit is not None
                and _proc_identity_matches(server_pid, server_start)):
            current_limit = _process_fsize(server_pid)
            if current_limit != original_server_limit:
                resource.prlimit(server_pid, resource.RLIMIT_FSIZE, original_server_limit)
                if _process_fsize(server_pid) != original_server_limit:
                    raise PublicationFaultError('Failed to restore owned HTTP process file-size limit')
        if service.process is not None and service.process.poll() is None:
            try:
                service.stop()
            except BaseException:
                pass
        try:
            progress.close()
        except BaseException:
            pass


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--api-src', type=Path, required=True)
    parser.add_argument('--api-scripts', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True,
                        help='Disposable copied CLI campaign Store; source campaign must remain untouched')
    parser.add_argument('--branch', required=True, help='Existing active Q4 CLI branch in copied Store')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-profile-sha256', required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--fsize-mib', type=int, default=64)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=1800)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_tokens <= 128:
        raise SystemExit('max-tokens must be between 1 and 128 for this bounded Q4 probe')
    if args.fsize_mib != 64 or args.ram_gib != 96 or args.disk_gib != 128:
        raise SystemExit('This probe is pinned to 64 MiB FSIZE, 96 GiB RAM, and 128 GiB disk settings')
    try:
        run_probe(args)
    except BaseException as exc:
        print(f'q4-publication-fault: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
