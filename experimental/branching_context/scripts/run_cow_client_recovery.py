"""Exercise COW HTTP resume, disconnect, exact retry, and restart recovery.

The production mode starts the real COW HTTP service with a pinned profile and
an existing Store. The explicit ``--synthetic-fixture`` mode exists only for
CPU tests of the client/server lifecycle; it is never real-model evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit


class RecoveryFailure(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _private_run_dir(path: Path) -> Path:
    path = path.expanduser().absolute()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name == 'posix':
        os.chmod(path, 0o700)
        if path.stat().st_mode & 0o077:
            raise PermissionError('Run directory must be private (0700)')
    return path


class ProgressTracker:
    """Atomically persist private phase/PID progress and raw HTTP stream events."""
    def __init__(self, run_dir: Path, mode: str, *, interval_seconds=30):
        self.run_dir = run_dir
        self.mode = mode
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.heartbeat_error = None
        self.phase_name = 'initialized'
        self.last_completed_step = None
        self.server_pids = []
        self.native_pids = []
        self.last_event = None
        self.status = 'RUNNING'
        self.error = None
        self.events_path = run_dir / 'http-events.jsonl'
        self.events_fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.write_progress()

    def _data(self):
        return dict(schema='cow-client-recovery-progress-v1', mode=self.mode,
            status=self.status, phase=self.phase_name,
            elapsed_seconds=round(time.monotonic() - self.started, 3),
            server_pids=list(self.server_pids), native_pids=list(self.native_pids),
            last_completed_step=self.last_completed_step, last_event=self.last_event,
            error=self.error)

    def write_progress(self):
        with self.lock:
            data = self._data()
            temporary = self.run_dir / f'.progress-{uuid.uuid4().hex}.tmp'
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.run_dir / 'progress.json')
                directory_fd = os.open(self.run_dir, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                raise

    def start(self):
        self.thread = threading.Thread(target=self._heartbeat, name='cow-recovery-heartbeat', daemon=True)
        self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval_seconds):
            try:
                self.write_progress()
            except BaseException as error:
                self.heartbeat_error = error
                return

    def check(self):
        if self.heartbeat_error is not None:
            raise RecoveryFailure(f'Progress heartbeat failed: {self.heartbeat_error}')

    def phase(self, name, *, completed=None):
        with self.lock:
            self.phase_name = name
            if completed is not None:
                self.last_completed_step = completed
        self.write_progress()
        self.check()

    def server_started(self, pid):
        with self.lock:
            self.server_pids.append(pid)
        self.write_progress()

    def native_started(self, pid):
        with self.lock:
            self.native_pids.append(pid)
        self.write_progress()

    def event(self, operation_id, event):
        value = dict(operation_id=operation_id, event=event)
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode('utf-8')
        offset = 0
        while offset < len(raw):
            offset += os.write(self.events_fd, raw[offset:])
        with self.lock:
            self.last_event = event.get('event')
        self.check()

    def flush_events(self):
        os.fsync(self.events_fd)

    def finish(self, status, error=None):
        with self.lock:
            self.status = status
            self.error = error
            self.phase_name = 'finished'
            if status == 'PASS':
                self.last_completed_step = 'pipeline_complete'
        self.write_progress()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
            if self.thread.is_alive():
                raise RecoveryFailure('Progress heartbeat thread did not terminate')
        os.fsync(self.events_fd)
        os.close(self.events_fd)


def checkpoint_fingerprint(store: Path, snapshot: str, *, synthetic_fixture=False) -> dict:
    """Hash the immutable source manifest and its referenced native payload."""
    directory = store / 'checkpoints'
    manifest_path = directory / (snapshot + '.json')
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RecoveryFailure('Source checkpoint manifest is missing or not a regular file')
    envelope = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest = envelope.get('manifest') if isinstance(envelope, dict) else None
    required = {'manifest'} if synthetic_fixture else {'manifest', 'checksum'}
    if (not isinstance(manifest, dict) or set(envelope) != required
            or (not synthetic_fixture and not isinstance(envelope.get('checksum'), str))):
        raise RecoveryFailure('Source checkpoint manifest envelope is malformed')
    if 'checksum' in envelope:
        canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=True, allow_nan=False).encode('ascii') + b'\n'
        if hashlib.sha256(canonical).hexdigest() != envelope['checksum']:
            raise RecoveryFailure('Source checkpoint manifest checksum mismatch')
    if manifest.get('snapshot') != snapshot:
        raise RecoveryFailure('Source checkpoint manifest identifies a different snapshot')
    payload_hash = manifest.get('payload_sha256')
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        raise RecoveryFailure('Source checkpoint manifest has no payload digest')
    payload_path = directory / (payload_hash + '.state')
    if not payload_path.is_file() or payload_path.is_symlink():
        raise RecoveryFailure('Source checkpoint payload is missing or not a regular file')
    rows = {}
    for path in (manifest_path, payload_path):
        stat = path.stat()
        rows[path.name] = dict(bytes=stat.st_size, sha256=_sha256(path),
                               mode=stat.st_mode & 0o777, inode=stat.st_ino)
    if rows[payload_path.name]['sha256'] != payload_hash:
        raise RecoveryFailure('Source checkpoint payload digest differs from its name/manifest')
    return dict(snapshot=snapshot, payload_sha256=payload_hash, files=rows)


class HttpClient:
    def __init__(self, base_url: str, *, timeout: float = 1800, progress=None):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Recovery harness requires an HTTP loopback service')
        self.host = parsed.hostname
        self.port = parsed.port
        self.prefix = parsed.path.rstrip('/')
        self.timeout = timeout
        self.progress = progress

    def _connection(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def get(self, path):
        connection = self._connection()
        connection.request('GET', self.prefix + path)
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        connection.close()
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            value = raw.decode('utf-8', errors='replace')
        return status, value

    def post_json(self, path, body):
        connection = self._connection()
        headers = {'Content-Type': 'application/json'}
        raw = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        connection.request('POST', self.prefix + path, body=raw, headers=headers)
        response = connection.getresponse()
        data = response.read()
        status = response.status
        connection.close()
        try:
            value = json.loads(data)
        except (TypeError, ValueError):
            value = data.decode('utf-8', errors='replace')
        return status, value

    def post_stream(self, path, body, *, disconnect_after_first_text=False):
        """Read NDJSON, or close the actual socket at its first consumed text token."""
        connection = self._connection()
        headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        raw = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        connection.request('POST', self.prefix + path, body=raw, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read().decode('utf-8', errors='replace')
            response.close()
            connection.close()
            raise RecoveryFailure(f'{path} returned HTTP {response.status}: {detail}')
        events = []
        try:
            while True:
                line = response.fp.readline()
                if not line:
                    break
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get('event'), str):
                    raise RecoveryFailure('Server emitted a malformed COW event')
                events.append(event)
                if self.progress is not None:
                    self.progress.event(body.get('operation_id'), event)
                if (disconnect_after_first_text and event.get('event') == 'text'
                        and type(event.get('token_id')) is int):
                    # The token has reached the HTTP client; closing here races
                    # cancellation against the server's durable publication.
                    connection.close()
                    if self.progress is not None:
                        self.progress.flush_events()
                    return events, True
                if event.get('event') in ('complete', 'error'):
                    break
        finally:
            response.close()
            connection.close()
            if self.progress is not None:
                self.progress.flush_events()
        return events, False


def _require_ok(status, value, operation):
    if status != 200:
        raise RecoveryFailure(f'{operation} returned HTTP {status}: {value}')
    return value


def _terminal_event(events):
    terminal = [event for event in events if event.get('event') in ('complete', 'error')]
    if len(terminal) != 1:
        raise RecoveryFailure(f'Expected one terminal stream event; found {len(terminal)}')
    return terminal[0]


def _assert_resume_complete(events, *, expected_revision: int, require_zero_prefill=True):
    final = _terminal_event(events)
    if final.get('event') != 'complete':
        raise RecoveryFailure(f'Resume did not complete durably: {final}')
    answer = final.get('answer')
    receipt = final.get('receipt')
    if not isinstance(answer, dict) or not isinstance(receipt, dict):
        raise RecoveryFailure('Complete event omitted answer or receipt')
    tokens = answer.get('tokens')
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise RecoveryFailure('Completed answer omitted valid generated token IDs')
    counters = answer.get('counter_deltas')
    if not isinstance(counters, dict):
        raise RecoveryFailure('Completed answer omitted native work counters')
    if type(counters.get('forward_tokens')) is not int or counters.get('forward_tokens') != len(tokens):
        raise RecoveryFailure('Zero-input resume forward-token delta differs from output count')
    prefill = counters.get('prefill_tokens')
    if require_zero_prefill and (type(prefill) is not int or prefill != 0):
        raise RecoveryFailure('Zero-input resume performed unexpected prefix prefill work')
    if prefill is not None and (type(prefill) is not int or prefill < 0):
        raise RecoveryFailure('Resume reported malformed prefix-prefill work')
    if (not isinstance(answer.get('snapshot'), str) or not answer.get('snapshot')
            or answer.get('snapshot') != receipt.get('snapshot')):
        raise RecoveryFailure('Answer and durable receipt identify different snapshots')
    branch = receipt.get('branch')
    if (not isinstance(branch, dict) or type(branch.get('revision')) is not int
            or branch.get('revision') != expected_revision + 1
            or branch.get('tip') != receipt.get('snapshot')):
        raise RecoveryFailure('Receipt branch revision/tip does not match the completed answer')
    return dict(event='complete', snapshot=receipt['snapshot'], branch=branch,
                output_tokens=tokens, counter_deltas=counters,
                text_bytes=len(answer.get('text', '').encode('utf-8')))


def _events_after_disconnect(events):
    if not events or not any(event.get('event') == 'text' and type(event.get('token_id')) is int
                             for event in events):
        raise RecoveryFailure('Socket closed before a consumed token reached the client')
    if any(event.get('event') == 'complete' for event in events):
        raise RecoveryFailure('Client disconnect helper read a complete event before closing')


def _event_summary(events):
    """Retain event ordering/token IDs without copying model text into reports."""
    return [dict(event=row.get('event'), token_id=row.get('token_id'),
                 text_bytes=len(row.get('text', '').encode('utf-8')))
            for row in events]


def _operation_status(client, operation, branch, deadline):
    while time.monotonic() < deadline:
        status, value = client.get('/cow/operation?operation=' + operation)
        if status == 200:
            if (not isinstance(value, dict) or value.get('operation_id') != operation or value.get('branch') != branch
                    or value.get('scope') != 'coordinator-process'):
                raise RecoveryFailure('Operation status is not bound to the requested operation and branch')
            if value.get('state') in ('durable', 'cancelled', 'failed'):
                return value
        elif status != 404:
            raise RecoveryFailure(f'Operation status returned HTTP {status}: {value}')
        time.sleep(.1)
    raise RecoveryFailure(f'Operation {operation} did not reach a terminal state')


def _classify_disconnect(operation, branch_head, source_snapshot, expected_operation, expected_branch):
    """Accept only cancellation with an unchanged head or a matching durable move."""
    if not isinstance(operation, dict) or type(operation.get('durable')) is not bool:
        raise RecoveryFailure('Operation status omitted its durable state')
    if (operation.get('operation_id') != expected_operation
            or operation.get('branch') != expected_branch
            or operation.get('scope') != 'coordinator-process'):
        raise RecoveryFailure('Disconnect status is not bound to the expected operation and branch')
    if operation['durable']:
        if operation.get('state') != 'durable':
            raise RecoveryFailure('Durable operation status has a non-durable state')
        receipt = operation.get('receipt')
        branch = receipt.get('branch') if isinstance(receipt, dict) else None
        if (not isinstance(branch, dict) or type(branch.get('revision')) is not int
                or branch.get('tip') != branch_head.get('tip')
                or branch_head.get('revision') != branch.get('revision')
                or branch.get('revision') != 1):
            raise RecoveryFailure('Durable disconnect outcome disagrees with the authoritative branch head')
        return 'publication_won'
    if operation.get('state') == 'cancelled':
        if branch_head.get('tip') != source_snapshot or branch_head.get('revision') != 0:
            raise RecoveryFailure('Cancelled request moved its branch before durable publication')
        return 'cancelled_before_publication'
    raise RecoveryFailure(f'Disconnected operation reached unexpected state: {operation}')


def _assert_retry_consistent(outcome, replay_events, operation, expected_operation, before_head,
                             after_head, before_history, after_history):
    """Bind a retry to canonical state without assuming canceled work was free."""
    replay = _terminal_event(replay_events)
    before_ids = [row.get('id') for row in before_history.get('messages', [])]
    after_ids = [row.get('id') for row in after_history.get('messages', [])]
    before_moves = [row.get('operation_id') for row in before_history.get('moves', [])]
    after_moves = [row.get('operation_id') for row in after_history.get('moves', [])]
    if (any(not isinstance(identity, str) for identity in before_ids + after_ids)
            or len(set(before_ids)) != len(before_ids)
            or len(set(after_ids)) != len(after_ids)):
        raise RecoveryFailure('Canonical history contains missing or duplicate message identities')
    if outcome == 'publication_won':
        if replay.get('event') != 'complete' or replay.get('receipt') != operation.get('receipt'):
            raise RecoveryFailure('Retry did not return the original durable receipt')
        if (after_head != before_head or after_ids != before_ids
                or after_moves != before_moves or expected_operation not in before_moves):
            raise RecoveryFailure('Durable retry changed the already-published branch/history')
        return dict(terminal='complete', canonical_messages=len(after_ids),
                    no_additional_work_required=True)
    if outcome != 'cancelled_before_publication':
        raise RecoveryFailure(f'Unknown disconnect outcome for retry: {outcome}')
    if replay.get('event') == 'error':
        if not replay.get('cancelled'):
            raise RecoveryFailure('Canceled retry returned a non-cancellation error')
        if after_head != before_head or after_ids != before_ids or after_moves != before_moves:
            raise RecoveryFailure('Canceled retry error changed canonical branch/history')
    elif replay.get('event') == 'complete':
        receipt = replay.get('receipt')
        branch = receipt.get('branch') if isinstance(receipt, dict) else None
        if (not isinstance(branch, dict) or branch.get('name') != before_head.get('name')
                or type(branch.get('revision')) is not int
                or type(before_head.get('revision')) is not int
                or branch.get('revision') != before_head.get('revision') + 1
                or branch.get('tip') != receipt.get('snapshot')
                or after_head.get('tip') != branch.get('tip')
                or after_head.get('revision') != branch.get('revision')):
            raise RecoveryFailure('Retried canceled work has no matching single branch publication')
        if after_ids != before_ids:
            raise RecoveryFailure('Retried canceled work duplicated or replaced canonical messages')
        if (after_moves[:len(before_moves)] != before_moves
                or after_moves[len(before_moves):] != [expected_operation]):
            raise RecoveryFailure('Retried canceled work is not bound to exactly one expected operation')
    else:
        raise RecoveryFailure(f'Canceled retry returned unexpected terminal event: {replay}')
    return dict(terminal=replay['event'], canonical_messages=len(after_ids),
                no_additional_work_required=False)


def _head_history(client, branch):
    status, head = client.get('/cow/head?branch=' + branch)
    _require_ok(status, head, 'read branch head')
    status, history = client.get('/cow/workspace/history?branch=' + branch)
    _require_ok(status, history, 'read canonical branch history')
    return head, history


def _fork(client, source, snapshot, new_branch, operation):
    status, result = client.post_json('/cow/fork', dict(name=new_branch, snapshot=snapshot,
        source_branch=source, operation_id=operation))
    _require_ok(status, result, 'fork active branch')
    head = result.get('branch') if isinstance(result.get('branch'), dict) else result
    if head.get('name') != new_branch or head.get('tip') != snapshot:
        raise RecoveryFailure('Fork receipt does not identify the requested source snapshot')
    return head


def _operation_ids(tag):
    nonce = uuid.uuid4().hex
    return dict(resume_branch=f'COWResume{nonce[:20]}', cancel_branch=f'COWCancel{nonce[:20]}',
        fork_resume='op-fork-r-' + nonce, fork_cancel='op-fork-c-' + nonce,
        resume='op-resume-' + nonce, cancel='op-cancel-' + nonce,
        followup='op-followup-' + nonce)


class ServiceProcess:
    """Own a loopback HTTP service and require its graceful terminal receipts."""
    def __init__(self, command, *, cwd, env, stdout_path, stderr_path, timeout, progress=None):
        self.command = list(command)
        self.cwd = str(cwd)
        self.env = dict(env)
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self.timeout = timeout
        self.progress = progress
        self.lines = queue.Queue()
        self.process = None
        self.reader = None
        self.ready = None
        self._stdout_stream = None
        self._owns_process_group = False

    def start(self):
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._stdout_stream = self.stdout_path.open('x', encoding='utf-8', newline='\n')
        stderr = self.stderr_path.open('xb')
        try:
            self.process = subprocess.Popen(self.command, cwd=self.cwd, env=self.env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                start_new_session=(os.name == 'posix'))
        finally:
            stderr.close()
        self._owns_process_group = os.name == 'posix'
        if self.progress is not None:
            self.progress.server_started(self.process.pid)

        def read_stdout():
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                self._stdout_stream.write(line)
                self._stdout_stream.flush()
                self.lines.put(line)
            self.lines.put(None)

        self.reader = threading.Thread(target=read_stdout, name='cow-recovery-stdout', daemon=True)
        self.reader.start()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RecoveryFailure(f'Service exited before ready with code {self.process.returncode}')
            try:
                line = self.lines.get(timeout=min(.25, max(.01, deadline-time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get('event') == 'ready':
                self.ready = value
                return value
        raise RecoveryFailure('Timed out waiting for COW service ready JSON')

    def _signal_owned_process(self, sig):
        if self.process is None:
            return False
        if self._owns_process_group:
            try:
                os.killpg(self.process.pid, sig)
                return True
            except ProcessLookupError:
                return False
        if self.process.poll() is None:
            self.process.send_signal(sig)
            return True
        return False

    def _owned_process_group_exists(self):
        if not self._owns_process_group or self.process is None:
            return False
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _finish_owned_process_group(self):
        """Reap descendants left behind after the HTTP leader exits."""
        if not self._owned_process_group_exists():
            return False
        self._signal_owned_process(signal.SIGTERM)
        deadline = time.monotonic() + 2.0
        while self._owned_process_group_exists() and time.monotonic() < deadline:
            time.sleep(.05)
        forced = True
        if self._owned_process_group_exists():
            self._signal_owned_process(signal.SIGKILL)
            deadline = time.monotonic() + 3.0
            while self._owned_process_group_exists() and time.monotonic() < deadline:
                time.sleep(.05)
        if self._owned_process_group_exists():
            raise RecoveryFailure('Owned service process group remains after forced cleanup')
        return forced

    def stop(self):
        if self.process is None:
            if self._stdout_stream is not None:
                self._stdout_stream.close()
            return dict(exit_code=None, graceful=False)
        forced = False
        if self.process.poll() is None:
            # Let the HTTP leader own ordinary shutdown so it can close its
            # native backend cleanly.  Signal the process group only when
            # escalation is required or descendants remain afterward.
            if self._owns_process_group:
                self.process.send_signal(signal.SIGINT)
            else:
                self._signal_owned_process(signal.SIGINT)
        try:
            code = self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            forced = True
            if os.name == 'posix':
                self._signal_owned_process(signal.SIGTERM)
            else:
                self.process.terminate()
            try:
                code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    self._signal_owned_process(signal.SIGKILL)
                else:
                    self.process.kill()
                code = self.process.wait(timeout=5)
        try:
            forced = self._finish_owned_process_group() or forced
        finally:
            if self.reader is not None:
                self.reader.join(timeout=5)
                if self.reader.is_alive():
                    raise RecoveryFailure('Service stdout reader did not terminate')
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self._stdout_stream is not None:
                self._stdout_stream.close()
        return dict(exit_code=code, graceful=(code == 0 and not forced), pid=self.process.pid,
                    command=self.command, forced_process_group_cleanup=forced)


def _trace_worker(trace_path: Path, service_pid: int):
    if not trace_path.is_file():
        raise RecoveryFailure('Pool trace is missing after service shutdown')
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    if (not isinstance(trace, dict) or not isinstance(trace.get('events'), list)
            or trace.get('overflowed') is not False
            or type(trace.get('event_count')) is not int
            or trace['event_count'] != len(trace['events'])
            or type(trace.get('process_id')) is not int
            or trace.get('process_id') != service_pid):
        raise RecoveryFailure('Pool trace is malformed, overflowed, or belongs to another service process')
    rows = trace['events']
    worker = [row for row in rows if row.get('event') == 'native_worker_closed']
    if len(worker) != 1:
        raise RecoveryFailure(f'Expected one native worker-close trace, found {len(worker)}')
    record = worker[0]
    if (type(record.get('process_id')) is not int or record.get('process_id') != service_pid
            or type(record.get('returncode')) is not int or record.get('returncode') != 0
            or record.get('close_succeeded') is not True):
        raise RecoveryFailure('Native worker did not close successfully with exit code zero')
    if type(record.get('pid')) is not int or record['pid'] == service_pid:
        raise RecoveryFailure('Native worker PID is missing or unexpectedly equals HTTP PID')
    return dict(native_pid=record['pid'], native_exit_code=record['returncode'],
                close_succeeded=True, trace_sha256=_sha256(trace_path), event_count=len(rows))


def _server_command(mode, store, run_dir, ordinal, profile=None, *, ram_gib=96, disk_gib=128):
    if mode == 'synthetic':
        return [sys.executable, '-m', 'serve.cow_browser_fixture', '--store', str(store), '--port', '0']
    trace = run_dir / f'pool-trace-{ordinal}.json'
    return [sys.executable, '-m', 'serve.cow_http', '--profile', str(profile), '--store', str(store),
        '--host', '127.0.0.1', '--port', '0', '--text', '--ram-gib', str(ram_gib),
        '--disk-gib', str(disk_gib), '--pool-trace-output', str(trace)]


def _ready_url(record):
    address = record.get('address')
    if isinstance(address, list) and len(address) >= 2:
        return f'http://{address[0]}:{address[1]}'
    if isinstance(address, tuple) and len(address) >= 2:
        return f'http://{address[0]}:{address[1]}'
    if isinstance(address, str):
        return address
    raise RecoveryFailure('Ready record contains an unsupported HTTP address')


def _stop_after_service_setup_failure(process, failure):
    """Stop a started service without hiding the startup/validation failure."""
    try:
        process.stop()
    except BaseException as cleanup_error:
        raise RecoveryFailure(
            f'Service setup failed ({type(failure).__name__}: {failure}); '
            f'cleanup also failed ({type(cleanup_error).__name__}: {cleanup_error})') from failure


def _start_service(mode, store, run_dir, ordinal, *, profile, timeout, ram_gib, disk_gib, progress):
    progress.phase(f'service_{ordinal}_starting')
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    import_roots = [str(repo_root), str(repo_root / 'experimental' / 'branching_context' / 'src')]
    prior_pythonpath = env.get('PYTHONPATH')
    if prior_pythonpath:
        import_roots.append(prior_pythonpath)
    env['PYTHONPATH'] = os.pathsep.join(import_roots)
    native_trace = run_dir / f'native-trace-{ordinal}.jsonl'
    if mode == 'real':
        descriptor = os.open(native_trace, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        env['CTT_NATIVE_TRACE'] = str(native_trace.absolute())
    command = _server_command(mode, store, run_dir, ordinal, profile,
                              ram_gib=ram_gib, disk_gib=disk_gib)
    process = ServiceProcess(command, cwd=repo_root, env=env,
        stdout_path=run_dir / f'server-{ordinal}.stdout.log',
        stderr_path=run_dir / f'server-{ordinal}.stderr.log', timeout=timeout,
        progress=progress)
    try:
        ready = process.start()
        base_url = _ready_url(ready)
        progress.phase(f'service_{ordinal}_ready')
        client = HttpClient(base_url, timeout=timeout, progress=progress)
        status, health = client.get('/health')
        _require_ok(status, health, 'health check')
        if mode == 'real' and health.get('service') != 'context-cow':
            raise RecoveryFailure('Started server is not the native COW HTTP service')
        if mode == 'real':
            expected_profile = json.loads(profile.read_text(encoding='utf-8')).get('execution_profile_id')
            if ready.get('execution_profile_id') != expected_profile:
                raise RecoveryFailure('Ready service profile identity differs from the pinned profile')
        return process, ready, client
    except BaseException as setup_error:
        _stop_after_service_setup_failure(process, setup_error)
        raise


def _run_pipeline(*, mode: str, store: Path, branch: str, run_dir: Path, progress,
                 profile: Path | None = None,
                 max_tokens: int = 8, timeout: float = 1800,
                 previous_native_pid: int | None = None,
                 ram_gib: int = 96, disk_gib: int = 128) -> dict:
    """Run restart/resume/cancellation checks against one persistent Store."""
    if mode not in ('real', 'synthetic'):
        raise ValueError('mode must be real or synthetic')
    if mode == 'real' and profile is None:
        raise ValueError('real mode requires a pinned serving profile')
    if mode == 'real' and previous_native_pid is None:
        raise ValueError('real mode requires the previous campaign native PID for restart comparison')
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    store = store.expanduser().absolute()
    if store.is_symlink():
        raise ValueError('Store path must not be a symbolic link')
    store = store.resolve(strict=True)
    if not store.is_dir():
        raise ValueError('Store must be an existing regular directory')
    profile_sha = None
    if profile is not None:
        profile = profile.expanduser().resolve(strict=True)
        profile_sha = _sha256(profile)
    names = _operation_ids('recovery')
    initial_head = None
    initial_history = None
    source_checkpoint_before = None
    service_records = []
    trace_records = []
    scope = ('real model/native HTTP service; single selected profile and loopback'
             if mode == 'real' else 'synthetic FlowModel over real HTTP/Store/coordinator; CPU-only, not model/GPU evidence')
    process, ready, client = _start_service(mode, store, run_dir, 1, profile=profile,
        timeout=timeout, ram_gib=ram_gib, disk_gib=disk_gib, progress=progress)
    try:
        progress.phase('read_existing_active_branch')
        initial_head, initial_history = _head_history(client, branch)
        if initial_history.get('controller_status') != 'active':
            raise RecoveryFailure('Selected branch has no saved active controller to resume')
        initial_revision = initial_head.get('revision')
        source_snapshot = initial_head.get('tip')
        if type(initial_revision) is not int or not isinstance(source_snapshot, str):
            raise RecoveryFailure('Selected branch head is malformed')
        source_checkpoint_before = checkpoint_fingerprint(store, source_snapshot,
            synthetic_fixture=(mode == 'synthetic'))

        progress.phase('fork_isolated_resume_and_disconnect_branches')
        resume_branch = names['resume_branch']
        cancel_branch = names['cancel_branch']
        _fork(client, branch, source_snapshot, resume_branch, names['fork_resume'])
        _fork(client, branch, source_snapshot, cancel_branch, names['fork_cancel'])

        resume_body = dict(branch=resume_branch, revision=0, operation_id=names['resume'],
                           max_tokens=max_tokens)
        progress.phase('resume_zero_input')
        resume_events, disconnected = client.post_stream('/cow/text/resume', resume_body)
        if disconnected:
            raise RecoveryFailure('Resume validation request unexpectedly used disconnect mode')
        resumed = _assert_resume_complete(resume_events, expected_revision=0,
            require_zero_prefill=(mode == 'real'))
        progress.phase('resume_zero_input', completed='resume_zero_input')
        status, source_after_resume = client.get('/cow/head?branch=' + branch)
        _require_ok(status, source_after_resume, 'recheck original branch after resume clone')
        if source_after_resume.get('tip') != source_snapshot or source_after_resume.get('revision') != initial_revision:
            raise RecoveryFailure('Resuming a fork moved the original active branch')

        cancel_body = dict(branch=cancel_branch, revision=0, operation_id=names['cancel'],
                           max_tokens=max_tokens)
        progress.phase('disconnect_after_first_consumed_token')
        cancel_events, disconnected = client.post_stream('/cow/text/resume', cancel_body,
                                                         disconnect_after_first_text=True)
        if not disconnected:
            raise RecoveryFailure('Disconnect probe completed without closing at a consumed text token')
        _events_after_disconnect(cancel_events)
        progress.phase('wait_for_authoritative_disconnect_outcome')
        operation = _operation_status(client, names['cancel'], cancel_branch, time.monotonic() + timeout)
        status, cancel_head = client.get('/cow/head?branch=' + cancel_branch)
        _require_ok(status, cancel_head, 'read branch after disconnected request')
        status, cancel_history = client.get('/cow/workspace/history?branch=' + cancel_branch)
        _require_ok(status, cancel_history, 'read canonical history after disconnected request')
        outcome = _classify_disconnect(operation, cancel_head, source_snapshot,
            names['cancel'], cancel_branch)
        progress.phase('disconnect_outcome_classified', completed='disconnect_' + outcome)

        native_trace = run_dir / 'native-trace-1.jsonl'
        trace_before_retry = native_trace.read_bytes() if mode == 'real' else None
        progress.phase('exact_retry')
        replay_events, replay_disconnected = client.post_stream('/cow/text/resume', cancel_body)
        if replay_disconnected:
            raise RecoveryFailure('Exact retry unexpectedly disconnected')
        status, retry_head = client.get('/cow/head?branch=' + cancel_branch)
        _require_ok(status, retry_head, 'read branch after exact retry')
        status, retry_history = client.get('/cow/workspace/history?branch=' + cancel_branch)
        _require_ok(status, retry_history, 'read canonical history after exact retry')
        retry_result = _assert_retry_consistent(outcome, replay_events, operation, names['cancel'],
            cancel_head, retry_head, cancel_history, retry_history)
        trace_unchanged = (native_trace.read_bytes() == trace_before_retry) if mode == 'real' else None
        if outcome == 'publication_won' and mode == 'real' and not trace_unchanged:
            raise RecoveryFailure('Durable exact retry added native worker commands or computation')
        progress.phase('exact_retry', completed='exact_retry_' + retry_result['terminal'])

        status, source_after_cancel = client.get('/cow/head?branch=' + branch)
        _require_ok(status, source_after_cancel, 'recheck original branch after disconnected request')
        if source_after_cancel.get('tip') != source_snapshot or source_after_cancel.get('revision') != initial_revision:
            raise RecoveryFailure('Disconnected fork request moved the original active branch')

        source_checkpoint_after = checkpoint_fingerprint(store, source_snapshot,
            synthetic_fixture=(mode == 'synthetic'))
        if source_checkpoint_after['files'] != source_checkpoint_before['files']:
            raise RecoveryFailure('Original active-source checkpoint bytes changed during resume/disconnect')
        first_stop = process.stop()
        service_records.append(dict(ready=ready, stop=first_stop))
        if first_stop['exit_code'] != 0:
            raise RecoveryFailure(f'First COW service did not exit cleanly: {first_stop}')
        if mode == 'real':
            worker_trace = _trace_worker(run_dir / 'pool-trace-1.json', first_stop['pid'])
            trace_records.append(worker_trace)
            progress.native_started(worker_trace['native_pid'])
        progress.phase('first_service_stopped', completed='first_service_stopped')

        # Restart from the same persistent Store and ensure the unchanged
        # original active branch remains resumable after cancellation/replay.
        progress.phase('restart_service')
        process2, ready2, client2 = _start_service(mode, store, run_dir, 2, profile=profile,
            timeout=timeout, ram_gib=ram_gib, disk_gib=disk_gib, progress=progress)
        try:
            progress.phase('verify_canonical_head_after_restart')
            restarted_head, restarted_history = _head_history(client2, branch)
            if (restarted_head.get('tip') != source_snapshot
                    or restarted_head.get('revision') != initial_revision
                    or restarted_history.get('controller_status') != 'active'):
                raise RecoveryFailure('Restart did not recover the exact original active branch')
            followup_body = dict(branch=branch, revision=initial_revision,
                operation_id=names['followup'], max_tokens=max_tokens)
            progress.phase('resume_after_restart')
            followup_events, did_disconnect = client2.post_stream('/cow/text/resume', followup_body)
            if did_disconnect:
                raise RecoveryFailure('Post-restart follow-up request unexpectedly disconnected')
            followup = _assert_resume_complete(followup_events, expected_revision=initial_revision,
                require_zero_prefill=(mode == 'real'))
            progress.phase('resume_after_restart', completed='restart_resume_completed')
            source_checkpoint_after_followup = checkpoint_fingerprint(store, source_snapshot,
                synthetic_fixture=(mode == 'synthetic'))
            if source_checkpoint_after_followup['files'] != source_checkpoint_before['files']:
                raise RecoveryFailure('Post-restart request changed the retained original source checkpoint')
            second_stop = process2.stop()
            service_records.append(dict(ready=ready2, stop=second_stop))
            if second_stop['exit_code'] != 0:
                raise RecoveryFailure(f'Restarted COW service did not exit cleanly: {second_stop}')
            if mode == 'real':
                worker_trace = _trace_worker(run_dir / 'pool-trace-2.json', second_stop['pid'])
                trace_records.append(worker_trace)
                progress.native_started(worker_trace['native_pid'])
        except BaseException:
            process2.stop()
            raise
    except BaseException:
        if process.process is not None and process.process.poll() is None:
            process.stop()
        raise

    distinct_from_prior = None
    if mode == 'real':
        native_pids = [row['native_pid'] for row in trace_records]
        if len(native_pids) != 2 or native_pids[0] == native_pids[1]:
            raise RecoveryFailure('Restart did not create a distinct native worker PID')
        if previous_native_pid is not None:
            distinct_from_prior = all(pid != previous_native_pid for pid in native_pids)
            if not distinct_from_prior:
                raise RecoveryFailure('New native worker PID matches the prior campaign worker')
    return dict(schema='cow-client-recovery-v1', status='PASS', mode=mode, scope=scope,
        numerical_qualification=dict(status='NOT_PERFORMED',
            reason='This lifecycle harness does not compare model logits or a numerical oracle.'),
        store=str(store), branch=branch, original_revision=initial_revision,
        original_snapshot=source_snapshot, profile_sha256=profile_sha,
        previous_native_pid=previous_native_pid, new_native_pids=[row['native_pid'] for row in trace_records],
        native_pids_distinct_from_previous=distinct_from_prior, service_instances=service_records,
        resume=resumed, disconnected_operation=dict(outcome=outcome, operation_id=names['cancel'],
            state=operation['state'], durable=operation['durable'], provisional_events=_event_summary(cancel_events),
            cancellation_coverage=(outcome == 'cancelled_before_publication'),
            retry_terminal=retry_result['terminal'], retry_canonical_message_count=retry_result['canonical_messages'],
            retry_no_additional_work_required=retry_result['no_additional_work_required'],
            retry_native_trace_unchanged=trace_unchanged),
        post_restart_followup=followup, source_checkpoint_before=source_checkpoint_before,
        source_checkpoint_after=source_checkpoint_after_followup,
        source_checkpoint_unchanged=True, worker_close_traces=trace_records)


def run_pipeline(*, mode: str, store: Path, branch: str, run_dir: Path,
                 profile: Path | None = None, max_tokens: int = 8,
                 timeout: float = 1800, previous_native_pid: int | None = None,
                 ram_gib: int = 96, disk_gib: int = 128,
                 heartbeat_interval_seconds: float = 30):
    run_dir = _private_run_dir(run_dir)
    progress = ProgressTracker(run_dir, mode,
        interval_seconds=heartbeat_interval_seconds).start()
    try:
        result = _run_pipeline(mode=mode, store=store, branch=branch, run_dir=run_dir,
            profile=profile, max_tokens=max_tokens, timeout=timeout,
            previous_native_pid=previous_native_pid, ram_gib=ram_gib,
            disk_gib=disk_gib, progress=progress)
        progress.check()
        progress.finish('PASS')
        return result
    except BaseException as error:
        progress.finish('FAIL', f'{type(error).__name__}: {error}')
        raise
    finally:
        progress.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--branch', required=True, help='Existing saved branch with an active controller')
    parser.add_argument('--run-dir', type=Path, required=True, help='New private output directory')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--profile', type=Path, help='Pinned production profile; starts real COW service')
    mode.add_argument('--synthetic-fixture', action='store_true', help='CPU-only synthetic HTTP fixture mode')
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--previous-native-pid', type=int)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    mode_value = 'synthetic' if args.synthetic_fixture else 'real'
    report_path = args.report
    if report_path is not None and report_path.exists():
        parser.error('--report must not already exist')
    try:
        value = run_pipeline(mode=mode_value, store=args.store, branch=args.branch,
            run_dir=args.run_dir, profile=args.profile,
            max_tokens=args.max_tokens, timeout=args.timeout_seconds,
            previous_native_pid=args.previous_native_pid, ram_gib=args.ram_gib, disk_gib=args.disk_gib)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            report_path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(value, sort_keys=True, separators=(',',':')))
        return 0
    except Exception as error:
        failure = dict(schema='cow-client-recovery-v1', status='FAIL', mode=mode_value,
                       error=type(error).__name__, message=str(error),
                       numerical_qualification=dict(status='NOT_PERFORMED',
                           reason='This lifecycle harness does not compare model logits or a numerical oracle.'))
        if report_path is not None and not report_path.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            report_path.write_text(json.dumps(failure, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(failure, sort_keys=True, separators=(',',':')), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
