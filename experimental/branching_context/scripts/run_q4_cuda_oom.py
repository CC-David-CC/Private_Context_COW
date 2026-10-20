"""Harness-only Q4 CUDA VMM allocation-failure probe.

The runner observes an actual greedy request in the pinned HTTP service. It
waits at the native ``generate`` send boundary, applies pressure with the
separately pinned VMM helper, records the unchanged native error and reads the
native counters on the same pipe. It never patches engine/API code or converts
an error response into success. CPU tests cover evidence guards only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote


class OOMProbeError(RuntimeError):
    """The observed allocation-failure protocol did not satisfy its contract."""


class ServiceStartupFailure(OOMProbeError):
    def __init__(self, startup_error, cleanup_receipt):
        super().__init__(f'Service failed during startup: {type(startup_error).__name__}: {startup_error}')
        self.cleanup_receipt = cleanup_receipt


def _start_service_owned(service):
    """Start a service and synchronously reap its owned group on readiness failure."""
    try:
        return service.start()
    except BaseException as startup_error:
        process = getattr(service, 'process', None)
        receipt = {'pid': process.pid if process is not None else None}
        cleanup_error = None
        try:
            receipt['stop'] = service.stop()
        except BaseException as exc:
            cleanup_error = exc
            receipt['stop_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        receipt['returncode'] = process.poll() if process is not None else None
        receipt['pid_absent'] = process is None or not Path('/proc', str(process.pid)).exists()
        if cleanup_error is not None:
            receipt['cleanup_failed'] = True
        raise ServiceStartupFailure(startup_error, receipt) from startup_error


def _object(value, label):
    if type(value) is not dict:
        raise OOMProbeError(f'{label} must be a JSON object')
    return value


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise OOMProbeError(f'{label} must be an integer >= {minimum}')
    return value


def validate_gate_record(value, *, operation_id: str, branch: str, native_pid: int,
                         expected_forward: int) -> dict:
    row = _object(value, 'generate gate')
    stats = _object(row.get('stats'), 'gate stats')
    request_start = _object(row.get('request_start_stats'), 'request-start stats')
    command = row.get('command')
    if (row.get('event') != 'before_generate' or row.get('operation_id') != operation_id
            or row.get('branch') != branch or row.get('native_pid') != native_pid
            or row.get('request_active') is not True or not isinstance(command, str)
            or re.fullmatch(r'generate [1-9][0-9]* [0-9]+(?: [0-9]+)*', command) is None
            or row.get('injected_commands') != ['pages', 'stats']
            or _integer(request_start.get('forward_tokens'), 'request-start forward_tokens') != 0
            or _integer(request_start.get('prefill_tokens'), 'request-start prefill_tokens') != 0
            or _integer(stats.get('forward_tokens'), 'gate forward_tokens') != expected_forward):
        raise OOMProbeError('Gate is not bound to a zero-prefill active generate request and fresh counters')
    if _integer(stats.get('prefill_tokens'), 'gate prefill_tokens') != 0:
        raise OOMProbeError('Gate observed prefix-prefill work before the failed generate')
    pages = row.get('pages')
    if type(pages) is not list:
        raise OOMProbeError('Gate lacks the injected native page-ownership witness')
    return dict(event=row['event'], operation_id=operation_id, branch=branch,
                native_pid=native_pid, command=command, stats=dict(stats),
                request_start_stats=dict(request_start), pages=pages,
                injected_commands=list(row['injected_commands']))


def validate_helper_ready(value) -> dict:
    row = _object(value, 'helper ready frame')
    if row.get('status') != 'ready' or _integer(row.get('granularity'), 'helper granularity', minimum=1) <= 0:
        raise OOMProbeError('Pressure helper did not report a valid VMM-ready device')
    return dict(status='ready', granularity=row['granularity'])


def validate_pressure_frame(value) -> dict:
    row = _object(value, 'pressure frame')
    for key in ('held_bytes', 'physical_allocations', 'free_before', 'free_after',
                'total_bytes', 'driver_failures', 'out_of_memory_failures', 'last_error'):
        _integer(row.get(key), f'pressure {key}')
    if (row.get('status') != 'full' or row['held_bytes'] <= 0
            or row['physical_allocations'] <= 0 or row['total_bytes'] < row['held_bytes']
            or row['free_after'] > row['free_before']
            or row['out_of_memory_failures'] <= 0 or row['last_error'] != 2
            or row.get('last_operation') != 'cuMemCreate'):
        raise OOMProbeError('Pressure helper did not end on a real cuMemCreate CUDA OOM')
    return dict(row)


def validate_helper_release(value, pressure) -> dict:
    released = _object(value, 'pressure release frame')
    row = _object(pressure, 'pressure frame')
    if (released.get('status') != 'released'
            or _integer(released.get('released_bytes'), 'released bytes') != row['held_bytes']
            or _integer(released.get('free_after'), 'free bytes after release') <= 0):
        raise OOMProbeError('Pressure helper did not release its exact allocation set')
    return dict(released)


def validate_helper_pressure(value, release) -> dict:
    row = validate_pressure_frame(value)
    validate_helper_release(release, row)
    return row


def _counter(stats, key):
    return _integer(_object(stats, 'native counters').get(key), f'native counter {key}')


def validate_native_oom(response, before, after, *, expected_forward: int) -> dict:
    raw = _object(response, 'original native response')
    before = _object(before, 'pre-pressure native stats')
    after = _object(after, 'post-error native stats')
    if raw.get('status') != 'error' or not isinstance(raw.get('message'), str) or not raw['message']:
        raise OOMProbeError('Native generate did not return its original explicit error frame')
    forward_before = _counter(before, 'forward_tokens')
    forward_after = _counter(after, 'forward_tokens')
    prefill_before = _counter(before, 'prefill_tokens')
    prefill_after = _counter(after, 'prefill_tokens')
    oom_before = _counter(before, 'vmm_out_of_memory_failures')
    oom_after = _counter(after, 'vmm_out_of_memory_failures')
    failures_before = _counter(before, 'vmm_driver_failures')
    failures_after = _counter(after, 'vmm_driver_failures')
    if (forward_before != expected_forward or forward_after != expected_forward
            or prefill_before != 0 or prefill_after != 0
            or oom_after <= oom_before or failures_after <= failures_before
            or _counter(after, 'vmm_last_driver_error') != 2
            or after.get('vmm_last_failed_operation') != 'cuMemCreate'):
        raise OOMProbeError('Same-pipe stats do not prove cuMemCreate OOM with zero generated forward work')
    ignored = {'vmm_driver_failures','vmm_out_of_memory_failures','vmm_last_driver_error',
               'vmm_last_failed_operation'}
    protected = {'consumed','prefill_tokens','saved_running_bytes','written_file_bytes','captures',
                 'snapshot_allocations','snapshot_releases','snapshot_live_images'}
    changed = {key: (before.get(key), after.get(key)) for key in protected
               if key in before and before.get(key) != after.get(key)}
    changed.update({key: (value, after.get(key)) for key, value in before.items()
               if key not in ignored and key not in protected and key in ('forward_tokens',)
               and type(value) in (int, float) and type(after.get(key)) in (int, float)
               and after[key] != value})
    if changed:
        raise OOMProbeError(f'Native work/capture counters changed during failed generate: {changed}')
    return dict(response=raw, forward_delta=forward_after-forward_before,
                oom_failure_delta=oom_after-oom_before,
                driver_failure_delta=failures_after-failures_before,
                stats_before=dict(before), stats_after=dict(after))


def validate_owned_process_exit(value, *, expected_pid: int, expected_start_time: str,
                                expected_exe: str) -> dict:
    row = _object(value, 'owned process exit')
    if (row.get('pid') != expected_pid or row.get('start_time') != expected_start_time
            or row.get('exe') != expected_exe or type(row.get('exit_code')) is not int
            or row['exit_code'] != 0 or row.get('pid_absent_after_wait') is not True):
        raise OOMProbeError('Owned helper process identity or terminal exit is not verified')
    return dict(row)


def validate_probe_result(result) -> str:
    row = _object(result, 'probe result')
    status = row.get('status')
    oom = _object(row.get('native_oom'), 'native OOM evidence')
    restart = row.get('restart_followup')
    if (oom.get('response', {}).get('status') != 'error'
            or oom.get('oom_failure_delta', 0) <= 0
            or type(restart) is not dict or restart.get('event') != 'complete'
            or not isinstance(restart.get('tokens'), list) or not restart['tokens']):
        raise OOMProbeError('Result lacks native OOM and successful clean-restart follow-up evidence')
    same = row.get('same_service_followup')
    if status == 'PASS':
        if (type(same) is not dict or same.get('event') != 'complete'
                or not isinstance(same.get('tokens'), list) or not same['tokens']):
            raise OOMProbeError('PASS requires same-service follow-up completion')
    elif status == 'FAIL-RECOVERY':
        if type(same) is not dict or same.get('event') not in ('error', 'failure'):
            raise OOMProbeError('FAIL-RECOVERY requires preserved same-service failure evidence')
    elif status == 'FAIL':
        if type(row.get('error')) is not dict:
            raise OOMProbeError('FAIL requires a preserved harness error')
    else:
        raise OOMProbeError('Probe result status must be PASS, FAIL-RECOVERY, or FAIL')
    return status


def validate_fault_http_events(events) -> dict:
    if type(events) is not list or not events or any(type(row) is not dict for row in events):
        raise OOMProbeError('Faulted HTTP request has no valid event sequence')
    terminal = [row for row in events if row.get('event') in ('complete','error','failure')]
    errors = [row for row in terminal if row.get('event') == 'error']
    if (len(terminal) != 1 or len(errors) != 1
            or any(row.get('event') in ('complete','durable','text','token','snapshot')
                   or 'receipt' in row for row in events)
            or not isinstance(errors[0].get('error'), str) or not errors[0]['error']):
        raise OOMProbeError('Faulted HTTP response acknowledged, emitted output, or lacked one explicit error')
    return dict(error=errors[0], event_count=len(events))


def validate_import_witness(value, expected_imports: dict) -> dict:
    row = _object(value, 'runtime import witness')
    modules = _object(row.get('modules'), 'imported runtime modules')
    required = {'context_time_travel.real','context_time_travel.coordinator',
        'context_time_travel.strata_backend','context_time_travel.execution_receipt',
        'context_time_travel.checkpoint_cache','serve.cow_http'}
    if row.get('event') != 'runtime_import_witness' or set(modules) != required:
        raise OOMProbeError('Runtime import witness lacks the exact required module set')
    for name, item in modules.items():
        item = _object(item, f'import witness {name}')
        path = item.get('path')
        sha = item.get('sha256')
        if (not isinstance(path, str) or not isinstance(sha, str)
                or expected_imports.get(path) != sha):
            raise OOMProbeError(f'Imported module origin/hash is not pinned: {name}')
    return dict(modules=modules)


def build_import_expectations(profile_imports, *, http_path, http_sha256, expected_http_sha256):
    """Keep API-profile pins distinct from the independently pinned HTTP source."""
    profile_imports = dict(profile_imports)
    path = str(Path(http_path).resolve(strict=True))
    if not isinstance(expected_http_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', expected_http_sha256) is None:
        raise OOMProbeError('Expected HTTP source SHA-256 is malformed')
    if http_sha256 != expected_http_sha256:
        raise OOMProbeError('HTTP source SHA-256 differs from the independent source pin')
    if path in profile_imports:
        raise OOMProbeError('HTTP source must be recorded separately from the execution-profile API closure')
    combined = dict(profile_imports)
    combined[path] = http_sha256
    source = {'path': path, 'sha256': http_sha256,
              'binding': 'independent_explicit_service_source_pin'}
    return combined, source


def run_status_exit_code(result) -> int:
    status = validate_probe_result(result)
    return 0 if status == 'PASS' else 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
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


def _finish_and_persist(result, output_path, progress, cleanup_failures, primary_error):
    """Persist terminal status after owned process cleanup and heartbeat shutdown."""
    if cleanup_failures:
        result['cleanup_failures'] = cleanup_failures
        result['status'] = 'FAIL'
        if primary_error is None:
            primary_error = OOMProbeError('Owned resource cleanup did not complete cleanly')
            result['error'] = {'type': type(primary_error).__name__, 'message': str(primary_error)}
    try:
        progress.finish(result.get('status', 'FAIL'), result.get('error', {}).get('message'))
    except BaseException as exc:
        result.setdefault('cleanup_failures', []).append({'resource': 'progress_finish',
            'type': type(exc).__name__, 'message': str(exc)})
        result['status'] = 'FAIL'
        if primary_error is None:
            primary_error = exc
    try:
        progress.close()
    except BaseException as exc:
        result.setdefault('cleanup_failures', []).append({'resource': 'progress_close',
            'type': type(exc).__name__, 'message': str(exc)})
        result['status'] = 'FAIL'
        if primary_error is None:
            primary_error = exc
    try:
        _atomic_json(output_path, result)
    except BaseException as exc:
        result['result_write_error'] = f'{type(exc).__name__}: {exc}'
        if primary_error is None:
            primary_error = exc
    return primary_error


def _exchange(control_path, event, *, wait=True, timeout=1800):
    """Send one framed service event to the parent controller."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(timeout)
        channel.connect(control_path)
        channel.sendall((json.dumps(event, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False) + '\n').encode())
        if not wait:
            return None
        raw = bytearray()
        while not raw.endswith(b'\n'):
            chunk = channel.recv(65536)
            if not chunk:
                raise OOMProbeError('Parent closed the control channel before releasing the native gate')
            raw.extend(chunk)
            if len(raw) > 1024 * 1024:
                raise OOMProbeError('Parent control response exceeded its frame limit')
        response = json.loads(raw)
        if type(response) is not dict or response.get('action') not in ('continue', 'abort'):
            raise OOMProbeError('Parent did not explicitly release or abort the generate gate')
        return response


def _wrapper_source():
    """Bootstrap observers in the owned HTTP process; engine/API files stay pinned."""
    return r'''import hashlib, importlib, json, os, signal, socket, sys, threading
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel import strata_backend
local = threading.local()
control = os.environ['Q4_CUDA_OOM_CONTROL']
target_operation = os.environ['Q4_CUDA_OOM_OPERATION']
target_branch = os.environ['Q4_CUDA_OOM_BRANCH']
original_run = ContextCoordinator._run
original_send = strata_backend._NativeRequest._send
original_read = strata_backend.StrataBackend._read
def exchange(event, wait=True):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(1800)
        channel.connect(control)
        channel.sendall((json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())
        if not wait: return
        raw = bytearray()
        while not raw.endswith(b'\n'):
            part = channel.recv(65536)
            if not part: raise RuntimeError('controller closed OOM gate')
            raw.extend(part)
            if len(raw) > 1048576: raise RuntimeError('controller frame too large')
        reply = json.loads(raw)
        if type(reply) is not dict or reply.get('action') not in ('continue','abort'):
            raise RuntimeError('controller did not release or abort OOM gate')
        if reply.get('action') == 'abort': raise RuntimeError('controller aborted native gate')
def direct(backend, command):
    backend.process.stdin.write(command+'\n')
    backend.process.stdin.flush()
    return original_read(backend)
def update_stats(backend, raw):
    if raw.get('status') != 'ok': raise RuntimeError('injected stats request failed')
    backend.stats.update({k:v for k,v in raw.items() if k!='status'})
    backend.stats['consumed_tokens'] = raw['forward_tokens']
    backend.stats['checkpoint_export_bytes'] = raw['saved_running_bytes'] + raw['written_file_bytes']
    backend.stats['checkpoint_import_bytes'] = raw['restored_running_bytes'] + raw['loaded_file_bytes']
def observed_run(self, model, job):
    previous = getattr(local, 'job', None)
    previous_stats = getattr(local, 'request_start_stats', None)
    local.job = (job.request.operation, job.branch)
    if local.job == (target_operation, target_branch):
        local.request_start_stats = model.backend.work_counters()
    try: return original_run(self, model, job)
    finally:
        local.job = previous
        local.request_start_stats = previous_stats
def observed_send(request, command):
    backend = request.backend
    identity = getattr(local, 'job', None)
    if (command.startswith('generate ') and identity == (target_operation, target_branch)
            and not getattr(local, 'oom_gate_sent', False)):
        local.oom_gate_sent = True
        pages = direct(backend, 'pages')
        if pages.get('status') != 'ok' or type(pages.get('pages')) is not list:
            raise RuntimeError('injected pages ownership witness failed')
        fresh = direct(backend, 'stats')
        update_stats(backend, fresh)
        local.gate_stats = dict(backend.stats)
        exchange({'event':'before_generate','operation_id':identity[0], 'branch':identity[1],
            'command':command,'native_pid':backend.process.pid,'request_active':backend._request_active is request,
            'stats':local.gate_stats,'request_start_stats':getattr(local,'request_start_stats',{}),
            'pages':pages['pages'],'injected_commands':['pages','stats']})
    return original_send(request, command)
def observed_read(backend):
    response = original_read(backend)
    identity = getattr(local, 'job', None)
    if (identity == (target_operation, target_branch) and getattr(local, 'oom_gate_sent', False)
            and response.get('status') == 'error' and not getattr(local, 'oom_error_sent', False)):
        local.oom_error_sent = True
        try:
            after = direct(backend, 'stats')
            update_stats(backend, after)
            event = {'event':'native_generate_error','operation_id':identity[0], 'branch':identity[1],
                'native_pid':backend.process.pid,'original_response':response,
                'stats_before':local.gate_stats,'stats_after':dict(backend.stats),
                'injected_command':'stats'}
        except BaseException as exc:
            event = {'event':'native_generate_error','operation_id':identity[0], 'branch':identity[1],
                'native_pid':backend.process.pid,'original_response':response,
                'stats_before':local.gate_stats,'stats_error':{'type':type(exc).__name__,'message':str(exc)},
                'injected_command':'stats'}
        try: exchange(event, wait=True)
        except BaseException: pass
    return response
ContextCoordinator._run = observed_run
strata_backend._NativeRequest._send = observed_send
strata_backend.StrataBackend._read = observed_read
modules = {name: importlib.import_module(name) for name in (
    'context_time_travel.real','context_time_travel.coordinator',
    'context_time_travel.strata_backend','context_time_travel.execution_receipt',
    'context_time_travel.checkpoint_cache','serve.cow_http')}
expected = json.loads(os.environ['Q4_CUDA_OOM_EXPECTED_IMPORTS'])
origin = {}
for name, module in modules.items():
    path = os.path.realpath(module.__file__)
    digest = hashlib.sha256(open(path,'rb').read()).hexdigest()
    origin[name] = {'path':path,'sha256':digest}
    if expected.get(path) != digest: raise RuntimeError('runtime import origin/hash mismatch: '+name)
exchange({'event':'runtime_import_witness','modules':origin}, wait=False)
sys.argv = ['serve.cow_http', *sys.argv[1:]]
modules['serve.cow_http'].main()
'''


def _write_wrapper(path: Path) -> Path:
    target = path / 'q4_cuda_oom_server_wrapper.py'
    raw = _wrapper_source().encode('utf-8')
    with target.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o600)
    return target


def _read_control(connection, limit=1024 * 1024):
    raw = bytearray()
    while not raw.endswith(b'\n'):
        block = connection.recv(65536)
        if not block:
            break
        raw.extend(block)
        if len(raw) > limit:
            raise OOMProbeError('Control event exceeded the bounded frame size')
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise OOMProbeError('Control event is not valid JSON') from exc
    return _object(value, 'control event')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--store', required=True)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--api-src', required=True)
    parser.add_argument('--api-scripts', required=True)
    parser.add_argument('--branch', default='CliA')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--expected-profile-sha256', required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--expected-http-sha256', required=True)
    parser.add_argument('--expected-helper-sha256', required=True)
    parser.add_argument('--pressure-helper', required=True)
    parser.add_argument('--expected-source-snapshot', required=True)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--preflight-only', action='store_true')
    return parser


def _read_line(stream, timeout, label):
    ready, _, _ = select.select([stream], [], [], timeout)
    if not ready:
        raise OOMProbeError(f'Timed out waiting for {label}')
    line = stream.readline()
    if not line:
        raise OOMProbeError(f'EOF waiting for {label}')
    try:
        value = json.loads(line)
    except ValueError as exc:
        raise OOMProbeError(f'{label} was not valid JSON') from exc
    return _object(value, label)


def _proc_identity(pid):
    proc = Path('/proc') / str(pid)
    raw_stat = (proc / 'stat').read_text(encoding='ascii')
    end = raw_stat.rfind(')')
    if end < 0:
        raise OOMProbeError('Cannot parse owned process identity')
    fields = raw_stat[end + 2:].split()
    return dict(pid=pid, parent_pid=int(fields[1]), start_time=fields[19],
                exe=os.readlink(proc / 'exe'))


def _control_event(listener, timeout, progress):
    listener.settimeout(min(1.0, timeout))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection, _ = listener.accept()
        except socket.timeout:
            progress.check()
            continue
        connection.settimeout(timeout)
        reader = connection.makefile('rb')
        try:
            raw = reader.readline(1024 * 1024 + 1)
        finally:
            reader.close()
        if len(raw) > 1024 * 1024 or not raw.endswith(b'\n'):
            connection.close()
            raise OOMProbeError('Service control frame is oversized or unterminated')
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            connection.close()
            raise OOMProbeError('Service control frame is invalid JSON') from exc
        return _object(event, 'service control event'), connection
    raise OOMProbeError('Timed out waiting for the owned service control event')


def _reply(connection, action):
    connection.sendall((json.dumps({'action': action}, separators=(',', ':'))+'\n').encode())


def _helper_start(path, log_path):
    stderr = log_path.open('xb')
    try:
        process = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding='utf-8', errors='replace', bufsize=1,
            start_new_session=True)
    finally:
        stderr.close()
    try:
        identity = _proc_identity(process.pid)
        if os.path.realpath(identity['exe']) != str(path):
            raise OOMProbeError('Pressure helper executable identity differs from pinned path')
        ready = _read_line(process.stdout, 30, 'pressure helper ready frame')
        validate_helper_ready(ready)
        return process, identity, ready
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        raise


def _helper_fill(process, timeout):
    if process.poll() is not None or process.stdin is None or process.stdout is None:
        raise OOMProbeError('Pressure helper exited before fill command')
    process.stdin.write('fill\n')
    process.stdin.flush()
    return _read_line(process.stdout, timeout, 'pressure helper fill frame')


def _helper_release(process, identity, timeout):
    if process.poll() is not None:
        raise OOMProbeError('Pressure helper exited before explicit release')
    process.stdin.write('release\n')
    process.stdin.flush()
    release = _read_line(process.stdout, timeout, 'pressure helper release frame')
    code = process.wait(timeout=timeout)
    try:
        process.stdin.close()
        process.stdout.close()
    except OSError:
        pass
    deadline = time.monotonic() + 2
    while Path('/proc', str(process.pid)).exists() and time.monotonic() < deadline:
        time.sleep(.02)
    absent = not Path('/proc', str(process.pid)).exists()
    return release, dict(pid=process.pid, start_time=identity['start_time'],
        exe=identity['exe'], exit_code=code, pid_absent_after_wait=absent)


def _stop_owned_helper(process, identity, timeout):
    """Release a helper normally, then stop only its exact owned PID if necessary."""
    receipt = {'identity': dict(identity), 'cleanup_errors': []}
    if process.poll() is None:
        try:
            release, terminal = _helper_release(process, identity, timeout)
            receipt.update(release=release, terminal=terminal)
            validate_owned_process_exit(terminal, expected_pid=identity['pid'],
                expected_start_time=identity['start_time'], expected_exe=identity['exe'])
            return receipt
        except BaseException as exc:
            receipt['cleanup_errors'].append({'type': type(exc).__name__, 'message': str(exc)})
    else:
        receipt['cleanup_errors'].append({'type': 'UnexpectedHelperExit',
            'message': f'Pressure helper exited before controlled release with {process.returncode}'})
    if process.poll() is None:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        except BaseException as exc:
            receipt['cleanup_errors'].append({'type': type(exc).__name__, 'message': str(exc)})
    receipt['forced_terminal'] = {'pid': process.pid,
        'start_time': identity.get('start_time'), 'exe': identity.get('exe'),
        'exit_code': process.poll(), 'pid_absent_after_wait': not Path('/proc', str(process.pid)).exists()}
    if receipt['forced_terminal']['exit_code'] is None or not receipt['forced_terminal']['pid_absent_after_wait']:
        receipt['cleanup_errors'].append({'type': 'OwnedProcessStillLive',
            'message': 'Pressure helper did not reach a verified terminal state'})
    return receipt


def _launch_service(*, runtime_root, api_src, api_scripts, profile, store, wrapper,
                    run_dir, control_path, operation, branch, expected_imports, args, progress, suffix):
    from run_cow_client_recovery import ServiceProcess
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(runtime_root), str(api_src), str(api_scripts)))
    env['Q4_CUDA_OOM_CONTROL'] = str(control_path)
    env['Q4_CUDA_OOM_OPERATION'] = operation
    env['Q4_CUDA_OOM_BRANCH'] = branch
    env['Q4_CUDA_OOM_EXPECTED_IMPORTS'] = json.dumps(expected_imports, sort_keys=True)
    env['CTT_NATIVE_TRACE'] = str(run_dir / f'native-trace-{suffix}.jsonl')
    (run_dir / f'native-trace-{suffix}.jsonl').touch(mode=0o600, exist_ok=False)
    trace_path = run_dir / f'pool-trace-{suffix}.json'
    command = [sys.executable, str(wrapper), '--profile', str(profile), '--store', str(store),
        '--host', '127.0.0.1', '--port', '0', '--text', '--ram-gib', str(args.ram_gib),
        '--disk-gib', str(args.disk_gib), '--pool-trace-output', str(trace_path)]
    service = ServiceProcess(command, cwd=runtime_root, env=env,
        stdout_path=run_dir / f'server-{suffix}.stdout.log',
        stderr_path=run_dir / f'server-{suffix}.stderr.log', timeout=args.timeout, progress=progress)
    ready = _start_service_owned(service)
    return service, ready, trace_path


def _stream_request(client, route, body, result_box):
    try:
        result_box['value'] = client.post_stream(route, body)
    except BaseException as exc:
        result_box['error'] = {'type': type(exc).__name__, 'message': str(exc)}


def _terminal_event(events):
    complete = [event for event in events if event.get('event') == 'complete']
    errors = [event for event in events if event.get('event') in ('error', 'failure')]
    if len(complete) == 1:
        return complete[0]
    if errors:
        return errors[-1]
    return {'event': 'eof', 'event_count': len(events)}


def preflight_configuration(args):
    """Validate profile, runtime imports, and separate HTTP pin without starting model/service."""
    if not sys.platform.startswith('linux'):
        raise OOMProbeError('Real CUDA OOM probe requires Linux')
    profile_path = Path(args.profile).expanduser().resolve(strict=True)
    store = Path(args.store).expanduser().resolve(strict=True)
    runtime_root = Path(args.runtime_root).expanduser().resolve(strict=True)
    api_src = Path(args.api_src).expanduser().resolve(strict=True)
    api_scripts = Path(args.api_scripts).expanduser().resolve(strict=True)
    helper_path = Path(args.pressure_helper).expanduser().resolve(strict=True)
    profile_sha = _sha256(profile_path)
    helper_sha = _sha256(helper_path)
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    if profile_sha != args.expected_profile_sha256 or helper_sha != args.expected_helper_sha256:
        raise OOMProbeError('Profile or pressure-helper hash differs from its pinned value')
    execution = profile.get('execution')
    if (type(execution) is not dict or execution.get('GPU') != [0]
            or execution.get('capacity') != 98304 or execution.get('prefill_chunk') != 8192
            or execution.get('kv_dtype') != 'FP16'):
        raise OOMProbeError('Selected profile is not the pinned Q4 single-GPU execution')
    command = execution.get('sut_command')
    if type(command) is not list or not command:
        raise OOMProbeError('Profile lacks its native SUT command')
    native = Path(command[0]).expanduser().resolve(strict=True)
    native_sha = _sha256(native)
    if native_sha != args.expected_native_sha256:
        raise OOMProbeError('Native SUT binary SHA-256 differs from pin')
    if not (store / 'refs.json').is_file() or not (store / 'checkpoints').is_dir():
        raise OOMProbeError('Disposable input Store is incomplete')
    script_dir = runtime_root / 'experimental' / 'branching_context' / 'scripts'
    http_path = runtime_root / 'serve' / 'cow_http.py'
    if not script_dir.is_dir() or not (script_dir / 'run_cow_client_recovery.py').is_file():
        raise OOMProbeError('Pinned runtime lacks its existing client-recovery helper')
    for path in (runtime_root, api_src, api_scripts, script_dir):
        if path.is_symlink():
            raise OOMProbeError(f'Runtime import path is a symlink: {path}')
    for path in (str(script_dir), str(runtime_root), str(api_src), str(api_scripts)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from run_q4_publication_fault import _runtime_closure
    closure = _runtime_closure(profile, runtime_root)
    profile_imports = {row['path']: row['sha256'] for row in closure['files'].values()}
    module_paths = {
        'context_time_travel.real': api_src / 'context_time_travel' / 'real.py',
        'context_time_travel.coordinator': api_src / 'context_time_travel' / 'coordinator.py',
        'context_time_travel.strata_backend': api_src / 'context_time_travel' / 'strata_backend.py',
        'context_time_travel.execution_receipt': api_src / 'context_time_travel' / 'execution_receipt.py',
        'context_time_travel.checkpoint_cache': api_src / 'context_time_travel' / 'checkpoint_cache.py',
    }
    for name, path in module_paths.items():
        resolved = str(path.resolve(strict=True))
        expected = profile_imports.get(resolved)
        if expected is None or _sha256(Path(resolved)) != expected:
            raise OOMProbeError(f'Execution-profile API closure does not pin current import: {name}')
    http_path = http_path.resolve(strict=True)
    http_sha = _sha256(http_path)
    expected_imports, service_source = build_import_expectations(profile_imports,
        http_path=http_path, http_sha256=http_sha, expected_http_sha256=args.expected_http_sha256)
    return dict(profile_path=str(profile_path), profile_sha256=profile_sha,
        native_binary=str(native), native_sha256=native_sha, helper_path=str(helper_path),
        helper_sha256=helper_sha, profile=profile, execution_profile_id=profile.get('execution_profile_id'),
        store=str(store), runtime_root=str(runtime_root), api_src=str(api_src),
        api_scripts=str(api_scripts), runtime_source_closure=closure,
        profile_api_imports=profile_imports, expected_imports=expected_imports,
        service_build_source={'http': service_source}, helper_script_dir=str(script_dir))


def run_probe(args):
    """Run one bounded Q4 greedy request under externally triggered real VMM pressure."""
    preflight = preflight_configuration(args)
    run_dir = Path(args.run_dir).expanduser().absolute()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    profile_path = Path(preflight['profile_path'])
    store = Path(preflight['store'])
    runtime_root = Path(preflight['runtime_root'])
    api_src = Path(preflight['api_src'])
    api_scripts = Path(preflight['api_scripts'])
    helper_path = Path(preflight['helper_path'])
    profile_sha = preflight['profile_sha256']
    helper_sha = preflight['helper_sha256']
    profile = preflight['profile']
    native = Path(preflight['native_binary'])
    native_sha = preflight['native_sha256']
    closure = preflight['runtime_source_closure']
    expected_imports = preflight['expected_imports']
    service_build_source = preflight['service_build_source']
    from run_q4_publication_fault import (
        _snapshot_file_fingerprint, _snapshot_record_fingerprint,
        _head_history, _generation_body, validate_source_checkpoint_unchanged,
    )
    from run_cow_client_recovery import (
        HttpClient, ProgressTracker, _ready_url, _fork,
    )
    from run_cow_client_recovery import ServiceProcess
    nonce = uuid.uuid4().hex
    fault_operation = 'op-cuda-oom-' + nonce
    branch = 'OOMQ4' + nonce[:16]
    restart_operation = 'op-cuda-restart-' + nonce
    result = dict(schema='q4-cuda-vmm-oom-v1', status='RUNNING',
        qualification_scope='single real Q4 greedy HTTP request; CUDA VMM cuMemCreate OOM injection; no numerical oracle',
        profile_path=str(profile_path), profile_sha256=profile_sha,
        native_binary=str(native), native_sha256=native_sha,
        pressure_helper=str(helper_path), pressure_helper_sha256=helper_sha,
        runtime_root=str(runtime_root), api_src=str(api_src), api_scripts=str(api_scripts),
        runtime_source_closure=closure, expected_imports=expected_imports,
        profile_api_imports=preflight['profile_api_imports'],
        service_build_source=service_build_source,
        store=str(store), source_branch=args.branch,
        expected_source_snapshot=args.expected_source_snapshot, target_branch=branch,
        fault_operation=fault_operation, restart_operation=restart_operation,
        max_tokens_fault=args.max_tokens, max_tokens_restart=8,
        runner_sha256=_sha256(Path(__file__).resolve()), steps=[])
    output_path = run_dir / 'result.json'
    wrapper = _write_wrapper(run_dir)
    socket_path = Path('/tmp') / f'q4-oom-{nonce[:16]}.sock'
    if socket_path.exists() or socket_path.is_symlink():
        raise OOMProbeError('Refusing a pre-existing controller socket path')
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(8)
    progress = ProgressTracker(run_dir, 'q4-cuda-vmm-oom')
    progress.start()
    service = helper = None
    helper_identity = None
    current_native = None
    first_client = None
    open_controls = []
    request_threads = []
    active_service_key = 'first_service'
    primary_error = None
    cleanup_failures = []
    result['first_service'] = {}
    try:
        progress.phase('start_q4_http_service')
        service, ready, trace_path = _launch_service(runtime_root=runtime_root,
            api_src=api_src, api_scripts=api_scripts, profile=profile_path, store=store,
            wrapper=wrapper, run_dir=run_dir, control_path=socket_path,
            operation=fault_operation, branch=branch, expected_imports=expected_imports,
            args=args, progress=progress, suffix='fault')
        import_event, import_connection = _control_event(listener, args.timeout, progress)
        result['first_service']['imports'] = validate_import_witness(import_event, expected_imports)
        import_connection.close()
        current_native_rows = sorted((Path('/proc') / str(service.process.pid) / 'task').glob('*/children'))
        child_pids = sorted({int(pid) for row in current_native_rows for pid in row.read_text().split()})
        if len(child_pids) != 1:
            raise OOMProbeError(f'Expected one owned native process after readiness, found {child_pids}')
        current_native = _proc_identity(child_pids[0])
        if current_native.get('parent_pid') != service.process.pid or current_native.get('exe') != str(native):
            raise OOMProbeError('Native worker PID/executable does not match the pinned Q4 child')
        result['first_service'].update(pid=service.process.pid, ready=ready,
            native_process=current_native, trace_path=str(trace_path))
        first_client = HttpClient(_ready_url(ready), timeout=args.timeout, progress=progress)
        status, health = first_client.get('/health')
        if status != 200 or health.get('service') != 'context-cow':
            raise OOMProbeError('Owned loopback service is not the pinned COW HTTP service')
        if ready.get('execution_profile_id') != profile.get('execution_profile_id'):
            raise OOMProbeError('HTTP service execution profile differs from pinned profile')
        source_head, source_history = _head_history(first_client, args.branch)
        if source_head.get('tip') != args.expected_source_snapshot:
            raise OOMProbeError('Source branch does not have the pinned checkpoint tip')
        if source_history.get('controller_status') != 'active':
            raise OOMProbeError('Pinned source branch must be active for zero-prefill resume fault injection')
        source_record = _snapshot_record_fingerprint(store, args.expected_source_snapshot)
        source_checkpoint = _snapshot_file_fingerprint(store, args.expected_source_snapshot)
        target_head = _fork(first_client, args.branch, args.expected_source_snapshot,
                            branch, 'op-cuda-oom-fork-' + nonce)
        target_head, target_history = _head_history(first_client, branch)
        route, fault_body = _generation_body(target_history, target_head, branch,
            fault_operation, args.max_tokens, 'Continue the selected Q4 test branch.')
        result['source_before'] = dict(head=source_head, snapshot_record_sha256=source_record,
                                       checkpoint_files=source_checkpoint)
        result['target_before'] = dict(head=target_head, controller=target_history.get('controller_status'),
                                       route=route, body=fault_body)
        request_box = {}
        request_thread = threading.Thread(target=_stream_request,
            args=(first_client, route, fault_body, request_box), name='q4-cuda-oom-http-request', daemon=True)
        request_threads.append(request_thread)
        progress.phase('fault_request_pending_before_native_generate')
        request_thread.start()
        gate_event, gate_connection = _control_event(listener, args.timeout, progress)
        open_controls.append(gate_connection)
        if gate_event.get('event') != 'before_generate':
            raise OOMProbeError('First service control event was not the expected native generate gate')
        gate = validate_gate_record(gate_event, operation_id=fault_operation, branch=branch,
                                    native_pid=current_native['pid'],
                                    expected_forward=0)
        result['generate_gate'] = gate
        progress.phase('native_generate_held_before_send')
        helper, helper_identity, helper_ready = _helper_start(helper_path, run_dir / 'pressure-helper.stderr.log')
        result['pressure_helper'] = dict(identity=helper_identity, ready=helper_ready)
        pressure = _helper_fill(helper, args.timeout)
        # Validate OOM before allowing the unchanged native generate command.
        validate_pressure_frame(pressure)
        result['pressure_helper']['fill'] = pressure
        _reply(gate_connection, 'continue')
        gate_connection.close()
        open_controls.remove(gate_connection)
        progress.phase('native_generate_under_real_cuda_pressure')
        error_event, error_connection = _control_event(listener, args.timeout, progress)
        open_controls.append(error_connection)
        if (error_event.get('event') != 'native_generate_error'
                or error_event.get('operation_id') != fault_operation
                or error_event.get('branch') != branch
                or error_event.get('native_pid') != current_native['pid']):
            raise OOMProbeError('Native error observation is not bound to the gated request/process')
        error_validation = None
        try:
            if error_event.get('stats_error'):
                raise OOMProbeError(f"Same-pipe stats read failed: {error_event['stats_error']}")
            native_oom = validate_native_oom(error_event.get('original_response'),
                error_event.get('stats_before'), error_event.get('stats_after'), expected_forward=0)
            result['native_oom'] = native_oom
        except BaseException as exc:
            error_validation = exc
            result['native_error_observation'] = error_event
        release_frame, helper_exit = _helper_release(helper, helper_identity, args.timeout)
        result['pressure_helper']['release'] = release_frame
        result['pressure_helper']['terminal'] = helper_exit
        validate_helper_pressure(pressure, release_frame)
        validate_owned_process_exit(helper_exit, expected_pid=helper_identity['pid'],
            expected_start_time=helper_identity['start_time'], expected_exe=helper_identity['exe'])
        helper = None
        _reply(error_connection, 'continue')
        error_connection.close()
        open_controls.remove(error_connection)
        if error_validation is not None:
            raise error_validation
        request_thread.join(timeout=args.timeout)
        if request_thread.is_alive():
            raise OOMProbeError('Faulted HTTP request thread remained live after native error')
        if 'value' not in request_box:
            raise OOMProbeError(f'Faulted HTTP request did not return: {request_box.get("error")}')
        fault_events, fault_disconnected = request_box['value']
        fault_evidence = validate_fault_http_events(fault_events)
        result['fault_response'] = dict(events=fault_events, disconnected=fault_disconnected,
                                        terminal=_terminal_event(fault_events), evidence=fault_evidence)
        source_after_fault, source_history_after_fault = _head_history(first_client, args.branch)
        target_after_fault, target_history_after_fault = _head_history(first_client, branch)
        validate_source_checkpoint_unchanged(source_checkpoint,
            _snapshot_file_fingerprint(store, args.expected_source_snapshot), phase='after native OOM')
        if (source_after_fault != source_head or target_after_fault != target_head
                or _snapshot_record_fingerprint(store, args.expected_source_snapshot) != source_record):
            raise OOMProbeError('Faulted generate changed canonical source or target branch state')
        result['canonical_after_fault'] = dict(source_head=source_after_fault,
            source_history_sha256=hashlib.sha256(json.dumps(source_history_after_fault, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(), target_head=target_after_fault,
            target_controller=target_history_after_fault.get('controller_status'))
        same_route, same_body = _generation_body(target_history_after_fault,
            target_after_fault, branch, 'op-cuda-same-service-' + nonce, 1,
            'Follow the branch after the failed native request.')
        same_box = {}
        same_thread = threading.Thread(target=_stream_request,
            args=(first_client, same_route, same_body, same_box),
            name='q4-cuda-oom-same-service-followup', daemon=True)
        request_threads.append(same_thread)
        same_thread.start()
        same_thread.join(timeout=min(args.timeout, 120))
        if same_thread.is_alive():
            result['same_service_followup'] = {'event': 'timeout'}
        elif 'value' in same_box:
            same_events, same_disconnected = same_box['value']
            result['same_service_followup'] = dict(**_terminal_event(same_events),
                disconnected=same_disconnected, events=same_events)
        else:
            result['same_service_followup'] = dict(event='error', error=same_box.get('error'))
        progress.phase('stop_poisoned_http_service')
        first_stop = service.stop()
        result['first_service']['stop'] = first_stop
        result['first_service']['native_pid_absent_after_stop'] = not Path('/proc', str(current_native['pid'])).exists()
        first_trace_file = Path(trace_path)
        if first_trace_file.is_file():
            try:
                trace_raw = json.loads(first_trace_file.read_text(encoding='utf-8'))
                result['first_service']['native_close_observation'] = {
                    'trace_sha256': _sha256(first_trace_file),
                    'event_count': trace_raw.get('event_count'),
                    'native_worker_closed': [row for row in trace_raw.get('events', [])
                        if isinstance(row, dict) and row.get('event') == 'native_worker_closed']}
            except BaseException as trace_error:
                result['first_service']['native_close_observation_error'] = {
                    'type': type(trace_error).__name__, 'message': str(trace_error)}
        service = None

        # A new server/native process proves the canonical source remains usable
        # after the known poisoned-adapter path. This is not same-process recovery.
        progress.phase('restart_http_service_after_native_protocol_poison')
        active_service_key = 'restart_service'
        service, ready, restart_trace = _launch_service(runtime_root=runtime_root,
            api_src=api_src, api_scripts=api_scripts, profile=profile_path, store=store,
            wrapper=wrapper, run_dir=run_dir, control_path=socket_path,
            operation=fault_operation, branch=branch, expected_imports=expected_imports,
            args=args, progress=progress, suffix='restart')
        import_event, import_connection = _control_event(listener, args.timeout, progress)
        result['restart_service'] = dict(imports=validate_import_witness(import_event, expected_imports),
                                         pid=service.process.pid, ready=ready)
        import_connection.close()
        restart_client = HttpClient(_ready_url(ready), timeout=args.timeout, progress=progress)
        restart_head, restart_history = _head_history(restart_client, branch)
        if restart_head != target_head:
            raise OOMProbeError('Restart did not reopen the exact pre-fault target head')
        restart_route, restart_body = _generation_body(restart_history, restart_head, branch,
            restart_operation, 8, 'Continue from the unchanged Q4 checkpoint after a clean restart.')
        restart_events, restart_disconnected = restart_client.post_stream(restart_route, restart_body)
        restart_terminal = _terminal_event(restart_events)
        if restart_terminal.get('event') != 'complete':
            raise OOMProbeError('Clean restart follow-up did not complete')
        answer = restart_terminal.get('answer')
        tokens = answer.get('tokens') if type(answer) is dict else None
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
            raise OOMProbeError('Restart completion lacks actual generated token IDs')
        result['restart_followup'] = dict(event='complete', tokens=tokens, events=restart_events,
            disconnected=restart_disconnected, route=restart_route, request=restart_body,
            answer=answer, trace_path=str(restart_trace))
        if result['same_service_followup'].get('event') in ('error', 'failure'):
            result['status'] = 'FAIL-RECOVERY'
            result['recovery_note'] = 'The expected poisoned adapter did not serve a same-process follow-up; clean restart succeeded.'
        elif result['same_service_followup'].get('event') != 'complete':
            result['status'] = 'FAIL'
            result['error'] = {'type': 'OOMProbeError',
                'message': 'Same-service follow-up ended without a classified HTTP error or completion'}
        else:
            result['status'] = 'PASS'
        validate_probe_result(result)
        restart_stop = service.stop()
        result['restart_service']['stop'] = restart_stop
        if restart_stop.get('exit_code') != 0:
            raise OOMProbeError('Clean-restart HTTP process did not exit with code zero')
        if Path('/proc', str(restart_stop.get('pid'))).exists():
            raise OOMProbeError('Clean-restart HTTP process remains after stop')
        service = None
        from run_cow_client_recovery import _trace_worker
        result['restart_service']['native_close_trace'] = _trace_worker(restart_trace, restart_stop['pid'])
    except BaseException as exc:
        primary_error = exc
        result['status'] = 'FAIL'
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        if isinstance(exc, ServiceStartupFailure):
            result.setdefault(active_service_key, {})['startup_cleanup'] = exc.cleanup_receipt
            if (not exc.cleanup_receipt.get('pid_absent', False)
                    or exc.cleanup_receipt.get('cleanup_failed', False)):
                cleanup_failures.append({'resource': f'{active_service_key}_startup_cleanup',
                    'type': 'OwnedServiceStillLive',
                    'message': 'Service readiness failed and the owned process group is not verified absent'})
    finally:
        for connection in list(open_controls):
            try:
                _reply(connection, 'abort')
            except BaseException as exc:
                cleanup_failures.append({'resource': 'control_connection',
                    'type': type(exc).__name__, 'message': str(exc)})
            try:
                connection.close()
            except BaseException as exc:
                cleanup_failures.append({'resource': 'control_connection_close',
                    'type': type(exc).__name__, 'message': str(exc)})
            open_controls.remove(connection)
        for thread in request_threads:
            try:
                thread.join(timeout=min(args.timeout, 30))
                if thread.is_alive():
                    cleanup_failures.append({'resource': thread.name,
                        'type': 'TimeoutError', 'message': 'Owned HTTP request thread remains live'})
            except BaseException as exc:
                cleanup_failures.append({'resource': thread.name,
                    'type': type(exc).__name__, 'message': str(exc)})
        if helper is not None:
            try:
                helper_cleanup = _stop_owned_helper(helper, helper_identity, min(args.timeout, 30))
                result['pressure_helper_cleanup'] = helper_cleanup
                cleanup_failures.extend({'resource': 'pressure_helper', **row}
                    for row in helper_cleanup.get('cleanup_errors', []))
            except BaseException as exc:
                cleanup_failures.append({'resource': 'pressure_helper_cleanup',
                    'type': type(exc).__name__, 'message': str(exc)})
            finally:
                for pipe_name in ('stdin', 'stdout'):
                    pipe = getattr(helper, pipe_name, None)
                    if pipe is not None:
                        try:
                            pipe.close()
                        except BaseException as exc:
                            cleanup_failures.append({'resource': f'pressure_helper_{pipe_name}_close',
                                'type': type(exc).__name__, 'message': str(exc)})
        if service is not None:
            try:
                stop = service.stop()
                result.setdefault(active_service_key, {})['stop'] = stop
                process = getattr(service, 'process', None)
                result[active_service_key]['pid_absent_after_stop'] = (
                    process is None or not Path('/proc', str(process.pid)).exists())
            except BaseException as exc:
                process = getattr(service, 'process', None)
                result.setdefault(active_service_key, {})['stop_error'] = {
                    'type': type(exc).__name__, 'message': str(exc),
                    'pid': process.pid if process is not None else None,
                    'returncode': process.poll() if process is not None else None,
                    'pid_absent': process is None or not Path('/proc', str(process.pid)).exists()}
                cleanup_failures.append({'resource': f'{active_service_key}_stop',
                    'type': type(exc).__name__, 'message': str(exc)})
        if listener is not None:
            try:
                listener.close()
            except BaseException as exc:
                cleanup_failures.append({'resource': 'control_listener_close',
                    'type': type(exc).__name__, 'message': str(exc)})
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_failures.append({'resource': 'control_socket_unlink',
                'type': type(exc).__name__, 'message': str(exc)})
        primary_error = _finish_and_persist(result, output_path, progress,
                                            cleanup_failures, primary_error)
    if primary_error is not None:
        raise primary_error
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.preflight_only:
            receipt = preflight_configuration(args)
            print(json.dumps({'status': 'PREPARED_NOT_RUN',
                'profile_sha256': receipt['profile_sha256'],
                'native_sha256': receipt['native_sha256'],
                'service_build_source': receipt['service_build_source'],
                'profile_api_imports': receipt['profile_api_imports']},
                sort_keys=True, separators=(',', ':')), flush=True)
            return 0
        result = run_probe(args)
        return run_status_exit_code(result)
    except BaseException as exc:
        print(f'q4-cuda-oom: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
