"""External, bounded attribution wrapper for the existing P4 lifecycle.

The wrapper observes Python/native request and checkpoint publication calls. It
does not modify the pinned API, native source, checkpoint representation, or
integrity checks. Inclusive timings contain nested calls; exclusive timings
subtract only observed children on the same thread.
"""
from contextlib import contextmanager
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time


class LifecycleProbe:
    def __init__(self, *, max_events=512, max_threads=32, max_stages=64):
        if type(max_events) is not int or max_events <= 0:
            raise ValueError('max_events must be a positive integer')
        if type(max_threads) is not int or max_threads <= 0:
            raise ValueError('max_threads must be a positive integer')
        if type(max_stages) is not int or max_stages <= 0:
            raise ValueError('max_stages must be a positive integer')
        self.max_events = max_events
        self.max_threads = max_threads
        self.max_stages = max_stages
        self._events = []
        self._stages = {}
        self._phase_stages = {}
        self._phase_context = None
        self._dropped_events = 0
        self._instrumentation_errors = 0
        self._counter_lock = threading.Lock()
        self._thread_state = threading.local()
        self._patches = []
        self._closed = False
        self._call_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for owner, attribute, descriptor in reversed(self._patches):
            setattr(owner, attribute, descriptor)
        self._patches.clear()

    def _stack(self):
        value = getattr(self._thread_state, 'stack', None)
        if value is None:
            value = []
            self._thread_state.stack = value
        return value

    def current_stage(self):
        stack = self._stack()
        return stack[-1]['stage'] if stack else None

    def set_phase(self, phase, *, root_index=None, repeat=None):
        """Record the runner's current phase for calls on every worker thread."""
        with self._counter_lock:
            previous = self._phase_context or {}
            phase_value = previous.get('phase') if phase is None else str(phase)
            if phase_value is None:
                self._phase_context = None
            else:
                self._phase_context = dict(phase=phase_value,
                    root_index=previous.get('root_index') if root_index is None else root_index,
                    repeat=previous.get('repeat') if repeat is None else repeat)

    @contextmanager
    def scope(self, stage, *, byte_count=None):
        """Create a nested measurement scope for an operation without a method hook."""
        started, frame = self._begin(stage)
        outcome = 'return'
        try:
            yield
        except BaseException:
            outcome = 'exception'
            raise
        finally:
            self._finish(stage, started, frame, outcome, byte_count)

    def _begin(self, stage):
        stack = self._stack()
        with self._counter_lock:
            self._call_id += 1
            call_id = self._call_id
            phase_context = None if self._phase_context is None else dict(self._phase_context)
        frame = dict(call_id=call_id, parent_call_id=stack[-1]['call_id'] if stack else None,
                     stage=stage, phase=phase_context, child_ns=0)
        stack.append(frame)
        return time.perf_counter_ns(), frame

    def _finish(self, stage, started, frame, outcome, byte_count):
        elapsed = max(0, time.perf_counter_ns() - started)
        stack = self._stack()
        if stack and stack[-1] is frame:
            stack.pop()
            if stack:
                stack[-1]['child_ns'] += elapsed
        exclusive = max(0, elapsed - frame['child_ns'])
        self._record(stage, elapsed, exclusive, outcome, byte_count, frame)

    def _record(self, stage, inclusive_ns, exclusive_ns, outcome, byte_count, frame):
        try:
            thread = threading.current_thread()
            thread_key = f'{thread.ident}:{thread.name}'
            with self._counter_lock:
                if stage not in self._stages and len(self._stages) >= self.max_stages:
                    stage = '<stage-overflow>'
                row = self._stages.setdefault(stage, dict(calls=0, errors=0, bytes=0,
                    inclusive_ns=0, exclusive_ns=0, threads={}))
                row['calls'] += 1
                row['errors'] += outcome == 'exception'
                row['bytes'] += byte_count if type(byte_count) is int and byte_count > 0 else 0
                row['inclusive_ns'] += inclusive_ns
                row['exclusive_ns'] += exclusive_ns
                threads = row['threads']
                if thread_key not in threads and len(threads) >= self.max_threads:
                    thread_key = '<thread-overflow>'
                thread_row = threads.setdefault(thread_key, dict(thread_id=thread.ident,
                    thread_name=thread.name, calls=0, errors=0, bytes=0,
                    inclusive_ns=0, exclusive_ns=0))
                thread_row['calls'] += 1
                thread_row['errors'] += outcome == 'exception'
                thread_row['bytes'] += byte_count if type(byte_count) is int and byte_count > 0 else 0
                thread_row['inclusive_ns'] += inclusive_ns
                thread_row['exclusive_ns'] += exclusive_ns
                if len(self._events) < self.max_events:
                    self._events.append(dict(call_id=frame['call_id'],
                        parent_call_id=frame['parent_call_id'], stage=stage,
                        phase=frame['phase'],
                        thread_id=thread.ident, thread_name=thread.name,
                        inclusive_ns=inclusive_ns, exclusive_ns=exclusive_ns,
                        byte_count=byte_count, outcome=outcome))
                else:
                    self._dropped_events += 1
                if frame['phase'] is not None:
                    context_key = json.dumps(frame['phase'], sort_keys=True, separators=(',', ':'))
                    if context_key not in self._phase_stages and len(self._phase_stages) >= self.max_stages:
                        context_key = '<phase-context-overflow>'
                        context = self._phase_stages.setdefault(context_key, dict(
                            phase=None, overflowed=True, stages={}))
                    else:
                        context = self._phase_stages.setdefault(context_key, dict(
                            phase=frame['phase'], overflowed=False, stages={}))
                    phase_stages = context['stages']
                    phase_stage = stage
                    if phase_stage not in phase_stages and len(phase_stages) >= self.max_stages:
                        phase_stage = '<stage-overflow>'
                    phase_row = phase_stages.setdefault(phase_stage, dict(calls=0, errors=0,
                        bytes=0, inclusive_ns=0, exclusive_ns=0, threads={}))
                    phase_row['calls'] += 1
                    phase_row['errors'] += outcome == 'exception'
                    phase_row['bytes'] += byte_count if type(byte_count) is int and byte_count > 0 else 0
                    phase_row['inclusive_ns'] += inclusive_ns
                    phase_row['exclusive_ns'] += exclusive_ns
                    phase_thread_key = f'{thread.ident}:{thread.name}'
                    if phase_thread_key not in phase_row['threads'] and len(phase_row['threads']) >= self.max_threads:
                        phase_thread_key = '<thread-overflow>'
                    phase_thread = phase_row['threads'].setdefault(phase_thread_key, dict(
                        thread_id=thread.ident, thread_name=thread.name, calls=0, errors=0,
                        bytes=0, inclusive_ns=0, exclusive_ns=0))
                    phase_thread['calls'] += 1
                    phase_thread['errors'] += outcome == 'exception'
                    phase_thread['bytes'] += byte_count if type(byte_count) is int and byte_count > 0 else 0
                    phase_thread['inclusive_ns'] += inclusive_ns
                    phase_thread['exclusive_ns'] += exclusive_ns
        except BaseException:
            # Diagnostics must not change a wrapped operation's return or error.
            try:
                with self._counter_lock:
                    self._instrumentation_errors += 1
            except BaseException:
                pass

    def patch(self, owner, attribute, stage, *, byte_count=None):
        """Wrap a method/function; stage may be a name or a callback returning a name/None."""
        descriptor = inspect.getattr_static(owner, attribute)
        if isinstance(descriptor, staticmethod):
            original = descriptor.__func__
            is_static = True
        elif isinstance(descriptor, classmethod):
            original = descriptor.__func__
            is_static = False
        else:
            original = descriptor
            is_static = False
        if not callable(original):
            raise TypeError(f'{attribute} is not callable')

        def timed(*args, **kwargs):
            try:
                selected = stage(*args, **kwargs) if callable(stage) else stage
            except BaseException:
                return original(*args, **kwargs)
            if selected is None:
                return original(*args, **kwargs)
            started, frame = self._begin(str(selected))
            outcome, result, error = 'return', None, None
            try:
                result = original(*args, **kwargs)
                return result
            except BaseException as caught:
                outcome, error = 'exception', caught
                raise
            finally:
                measured_bytes = None
                if byte_count is not None and outcome == 'return':
                    try:
                        measured_bytes = byte_count(args, kwargs, result, error)
                    except BaseException:
                        with self._counter_lock:
                            self._instrumentation_errors += 1
                self._finish(str(selected), started, frame, outcome, measured_bytes)

        self._patches.append((owner, attribute, descriptor))
        setattr(owner, attribute, staticmethod(timed) if is_static else timed)

    def snapshot(self):
        with self._counter_lock:
            return dict(schema='p4-checkpoint-lifecycle-diagnostics-v1',
                max_events=self.max_events, max_threads_per_stage=self.max_threads,
                max_stages=self.max_stages,
                events=[dict(row) for row in self._events], dropped_events=self._dropped_events,
                instrumentation_errors=self._instrumentation_errors,
                stages={name:dict(row,threads=[dict(item) for item in row['threads'].values()])
                        for name,row in sorted(self._stages.items())},
                phase_stages={name:dict(context,
                    stages={stage:dict(row,threads=[dict(item) for item in row['threads'].values()])
                            for stage,row in sorted(context['stages'].items())})
                    for name,context in sorted(self._phase_stages.items())})


def _checkpoint_size(checkpoint):
    try:
        return len(checkpoint.blob) + 4 * len(checkpoint.frontier)
    except (AttributeError, TypeError):
        return None


def _checkpoint_argument_size(args, index):
    try:
        return _checkpoint_size(args[index])
    except IndexError:
        return None


def _prepared_checkpoint_size(args, result):
    try:
        saved, blob = args[2], args[3]
        return len(blob) + 4 * len(saved.frontier)
    except (AttributeError, IndexError, TypeError):
        return None


def _prepared_result_size(args, result):
    try:
        checkpoint = result.checkpoint
        return len(checkpoint.blob) + 4 * len(checkpoint.frontier)
    except (AttributeError, TypeError):
        return None


def _manifest_checkpoint_size(args, result):
    try:
        return result['state_bytes'] + 4 * result['logits_count']
    except (KeyError, TypeError):
        return None


def install_lifecycle_observers(probe, real_module, backend_module, pools_module):
    """Install bounded observation hooks on the frozen lifecycle's Python methods."""
    StrataBackend = backend_module.StrataBackend
    BranchingModel = real_module.BranchingModel
    Checkpoint = real_module.Checkpoint

    def native_request_stage(self, command):
        if not isinstance(command, str):
            return None
        verb = command.partition(' ')[0]
        return 'native_request_' + verb if verb in ('write', 'load') else None

    probe.patch(StrataBackend, '_request', native_request_stage,
        byte_count=lambda args, kwargs, result, error: len(args[1].encode('utf-8'))
        if len(args) > 1 and isinstance(args[1], str) else None)
    probe.patch(StrataBackend, 'serialize', 'backend_serialize',
        byte_count=lambda args, kwargs, result, error: len(result) if isinstance(result, bytes) else None)
    probe.patch(Path, 'read_bytes',
        lambda self: 'export_state_readback' if isinstance(self, Path) and self.name == 'export.state' else None,
        byte_count=lambda args, kwargs, result, error: len(result) if isinstance(result, bytes) else None)
    probe.patch(Checkpoint, 'digest', 'checkpoint_digest',
        byte_count=lambda args, kwargs, result, error:
        (len(args[0]) if isinstance(args[0], bytes) else len(getattr(args[0], 'identity', b'')))
        + 4 * len(args[1]))
    probe.patch(pools_module, 'checkpoint_digest', 'pool_checkpoint_digest',
        byte_count=lambda args, kwargs, result, error: _checkpoint_size(args[0]))
    probe.patch(pools_module.SharedCheckpointPool, 'admit', 'shared_pool_admit',
        byte_count=lambda args, kwargs, result, error: _checkpoint_argument_size(args, 2))
    probe.patch(pools_module.SharedCheckpointPool, '_admit', 'shared_pool_internal_admit',
        byte_count=lambda args, kwargs, result, error: _checkpoint_argument_size(args, 3))
    probe.patch(BranchingModel, '_prepared_checkpoint', 'prepared_checkpoint',
        byte_count=lambda args, kwargs, result, error: _prepared_checkpoint_size(args, result))
    probe.patch(BranchingModel, '_publish_checkpoint_parts',
        lambda self, destination, parts: 'payload_publish' if Path(destination).suffix == '.state' else 'other_parts_publish',
        byte_count=lambda args, kwargs, result, error: sum(map(len, args[2])) if len(args) > 2 else None)
    probe.patch(BranchingModel, '_publish_checkpoint_file',
        lambda self, destination, data: 'manifest_publish' if Path(destination).suffix == '.json' else 'other_file_publish',
        byte_count=lambda args, kwargs, result, error: len(args[2]) if len(args) > 2 else None)
    probe.patch(BranchingModel, 'prepare_checkpoint', 'prepare_checkpoint',
        byte_count=lambda args, kwargs, result, error: _prepared_result_size(args, result))
    probe.patch(BranchingModel, 'publish_prepared_checkpoint', 'publish_prepared_checkpoint')
    probe.patch(BranchingModel, 'validate_checkpoint', 'validate_checkpoint',
        byte_count=lambda args, kwargs, result, error: _manifest_checkpoint_size(args, result))
    probe.patch(BranchingModel, '_load_checkpoint', 'load_checkpoint',
        byte_count=lambda args, kwargs, result, error: _checkpoint_size(result))

    def fsync_stage(fd):
        parent = probe.current_stage()
        if parent not in ('payload_publish', 'manifest_publish'):
            return None
        try:
            is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        except (OSError, TypeError):
            is_directory = False
        label = 'payload' if parent == 'payload_publish' else 'manifest'
        return label + ('_directory_fsync' if is_directory else '_file_fsync')

    probe.patch(os, 'fsync', fsync_stage)


def install_runner_phase_observer(probe, runner_module):
    """Observe P4 heartbeat phase changes without editing the runner."""
    owner = runner_module.RunHeartbeat
    attribute = 'update'
    descriptor = inspect.getattr_static(owner, attribute)
    if isinstance(descriptor, (staticmethod, classmethod)):
        raise TypeError('RunHeartbeat.update must be an instance method')
    if not callable(descriptor):
        raise TypeError('RunHeartbeat.update is not callable')

    def observed(self, phase, *, root_index=None, repeat=None):
        result = descriptor(self, phase, root_index=root_index, repeat=repeat)
        probe.set_phase(phase, root_index=root_index, repeat=repeat)
        return result

    probe._patches.append((owner, attribute, descriptor))
    setattr(owner, attribute, observed)


def storage_preflight(path, required_free_bytes, *, disk_usage=shutil.disk_usage):
    """Measure free bytes on the intended output filesystem without writing there."""
    if type(required_free_bytes) is not int or required_free_bytes <= 0:
        raise ValueError('required free bytes must be a positive integer')
    directory = Path(path).resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    usage = disk_usage(directory)
    return dict(path=str(directory), total_bytes=usage.total,
        used_bytes=usage.used, available_bytes=usage.free,
        required_free_bytes=required_free_bytes,
        passes=usage.free >= required_free_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='clean frozen Context COW source root')
    parser.add_argument('--diagnostics', type=Path, required=True,
                        help='diagnostics JSON output outside frozen source')
    parser.add_argument('--max-events', type=int, default=512)
    parser.add_argument('--max-threads', type=int, default=32)
    parser.add_argument('--storage-check-path', type=Path,
                        help='existing directory on the P4 output filesystem')
    parser.add_argument('--required-free-bytes', type=int,
                        help='external disk gate; checked before invoking P4')
    args, runner_args = parser.parse_known_args(argv)
    root = args.source_root.resolve(strict=True)
    destination = args.diagnostics.resolve()
    temporary = destination.with_name(destination.name+'.tmp')
    if destination == root or root in destination.parents:
        parser.error('diagnostics output must be outside the frozen source root')
    if destination.exists() or temporary.exists():
        parser.error('diagnostics output and temporary paths must be new')
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = root/'experimental/branching_context/src'
    scripts = root/'experimental/branching_context/scripts'
    if not src.is_dir() or not scripts.is_dir():
        parser.error('source root is missing the Context COW source tree')
    if (args.storage_check_path is None) != (args.required_free_bytes is None):
        parser.error('--storage-check-path and --required-free-bytes must be supplied together')
    gate = None
    if args.storage_check_path is not None:
        try:
            gate = storage_preflight(args.storage_check_path, args.required_free_bytes)
        except (OSError, ValueError) as caught:
            parser.error(f'invalid storage preflight: {type(caught).__name__}: {caught}')
        if not gate['passes']:
            invocation = dict(source_root=str(root), runner='p4_return_latency.main',
                runner_args=runner_args, status='PREFLIGHT_REJECTED',
                error='insufficient free bytes; P4 runner not imported or invoked',
                storage_preflight=gate,
                measurements=dict(schema='p4-checkpoint-lifecycle-diagnostics-v1',
                    events=[], phase_stages={}, stages={}, dropped_events=0,
                    instrumentation_errors=0))
            temporary.write_text(json.dumps(invocation, sort_keys=True, indent=2)+'\n', encoding='utf-8')
            os.replace(temporary, destination)
            print(json.dumps(dict(status='PREFLIGHT_REJECTED', diagnostics=str(destination),
                available_bytes=gate['available_bytes'], required_free_bytes=gate['required_free_bytes'])), flush=True)
            return 2
    for path in (str(src), str(scripts)):
        if path not in sys.path:
            sys.path.insert(0, path)

    real_module = importlib.import_module('context_time_travel.real')
    backend_module = importlib.import_module('context_time_travel.strata_backend')
    pools_module = importlib.import_module('context_time_travel.pools')
    runner = importlib.import_module('p4_return_latency')
    probe = LifecycleProbe(max_events=args.max_events, max_threads=args.max_threads)
    install_lifecycle_observers(probe, real_module, backend_module, pools_module)
    install_runner_phase_observer(probe, runner)
    invocation = dict(source_root=str(root), runner='p4_return_latency.main',
                      runner_args=runner_args, status='RUNNING')
    if gate is not None:
        invocation['storage_preflight'] = gate
    exit_code = None
    error = None
    try:
        exit_code = runner.main(runner_args)
        invocation.update(status='RETURNED', exit_code=exit_code)
    except BaseException as caught:
        error = f'{type(caught).__name__}: {caught}'
        invocation.update(status='RAISED', error=error)
    finally:
        probe.close()
        invocation['measurements'] = probe.snapshot()
        temporary.write_text(json.dumps(invocation, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        os.replace(temporary, destination)
    if error is not None:
        print(json.dumps(dict(status='RAISED', diagnostics=str(destination), error=error)), flush=True)
        return 1
    print(json.dumps(dict(status='RETURNED', exit_code=exit_code,
        diagnostics=str(destination))), flush=True)
    return exit_code if type(exit_code) is int else 1


if __name__ == '__main__':
    raise SystemExit(main())
