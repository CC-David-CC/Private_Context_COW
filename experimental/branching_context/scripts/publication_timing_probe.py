"""Test-only method timing hooks for checkpoint publication attribution.

This module adds no model calls and does not skip, replace, or reorder any
production operations. It records phase spans from monotonic_ns in the HTTP
service process. Events contain only phase labels, thread/span IDs, outcome
classes, numeric timestamps and an existing snapshot ID for correlation.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import itertools
import json
import os
from pathlib import Path
import stat
import threading
import time


class TimingRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.status_path = self.path.with_suffix(".status.json")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        if os.name != "nt":
            os.chmod(self.path, 0o600)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._ids = itertools.count(1)
        self._closed = False
        self._instrumentation_error = False

    def _stack(self):
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def set_thread_correlation(self, value):
        self._local.correlation = value

    def clear_thread_correlation(self):
        self._local.correlation = None

    @property
    def current_correlation(self):
        stack = self._stack()
        if stack:
            return stack[-1]["correlation_id"]
        return getattr(self._local, "correlation", None)

    def in_scope(self, labels):
        wanted = set(labels)
        return any(frame["phase"] in wanted for frame in self._stack())

    def _begin(self, phase, correlation_id=None):
        stack = self._stack()
        if correlation_id is None:
            correlation_id = self.current_correlation
        span = {
            "span_id": next(self._ids),
            "parent_span_id": stack[-1]["span_id"] if stack else None,
            "phase": phase,
            "correlation_id": correlation_id if isinstance(correlation_id, str) else None,
            "thread_id": threading.get_ident(),
            "depth": len(stack),
            "start_monotonic_ns": time.monotonic_ns(),
        }
        stack.append(span)
        return span

    def _finish(self, span, outcome, fields=None):
        # Instrumentation must never replace a production return or exception.
        # A missing/partial trace is rejected by the offline summarizer instead.
        try:
            end = time.monotonic_ns()
            stack = self._stack()
            if not stack or stack[-1] is not span:
                self._instrumentation_error = True
                return
            stack.pop()
            row = {
                "schema": "cow-publication-phase-v1",
                "span_id": span["span_id"],
                "parent_span_id": span["parent_span_id"],
                "phase": span["phase"],
                "correlation_id": span["correlation_id"],
                "thread_id": span["thread_id"],
                "depth": span["depth"],
                "start_monotonic_ns": span["start_monotonic_ns"],
                "end_monotonic_ns": end,
                "duration_ns": max(0, end - span["start_monotonic_ns"]),
                "outcome": outcome,
            }
            if fields:
                row.update(fields)
            raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            with self._lock:
                if self._closed:
                    return
                self._stream.write(raw)
                self._stream.flush()
        except BaseException:
            self._instrumentation_error = True

    @contextmanager
    def span(self, phase, correlation_id=None):
        try:
            span = self._begin(phase, correlation_id)
        except BaseException:
            self._instrumentation_error = True
            span = None
        try:
            yield span
        except BaseException as exc:
            if span is not None:
                self._finish(span, "exception:" + type(exc).__name__)
            raise
        else:
            if span is not None:
                self._finish(span, "ok")

    def wrap_method(self, owner, method_name, phase, correlation=None):
        original = getattr(owner, method_name)
        recorder = self

        @wraps(original)
        def timed(self_obj, *args, **kwargs):
            try:
                key = correlation(self_obj, args, kwargs) if correlation else None
            except BaseException:
                recorder._instrumentation_error = True
                key = None
            with recorder.span(phase, key):
                return original(self_obj, *args, **kwargs)

        setattr(owner, method_name, timed)
        return original

    def wrap_reservation(self, owner, method_name, phase):
        original = getattr(owner, method_name)
        recorder = self

        @wraps(original)
        def timed(self_obj, *args, **kwargs):
            inner = original(self_obj, *args, **kwargs)
            try:
                correlation_id = recorder.current_correlation
            except BaseException:
                recorder._instrumentation_error = True
                correlation_id = None
            return _TimedReservation(recorder, inner, phase, correlation_id)

        setattr(owner, method_name, timed)
        return original

    def wrap_filesystem(self, os_module):
        original_fsync = os_module.fsync
        original_replace = os_module.replace
        recorder = self
        publish_scopes = {
            "writer.publish_total", "checkpoint.save_prepared_branch",
            "checkpoint.publish_prepared_checkpoint",
        }

        @wraps(original_fsync)
        def timed_fsync(fd):
            try:
                publishing = recorder.in_scope(publish_scopes)
            except BaseException:
                recorder._instrumentation_error = True
                publishing = False
            if not publishing:
                return original_fsync(fd)
            try:
                is_directory = stat.S_ISDIR(os_module.fstat(fd).st_mode)
            except BaseException:
                recorder._instrumentation_error = True
                is_directory = False
            label = "filesystem.fsync_directory" if is_directory else "filesystem.fsync_file"
            with recorder.span(label):
                return original_fsync(fd)

        @wraps(original_replace)
        def timed_replace(source, destination):
            try:
                publishing = recorder.in_scope(publish_scopes)
            except BaseException:
                recorder._instrumentation_error = True
                publishing = False
            if not publishing:
                return original_replace(source, destination)
            with recorder.span("filesystem.atomic_replace"):
                return original_replace(source, destination)

        os_module.fsync = timed_fsync
        os_module.replace = timed_replace
        return ((os_module, "fsync", original_fsync), (os_module, "replace", original_replace))

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._stream.flush()
            except BaseException:
                self._instrumentation_error = True
            try:
                self._stream.close()
            except BaseException:
                self._instrumentation_error = True
            try:
                with self.status_path.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump({"schema": "cow-publication-phase-status-v1",
                               "instrumentation_error": self._instrumentation_error},
                              stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    stream.write("\n")
                    stream.flush()
            except BaseException:
                # The offline runner requires the sidecar; its absence or
                # malformed content invalidates the diagnostic run.
                self._instrumentation_error = True


class _TimedReservation:
    def __init__(self, recorder, inner, phase, correlation_id):
        self.recorder = recorder
        self.inner = inner
        self.phase = phase
        self.correlation_id = correlation_id
        self.span = None
        self.acquired_ns = None
        self.wait_ns = None

    def __enter__(self):
        try:
            self.span = self.recorder._begin(self.phase, self.correlation_id)
            wait_start = time.monotonic_ns()
        except BaseException:
            self.recorder._instrumentation_error = True
            self.span = None
            wait_start = None
        try:
            value = self.inner.__enter__()
        except BaseException as exc:
            if self.span is not None:
                fields = {"held_ns": 0}
                if wait_start is not None:
                    try:
                        fields["wait_ns"] = max(0, time.monotonic_ns() - wait_start)
                    except BaseException:
                        self.recorder._instrumentation_error = True
                self.recorder._finish(self.span, "exception:" + type(exc).__name__, fields)
            self.span = None
            raise
        if self.span is None:
            return value
        try:
            self.acquired_ns = time.monotonic_ns()
        except BaseException:
            self.recorder._instrumentation_error = True
            self.span = None
            return value
        if wait_start is None:
            self.recorder._instrumentation_error = True
            self.span = None
            return value
        self.wait_ns = max(0, self.acquired_ns - wait_start)
        return value

    def __exit__(self, exc_type, exc, tb):
        if self.span is None:
            return self.inner.__exit__(exc_type, exc, tb)
        if self.acquired_ns is None:
            self.recorder._instrumentation_error = True
            return self.inner.__exit__(exc_type, exc, tb)
        try:
            held = max(0, time.monotonic_ns() - self.acquired_ns)
        except BaseException:
            self.recorder._instrumentation_error = True
            held = None
        try:
            result = self.inner.__exit__(exc_type, exc, tb)
        except BaseException as error:
            fields = {"wait_ns": self.wait_ns}
            if held is not None:
                fields["held_ns"] = held
            self.recorder._finish(self.span, "exception:" + type(error).__name__, fields)
            self.span = None
            raise
        outcome = "exception:" + exc_type.__name__ if exc_type is not None else "ok"
        fields = {"wait_ns": self.wait_ns}
        if held is not None:
            fields["held_ns"] = held
        self.recorder._finish(self.span, outcome, fields)
        self.span = None
        return result


def _snapshot_from_prepared(_self, args, kwargs):
    prepared = args[2] if len(args) > 2 else kwargs.get("prepared")
    return getattr(prepared, "snapshot", None)


def _snapshot_from_publish(_self, args, kwargs):
    prepared = args[2] if len(args) > 2 else kwargs.get("prepared")
    return getattr(prepared, "snapshot", None)


def install_timing_wrappers(real_module, coordinator_module, store_module,
                            pools_module, output_path):
    """Wrap current publication methods without changing their return/error behavior."""
    recorder = TimingRecorder(output_path)
    real_class = real_module.BranchingModel
    coordinator_class = coordinator_module.ContextCoordinator
    request_class = coordinator_module.Request
    store_class = store_module.Store
    shared_pool = pools_module.SharedCheckpointPool
    disk_pool = pools_module.DiskCheckpointPool

    targets = [
        (real_class, "prepare_checkpoint", "worker.prepare_checkpoint",
         lambda self, args, kwargs: args[0] if args else kwargs.get("snapshot")),
        (real_class, "_prepared_envelope", "checkpoint.envelope_validation",
         lambda self, args, kwargs: args[0] if args else kwargs.get("snapshot")),
        (real_class, "_publish_checkpoint_parts", "checkpoint.payload_write_fsync_rename_dirsync", None),
        (real_class, "_publish_checkpoint_file", "checkpoint.manifest_write_fsync_rename_dirsync", None),
        (real_class, "validate_checkpoint", "checkpoint.saved_validation",
         lambda self, args, kwargs: args[0] if args else kwargs.get("snapshot")),
        (real_class, "publish_prepared_checkpoint", "checkpoint.publish_prepared_checkpoint",
         lambda self, args, kwargs: getattr(args[0] if args else kwargs.get("prepared"), "snapshot", None)),
        (real_class, "save_prepared_branch", "checkpoint.save_prepared_branch", _snapshot_from_prepared),
        (coordinator_class, "_publish", "writer.publish_total", _snapshot_from_publish),
        (store_class, "move", "store.move_receipt",
         lambda self, args, kwargs: args[2] if len(args) > 2 else kwargs.get("snapshot")),
        (request_class, "emit", "request.event_emit", None),
    ]
    originals = []
    for owner, name, phase, key in targets:
        if name == "emit":
            original = getattr(owner, name)
            @wraps(original)
            def emit(self_obj, *args, __original=original, **kwargs):
                event = args[0] if args else kwargs.get("event")
                if event not in ("snapshot", "durable"):
                    return __original(self_obj, *args, **kwargs)
                answer = kwargs.get("answer")
                receipt = kwargs.get("receipt")
                correlation_id = (answer.get("snapshot") if isinstance(answer, dict) else None)
                if correlation_id is None:
                    correlation_id = kwargs.get("snapshot")
                if correlation_id is None and isinstance(receipt, dict):
                    correlation_id = receipt.get("snapshot")
                with recorder.span("request." + event + "_emit", correlation_id):
                    result = __original(self_obj, *args, **kwargs)
                if event == "snapshot":
                    try:
                        recorder.clear_thread_correlation()
                    except BaseException:
                        recorder._instrumentation_error = True
                return result
            setattr(owner, name, emit)
            originals.append((owner, name, original))
        else:
            originals.append((owner, name, recorder.wrap_method(owner, name, phase, key)))
    originals.append((real_module, "_sync_dir", recorder.wrap_method(real_module, "_sync_dir",
        "checkpoint.directory_sync")))
    originals.append((shared_pool, "reserve", recorder.wrap_reservation(shared_pool, "reserve",
        "memory_pool.reserve")))
    originals.append((disk_pool, "reserve", recorder.wrap_reservation(disk_pool, "reserve",
        "disk_pool.reserve")))

    # Keep the worker's pool reservation correlated with the snapshot it is
    # preparing. _run already returns that canonical content-addressed ID.
    original_run = getattr(coordinator_class, "_run")
    @wraps(original_run)
    def run(self_obj, *args, **kwargs):
        try:
            recorder.clear_thread_correlation()
        except BaseException:
            recorder._instrumentation_error = True
        result = original_run(self_obj, *args, **kwargs)
        try:
            snapshot = result.get("snapshot") if isinstance(result, dict) else None
            recorder.set_thread_correlation(snapshot if isinstance(snapshot, str) else None)
        except BaseException:
            recorder._instrumentation_error = True
        return result
    setattr(coordinator_class, "_run", run)
    originals.append((coordinator_class, "_run", original_run))

    os_originals = recorder.wrap_filesystem(real_module.os)
    originals.extend(os_originals)

    recorder.installed_targets = [
        "BranchingModel.prepare_checkpoint", "BranchingModel._prepared_envelope",
        "BranchingModel._publish_checkpoint_parts", "BranchingModel._publish_checkpoint_file",
        "BranchingModel.publish_prepared_checkpoint", "BranchingModel.validate_checkpoint",
        "BranchingModel.save_prepared_branch", "ContextCoordinator._run",
        "ContextCoordinator._publish", "Request.emit(snapshot|durable)", "Store.move",
        "SharedCheckpointPool.reserve", "DiskCheckpointPool.reserve", "real._sync_dir",
        "os.fsync(publication scope only)", "os.replace(publication scope only)",
    ]

    def restore():
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
    recorder.restore = restore
    return recorder


def install_from_loaded(output_path):
    """Called only in the owned HTTP child after the pinned modules are imported."""
    import atexit
    import importlib
    real_module = importlib.import_module("context_time_travel.real")
    coordinator_module = importlib.import_module("context_time_travel.coordinator")
    store_module = importlib.import_module("context_time_travel.store")
    pools_module = importlib.import_module("context_time_travel.pools")
    recorder = install_timing_wrappers(real_module, coordinator_module, store_module,
                                        pools_module, output_path)
    atexit.register(recorder.close)
    return recorder
