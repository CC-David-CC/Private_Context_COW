"""CPU-only tests for the real COW HTTP recovery client against FlowModel."""
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_cow_client_recovery import (  # noqa: E402
    HttpClient,
    RecoveryFailure,
    _assert_resume_complete,
    _classify_disconnect,
    _assert_retry_consistent,
    _trace_worker,
    checkpoint_fingerprint,
    ProgressTracker,
    run_pipeline,
)
import run_cow_client_recovery as recovery  # noqa: E402
from serve.cow_browser_fixture import CowBrowserFixtureServer  # noqa: E402


class CowClientRecoveryUnitTests(unittest.TestCase):
    def test_resume_accepts_only_zero_prefill_exact_output_work_and_bound_receipt(self):
        events = [dict(event='complete', answer=dict(snapshot='s1', tokens=[7, 8],
            counter_deltas=dict(forward_tokens=2, prefill_tokens=0), text='ok'),
            receipt=dict(snapshot='s1', branch=dict(name='b', revision=5, tip='s1')))]
        result = _assert_resume_complete(events, expected_revision=4)
        self.assertEqual(result['output_tokens'], [7, 8])
        self.assertEqual(result['text_bytes'], 2)

    def test_resume_rejects_nonzero_prefix_work_and_contradictory_receipt(self):
        base = dict(event='complete', answer=dict(snapshot='s1', tokens=[7],
            counter_deltas=dict(forward_tokens=1, prefill_tokens=1), text='x'),
            receipt=dict(snapshot='s1', branch=dict(name='b', revision=1, tip='s1')))
        with self.assertRaisesRegex(RecoveryFailure, 'prefill'):
            _assert_resume_complete([base], expected_revision=0)
        base['answer']['counter_deltas']['prefill_tokens'] = 0
        base['receipt']['branch']['tip'] = 'different'
        with self.assertRaisesRegex(RecoveryFailure, 'revision/tip'):
            _assert_resume_complete([base], expected_revision=0)

    def test_resume_rejects_empty_output_and_boolean_counter(self):
        base = dict(event='complete', answer=dict(snapshot='s1', tokens=[],
            counter_deltas=dict(forward_tokens=0, prefill_tokens=0), text=''),
            receipt=dict(snapshot='s1', branch=dict(name='b', revision=1, tip='s1')))
        with self.assertRaisesRegex(RecoveryFailure, 'valid generated token'):
            _assert_resume_complete([base], expected_revision=0)
        base['answer']['tokens'] = [7]
        base['answer']['counter_deltas']['forward_tokens'] = True
        with self.assertRaisesRegex(RecoveryFailure, 'forward-token'):
            _assert_resume_complete([base], expected_revision=0)

        base['answer']['counter_deltas']['forward_tokens'] = 1
        del base['answer']['counter_deltas']['prefill_tokens']
        with self.assertRaisesRegex(RecoveryFailure, 'prefill'):
            _assert_resume_complete([base], expected_revision=0)

    def test_checkpoint_fingerprint_validates_real_manifest_envelope(self):
        with tempfile.TemporaryDirectory(prefix='cow-fingerprint-') as temp:
            directory = Path(temp) / 'checkpoints'
            directory.mkdir()
            snapshot = 's' * 64
            payload = b'native-state'
            payload_sha = hashlib.sha256(payload).hexdigest()
            (directory / f'{payload_sha}.state').write_bytes(payload)
            manifest = dict(snapshot=snapshot, payload_sha256=payload_sha)
            canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=True, allow_nan=False).encode('ascii') + b'\n'
            (directory / f'{snapshot}.json').write_text(json.dumps(dict(
                manifest=manifest, checksum=hashlib.sha256(canonical).hexdigest())), encoding='utf-8')
            result = checkpoint_fingerprint(Path(temp), snapshot)
            self.assertEqual(result['payload_sha256'], payload_sha)
            envelope_path = directory / f'{snapshot}.json'
            envelope = json.loads(envelope_path.read_text(encoding='utf-8'))
            envelope['manifest']['payload_sha256'] = '0' * 64
            envelope_path.write_text(json.dumps(envelope), encoding='utf-8')
            with self.assertRaisesRegex(RecoveryFailure, 'checksum mismatch'):
                checkpoint_fingerprint(Path(temp), snapshot)

    def test_pool_trace_parser_requires_wrapped_nonoverflowed_service_trace(self):
        with tempfile.TemporaryDirectory(prefix='cow-pool-trace-') as temp:
            path = Path(temp) / 'pool.json'
            event = dict(event='native_worker_closed', process_id=100,
                pid=200, returncode=0, close_succeeded=True)
            trace = dict(process_id=100, event_count=1, overflowed=False, events=[event])
            path.write_text(json.dumps(trace), encoding='utf-8')
            self.assertEqual(_trace_worker(path, 100)['native_pid'], 200)
            trace['overflowed'] = True
            path.write_text(json.dumps(trace), encoding='utf-8')
            with self.assertRaisesRegex(RecoveryFailure, 'overflowed'):
                _trace_worker(path, 100)
            trace['overflowed'] = False
            trace['events'][0]['returncode'] = False
            path.write_text(json.dumps(trace), encoding='utf-8')
            with self.assertRaisesRegex(RecoveryFailure, 'exit code zero'):
                _trace_worker(path, 100)

    def test_progress_heartbeat_is_atomic_and_keeps_raw_events_and_failure_phase(self):
        with tempfile.TemporaryDirectory(prefix='cow-progress-') as temp:
            run_dir = Path(temp) / 'run'
            run_dir.mkdir(mode=0o700)
            progress = ProgressTracker(run_dir, 'synthetic', interval_seconds=.01).start()
            try:
                progress.server_started(101)
                progress.phase('retry_wait', completed='resume_zero_input')
                progress.event('op-progress', dict(event='text', token_id=7, text='private'))
                time.sleep(.04)
                snapshot = json.loads((run_dir / 'progress.json').read_text(encoding='utf-8'))
                self.assertEqual(snapshot['phase'], 'retry_wait')
                self.assertEqual(snapshot['server_pids'], [101])
                self.assertEqual(snapshot['last_completed_step'], 'resume_zero_input')
                self.assertGreater(snapshot['elapsed_seconds'], 0)
            finally:
                progress.finish('FAIL', 'simulated bounded test failure')
                progress.close()
            terminal = json.loads((run_dir / 'progress.json').read_text(encoding='utf-8'))
            self.assertEqual(terminal['status'], 'FAIL')
            self.assertEqual(terminal['error'], 'simulated bounded test failure')
            events = [json.loads(line) for line in (run_dir / 'http-events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(events[0]['operation_id'], 'op-progress')
            self.assertEqual(events[0]['event']['text'], 'private')
            self.assertFalse(any(path.name.startswith('.progress-') for path in run_dir.iterdir()))

    def test_disconnect_classifier_distinguishes_cancel_from_publication_race(self):
        unchanged = dict(name='cancel-copy', tip='source', revision=0)
        cancelled = dict(operation_id='op-cancel', branch='cancel-copy', scope='coordinator-process',
                         state='cancelled', durable=False)
        self.assertEqual(_classify_disconnect(cancelled, unchanged, 'source',
                                               'op-cancel', 'cancel-copy'),
                         'cancelled_before_publication')
        published = dict(operation_id='op-cancel', branch='cancel-copy', scope='coordinator-process',
            state='durable', durable=True,
            receipt=dict(snapshot='next', branch=dict(name='cancel-copy', tip='next', revision=1)))
        self.assertEqual(_classify_disconnect(published,
            dict(name='cancel-copy', tip='next', revision=1), 'source',
            'op-cancel', 'cancel-copy'), 'publication_won')
        with self.assertRaisesRegex(RecoveryFailure, 'moved its branch'):
            _classify_disconnect(cancelled, dict(name='cancel-copy', tip='next', revision=1),
                                 'source', 'op-cancel', 'cancel-copy')
        with self.assertRaisesRegex(RecoveryFailure, 'authoritative branch head'):
            _classify_disconnect(published, unchanged, 'source', 'op-cancel', 'cancel-copy')
        with self.assertRaisesRegex(RecoveryFailure, 'bound to the expected'):
            _classify_disconnect(dict(cancelled, branch='other'), unchanged, 'source',
                                 'op-cancel', 'cancel-copy')

    def test_durable_retry_requires_same_receipt_and_no_canonical_change(self):
        head = dict(name='b-cancel', tip='s1', revision=1)
        history = dict(messages=[dict(id='assistant-1', role='assistant')],
                        moves=[dict(operation_id='op-cancel')])
        operation = dict(receipt=dict(snapshot='s1', branch=head))
        events = [dict(event='complete', receipt=operation['receipt'])]
        result = _assert_retry_consistent('publication_won', events, operation, 'op-cancel',
                                          head, head, history, history)
        self.assertTrue(result['no_additional_work_required'])
        with self.assertRaisesRegex(RecoveryFailure, 'original durable receipt'):
            _assert_retry_consistent('publication_won',
                [dict(event='complete', receipt=dict(snapshot='other'))], operation, 'op-cancel',
                head, head, history, history)

    def test_canceled_retry_allows_recompute_but_rejects_duplicate_messages(self):
        before_head = dict(name='b-cancel', tip='s0', revision=0)
        after_head = dict(name='b-cancel', tip='s1', revision=1)
        before_history = dict(messages=[dict(id='assistant-active', role='assistant')], moves=[])
        after_history = dict(messages=[dict(id='assistant-active', role='assistant')],
                             moves=[dict(operation_id='op-cancel')])
        receipt = dict(snapshot='s1', branch=after_head)
        result = _assert_retry_consistent('cancelled_before_publication',
            [dict(event='complete', receipt=receipt)], {}, 'op-cancel', before_head, after_head,
            before_history, after_history)
        self.assertFalse(result['no_additional_work_required'])
        with self.assertRaisesRegex(RecoveryFailure, 'duplicated or replaced'):
            _assert_retry_consistent('cancelled_before_publication',
                [dict(event='complete', receipt=receipt)], {}, 'op-cancel', before_head, after_head,
                before_history, dict(messages=[dict(id='assistant-active', role='assistant'),
                                               dict(id='assistant-duplicate', role='assistant')]))


class CowClientRecoverySyntheticHTTPTests(unittest.TestCase):
    def _seed_active_branch(self, store):
        fixture = CowBrowserFixtureServer(store_path=store).start()
        client = HttpClient(fixture.base_url)
        try:
            code, namespace = client.post_json('/cow/namespace',
                dict(title='Recovery fixture', operation_id='op-recovery-seed-namespace'))
            self.assertEqual(code, 200)
            namespace_id = namespace['namespace_id']
            code, prepared = client.post_json('/cow/text/prepare', dict(
                branch='RecoverySeed', namespace_id=namespace_id,
                messages=[dict(role='user', content='synthetic active fixture')],
                operation_id='op-recovery-seed-prepare', stream=False))
            self.assertEqual(code, 200)
            self.assertEqual(prepared['event'], 'complete')
            source = fixture.coordinator.store.head('RecoverySeed')
            code, answer = client.post_json('/cow/text/answer', dict(
                branch='RecoverySeed', revision=source.revision,
                operation_id='op-recovery-seed-answer', max_tokens=1, stream=False))
            self.assertEqual(code, 200)
            self.assertEqual(answer['event'], 'complete')
            self.assertEqual(fixture.workspace.history('RecoverySeed')['controller_status'], 'active')
            return namespace_id
        finally:
            fixture.close()

    def test_subprocess_restart_resume_disconnect_and_followup_use_canonical_store(self):
        with tempfile.TemporaryDirectory(prefix='cow-client-recovery-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            self._seed_active_branch(store)
            result = run_pipeline(mode='synthetic', store=store, branch='RecoverySeed',
                run_dir=root / 'run', max_tokens=8, timeout=30)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['mode'], 'synthetic')
            self.assertEqual(result['numerical_qualification']['status'], 'NOT_PERFORMED')
            self.assertIsNone(result['disconnected_operation']['retry_native_trace_unchanged'])
            self.assertIn(result['disconnected_operation']['outcome'],
                          ('publication_won', 'cancelled_before_publication'))
            self.assertEqual(result['disconnected_operation']['cancellation_coverage'],
                result['disconnected_operation']['outcome'] == 'cancelled_before_publication')
            self.assertTrue(result['source_checkpoint_unchanged'])
            self.assertNotIn('prefill_tokens', result['post_restart_followup']['counter_deltas'])
            self.assertEqual(len(result['service_instances']), 2)
            self.assertTrue(all(row['stop']['exit_code'] == 0 for row in result['service_instances']))


class CowClientRecoveryServiceStartupCleanupTests(unittest.TestCase):
    def _progress(self, run_dir):
        run_dir.mkdir(mode=0o700)
        return ProgressTracker(run_dir, 'synthetic', interval_seconds=60)

    def _capture_service_process(self):
        created = []
        base = recovery.ServiceProcess

        class CapturingServiceProcess(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        return CapturingServiceProcess, created

    @staticmethod
    def _assert_pid_gone(testcase, pid):
        with testcase.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_started_real_http_fixture_is_stopped_when_health_validation_fails(self):
        with tempfile.TemporaryDirectory(prefix='cow-startup-health-cleanup-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            progress = self._progress(root / 'run')
            captured_class, processes = self._capture_service_process()
            real_client = recovery.HttpClient
            observed_health_status = []

            class RejectHealthClient(real_client):
                def get(self, path):
                    status, value = super().get(path)
                    if path == '/health':
                        observed_health_status.append(status)
                        return 503, dict(error='synthetic validation rejection')
                    return status, value

            try:
                with patch.object(recovery, 'ServiceProcess', captured_class), \
                     patch.object(recovery, 'HttpClient', RejectHealthClient):
                    with self.assertRaisesRegex(RecoveryFailure, 'health check returned HTTP 503'):
                        recovery._start_service('synthetic', store, root / 'run', 1,
                            profile=None, timeout=10, ram_gib=1, disk_gib=1, progress=progress)
                self.assertEqual(len(processes), 1)
                process = processes[0]
                self.assertIsNotNone(process.ready)
                self.assertEqual(observed_health_status, [200])
                self.assertIsNotNone(process.process.poll())
                self._assert_pid_gone(self, process.process.pid)
            finally:
                progress.close()

    def test_started_fixture_is_stopped_when_pinned_profile_read_fails(self):
        with tempfile.TemporaryDirectory(prefix='cow-startup-profile-cleanup-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            run_dir = root / 'run'
            progress = self._progress(run_dir)
            profile = root / 'profile.json'
            profile.write_text('{not valid json', encoding='utf-8')
            captured_class, processes = self._capture_service_process()
            actual_command = [sys.executable, '-m', 'serve.cow_browser_fixture',
                              '--store', str(store), '--port', '0']
            real_client = recovery.HttpClient

            class RealHealthAsNativeClient(real_client):
                def get(self, path):
                    status, value = super().get(path)
                    if path == '/health' and status == 200:
                        return status, dict(value, service='context-cow')
                    return status, value

            try:
                with patch.object(recovery, 'ServiceProcess', captured_class), \
                     patch.object(recovery, 'HttpClient', RealHealthAsNativeClient), \
                     patch.object(recovery, '_server_command', return_value=actual_command):
                    with self.assertRaises(json.JSONDecodeError):
                        recovery._start_service('real', store, run_dir, 1,
                            profile=profile, timeout=10, ram_gib=1, disk_gib=1, progress=progress)
                self.assertEqual(len(processes), 1)
                process = processes[0]
                self.assertIsNotNone(process.ready)
                self._assert_pid_gone(self, process.process.pid)
            finally:
                progress.close()

    def test_readiness_timeout_stops_the_started_service(self):
        with tempfile.TemporaryDirectory(prefix='cow-startup-ready-cleanup-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            run_dir = root / 'run'
            progress = self._progress(run_dir)
            captured_class, processes = self._capture_service_process()
            command = [sys.executable, '-c', 'import time; time.sleep(60)']
            try:
                with patch.object(recovery, 'ServiceProcess', captured_class), \
                     patch.object(recovery, '_server_command', return_value=command):
                    with self.assertRaisesRegex(RecoveryFailure, 'Timed out waiting'):
                        recovery._start_service('synthetic', store, run_dir, 1,
                            profile=None, timeout=.2, ram_gib=1, disk_gib=1, progress=progress)
                self.assertEqual(len(processes), 1)
                process = processes[0]
                self.assertIsNotNone(process.process.poll())
                self._assert_pid_gone(self, process.process.pid)
            finally:
                progress.close()

    def test_process_launch_failure_closes_open_log_streams_and_preserves_error(self):
        with tempfile.TemporaryDirectory(prefix='cow-startup-popen-cleanup-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            run_dir = root / 'run'
            progress = self._progress(run_dir)
            captured_class, processes = self._capture_service_process()
            captured_stderr = []

            def fail_popen(*args, **kwargs):
                captured_stderr.append(kwargs['stderr'])
                raise FileNotFoundError('synthetic executable launch failure')

            try:
                with patch.object(recovery, 'ServiceProcess', captured_class), \
                     patch.object(recovery, '_server_command', return_value=['missing-service']), \
                     patch.object(recovery.subprocess, 'Popen', side_effect=fail_popen):
                    with self.assertRaisesRegex(FileNotFoundError, 'synthetic executable launch failure'):
                        recovery._start_service('synthetic', store, run_dir, 1,
                            profile=None, timeout=1, ram_gib=1, disk_gib=1, progress=progress)
                self.assertEqual(len(processes), 1)
                self.assertEqual(len(captured_stderr), 1)
                self.assertTrue(captured_stderr[0].closed)
                self.assertTrue(processes[0]._stdout_stream.closed)
            finally:
                progress.close()

    @unittest.skipUnless(os.name == 'posix', 'owned process-group cleanup is POSIX-specific')
    def test_readiness_timeout_stops_only_the_owned_service_process_group(self):
        with tempfile.TemporaryDirectory(prefix='cow-startup-group-cleanup-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            run_dir = root / 'run'
            progress = self._progress(run_dir)
            captured_class, processes = self._capture_service_process()
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                         start_new_session=True,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child_code = 'import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)'
            parent_code = (
                'import json,signal,subprocess,sys,time; '
                'signal.signal(signal.SIGINT, signal.SIG_IGN); '
                f'child=subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
                'print(json.dumps({"owned_child_pid":child.pid}), flush=True); '
                'time.sleep(60)')
            child_pid = None
            try:
                with patch.object(recovery, 'ServiceProcess', captured_class), \
                     patch.object(recovery, '_server_command',
                                  return_value=[sys.executable, '-c', parent_code]):
                    with self.assertRaisesRegex(RecoveryFailure, 'Timed out waiting'):
                        recovery._start_service('synthetic', store, run_dir, 1,
                            profile=None, timeout=.2, ram_gib=1, disk_gib=1, progress=progress)
                lines = (run_dir / 'server-1.stdout.log').read_text(encoding='utf-8').splitlines()
                child_pid = json.loads(next(line for line in lines if 'owned_child_pid' in line))['owned_child_pid']
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].process.poll())
                self._assert_pid_gone(self, child_pid)
                self.assertIsNone(unrelated.poll(), 'cleanup signalled a process outside the owned service group')
            finally:
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if unrelated.poll() is None:
                    unrelated.terminate()
                    unrelated.wait(timeout=5)
                progress.close()

    @unittest.skipUnless(os.name == 'posix', 'owned process-group behavior is POSIX-specific')
    def test_graceful_stop_signals_only_leader_before_it_closes_child(self):
        with tempfile.TemporaryDirectory(prefix='cow-graceful-child-cleanup-') as temp:
            root = Path(temp)
            events = root / 'events.log'
            child_ready = root / 'child.ready'
            child_code = '\n'.join((
                'import signal, time',
                f'path = {str(events)!r}',
                f'ready_path = {str(child_ready)!r}',
                'def log(name):',
                '    with open(path, "a", encoding="utf-8") as stream: stream.write(name + "\\n")',
                'def on_sigint(signum, frame): log("child_sigint")',
                'def on_sigterm(signum, frame):',
                '    log("child_sigterm")',
                '    raise SystemExit(0)',
                'signal.signal(signal.SIGINT, on_sigint)',
                'signal.signal(signal.SIGTERM, on_sigterm)',
                'open(ready_path, "w", encoding="utf-8").close()',
                'time.sleep(60)',
            ))
            parent_code = '\n'.join((
                'import json, signal, subprocess, sys, time',
                f'path = {str(events)!r}',
                'def log(name):',
                '    with open(path, "a", encoding="utf-8") as stream: stream.write(name + "\\n")',
                f'child = subprocess.Popen([sys.executable, "-c", {child_code!r}])',
                'def on_sigint(signum, frame):',
                '    log("leader_sigint")',
                '    child.terminate()',
                '    child.wait(timeout=5)',
                '    log("parent_closed_child")',
                '    raise SystemExit(0)',
                'signal.signal(signal.SIGINT, on_sigint)',
                'deadline = time.monotonic() + 5',
                'while not __import__("os").path.exists(' + repr(str(child_ready)) + ') and time.monotonic() < deadline: time.sleep(.01)',
                'if not __import__("os").path.exists(' + repr(str(child_ready)) + '): raise RuntimeError("child did not become ready")',
                'print(json.dumps({"event": "ready", "child_pid": child.pid}), flush=True)',
                'time.sleep(60)',
            ))
            service = recovery.ServiceProcess(
                [sys.executable, '-c', parent_code], cwd=root, env=os.environ,
                stdout_path=root / 'service.stdout.log', stderr_path=root / 'service.stderr.log',
                timeout=3)
            child_pid = None
            try:
                ready = service.start()
                child_pid = ready['child_pid']
                result = service.stop()
                self.assertEqual(result['exit_code'], 0)
                self.assertTrue(result['graceful'])
                self.assertFalse(result['forced_process_group_cleanup'])
                observed = events.read_text(encoding='utf-8').splitlines()
                self.assertIn('leader_sigint', observed)
                self.assertIn('parent_closed_child', observed)
                self.assertIn('child_sigterm', observed)
                self.assertNotIn('child_sigint', observed)
                self._assert_pid_gone(self, child_pid)
            finally:
                if service.process is not None and service.process.poll() is None:
                    service._signal_owned_process(signal.SIGKILL)
                    service.process.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_cleanup_failure_reports_both_validation_and_cleanup_errors(self):
        class BrokenCleanupProcess:
            process = None

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return dict(address='http://127.0.0.1:1', execution_profile_id='wrong')

            def stop(self):
                raise OSError('synthetic cleanup failure')

        class RejectingClient:
            def __init__(self, *args, **kwargs):
                pass

            def get(self, path):
                return 503, dict(error='synthetic health failure')

        with tempfile.TemporaryDirectory(prefix='cow-startup-cleanup-diagnostic-') as temp:
            root = Path(temp)
            store = root / 'store'
            store.mkdir(mode=0o700)
            run_dir = root / 'run'
            progress = self._progress(run_dir)
            try:
                with patch.object(recovery, 'ServiceProcess', BrokenCleanupProcess), \
                     patch.object(recovery, 'HttpClient', RejectingClient):
                    with self.assertRaises(RecoveryFailure) as caught:
                        recovery._start_service('synthetic', store, run_dir, 1,
                            profile=None, timeout=1, ram_gib=1, disk_gib=1, progress=progress)
                message = str(caught.exception)
                self.assertIn('health check returned HTTP 503', message)
                self.assertIn('synthetic cleanup failure', message)
                self.assertIsInstance(caught.exception.__cause__, RecoveryFailure)
            finally:
                progress.close()


if __name__ == '__main__':
    unittest.main()
