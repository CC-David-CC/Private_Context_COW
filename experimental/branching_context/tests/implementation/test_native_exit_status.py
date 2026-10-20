import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest


API = Path(__file__).resolve().parents[2]
SCRIPTS = API / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('native_request_probe_exit_test', SCRIPTS / 'native_request_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
pool_spec = importlib.util.spec_from_file_location('pool_campaign_exit_test', SCRIPTS / 'pool_campaign.py')
pool = importlib.util.module_from_spec(pool_spec)
pool_spec.loader.exec_module(pool)


class FakeProcess:
    def __init__(self, code):
        self.code = code
        self.pid = 12345

    def poll(self):
        return self.code


class NativeExitStatusTests(unittest.TestCase):
    def test_nonzero_native_exit_fails_record_and_preserves_numeric_code(self):
        record = {'status': 'PASS'}
        process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(17)'])
        process.wait(timeout=10)
        ok = probe._record_native_shutdown(record, {'0': process}, lambda: None)
        self.assertFalse(ok)
        self.assertEqual(record['status'], 'FAIL')
        self.assertEqual(record['native_exit_codes'], {'0': 17})
        self.assertTrue(record['native_exit_observed'])
        self.assertFalse(record['native_exit_success'])

    def test_zero_native_exit_is_recorded_as_success(self):
        record = {'status': 'PASS'}
        ok = probe._record_native_shutdown(record, {'0': FakeProcess(0)}, lambda: None)
        self.assertTrue(ok)
        self.assertEqual(record['native_exit_codes'], {'0': 0})
        self.assertTrue(record['native_exit_success'])
        self.assertEqual(record['status'], 'PASS')

    def test_close_exception_still_records_exit_code_and_fails(self):
        record = {'status': 'PASS'}

        def close():
            raise RuntimeError('close failed after process shutdown')

        ok = probe._record_native_shutdown(record, {'0': FakeProcess(0)}, close)
        self.assertFalse(ok)
        self.assertEqual(record['native_exit_codes'], {'0': 0})
        self.assertIn('close failed', record['cleanup_error'])
        self.assertEqual(record['status'], 'FAIL')

    def test_phase_finalizer_stops_heartbeat_and_publishes_on_close_failure(self):
        with self.subTest('failure is durable and heartbeat is drained'):
            import tempfile

            class FakeCoordinator:
                _workers = {0: {'model': type('Model', (), {
                    'backend': type('Backend', (), {'process': FakeProcess(23)})()
                })()}}

                def close(self):
                    raise RuntimeError('native shutdown failed')

            stop = threading.Event()

            class FakeThread:
                joined = False

                def join(self):
                    self.joined = True

            thread = FakeThread()
            with tempfile.TemporaryDirectory() as temp:
                destination = Path(temp) / 'phase.json'
                record = {'status': 'PASS', 'phase': 'prepare'}
                with self.assertRaises(SystemExit) as exit_result:
                    pool._finish_phase(record, destination, 'prepare', FakeCoordinator(), None,
                                       stop, thread, 1.0)
                self.assertEqual(exit_result.exception.code, 1)
                self.assertTrue(stop.is_set())
                self.assertTrue(thread.joined)
                written = json.loads(destination.read_text())
                self.assertEqual(written['status'], 'FAIL')
                self.assertEqual(written['native_exit_codes'], {'0': 23})
                self.assertIn('native shutdown failed', written['cleanup_error'])

    def test_finalizer_preserves_active_phase_exception_and_still_publishes(self):
        import tempfile

        class FakeCoordinator:
            _workers = {0: {'model': type('Model', (), {
                'backend': type('Backend', (), {'process': FakeProcess(23)})()
            })()}}

            def close(self):
                raise RuntimeError('close failed')

        stop = threading.Event()

        class FakeThread:
            def join(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'phase.json'
            record = {'status': 'FAIL', 'phase': 'resume', 'error': 'phase assertion'}
            try:
                raise ValueError('phase assertion')
            except ValueError as original:
                pool._finish_phase(record, destination, 'resume', FakeCoordinator(), None,
                                   stop, FakeThread(), 1.0)
                self.assertEqual(str(original), 'phase assertion')
            written = json.loads(destination.read_text())
            self.assertEqual(written['error'], 'phase assertion')
            self.assertEqual(written['native_exit_codes'], {'0': 23})
            self.assertTrue(stop.is_set())


if __name__ == '__main__':
    unittest.main()
