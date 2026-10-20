"""The CLI timing client must qualify numeric native-worker exit status."""
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import client_stream_speed


class NativeExitStatusTests(unittest.TestCase):
    def run_process(self, code):
        process = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
        record = {}
        success = client_stream_speed._finalize_native_worker(record, process, process.wait)
        return success, record

    def test_zero_exit_is_recorded_as_success(self):
        success, record = self.run_process(0)
        self.assertTrue(success)
        self.assertEqual(record["native_exit_codes"], {"worker": 0})
        self.assertIs(record["native_exit_success"], True)

    def test_nonzero_exit_is_recorded_as_failure(self):
        success, record = self.run_process(7)
        self.assertFalse(success)
        self.assertEqual(record["native_exit_codes"], {"worker": 7})
        self.assertIs(record["native_exit_success"], False)
        self.assertEqual(record["status"], "FAIL")

    def test_unobserved_exit_cannot_pass(self):
        class RunningProcess:
            pid = 123

            def poll(self):
                return None

        record = {}
        success = client_stream_speed._finalize_native_worker(
            record, RunningProcess(), lambda: None)
        self.assertFalse(success)
        self.assertIs(record["native_exit_observed"], False)
        self.assertEqual(record["native_exit_codes"], {"worker": None})
        self.assertEqual(record["status"], "FAIL")

    def test_close_error_cannot_pass_even_with_zero_exit(self):
        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])

        def fail_close():
            process.wait()
            raise RuntimeError("close failed")

        record = {}
        success = client_stream_speed._finalize_native_worker(record, process, fail_close)
        self.assertFalse(success)
        self.assertEqual(record["native_exit_codes"], {"worker": 0})
        self.assertEqual(record["cleanup_error"], "close failed")
        self.assertEqual(record["status"], "FAIL")

    def test_boolean_poll_result_is_not_a_numeric_exit_code(self):
        class BooleanProcess:
            pid = 124

            def poll(self):
                return False

        record = {}
        success = client_stream_speed._finalize_native_worker(
            record, BooleanProcess(), lambda: None)
        self.assertFalse(success)
        self.assertIs(record["native_exit_observed"], False)

    def test_client_requires_observed_zero_code(self):
        self.assertTrue(client_stream_speed._native_worker_succeeded(
            {"native_exit_codes": {"worker": 0}, "native_exit_success": True}))
        self.assertFalse(client_stream_speed._native_worker_succeeded(
            {"native_exit_observed": True}))
        self.assertFalse(client_stream_speed._native_worker_succeeded(
            {"native_exit_codes": {"worker": 7}, "native_exit_success": False}))


if __name__ == "__main__":
    unittest.main()
