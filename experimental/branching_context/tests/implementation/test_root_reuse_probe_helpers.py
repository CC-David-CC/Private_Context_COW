"""CPU-only controls for the duplicate-root GPU harness; no inference occurs."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

API = Path(__file__).resolve().parents[2]
SCRIPTS = API / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_root_reuse_probe.py"
SPEC = importlib.util.spec_from_file_location("root_reuse_probe_under_test", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class _Tokenizer:
    tokens = ["a", "b", "c", "d"]

    def token_bytes(self, token):
        return {0: b"a", 1: b"b", 2: b"c", 3: b"d"}[token]


class _Process:
    def __init__(self, pid, code):
        self.pid = pid
        self.code = code

    def poll(self):
        return self.code


class RootReuseProbeHelperTests(unittest.TestCase):
    def test_root_reuse_uses_post_capture_load_counters(self):
        answer = {"exact_materialization_reused": True,
                  "counter_deltas": {"forward_tokens": 0, "loaded_file_bytes": 0}}
        observed = PROBE.validate_root_create_observation(answer, True, 0,
            {"loaded_file_bytes": 0, "forward_tokens": 0},
            {"loaded_file_bytes": 4096, "forward_tokens": 0}, require_disk_read=True)
        self.assertEqual(observed["loaded_file_bytes_after_capture"], 4096)
        self.assertEqual(observed["coordinator_forward_tokens"], 0)

    def test_root_reuse_rejects_missing_reuse_evidence_and_forward_work(self):
        base = {"counter_deltas": {"forward_tokens": 0}}
        with self.assertRaisesRegex(AssertionError, "reuse evidence"):
            PROBE.validate_root_create_observation({**base, "exact_materialization_reused": False},
                True, 0, {}, {})
        with self.assertRaisesRegex(AssertionError, "forward counter"):
            PROBE.validate_root_create_observation({**base, "exact_materialization_reused": True},
                True, 128, {}, {})
        with self.assertRaisesRegex(AssertionError, "read saved checkpoint"):
            PROBE.validate_root_create_observation({**base, "exact_materialization_reused": True},
                True, 0, {"loaded_file_bytes": 0, "forward_tokens": 0},
                {"loaded_file_bytes": 0, "forward_tokens": 0}, True)

    def test_post_capture_work_must_match_coordinator_root_work(self):
        answer = {"exact_materialization_reused": True,
                  "counter_deltas": {"forward_tokens": 0}}
        with self.assertRaisesRegex(AssertionError, "post-capture native forward"):
            PROBE.validate_root_create_observation(answer, True, 0,
                {"forward_tokens": 0}, {"forward_tokens": 128})

    def test_controller_reconstruction_uses_fake_tokenizer_and_full_history(self):
        tokenizer = _Tokenizer()
        actual = PROBE.expected_controller("answer-A", [0, 1], [2, 3], tokenizer, [], len(tokenizer.tokens))
        self.assertEqual(actual["seen"], {"0": 1, "1": 1, "2": 1, "3": 1})
        self.assertEqual(actual["accepted_in_scope"], 2)
        self.assertEqual(actual["next_draw"], 0)

    def test_shutdown_control_requires_numeric_zero_from_fake_process(self):
        from native_request_probe import _record_native_shutdown
        success = {}
        self.assertTrue(_record_native_shutdown(success, {0: _Process(123, 0)}, lambda: None))
        self.assertEqual(success["native_exit_codes"], {"0": 0})
        failure = {}
        self.assertFalse(_record_native_shutdown(failure, {0: _Process(124, 7)}, lambda: None))
        self.assertEqual(failure["native_exit_codes"], {"0": 7})

    def test_vector_mismatch_reports_error_and_retains_both_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            actual = bytearray(PROBE.FRONTIER_WIDTH)
            expected = bytearray(PROBE.FRONTIER_WIDTH)
            actual[:4] = b"\x00\x00\x80\x3f"
            expected[:4] = b"\x00\x00\x00\x40"
            with self.assertRaises(PROBE.VectorMismatch) as caught:
                PROBE.compare_vectors(bytes(actual), bytes(expected), "test", output)
            self.assertEqual(caught.exception.maximum, 1.0)
            self.assertTrue((output / "vector-mismatches" / "test.actual.f32").is_file())
            self.assertTrue((output / "vector-mismatches" / "test.expected.f32").is_file())


if __name__ == "__main__":
    unittest.main()
