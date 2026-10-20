"""Fail-closed measurement helpers; synthetic controls are not GPU evidence."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_grammar_long_probe.py'
spec = importlib.util.spec_from_file_location('grammar_long_probe', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class GrammarLongProbeTests(unittest.TestCase):
    def test_failed_native_cleanup_cannot_qualify_a_successful_python_exit(self):
        valid = dict(status='PASS', native_exit_code=0, native_pid_absent=True, failures=[])
        probe.require_terminal_result(valid)
        for change in [dict(status='FAIL'), dict(native_exit_code=False), dict(native_exit_code=1),
                       dict(native_pid_absent=1), dict(failures=['late failure']),
                       dict(native_cleanup_errors=['quit failed'])]:
            with self.subTest(change=change), self.assertRaises(AssertionError):
                probe.require_terminal_result(dict(valid, **change))

    def values(self):
        result = dict(tokens=[1], stopped=False, grammar_complete=False)
        reference = dict(result, controller={'next_draw': 0, 'seen': {'1': 1}})
        return result, copy.deepcopy(reference['controller']), reference

    def test_controller_record_identity_distinguishes_boolean_from_integer(self):
        result, controller, reference = self.values()
        controller['next_draw'] = False
        with self.assertRaises(AssertionError):
            probe.validate_answer(result, controller, [5, 1], reference, [5])

    def test_token_and_scope_flags_require_exact_types(self):
        for field, value in [('tokens', [True]), ('stopped', 0), ('grammar_complete', 0)]:
            result, controller, reference = self.values()
            result[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                probe.validate_answer(result, controller, [5, 1], reference, [5])

    def test_matching_records_pass_but_sibling_history_and_controller_do_not(self):
        result, controller, reference = self.values()
        probe.validate_answer(result, controller, [5, 1], reference, [5])
        with self.assertRaises(AssertionError):
            probe.validate_answer(result, controller, [6, 1], reference, [5])
        controller['seen']['1'] = 2
        with self.assertRaises(AssertionError):
            probe.validate_answer(result, controller, [5, 1], reference, [5])

    def test_sampled_counts_keep_actual_early_end_and_reject_invalid_counts(self):
        self.assertEqual(probe.sample_counts(9), [1, 9])
        self.assertEqual(probe.sample_counts(64), [1, 16, 64])
        self.assertEqual(probe.sample_counts(128), [1, 16, 64, 128])
        for count in [0, -1, True, 1.0]:
            with self.assertRaises(ValueError):
                probe.sample_counts(count)

    def test_reference_sample_requires_full_finite_bytes_and_contained_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'frontier.f32'
            raw = struct.pack('<ff', 1., 2.)
            path.write_bytes(raw)
            descriptor = dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), consumed=5)
            self.assertEqual(probe.checked_sample(root, descriptor, 2), raw)
            for changed in [dict(path='../frontier.f32'), dict(bytes=4), dict(sha256='0'*64)]:
                with self.assertRaises(ValueError):
                    probe.checked_sample(root, dict(descriptor, **changed), 2)
            path.write_bytes(struct.pack('<ff', 1., float('nan')))
            descriptor['sha256'] = probe.sha(path)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                probe.checked_sample(root, descriptor, 2)

    def test_changed_frontier_is_an_error_even_at_one_float(self):
        raw = struct.pack('<ff', 1., 2.)
        probe.compare_frontier(raw, raw, 'control')
        with self.assertRaisesRegex(AssertionError, 'max_abs_error=1.0'):
            probe.compare_frontier(raw, struct.pack('<ff', 1., 3.), 'changed')

    def test_audit_wrappers_restore_original_operations_after_exception(self):
        def original(*args):
            raise RuntimeError('native failure')
        backend = SimpleNamespace(consume=original, consume_generated=original)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                with probe.capture_samples(backend, Path(directory), 'A', [1], [1], {}):
                    backend.consume(SimpleNamespace(), [1])
        self.assertIs(backend.consume, original)
        self.assertIs(backend.consume_generated, original)


if __name__ == '__main__':
    unittest.main()
