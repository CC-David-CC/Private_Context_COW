"""CPU-only checks for numeric native shutdown receipts in resident campaigns."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

BRANCHING_CONTEXT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(os.environ.get('RESIDENT_RESUME_SCRIPTS', BRANCHING_CONTEXT / 'scripts')).resolve()
SCRIPT = Path(os.environ.get('RESIDENT_RESUME_SOURCE', SCRIPTS / 'resident_resume.py')).resolve()
if not SCRIPTS.is_dir() or not SCRIPT.is_file():
    raise RuntimeError('resident_resume test script paths are not available')
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('resident_resume_under_test', SCRIPT)
resident = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resident)


class FakeProcess:
    def __init__(self, code):
        self.pid = 424242
        self.code = code

    def poll(self):
        return self.code


class FakeBackend:
    def __init__(self, code):
        self.process = FakeProcess(code)
        self.gpu_id = 0

    def close(self):
        pass

    def work_counters(self):
        return {}


class FakeStore:
    def fork(self, *args, **kwargs):
        return None

    def head(self, name):
        return types.SimpleNamespace(revision=0)


class FakeModel:
    def __init__(self, directory, code):
        self.store = FakeStore()
        self.backend = FakeBackend(code)
        self.checkpoint_directory = directory / 'checkpoints'
        self.checkpoint_directory.mkdir()
        payload = b'synthetic-payload'
        import hashlib
        digest = hashlib.sha256(payload).hexdigest()
        (self.checkpoint_directory / (digest + '.state')).write_bytes(payload)
        (self.checkpoint_directory / 'root.json').write_text(
            json.dumps({'manifest': {'payload_sha256': digest}}))

    def create(self, ids):
        return 'root'

    def save_branch(self, *args, **kwargs):
        return {'saved': True}

    def run_greedy_request(self, root, suffix, count, stream=None):
        return {'snapshot': 'snap-' + stream + '-' + str(count),
                'tokens': list(range(100, 100 + count))}


class NativeShutdownTests(unittest.TestCase):
    def _exercise_prepare(self, code):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'case'
            output.mkdir()
            (output / 'fixture.json').write_text(json.dumps(
                {'prefix': [1], 'A': [2], 'B': [3]}))
            model = FakeModel(Path(temp), code)
            resident.open_model = lambda args: (model, 0.0)
            resident.capture = lambda model, sid, out, label: self._capture(out, label, sid)
            args = types.SimpleNamespace(phase='prepare', output=output, generated=5)
            resident.execute(args)
            return json.loads((output / 'prepare.json').read_text())

    @staticmethod
    def _capture(output, label, sid):
        name = label + '.f32'
        (output / name).write_bytes(b'frontier')
        return {'frontier': name, 'schedule_sha256': 'a' * 64,
                'history': [sid], 'controller': {}, 'consumed': 1}

    def test_execute_rejects_nonzero_native_exit(self):
        with self.assertRaises(RuntimeError):
            self._exercise_prepare(7)

    def test_execute_records_zero_native_exit(self):
        record = self._exercise_prepare(0)
        self.assertEqual(record['status'], 'PASS')
        self.assertEqual(record.get('native_exit_codes'), {'0': 0})
        self.assertIs(record.get('native_exit_success'), True)

    def test_require_shutdown_rejects_missing_or_nonzero_or_bool(self):
        for record in (
            {'native_exit_success': True, 'native_exit_codes': {}},
            {'native_exit_success': True, 'native_exit_codes': {'0': None}},
            {'native_exit_success': True, 'native_exit_codes': {'0': 7}},
            {'native_exit_success': True, 'native_exit_codes': {'0': True}},
            {'native_exit_success': 1, 'native_exit_codes': {'0': 0}},
        ):
            with self.subTest(record=record), self.assertRaises(AssertionError):
                resident.require_native_shutdown(record, 'fixture')

    def test_phase_summary_propagates_exact_numeric_codes(self):
        child = {'native_exit_observed': True, 'native_exit_success': True,
                 'native_exit_codes': {'0': 0, '1': 0}}
        summary = resident.native_phase_summary('resume', 9, 0, child)
        self.assertEqual(summary['native_exit_codes'], {'0': 0, '1': 0})
        self.assertTrue(resident.require_phase_success('resume', 0,
                         dict(status='PASS', **child)))

    def test_phase_success_rejects_nonzero_native_code_even_if_python_passes(self):
        child = {'status': 'PASS', 'native_exit_observed': True,
                 'native_exit_success': False, 'native_exit_codes': {'0': 7}}
        with self.assertRaises(AssertionError):
            resident.require_phase_success('resume', 0, child)


if __name__ == '__main__':
    unittest.main()
