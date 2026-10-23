import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.pcc_foundation_session import DenseCandidateSession, admit_samples, runtime_contract, stop_owned


class FoundationSessionTests(unittest.TestCase):
    def samples(self):
        return [[dict(index=2, uuid='GPU-selected', free_mib=40000, activity=10),
                 dict(index=3, uuid='GPU-other', free_mib=80000, activity=0)] for _ in range(5)]

    def test_selected_gpu_activity_memory_and_identity_are_independent(self):
        values = self.samples()
        self.assertEqual(admit_samples(values, 'GPU-selected', maximum_activity=50, minimum_free_mib=32768), 2)
        for rows in values:
            rows[0]['activity'] = 50
        with self.assertRaises(RuntimeError):
            admit_samples(values, 'GPU-selected', maximum_activity=50, minimum_free_mib=32768)
        values = self.samples(); values[-1][0]['free_mib'] = 32000
        with self.assertRaises(RuntimeError):
            admit_samples(values, 'GPU-selected', maximum_activity=50, minimum_free_mib=32768)
        values = self.samples(); values[-1][0]['index'] = 4
        with self.assertRaises(RuntimeError):
            admit_samples(values, 'GPU-selected', maximum_activity=50, minimum_free_mib=32768)

    def test_missing_samples_or_uuid_are_not_admitted(self):
        for values in (self.samples()[:4], self.samples() + self.samples()[:1]):
            with self.assertRaises(ValueError):
                admit_samples(values, 'GPU-selected', maximum_activity=50, minimum_free_mib=32768)
        with self.assertRaises(ValueError):
            admit_samples(self.samples(), 'GPU-unknown', maximum_activity=50, minimum_free_mib=32768)

    def profile(self):
        return dict(model_path='/owned/base.gguf', model_weights_sha256='a'*64,
            geometry={'layers':32}, native_build={'sha256':{'native':'b'*64}},
            bridge_library='/owned/build/libqwen35ctt.so', execution=dict(
                GPU=[2], physical_sharing=False, capacity=8192, prefill_chunk=128,
                threads=4, f32_activations=True))

    def configuration(self):
        return dict(host='127.0.0.1', openai_port=12345, openai_max_output=512,
                    openai_thinking=False, openai_model='test')

    def test_runtime_contract_binds_code_and_budgets_but_not_adapter(self):
        profile = self.profile(); config = self.configuration()
        original = runtime_contract(profile, config)
        adapted = copy.deepcopy(profile); adapted['execution']['adapter'] = {'sha256':'c'*64}
        self.assertEqual(runtime_contract(adapted, config), original)
        changed = copy.deepcopy(profile); changed['native_build']['sha256']['native'] = 'd'*64
        self.assertNotEqual(runtime_contract(changed, config), original)
        self.assertNotEqual(runtime_contract(profile, dict(config, openai_thinking=True)), original)
        self.assertNotEqual(runtime_contract(profile, dict(config, openai_max_output=1024)), original)

    def test_wrong_candidate_or_changed_runtime_cannot_be_labeled_valid(self):
        session = DenseCandidateSession.__new__(DenseCandidateSession)
        session.profile = self.profile(); session.configuration = self.configuration()
        session.contract = runtime_contract(session.profile, session.configuration)
        session.runtime_id = 'e'*64
        candidate = dict(mode='lora', candidate_path='/owned/adapter.gguf', candidate_sha256='c'*64)
        with patch('experimental.branching_context.scripts.prepare_qwen35_profile.make_profile', return_value=session.profile):
            with self.assertRaisesRegex(ValueError, 'exact requested candidate'):
                session._profile(candidate, Path('/unused'))
        correct = copy.deepcopy(session.profile)
        correct['adapter'] = {'sha256':'c'*64}; correct['execution']['adapter'] = {'sha256':'c'*64}
        with patch('experimental.branching_context.scripts.prepare_qwen35_profile.make_profile', return_value=correct):
            _, receipt = session._profile(candidate, Path('/unused'))
            self.assertEqual(receipt, dict(base_sha256='a'*64, adapter_sha256='c'*64, runtime_id='e'*64))
        correct['execution']['capacity'] = 16384
        with patch('experimental.branching_context.scripts.prepare_qwen35_profile.make_profile', return_value=correct):
            with self.assertRaisesRegex(ValueError, 'frozen baseline'):
                session._profile(candidate, Path('/unused'))

    def test_network_exposure_or_recursive_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            profile = directory/'profile.json'; profile.write_text(json.dumps(self.profile()))
            for extra in (dict(host='0.0.0.0'), dict(training_config='/owned/training.json')):
                (directory/'server.json').write_text(json.dumps(dict(self.configuration(), profile=str(profile), **extra)))
                with self.assertRaises(ValueError):
                    DenseCandidateSession(directory, python=sys.executable, gpu_uuid='GPU-selected')

    def test_entry_imports_runtime_in_a_fresh_isolated_process(self):
        root = Path(__file__).resolve().parents[1]
        code = """import sys
sys.path.insert(0, sys.argv[1])
from tools import cow_server
cow_server.main = lambda: print('EXISTING_COW_ENTRY_REACHED')
sys.argv = ['session', '/unused/private-server', '/unused/private-trace']
from tools.pcc_foundation_session import main
main()
"""
        result = subprocess.run([sys.executable, '-I', '-c', code, str(root)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('EXISTING_COW_ENTRY_REACHED', result.stdout)

    def test_owned_process_is_stopped_and_reaped(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True)
        try:
            stop_owned(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
