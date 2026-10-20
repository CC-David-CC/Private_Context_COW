"""Probe failure guards and real CPU child cleanup; no GPU qualification."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('grammar_probe',
    Path(__file__).resolve().parents[2] / 'scripts/run_grammar_first_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class GrammarProbeTests(unittest.TestCase):
    def test_incomplete_actions_cannot_qualify(self):
        engine = SimpleNamespace(eos_tokens=(1,))
        tokenizer = SimpleNamespace(token_bytes=lambda token: b'{}')
        for complete, stopped in [(False, False), (True, False), (False, True)]:
            with self.assertRaises(AssertionError):
                probe.completed_action(dict(tokens=[0], grammar_complete=complete, stopped=stopped), engine, tokenizer)
        self.assertEqual(probe.completed_action(dict(tokens=[0, 1], grammar_complete=True, stopped=True),
                                               engine, tokenizer), {})

    def test_timeout_stops_owned_child_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_script = root / 'child.py'
            child_pid = root / 'child.pid'
            child_script.write_text(
                'import subprocess,sys,time\n'
                'from pathlib import Path\n'
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                'Path(sys.argv[1]).write_text(str(p.pid))\n'
                'time.sleep(60)\n')
            record = probe.run_stage([sys.executable, str(child_script), str(child_pid)],
                                     root / 'stdout', root / 'stderr', timeout=0.5)
            self.assertTrue(record['timed_out'])
            self.assertTrue(record['pid_absent'])
            self.assertEqual(record['live_group_pids'], [])
            pid = int(child_pid.read_text())
            stat = Path('/proc', str(pid), 'stat')
            self.assertTrue(not stat.exists() or stat.read_text().split(')')[-1].split()[0] == 'Z')

    def test_close_failure_still_records_terminal_failure(self):
        class Backend:
            process = SimpleNamespace(pid=99999999, returncode=17, poll=lambda: 17)
            def close(self):
                raise RuntimeError('test close failure')
        record = {'status': 'PASS', 'failures': []}
        probe.finish_backend(Backend(), record)
        self.assertEqual(record['status'], 'FAIL')
        self.assertIn('test close failure', json.dumps(record['failures']))
        self.assertEqual(record['native_exit_code'], 17)


if __name__ == '__main__':
    unittest.main()
