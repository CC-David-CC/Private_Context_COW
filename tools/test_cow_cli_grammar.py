"""Subprocess CLI-to-HTTP tests for named server-owned GBNF states."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json
import threading

from serve.cow_grammar import GrammarRegistry
from serve.cow_http import make_server
from serve.test_cow_grammar_http import _NativeAdapter, _TextAdapter
from test_grammar_first import TinyByteTokenizer


class CowGrammarCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / 'grammars.json'
        self.config.write_text(json.dumps({'schema': 1, 'states': {
            'READ': {'format': 'gbnf', 'source': 'root ::= "A" "B"'},
            'EDIT': {'format': 'gbnf', 'source': 'root ::= "X" "Y"'},
        }}), encoding='utf-8')
        registry = GrammarRegistry.load(self.config, TinyByteTokenizer(),
                                        [TinyByteTokenizer.EOS])
        self.native = _NativeAdapter()
        self.text = _TextAdapter(registry)
        self.server = make_server(self.native, api_key='grammar-test', text_adapter=self.text)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def run_cli(self, *args):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env['COW_API_KEY'] = 'grammar-test'
        env['COW_CLI_STATE_DIR'] = str(Path(self.temp.name) / 'journal')
        env['PYTHONPATH'] = os.pathsep.join((str(root), str(root / 'experimental' / 'branching_context' / 'src'),
            env.get('PYTHONPATH', '')))
        return subprocess.run([sys.executable, str(root / 'tools' / 'cow_cli.py'),
            '--url', self.url, *args], capture_output=True, text=True,
            env=env, timeout=20)

    def test_cli_sends_only_registry_state_name_and_continues_saved_parser(self):
        first = self.run_cli('ask', '--branch', 'A', '--text', 'read this',
            '--max-tokens', '1', '--operation-id', 'cli-read-1', '--grammar-state', 'READ')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn('A', first.stdout)
        self.assertNotIn('root ::= ', first.stdout + first.stderr)
        self.assertEqual(self.text.calls[0][0:2], ('A', 'READ'))

        continued = self.run_cli('continue', '--branch', 'A', '--max-tokens', '1',
                                 '--operation-id', 'cli-read-2',
                                 '--temperature', '0.7', '--min-p', '0.0')
        self.assertEqual(continued.returncode, 0, continued.stderr)
        self.assertEqual(self.text.calls[1][0:2], ('A', 'READ'))
        self.assertEqual(self.text.calls[0][2], self.text.calls[1][2])
        self.assertEqual(self.text.calls[1][3], {'temperature':0.7,'min_p':0.0})

    def test_cli_forked_branches_can_name_different_host_grammars(self):
        for branch, state in (('A', 'READ'), ('B', 'EDIT')):
            completed = self.run_cli('ask', '--branch', branch, '--text', 'task',
                '--max-tokens', '1', '--operation-id', 'cli-' + branch,
                '--grammar-state', state)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual([(row[0], row[1]) for row in self.text.calls],
                         [('A', 'READ'), ('B', 'EDIT')])
        self.assertEqual(self.text.work,
                         [('A', ord('A')), ('B', ord('X'))])
        for branch, expected_token in (('A', 'B'), ('B', 'Y')):
            continued = self.run_cli('continue', '--branch', branch, '--max-tokens', '1',
                                     '--operation-id', 'continue-' + branch)
            self.assertEqual(continued.returncode, 0, continued.stderr)
            self.assertIn(expected_token, continued.stdout)
        self.assertEqual([(row[0], row[1]) for row in self.text.calls],
                         [('A', 'READ'), ('B', 'EDIT'), ('A', 'READ'), ('B', 'EDIT')])
        self.assertEqual(self.text.work,
                         [('A', ord('A')), ('B', ord('X')), ('A', ord('B')), ('B', ord('Y'))])

    def test_cli_rejects_unknown_state_without_submitting_work(self):
        completed = self.run_cli('ask', '--branch', 'A', '--text', 'task',
            '--max-tokens', '1', '--operation-id', 'cli-bad', '--grammar-state', 'MISSING')
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('Unknown host-selected grammar state', completed.stderr)
        self.assertEqual(self.text.calls, [])


if __name__ == '__main__':
    unittest.main()
