"""Real argparse regression for a legal leading-dash credential, no GPU fixture."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from run_cli_grammar_probe import GrammarProbe


class ApiKeyArgumentTests(unittest.TestCase):
    def test_leading_dash_key_is_one_unambiguous_option_value_and_stays_redacted(self):
        class CommandInspected(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / 'registry.json'
            registry.write_text('{}')
            args = SimpleNamespace(runtime_root=root, run_dir=root, profile=root / 'profile.json',
                store=root / 'store', dependency_root=None, ram_gib=1, disk_gib=1)
            probe = GrammarProbe(args, _test_dependencies={'api_key': '-known-test-credential'})
            observed = []
            def inspect(command, **_kwargs):
                observed.extend(command)
                suffix = command[command.index('--profile'):]
                parsed = subprocess.run([sys.executable, '-c',
                    'import argparse; p=argparse.ArgumentParser(); p.add_argument("--api-key",required=True);'
                    ' a,_=p.parse_known_args(); assert a.api_key=="-known-test-credential"', *suffix],
                    capture_output=True, timeout=5)
                self.assertEqual(parsed.returncode, 0, parsed.stderr.decode())
                raise CommandInspected()
            probe.ServiceProcess = inspect
            with self.assertRaises(CommandInspected):
                probe.start_service(registry)
            from run_cli_grammar_probe import redact_argv
            self.assertNotIn(probe.secret, str(redact_argv(observed, probe.secret)))


if __name__ == '__main__':
    unittest.main()
