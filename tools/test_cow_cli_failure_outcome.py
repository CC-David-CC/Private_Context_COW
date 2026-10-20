"""Exit 2 alone cannot distinguish rejection from a lost durable reply."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from tools import cow_cli
import tools.test_cow_cli as fixture


class MachineFailureOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CowCliTests()
        self.case.setUpClass()
        self.addCleanup(self.case.tearDownClass)
        self.case.setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def cli(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cow_cli.main(['--url', self.case.url, '--machine-result',
                                 '--state-dir', self.temp.name, *arguments])
        return code, out.getvalue(), err.getvalue()

    def test_eof_after_dispatch_reports_indeterminate_and_keeps_original_intent(self):
        fixture.FakeCowHandler.stream_mode = 'eof'
        code, out, error = self.cli('ask', '--branch', 'b-test', '--text', 'Question',
                                     '--operation-id', 'lost')
        self.assertEqual(code, 2, error)  # Preserve the existing CLI exit contract.
        value = json.loads(out)
        self.assertEqual(value['schema'], 1)
        self.assertEqual(value['error']['outcome'], 'indeterminate')
        self.assertEqual(value['error']['operation_id'], 'lost')
        self.assertEqual(cow_cli.RequestJournal(self.temp.name).get('lost')['state'], 'pending')
        self.assertEqual(sum(row[0] == 'POST' for row in fixture.FakeCowHandler.calls), 1)

    def test_explicit_http_rejection_is_distinguished_from_transport_uncertainty(self):
        path = Path(self.temp.name) / 'context.txt'
        path.write_text('Literal context.')
        code, out, error = self.cli('load', '--branch', 'rejected', '--file', str(path),
                                     '--namespace-id', 'namespace', '--operation-id', 'rejected')
        self.assertEqual(code, 2, error)
        value = json.loads(out)
        self.assertEqual(value['error']['outcome'], 'rejected')
        self.assertEqual(value['error']['operation_id'], 'rejected')
        self.assertIn('HTTP 409', value['error']['message'])


if __name__ == '__main__':
    unittest.main()
