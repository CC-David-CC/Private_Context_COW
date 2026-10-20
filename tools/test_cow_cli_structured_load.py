"""Explicit structured load preserves wire roles; CPU real HTTP, not GPU proof."""
import json
import unittest

from tools import cow_cli
import tools.test_cow_cli_images as fixture


class StructuredLoadTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CowCliImagesTests()
        self.addCleanup(self.case.doCleanups)
        self.case.setUp()
        self.path = self.case.folder / 'messages.json'

    def load(self, flag, operation):
        return self.case.cli('load', flag, str(self.path), '--namespace-id', self.case.flow.namespace,
                             '--branch', operation, '--operation-id', operation)

    def test_structured_messages_keep_roles_content_and_order_in_actual_post(self):
        messages = [dict(role='system', content='Goal: keep exact facts.'),
                    dict(role='user', content='Hash is abc.\nSnowman \u2603'),
                    dict(role='assistant', content='Observed, not authorized.')]
        self.path.write_text(json.dumps(messages), encoding='utf-8')
        code, _, error = self.load('--messages-file', 'structured')
        self.assertEqual(code, 0, error)
        body = json.loads(cow_cli.RequestJournal(self.case.journal).get('structured')['body_json'])
        self.assertEqual(body['messages'], messages)

    def test_old_file_option_keeps_json_as_literal_plain_system_text(self):
        literal = '[{"role":"user","content":"literal JSON, not messages"}]\n'
        self.path.write_text(literal, encoding='utf-8')
        code, _, error = self.load('--file', 'literal')
        self.assertEqual(code, 0, error)
        body = json.loads(cow_cli.RequestJournal(self.case.journal).get('literal')['body_json'])
        self.assertEqual(body['messages'], [dict(role='system', content=literal)])

    def test_invalid_structured_inputs_reject_before_journal_or_model_mutation(self):
        before = self.case.fixture.inventory()
        invalid = ['[]', '{}', '[{"role":"tool","content":"x"}]',
                   '[{"role":"user","content":42}]',
                   '[{"role":"user","content":"x","grant":true}]',
                   '[{"role":"user","role":"system","content":"x"}]',
                   '[{"role":"user","content":"x"}', ' ' * (1024**2 + 1)]
        for value in invalid:
            with self.subTest(value=value[:80]):
                self.path.write_text(value, encoding='utf-8')
                code, _, error = self.load('--messages-file', 'invalid-structured')
                self.assertEqual(code, 2, error)
                self.assertIsNone(cow_cli.RequestJournal(self.case.journal).get('invalid-structured'))
                self.assertEqual(self.case.fixture.inventory(), before)


if __name__ == '__main__':
    unittest.main()
