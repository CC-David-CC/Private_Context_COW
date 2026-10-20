"""CPU-only contract tests for a multi-source reusable-context browser run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


BUILDER = (Path(__file__).parents[1] / 'experimental' / 'branching_context' /
           'scripts' / 'build_cow_browser_context_fixture.py')
spec = importlib.util.spec_from_file_location('cow_browser_context_fixture_builder', BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def _fixture(name, body_sha, profile='profile-1', root_tokens=8190,
             suffix_a=510, suffix_b=525):
    return {
        'schema': 'cow-browser-context-fixture-v1',
        'status': 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY',
        'name': name,
        'profile_sha256': 'profile-sha',
        'execution_profile_id': profile,
        'renderer_id': 'renderer-1',
        'source_body_sha256': body_sha,
        'source_body_bytes': 24000,
        'root_token_count': root_tokens,
        'append_user_separator_token_count': 1,
        'root_token_ids_sha256': 'root-' + name,
        'branches': {
            'A': {'suffix_token_count': suffix_a, 'append_user_suffix_token_count': suffix_a + 1,
                  'question': 'review A ' + name},
            'B': {'suffix_token_count': suffix_b, 'append_user_suffix_token_count': suffix_b + 1,
                  'question': 'review B ' + name},
        },
        'source_receipts': [{'path': name + '/source', 'sha256': body_sha}],
    }


class ReusableContextSetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'fixture': _fixture('engine', 'a' * 64), 'context_file': 'engine.cc',
             'attachment_name': 'engine.cc'},
            {'fixture': _fixture('store', 'b' * 64), 'context_file': 'store.py',
             'attachment_name': 'store.py'},
            {'fixture': _fixture('ui', 'c' * 64), 'context_file': 'workspace.js',
             'attachment_name': 'workspace.js'},
        ]

    def test_accepts_three_distinct_real_source_contexts_with_common_profile(self):
        normalized = builder.validate_reusable_context_set(self.rows)
        self.assertEqual(normalized['context_names'], ['engine', 'store', 'ui'])
        self.assertEqual(normalized['root_token_counts'], [8190, 8190, 8190])
        self.assertEqual(normalized['question_suffix_counts'], [[510, 525]] * 3)
        self.assertEqual(normalized['execution_profile_id'], 'profile-1')

    def test_rejects_duplicate_or_missing_source_contexts(self):
        duplicate = [dict(row, fixture=_fixture(name, 'a' * 64))
                     for row, name in zip(self.rows, ('engine', 'store', 'ui'))]
        with self.assertRaisesRegex(ValueError, 'distinct source'):
            builder.validate_reusable_context_set(duplicate)
        with self.assertRaisesRegex(ValueError, 'exactly three'):
            builder.validate_reusable_context_set(self.rows[:2])

    def test_rejects_profile_and_renderer_mismatch(self):
        rows = list(self.rows)
        rows[2] = dict(rows[2], fixture=_fixture('ui', 'c' * 64, profile='profile-2'))
        with self.assertRaisesRegex(ValueError, 'execution profile'):
            builder.validate_reusable_context_set(rows)
        rows = list(self.rows)
        rows[1] = dict(rows[1], fixture=dict(rows[1]['fixture'], renderer_id='renderer-2'))
        with self.assertRaisesRegex(ValueError, 'renderer'):
            builder.validate_reusable_context_set(rows)

    def test_rejects_unbounded_or_non_source_fixture_rows(self):
        rows = list(self.rows)
        rows[0] = dict(rows[0], fixture=_fixture('engine', 'a' * 64, root_tokens=11000))
        with self.assertRaisesRegex(ValueError, 'root token count'):
            builder.validate_reusable_context_set(rows)
        rows = list(self.rows)
        rows[1] = dict(rows[1], fixture=dict(rows[1]['fixture'], source_receipts=[]))
        with self.assertRaisesRegex(ValueError, 'source provenance'):
            builder.validate_reusable_context_set(rows)
        rows = list(self.rows)
        rows[0] = dict(rows[0], fixture=dict(rows[0]['fixture'],
            append_user_separator_token_count=0))
        with self.assertRaisesRegex(ValueError, 'append-user separator'):
            builder.validate_reusable_context_set(rows)
        rows = list(self.rows)
        bad_fixture = dict(rows[1]['fixture'])
        bad_fixture['branches'] = dict(bad_fixture['branches'])
        bad_fixture['branches']['A'] = dict(bad_fixture['branches']['A'],
                                             append_user_suffix_token_count=512)
        rows[1] = dict(rows[1], fixture=bad_fixture)
        with self.assertRaisesRegex(ValueError, 'stopped-turn suffix'):
            builder.validate_reusable_context_set(rows)

    def test_rejects_duplicate_attachment_names_and_empty_questions(self):
        rows = list(self.rows)
        rows[2] = dict(rows[2], attachment_name='engine.cc')
        with self.assertRaisesRegex(ValueError, 'attachment names'):
            builder.validate_reusable_context_set(rows)
        rows = list(self.rows)
        rows[0] = dict(rows[0], fixture=dict(rows[0]['fixture'], branches={
            'A': {'suffix_token_count': 510, 'question': ''},
            'B': {'suffix_token_count': 525, 'question': 'B'}}))
        with self.assertRaisesRegex(ValueError, 'question'):
            builder.validate_reusable_context_set(rows)

    def test_load_context_set_binds_fixture_files_and_server_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = []
            for row in self.rows:
                fixture = row['fixture']
                folder = root / fixture['name']
                folder.mkdir()
                body = (fixture['name'] + '\n').encode()
                fixture = dict(fixture, source_body_sha256=builder.sha(body),
                               source_body_bytes=len(body))
                (folder / 'source.txt').write_bytes(body)
                raw = json.dumps(fixture, sort_keys=True, separators=(',', ':')).encode()
                (folder / 'fixture.json').write_bytes(raw)
                entries.append({'name': fixture['name'],
                    'fixture_file': fixture['name'] + '/fixture.json',
                    'context_file': fixture['name'] + '/source.txt',
                    'attachment_name': row['attachment_name'],
                    'fixture_sha256': builder.sha(raw),
                    'source_body_sha256': fixture['source_body_sha256'],
                    'root_token_count': fixture['root_token_count']})
            manifest = {'schema': 'cow-browser-context-set-v1',
                'status': 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY',
                'execution_profile_id': 'profile-1', 'renderer_id': 'renderer-1',
                'contexts': entries}
            path = root / 'fixture-set.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            loaded = builder.load_reusable_context_set(path, 'profile-1')
            self.assertEqual([row['fixture']['name'] for row in loaded], ['engine', 'store', 'ui'])
            with self.assertRaisesRegex(ValueError, 'execution profile'):
                builder.load_reusable_context_set(path, 'profile-other')
            escaped = dict(manifest, contexts=[dict(entry) for entry in entries])
            escaped['contexts'][0]['context_file'] = '../source.txt'
            escaped_path = root / 'escaped-set.json'
            escaped_path.write_text(json.dumps(escaped), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'inside its fixture directory'):
                builder.load_reusable_context_set(escaped_path, 'profile-1')
            (root / 'store' / 'source.txt').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source context file digest'):
                builder.load_reusable_context_set(path, 'profile-1')


if __name__ == '__main__':
    unittest.main()
