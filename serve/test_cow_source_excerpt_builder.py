"""CPU tests for bounded, hash-checked source excerpt derivation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (Path(__file__).parents[1] / 'experimental' / 'branching_context' /
          'scripts' / 'derive_cow_browser_source_set.py')
spec = importlib.util.spec_from_file_location('cow_source_excerpt_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class BoundedSourceExcerptTests(unittest.TestCase):
    def _evidence(self, root):
        rows = []
        for index, name in enumerate(builder.QUESTIONS):
            source_root = root / name
            source_root.mkdir(exist_ok=True)
            files = []
            for number in range(2):
                relative = f'{name}/source-{number}.py'
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                raw = ((f'# {name} source file {number}\n' +
                        f'def function_{number}():\n    return {index + number}\n') * 8).encode()
                path.write_bytes(raw)
                files.append({'path': str(path), 'relative_path': relative,
                    'source_root': str(root), 'bytes': len(raw), 'sha256': _sha(raw)})
            body = f'Original source bundle {name} ' * 8
            rows.append({'name': name, 'source_root': str(root),
                'base_token_count': 10000, 'root_messages': [{'role':'user','content':body}],
                'included_files_in_order': files})
        return {'contexts': rows}

    def test_derives_three_distinct_source_contexts_with_exact_whole_line_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source-evidence.json'
            source.write_text(json.dumps(self._evidence(root)), encoding='utf-8')
            output = root / 'bounded.json'
            result = builder.derive(source, output, target_tokens=6000)
            self.assertEqual([row['name'] for row in result['contexts']], list(builder.QUESTIONS))
            self.assertEqual(len({row['source_body_sha256'] for row in result['contexts']}), 3)
            for row in result['contexts']:
                self.assertEqual(row['root_messages'][0]['role'], 'user')
                self.assertNotEqual(row['A_messages'][-1]['content'], row['B_messages'][-1]['content'])
                self.assertTrue(row['source_receipts'])
                for receipt in row['source_receipts']:
                    raw = Path(receipt['path']).read_bytes()
                    lines = raw.decode('utf-8').splitlines(keepends=True)
                    excerpt = ''.join(lines[receipt['line_start_1based']-1:
                                             receipt['line_end_1based']]).encode()
                    self.assertEqual(receipt['byte_start'], 0)
                    self.assertEqual(receipt['byte_end'], len(excerpt))
                    self.assertEqual(receipt['excerpt_sha256'], _sha(excerpt))
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['source_evidence_sha256'],
                             result['source_evidence_sha256'])

    def test_rejects_source_file_drift_from_recorded_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source-evidence.json'
            evidence = self._evidence(root)
            source.write_text(json.dumps(evidence), encoding='utf-8')
            first = Path(evidence['contexts'][0]['included_files_in_order'][0]['path'])
            first.write_text('changed after evidence', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source file changed'):
                builder.derive(source, root / 'bounded.json', target_tokens=6000)

    def test_rejects_duplicate_contexts_and_invalid_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source-evidence.json'
            evidence = self._evidence(root)
            evidence['contexts'][2]['name'] = evidence['contexts'][0]['name']
            source.write_text(json.dumps(evidence), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'three supported source domains'):
                builder.derive(source, root / 'duplicate.json', target_tokens=6000)
            evidence = self._evidence(root)
            source.write_text(json.dumps(evidence), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'between 6000 and 9000'):
                builder.derive(source, root / 'invalid.json', target_tokens=100)


if __name__ == '__main__':
    unittest.main()
