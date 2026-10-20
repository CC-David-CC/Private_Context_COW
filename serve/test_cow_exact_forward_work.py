"""CPU-only checks for exact suffix plus generated-token accounting."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SMOKE = Path(__file__).parent / 'web' / 'cow_native_smoke.py'
spec = importlib.util.spec_from_file_location('cow_native_smoke_forward_work', SMOKE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class ExactForwardWorkTests(unittest.TestCase):
    def test_zero_suffix_resume_accepts_only_generated_forward_work(self):
        answer = {'counter_deltas': {'forward_tokens': 3}, 'tokens': [7, 8, 9]}
        counters, tokens = smoke._validate_exact_forward_work(answer, 0)
        self.assertEqual(counters['forward_tokens'], len(tokens))

    def test_zero_suffix_resume_rejects_unexpected_prefix_work(self):
        answer = {'counter_deltas': {'forward_tokens': 4}, 'tokens': [7, 8, 9]}
        with self.assertRaisesRegex(RuntimeError, 'not exact suffix plus generated'):
            smoke._validate_exact_forward_work(answer, 0)

    def test_positive_suffix_still_requires_suffix_plus_generated_work(self):
        answer = {'counter_deltas': {'forward_tokens': 5}, 'tokens': [7, 8, 9]}
        counters, tokens = smoke._validate_exact_forward_work(answer, 2)
        self.assertEqual(counters['forward_tokens'], 2 + len(tokens))

    def test_rejects_missing_noninteger_boolean_and_negative_counts(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks exact integer'):
            smoke._validate_exact_forward_work({'tokens': []}, 0)
        for suffix in (-1, True, False, 0.0, '0', None):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(RuntimeError,
                    'lacks exact integer'):
                smoke._validate_exact_forward_work(
                    {'counter_deltas': {'forward_tokens': 0}, 'tokens': []}, suffix)
        for forward in (True, False, 0.0, '0', -1):
            with self.subTest(forward=forward), self.assertRaisesRegex(RuntimeError,
                    'lacks exact integer'):
                smoke._validate_exact_forward_work(
                    {'counter_deltas': {'forward_tokens': forward}, 'tokens': []}, 0)

    def test_completed_exchange_is_saved_before_later_work_validation(self):
        exchange = {'request': {'branch': 'branch-A'}, 'post_data': '{"branch":"branch-A"}',
            'post_data_sha256': 'request-hash', 'request_id': 'request-1',
            'events': [{'event': 'complete'}],
            'complete': {'answer': {'counter_deltas': {'forward_tokens': 3}, 'tokens': [1, 2, 3]}},
            'client_ack': {'accepted': True}, 'client_stream_ndjson': '{"event":"complete"}\n'}
        with tempfile.TemporaryDirectory() as temp:
            files = smoke._persist_completed_exchange(Path(temp), 'old-branch', exchange)
            self.assertEqual(Path(temp, files['request']).read_text(encoding='utf-8'), exchange['post_data'])
            self.assertEqual(Path(temp, files['client_stream']).read_text(encoding='utf-8'),
                             exchange['client_stream_ndjson'])
            receipt = Path(temp, files['exchange']).read_text(encoding='utf-8')
            self.assertIn('forward_tokens', receipt)
            self.assertIn('request-1', receipt)

    def test_old_branch_persists_completed_exchange_before_forward_validation(self):
        stream = '{"event":"complete"}\n'
        exchange = {'request': {'branch': 'branch-A'}, 'post_data': '{"branch":"branch-A"}',
            'post_data_sha256': 'request-hash', 'request_id': 'request-1',
            'events': [{'event': 'complete'}],
            'complete': {'answer': {'counter_deltas': {'forward_tokens': 9}, 'tokens': [1, 2, 3]},
                         'receipt': {'snapshot': 'next-tip'}},
            'client_ack': {'accepted': True, 'raw_ndjson_sha256': hashlib.sha256(
                stream.encode()).hexdigest(), 'raw_ndjson_bytes': len(stream.encode())},
            'client_stream_ndjson': stream}

        class FakeDevtools:
            def evaluate(self, expression):
                if 'const c=ContextWorkspace.current' in expression:
                    return {'id': 'branch-A', 'revision': 2, 'tip': 'source-tip',
                            'status': 'active'}
                return True

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)

            def fail_after_check(answer, suffix_count):
                saved = json.loads((folder / 'old-branch-exchange.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['complete']['answer']['counter_deltas']['forward_tokens'], 9)
                self.assertTrue((folder / 'old-branch-client-stream.ndjson').is_file())
                raise RuntimeError('expected validation stop after evidence persistence')

            with mock.patch.object(smoke, '_wait_for_network_completion', return_value='request-1'), \
                 mock.patch.object(smoke, '_network_exchange', return_value=exchange), \
                 mock.patch.object(smoke, '_validate_exact_forward_work', side_effect=fail_after_check):
                with self.assertRaisesRegex(RuntimeError, 'after evidence persistence'):
                    smoke._continue_old_branch(FakeDevtools(), 'branch-A', 'source-tip', 0,
                                               folder, 'unused on active resume', 0)

    def test_context_set_progress_keeps_completed_receipts_incrementally(self):
        with tempfile.TemporaryDirectory() as temp:
            smoke._write_context_set_progress(Path(temp), [{'name': 'engine', 'branches': ['A', 'B']}],
                                              3, [{'id': 'ENGINE-CONTEXT', 'status': 'PASS'}])
            path = Path(temp) / 'context-set-progress.json'
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'IN_PROGRESS')
            self.assertEqual(saved['completed_context_count'], 1)
            self.assertEqual(saved['completed_contexts'][0]['name'], 'engine')
            self.assertEqual(saved['checks_so_far'][0]['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
