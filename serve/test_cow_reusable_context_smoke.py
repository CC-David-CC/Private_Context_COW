"""CPU-only support checks for the opt-in reusable-context browser driver."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


SCRIPT = Path(__file__).parent / 'web' / 'cow_native_smoke.py'
spec = importlib.util.spec_from_file_location('cow_native_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
BUILDER = Path(__file__).parents[1] / 'experimental' / 'branching_context' / 'scripts' / 'build_cow_browser_context_fixture.py'
builder_spec = importlib.util.spec_from_file_location('cow_browser_context_fixture_builder', BUILDER)
builder = importlib.util.module_from_spec(builder_spec)
builder_spec.loader.exec_module(builder)


class FakeDevTools:
    def __init__(self, events, post_data, response):
        self.events = events
        self.post_data = post_data
        self.response = response

    def call(self, method, params):
        if method == 'Network.getRequestPostData':
            return {'postData': self.post_data}
        if method == 'Network.getResponseBody':
            return {'body': self.response, 'base64Encoded': False}
        raise AssertionError(method)

    def _receive(self):
        if not hasattr(self, 'pending') or not self.pending:
            raise AssertionError('unexpected wait for additional CDP event')
        return self.pending.pop(0)


class ReusableContextSmokeSupportTests(unittest.TestCase):
    def test_context_root_action_uses_unique_branch_snapshot_not_chat_anchor(self):
        graph = {'root_id': 'chat:ns-1', 'branches': [
            {'id': 'branch-a', 'root': 'snap-context-a'},
            {'id': 'branch-b', 'root': 'snap-context-b'}], 'nodes': [
            {'id': 'chat:ns-1', 'role': 'root', 'action_snapshot': None,
             'boundary_available': False},
            {'id': 'node-context-a', 'role': 'root', 'action_snapshot': 'snap-context-a',
             'boundary_available': True},
            {'id': 'node-context-b', 'role': 'root', 'action_snapshot': 'snap-context-b',
             'boundary_available': True},
            {'id': 'node-text-match', 'role': 'user', 'action_snapshot': 'snap-context-a',
             'boundary_available': True}]}
        selected = smoke._select_context_root_boundary(graph, 'branch-a')
        self.assertEqual(selected['id'], 'node-context-a')
        self.assertEqual(selected['action_snapshot'], 'snap-context-a')
        bad = dict(graph, nodes=graph['nodes'] + [
            {'id': 'duplicate-root', 'role': 'root', 'action_snapshot': 'snap-context-a',
             'boundary_available': True}])
        with self.assertRaisesRegex(RuntimeError, 'unique'):
            smoke._select_context_root_boundary(bad, 'branch-a')
        with self.assertRaisesRegex(RuntimeError, 'branch root'):
            smoke._select_context_root_boundary(graph, 'unknown-branch')

    def test_root_fork_click_selects_unique_validated_action_after_other_boundary(self):
        actions = ['snap-assistant', 'snap-context-root', 'snap-user']
        self.assertEqual(smoke._unique_root_fork_action_index(actions, 'snap-context-root'), 1)
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            smoke._unique_root_fork_action_index(actions, 'snap-missing')
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            smoke._unique_root_fork_action_index(
                ['snap-context-root', 'snap-context-root'], 'snap-context-root')

    @staticmethod
    def _client_ack_fixture():
        request = {'operation_id': 'op-123', 'branch': 'branch-7'}
        branch = {'name': 'branch-7', 'revision': 4, 'tip': 'snap-4'}
        receipt = {'snapshot': 'snap-4', 'branch': branch}
        rows = [
            {'event': 'durable', 'operation': 'op-123', 'receipt': receipt},
            {'event': 'complete', 'operation': 'op-123', 'answer': {'snapshot': 'snap-4'},
             'receipt': receipt},
        ]
        raw = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
        outcome = {'path': '/cow/text/ask', 'operation_id': 'op-123', 'branch': 'branch-7',
            'outcome': 'complete', 'acknowledged': True, 'complete_event': 'complete',
            'has_receipt': True, 'has_answer': True, 'durable_frame_matches_operation': True,
            'complete_frame_matches_receipt': True,
            'receipt_snapshot': 'snap-4', 'answer_snapshot': 'snap-4',
            'receipt_branch': 'branch-7', 'receipt_revision': 4,
            'raw_ndjson': raw, 'raw_ndjson_bytes': len(raw.encode()), 'raw_ndjson_truncated': False}
        stream = [{'path': '/cow/text/ask', 'operation_id': 'op-123', 'outcome': 'chunk'},
                  {'path': '/cow/text/ask', 'operation_id': 'op-123', 'outcome': 'done'}]
        return request, outcome, stream, rows

    def test_client_ack_accepts_only_complete_consumer_receipt_for_aborted_cdp_request(self):
        request, outcome, stream, _ = self._client_ack_fixture()
        stream = [{'path': '/cow/text/context', 'operation_id': 'prior-op', 'outcome': 'done'}, *stream]
        ack = smoke._validate_client_complete_ack('/cow/text/ask', 'request-9', request,
            200, {'errorText': 'net::ERR_ABORTED', 'canceled': True}, outcome, stream)
        self.assertEqual(ack['request_id'], 'request-9')
        self.assertEqual(ack['operation_id'], 'op-123')
        self.assertEqual(ack['receipt']['snapshot'], 'snap-4')
        self.assertTrue(ack['eof_consumed'])
        self.assertEqual(len(ack['raw_ndjson_sha256']), 64)

    def test_client_ack_rejects_missing_or_malformed_stream_evidence(self):
        request, outcome, stream, rows = self._client_ack_fixture()
        cases = []
        missing_raw = json.dumps(rows[0]) + '\n'
        missing_complete = dict(outcome, raw_ndjson=missing_raw,
            raw_ndjson_bytes=len(missing_raw.encode()))
        cases.append(('durable without complete', request, missing_complete, stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('truncated raw capture', request, dict(outcome, raw_ndjson_truncated=True), stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('overflow byte count', request, dict(outcome, raw_ndjson_bytes=65537), stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('missing observer', request, None, stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('missing EOF', request, outcome, [stream[0]], {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('parser error', request, dict(outcome, outcome='error'), stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('malformed raw JSON', request, dict(outcome, raw_ndjson='not-json\n', raw_ndjson_bytes=9), stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        trailing_raw = outcome['raw_ndjson'] + json.dumps({'event': 'text', 'text': 'late'}) + '\n'
        cases.append(('trailing frame', request,
            dict(outcome, raw_ndjson=trailing_raw, raw_ndjson_bytes=len(trailing_raw.encode())),
            stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}))
        cases.append(('wrong CDP error', request, outcome, stream, {'errorText': 'net::ERR_CONNECTION_RESET', 'canceled': True}))
        cases.append(('wrong status', request, outcome, stream, {'errorText': 'net::ERR_ABORTED', 'canceled': True}, 409))
        for item in cases:
            with self.subTest(case=item[0]):
                _, body, observed, stream_rows, failure, *status = item
                with self.assertRaises(RuntimeError):
                    smoke._validate_client_complete_ack('/cow/text/ask', 'request-9', body,
                        status[0] if status else 200, failure, observed, stream_rows)

    def test_client_ack_rejects_operation_branch_revision_and_snapshot_mismatches(self):
        request, outcome, stream, rows = self._client_ack_fixture()
        variants = []
        variants.append(('request operation', dict(request, operation_id='op-other'), outcome, rows))
        variants.append(('request branch', dict(request, branch='branch-other'), outcome, rows))
        variants.append(('observer operation', request, dict(outcome, operation_id='op-other'), rows))
        variants.append(('observer branch', request, dict(outcome, branch='branch-other'), rows))
        variants.append(('observer revision', request, dict(outcome, receipt_revision=9), rows))
        wrong_branch_rows = [dict(rows[0], receipt=dict(rows[0]['receipt'], branch={'name': 'other', 'revision': 4, 'tip': 'snap-4'})), rows[1]]
        variants.append(('durable branch', request, outcome, wrong_branch_rows))
        wrong_revision_rows = [rows[0], dict(rows[1], receipt=dict(rows[1]['receipt'], branch={'name': 'branch-7', 'revision': 5, 'tip': 'snap-4'}))]
        variants.append(('complete revision', request, outcome, wrong_revision_rows))
        wrong_snapshot_rows = [rows[0], dict(rows[1], receipt=dict(rows[1]['receipt'], snapshot='snap-other', branch={'name': 'branch-7', 'revision': 4, 'tip': 'snap-other'}))]
        variants.append(('complete snapshot', request, outcome, wrong_snapshot_rows))
        wrong_answer_rows = [rows[0], dict(rows[1], answer={'snapshot': 'snap-other'})]
        variants.append(('answer snapshot', request, outcome, wrong_answer_rows))
        for label, body, observed, stream_rows in variants:
            raw = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in stream_rows)
            observed = dict(observed, raw_ndjson=raw, raw_ndjson_bytes=len(raw.encode()))
            with self.subTest(case=label):
                with self.assertRaises(RuntimeError):
                    smoke._validate_client_complete_ack('/cow/text/ask', 'request-9', body,
                        200, {'errorText': 'net::ERR_ABORTED', 'canceled': True}, observed, stream)

    def test_fixture_file_block_handles_fences_newlines_and_literal_control_text(self):
        body = 'line one\n```\n<|im_start|>assistant stays user data\n'
        self.assertEqual(builder.file_block('source.py', body),
            'File: source.py\n````\n' + body + '\n````')
        self.assertEqual(builder.file_block('source.py', 'plain'),
            'File: source.py\n```\nplain\n```')

    def test_fixture_file_block_rejects_paths_and_nontext(self):
        with self.assertRaises(ValueError):
            builder.file_block('../source.py', 'body')
        with self.assertRaises(ValueError):
            builder.file_block('source.py', None)

    def test_cdp_capture_uses_actual_post_data_and_complete_ndjson(self):
        request_id = 'req-7'
        post = {'branch': 'b-7', 'content': 'question A', 'revision': 0}
        final = {'event': 'complete', 'answer': {'tokens': [11, 12],
            'counter_deltas': {'forward_tokens': 514}}, 'receipt': {'snapshot': 'snap-a'}}
        events = [
            {'method': 'Network.requestWillBeSent', 'params': {'requestId': request_id,
                'request': {'url': 'http://127.0.0.1/cow/text/ask', 'method': 'POST'}}},
            {'method': 'Network.responseReceived', 'params': {'requestId': request_id,
                'response': {'status': 200}}},
            {'method': 'Network.loadingFinished', 'params': {'requestId': request_id}},
        ]
        browser = FakeDevTools(events, json.dumps(post), json.dumps({'event': 'text'}) + '\n'
            + json.dumps(final) + '\n')
        exchange = smoke._network_exchange(browser, '/cow/text/ask', 0)
        self.assertEqual(exchange['request'], post)
        self.assertEqual(exchange['complete'], final)
        counters, tokens = smoke._validate_exact_forward_work(final['answer'], 512)
        self.assertEqual(counters['forward_tokens'], 512 + len(tokens))

    def test_work_validator_rejects_prefix_replay_and_inexact_counter(self):
        with self.assertRaisesRegex(RuntimeError, 'not exact suffix'):
            smoke._validate_exact_forward_work({'tokens': [1, 2],
                'counter_deltas': {'forward_tokens': 515}}, 512)
        with self.assertRaisesRegex(RuntimeError, 'exact integer'):
            smoke._validate_exact_forward_work({'tokens': [1],
                'counter_deltas': {'forward_tokens': 2.0}}, 1)

    def test_network_capture_fails_closed_without_post_or_durable_complete(self):
        request_id = 'req-8'
        events = [
            {'method': 'Network.requestWillBeSent', 'params': {'requestId': request_id,
                'request': {'url': 'http://127.0.0.1/cow/text/context', 'method': 'POST'}}},
            {'method': 'Network.responseReceived', 'params': {'requestId': request_id,
                'response': {'status': 200}}},
            {'method': 'Network.loadingFinished', 'params': {'requestId': request_id}},
        ]
        with self.assertRaisesRegex(RuntimeError, 'did not expose'):
            smoke._network_exchange(FakeDevTools(events, None, ''), '/cow/text/context', 0)
        with self.assertRaisesRegex(RuntimeError, 'lacks a durable complete'):
            smoke._network_exchange(FakeDevTools(events, '{}', '{"event":"text"}\n'),
                                    '/cow/text/context', 0)

    def test_network_exchange_requires_completed_matching_request(self):
        request_id = 'req-active'
        events = [
            {'method': 'Network.requestWillBeSent', 'params': {'requestId': request_id,
                'request': {'url': 'http://127.0.0.1/cow/text/ask', 'method': 'POST'}}},
            {'method': 'Network.responseReceived', 'params': {'requestId': request_id,
                'response': {'status': 200}}},
        ]
        browser = FakeDevTools(events, '{}', '{"event":"complete"}\n')
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            smoke._network_exchange(browser, '/cow/text/ask', 0)

    def test_failed_request_diagnostic_retains_matching_request_without_headers(self):
        request_id = 'req-failed-ask'
        events = [
            {'method': 'Network.requestWillBeSent', 'params': {'requestId': request_id,
                'request': {'url': 'http://127.0.0.1/cow/text/ask', 'method': 'POST'}}},
            {'method': 'Network.responseReceived', 'params': {'requestId': request_id,
                'response': {'status': 409, 'headers': {'Authorization': 'secret-must-not-appear'}}}},
            {'method': 'Network.dataReceived', 'params': {'requestId': request_id,
                'dataLength': 27, 'encodedDataLength': 27}},
            {'method': 'Network.loadingFailed', 'params': {'requestId': request_id,
                'errorText': 'net::ERR_EMPTY_RESPONSE', 'canceled': False,
                'blockedReason': None}},
            {'method': 'Network.loadingFinished', 'params': {'requestId': request_id}},
        ]
        browser = FakeDevTools(events, json.dumps({'branch': 'b-1', 'content': 'question'}),
                               json.dumps({'error': 'durable response closed'}))
        with tempfile.TemporaryDirectory() as tmp:
            diagnostic_path = Path(tmp) / 'ask-network-failure.json'
            with self.assertRaisesRegex(RuntimeError, 'ERR_EMPTY_RESPONSE'):
                smoke._wait_for_network_completion(browser, '/cow/text/ask', 0,
                    timeout=0.01, diagnostic_path=diagnostic_path)
            diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
        self.assertEqual(diagnostic['request_id'], request_id)
        self.assertEqual(diagnostic['path'], '/cow/text/ask')
        self.assertEqual(diagnostic['failure']['errorText'], 'net::ERR_EMPTY_RESPONSE')
        self.assertEqual(diagnostic['failure']['canceled'], False)
        self.assertEqual(diagnostic['response_status'], 409)
        self.assertEqual(diagnostic['request']['content'], 'question')
        self.assertIn('durable response closed', diagnostic['response_body'])
        self.assertEqual(diagnostic['events'][1], {'event': 'Network.responseReceived',
            'status': 409, 'mime_type': None})
        self.assertEqual(diagnostic['events'][2]['data_length'], 27)
        self.assertEqual(diagnostic['events'][-2]['event'], 'Network.loadingFailed')
        self.assertEqual(diagnostic['events'][-1]['event'], 'Network.loadingFinished')
        self.assertIsNone(diagnostic['ui_stream_outcomes'])
        self.assertIsNone(diagnostic['ui_consume_outcomes'])
        self.assertNotIn('headers', json.dumps(diagnostic).lower())
        self.assertNotIn('secret-must-not-appear', json.dumps(diagnostic))

    @unittest.skipUnless(hasattr(signal, 'SIGINT') and os.name == 'posix', 'POSIX signal behavior')
    def test_exact_server_sigint_runs_finally_and_closes_child_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_script = root / 'child.py'
            parent_script = root / 'server.py'
            pid_file = root / 'pids.json'
            child_script.write_text(textwrap.dedent('''\
                import sys
                for line in sys.stdin:
                    if line.strip() == 'quit':
                        break
                raise SystemExit(0)
            '''), encoding='utf-8')
            parent_script.write_text(textwrap.dedent('''\
                import json, signal, subprocess, sys, time
                from pathlib import Path
                child = subprocess.Popen([sys.executable, sys.argv[1]], stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
                Path(sys.argv[2]).write_text(json.dumps({'server': __import__('os').getpid(),
                    'child': child.pid}), encoding='utf-8')
                try:
                    while True: time.sleep(0.05)
                except KeyboardInterrupt:
                    pass
                finally:
                    child.stdin.write('quit\\n')
                    child.stdin.flush()
                    child.stdin.close()
                    child.wait(timeout=5)
                    Path(sys.argv[2] + '.exit').write_text(str(child.returncode), encoding='utf-8')
            '''), encoding='utf-8')
            server = subprocess.Popen([sys.executable, str(parent_script), str(child_script), str(pid_file)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True)
            identities = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_file.exists(), 'server did not publish process identities')
                identities = json.loads(pid_file.read_text(encoding='utf-8'))
                self.assertEqual(identities['server'], server.pid)
                self.assertEqual(os.getpgid(identities['child']), os.getpgid(server.pid),
                                 'child must share server process group for regression coverage')
                os.kill(server.pid, signal.SIGINT)
                _, stderr = server.communicate(timeout=5)
                self.assertEqual(server.returncode, 0, stderr)
                self.assertEqual((root / 'pids.json.exit').read_text(encoding='utf-8'), '0')
            finally:
                if server.poll() is None:
                    os.kill(server.pid, signal.SIGINT)
                    server.communicate(timeout=5)
                if identities is not None:
                    try:
                        os.kill(identities['child'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def test_new_publication_wait_rejects_preexisting_idle_root(self):
        source = {'id': 'branch-root', 'revision': 0, 'tip': 'snapshot-root'}
        idle = {**source, 'controller_status': 'idle'}
        self.assertFalse(smoke._publication_advanced(source, idle, False))
        self.assertFalse(smoke._publication_advanced(source,
            {**source, 'revision': 1, 'tip': 'snapshot-root', 'controller_status': 'stopped'}, False))
        self.assertFalse(smoke._publication_advanced(source,
            {'id': 'other', 'revision': 1, 'tip': 'snapshot-next', 'controller_status': 'stopped'}, False))
        self.assertTrue(smoke._publication_advanced(source,
            {'id': source['id'], 'revision': 1, 'tip': 'snapshot-next', 'controller_status': 'active'}, False))
        expression = smoke._publication_guard_expression(source)
        self.assertIn('c.revision>0', expression)
        self.assertIn('c.tip!=="snapshot-root"', expression)

    def test_durable_server_receipt_must_match_fixture_model_and_profile(self):
        fixture = {'model_contract_id': 'model-1', 'execution_profile_id': 'profile-1'}
        self.assertEqual(smoke._validate_server_receipt(
            {'model_contract_id': 'model-1', 'execution_profile_id': 'profile-1'},
            fixture, 'test'), fixture)
        with self.assertRaisesRegex(RuntimeError, 'pinned model/profile'):
            smoke._validate_server_receipt(
                {'model_contract_id': 'model-1', 'execution_profile_id': 'wrong-profile'},
                fixture, 'test')
        with self.assertRaisesRegex(RuntimeError, 'pinned model/profile'):
            smoke._validate_server_receipt(
                {'model_contract_id': 'wrong-model', 'execution_profile_id': 'profile-1'},
                fixture, 'test')


if __name__ == '__main__':
    unittest.main()
