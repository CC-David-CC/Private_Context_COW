"""HTTP boundary tests for server-owned grammar names (CPU transport only)."""
from concurrent.futures import Future
import json
import queue
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from context_time_travel.store import InvalidOperation
from serve.cow_grammar import GrammarRegistry
from serve.cow_http import make_server
from test_grammar_first import TinyByteTokenizer


class _NativeAdapter:
    def __init__(self):
        self.work = []

    def head(self, branch):
        return dict(name=branch, revision=0, tip='root', root='root')


class _TextAdapter:
    """HTTP seam fixture: registry is host-loaded and only names cross HTTP."""
    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.work = []
        self.branch_state = {}
        self.branch_history = {}

    def ask(self, branch, content, revision, operation_id, max_tokens, *,
            grammar_state=None, cancelled=None, settings=None):
        if branch in self.branch_state:
            raise InvalidOperation('Initial grammar ask cannot overwrite branch state')
        state = self.registry.start(grammar_state, operation_id, (), 259)
        session = self.registry.restore(state, ())
        token = session.choose([0.0] * 259)
        state = session.consume(state, token, bytes([token]), False)
        self.branch_history[branch] = (token,)
        self.branch_state[branch] = state
        self.work.append((branch, token))
        self.calls.append((branch, grammar_state, state['constraint']['engine_id'], settings))
        return self._request(operation_id, branch, token)

    def resume(self, branch, revision, operation_id, max_tokens, *,
               grammar_state=None, cancelled=None, settings=None):
        state = self.branch_state.get(branch)
        if state is None:
            raise InvalidOperation('No grammar controller exists for branch')
        saved = state['constraint']['agent_state']
        if grammar_state is not None and grammar_state != saved:
            raise InvalidOperation('Resume cannot change the host-selected agent state')
        history = self.branch_history[branch]
        restored = self.registry.restore(state, history)
        token = restored.choose([0.0] * 259)
        state = restored.consume(state, token, bytes([token]), False)
        self.branch_history[branch] = history + (token,)
        self.branch_state[branch] = state
        self.work.append((branch, token))
        self.calls.append((branch, saved, state['constraint']['engine_id'], settings))
        return self._request(operation_id, branch, token)

    def _request(self, operation_id, branch, token):
        request = type('Request', (), {})()
        request.events = queue.SimpleQueue()
        request.events.put(dict(event='token', token=token))
        request.snapshot = Future()
        request.snapshot.set_result(dict(tokens=[token], text=chr(token)))
        request.durable = Future()
        request.durable.set_result(dict(operation_id=operation_id, snapshot='snap-' + branch,
            branch=dict(name=branch, revision=1, tip='snap-' + branch, root='root')))

        class View:
            def __init__(self, value): self.request = value
            @property
            def snapshot(self): return self.request.snapshot
            @property
            def durable(self): return self.request.durable
            def decode_event(self, event):
                token = event['token']
                return [dict(event='text', token_id=token, text=chr(token))]
            def text_result(self): return chr(token)

        return View(request)


class CowGrammarHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / 'grammars.json'
        self.config.write_text(json.dumps({'schema': 1, 'states': {
            'READ': {'format': 'gbnf', 'source': 'root ::= "A" "B"'},
            'EDIT': {'format': 'gbnf', 'source': 'root ::= "X" "Y"'},
        }}), encoding='utf-8')
        tokenizer = TinyByteTokenizer()
        self.registry = GrammarRegistry.load(self.config, tokenizer,
                                             [TinyByteTokenizer.EOS])
        self.native = _NativeAdapter()
        self.text = _TextAdapter(self.registry)
        self.server = make_server(self.native, api_key='grammar-test', text_adapter=self.text)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def call(self, path, body, *, authenticated=True):
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers['Authorization'] = 'Bearer grammar-test'
        request = urllib.request.Request(self.url + path,
            data=json.dumps(body).encode('utf-8'), method='POST',
            headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read().decode('utf-8')

    def test_generation_http_accepts_only_named_state_and_restores_same_parser(self):
        code, raw = self.call('/cow/text/ask', dict(branch='A', content='q', revision=0,
            operation_id='read-1', max_tokens=1, grammar_state='READ', stream=True))
        self.assertEqual(code, 200, raw)
        self.assertEqual(self.text.calls[0][0:2], ('A', 'READ'))
        self.assertNotIn('source', json.dumps(self.text.calls))

        code, raw = self.call('/cow/text/resume', dict(branch='A', revision=1,
            operation_id='read-2', max_tokens=1, stream=True))
        self.assertEqual(code, 200, raw)
        self.assertEqual(self.text.calls[1][0:2], ('A', 'READ'))
        self.assertEqual(self.text.calls[0][2], self.text.calls[1][2])
        self.assertEqual(self.text.work, [('A', ord('A')), ('A', ord('B'))])
        self.assertEqual(self.native.work, [], 'HTTP fixture must not claim native inference')

    def test_http_forwards_explicit_sampler_override_with_saved_grammar_state(self):
        code, raw = self.call('/cow/text/ask', dict(branch='sampled', content='q', revision=0,
            operation_id='sampled-read-1', max_tokens=1, grammar_state='READ', stream=False))
        self.assertEqual(code, 200, raw)
        code, raw = self.call('/cow/text/resume', dict(branch='sampled', revision=1,
            operation_id='sampled-read-2', max_tokens=1, settings={'temperature':0.7,'min_p':0.0},
            stream=False))
        self.assertEqual(code, 200, raw)
        self.assertEqual(self.text.calls[1][1], 'READ')
        self.assertEqual(self.text.calls[1][3], {'temperature':0.7,'min_p':0.0})

    def test_wrong_or_unknown_host_state_fails_before_generation_work(self):
        code, raw = self.call('/cow/text/ask', dict(branch='unknown', content='q', revision=0,
            operation_id='unknown-state', max_tokens=1, grammar_state='MISSING', stream=False))
        self.assertIn(code, (400, 409), raw)
        self.assertIn('Unknown host-selected grammar state', raw)
        self.assertEqual(self.text.calls, [])
        self.assertEqual(self.native.work, [])

    def test_active_branch_cannot_change_grammar_state_mid_parser(self):
        code, raw = self.call('/cow/text/ask', dict(branch='A', content='q', revision=0,
            operation_id='read-first', max_tokens=1, grammar_state='READ', stream=False))
        self.assertEqual(code, 200, raw)
        before = list(self.text.work)
        code, raw = self.call('/cow/text/resume', dict(branch='A', revision=1,
            operation_id='wrong-transition', max_tokens=1, grammar_state='EDIT', stream=False))
        self.assertIn(code, (400, 409), raw)
        self.assertIn('Resume cannot change the host-selected agent state', raw)
        self.assertEqual(self.text.work, before)

    def test_host_state_sessions_are_independent_for_forked_branches(self):
        read = self.registry.start('READ', 'read-A', (), 259)
        edit = self.registry.start('EDIT', 'edit-B', (), 259)
        read_session = self.registry.restore(read, ())
        edit_session = self.registry.restore(edit, ())
        logits = [0.0] * 259
        read_token = read_session.choose(logits)
        edit_token = edit_session.choose(logits)
        self.assertEqual(read_token, ord('A'))
        self.assertEqual(edit_token, ord('X'))
        read = read_session.consume(read, read_token, b'A', False)
        self.assertEqual(read_session.choose(logits), ord('B'))
        self.assertEqual(edit_session.choose(logits), ord('X'))
        self.assertEqual(edit['constraint']['agent_state'], 'EDIT')

    def test_unauthorized_request_never_reaches_text_adapter(self):
        code, raw = self.call('/cow/text/ask', dict(branch='A', content='q', revision=0,
            operation_id='unauthorized', max_tokens=1, grammar_state='READ'),
            authenticated=False)
        self.assertEqual(code, 401, raw)
        self.assertEqual(self.text.calls, [])

    def test_legacy_text_adapter_does_not_receive_grammar_keyword_when_omitted(self):
        delegate = self.text
        calls = []

        class LegacyTextAdapter:
            # This is the pre-grammar adapter call surface. Omitted optional
            # grammar input must preserve compatibility with injected adapters.
            def ask(self, branch, content, revision, operation_id, max_tokens, *,
                    settings=None, cancelled=None):
                calls.append((branch, content, revision, operation_id, max_tokens))
                return delegate._request(operation_id, branch, ord('A'))

        server = make_server(self.native, api_key='grammar-test', text_adapter=LegacyTextAdapter())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f'http://127.0.0.1:{server.server_port}/cow/text/ask',
                data=json.dumps(dict(branch='legacy', content='q', revision=0,
                    operation_id='legacy-op', max_tokens=1, stream=False)).encode('utf-8'),
                method='POST', headers={'Content-Type':'application/json',
                    'Authorization':'Bearer grammar-test'})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read().decode('utf-8'))['answer']['tokens'], [ord('A')])
            self.assertEqual(calls, [('legacy','q',0,'legacy-op',1)])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_invalid_startup_registry_fails_before_coordinator_or_model_start(self):
        from serve import cow_http
        from context_time_travel.coordinator import ContextCoordinator

        profile_path = Path(self.temp.name) / 'startup-profile.json'
        profile_path.write_text(json.dumps({
            'model_path':'unused.gguf',
            'execution':{'GPU':[0], 'eos_tokens':[TinyByteTokenizer.EOS]},
            'model_contract_id':'cpu-test',
            'execution_profile_id':'cpu-execution',
        }), encoding='utf-8')
        invalid_registry = Path(self.temp.name) / 'invalid-startup-registry.json'
        invalid_registry.write_text('{"schema":1,"states":{"READ":{"format":"gbnf","source":""}}}',
                                    encoding='utf-8')
        argv = ['serve.cow_http', '--profile', str(profile_path), '--store',
                str(Path(self.temp.name) / 'store'), '--text', '--grammar-registry',
                str(invalid_registry)]
        with patch('sys.argv', argv), patch.object(ContextCoordinator, '__init__') as coordinator_init:
            # Tokenizer loading is outside this test's scope; prove registry
            # validation precedes coordinator.ready(), which starts workers.
            from tools.strata_tokenizer import Tokenizer
            with patch.object(Tokenizer, 'from_gguf', return_value=TinyByteTokenizer()):
                with self.assertRaises(InvalidOperation):
                    cow_http.main()
        coordinator_init.assert_not_called()
        self.assertEqual(self.text.work, [])

    def test_client_grammar_source_or_path_is_rejected_before_work(self):
        for name, value in (('grammar_source', 'root ::= "A"'),
                            ('grammar_path', '/server/private/grammar.gbnf')):
            with self.subTest(field=name):
                code, raw = self.call('/cow/text/ask', dict(branch='A', content='q', revision=0,
                    operation_id='client-' + name, max_tokens=1, grammar_state='READ',
                    **{name: value}))
                self.assertEqual(code, 400, raw)
                self.assertIn(name, raw)
                self.assertEqual(self.text.calls, [])
                self.assertEqual(self.text.work, [])


if __name__ == '__main__':
    unittest.main()
