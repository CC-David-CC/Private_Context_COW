"""Real sockets and canonical Store projection; synthetic computation only."""
import json
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_workspace import CowWorkspace
from serve import test_cow_http as http_fixture
from serve import test_cow_presentation_flow as flow_fixture


class WorkspaceHTTPTests(unittest.TestCase):
    call = http_fixture.CowHTTPTests.call

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gate = threading.Event(); self.gate.set()
        self.models = {}
        def factory(gpu):
            model = flow_fixture.FlowModel(self.temp.name, gpu, (0,), [], self.gate)
            self.models[gpu] = model
            return model
        self.coordinator = ContextCoordinator(self.temp.name,
            PoolConfig(gpus=(0,), ram_bytes=1024**2, disk_bytes=8*1024**2,
                       min_available_bytes=0), factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)
        self.text = flow_fixture.CowPresentationFlowTests.make_text(self.coordinator)
        self.workspace = CowWorkspace(self.coordinator.store, self.text.tokenizer)
        self.server = make_server(CowAdapter(self.coordinator), api_key='secret',
                                  text_adapter=self.text, workspace=self.workspace)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def close_server(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(5)

    def namespace(self):
        code, raw = self.call('/cow/namespace',
                              dict(title='Code', operation_id='namespace-one'))
        self.assertEqual(code, 200)
        return json.loads(raw)['namespace_id']

    def test_empty_namespace_has_no_invented_branch_or_inference(self):
        ns = self.namespace()
        before = dict(self.models[0].stats)
        code, raw = self.call('/cow/workspace/chats')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['chats'][0]['branch_count'], 0)
        code, raw = self.call('/cow/workspace/branches?chat=' + ns)
        self.assertEqual(code, 200); self.assertEqual(json.loads(raw), dict(branches=[]))
        self.assertEqual(self.models[0].stats, before)

    def test_history_is_canonical_and_only_committed_output_is_projected(self):
        ns = self.namespace()
        code, raw = self.call('/cow/text/prepare', dict(branch='chat',
            messages=[dict(role='user', content='hello')], namespace_id=ns,
            operation_id='prepare-one', stream=False))
        self.assertEqual(code, 200)
        head = self.coordinator.store.head('chat')
        code, raw = self.call('/cow/text/answer', dict(branch='chat',
            revision=head.revision, operation_id='answer-op', max_tokens=1, stream=False))
        self.assertEqual(code, 200)
        before = dict(self.models[0].stats)
        code, raw = self.call('/cow/workspace/history?branch=chat')
        self.assertEqual(code, 200)
        history = json.loads(raw)
        self.assertEqual(history, self.workspace.history('chat'))
        self.assertEqual([m['text'] for m in history['messages']], ['hello', 'hi'])
        self.assertEqual(history['controller_status'], 'active')
        self.assertTrue(history['messages'][-1]['incomplete'])
        self.assertTrue(history['messages'][-1]['boundary_available'])
        self.assertEqual(self.models[0].stats, before)
        self.assertEqual(list(self.coordinator.store.root.rglob('*.sqlite*')), [])

    def test_projection_auth_origin_query_and_missing_scope(self):
        for path in ('/cow/workspace/chats', '/cow/workspace/branches?chat=x',
                     '/cow/workspace/history?branch=x'):
            self.assertEqual(self.call(path, key='wrong')[0], 401)
            self.assertEqual(self.call(path, origin='http://foreign.example')[0], 403)
        for path in ('/cow/workspace/chats?other=x', '/cow/workspace/branches',
                     '/cow/workspace/branches?chat=a&chat=b',
                     '/cow/workspace/history?branch=a&extra=b'):
            self.assertEqual(self.call(path)[0], 400)
        self.assertEqual(self.call('/cow/workspace/history?branch=missing')[0], 404)
        self.assertEqual(self.call('/cow/workspace/branches?chat=' + 'a'*32)[0], 404)

    def test_same_origin_ui_assets_and_native_capability_discovery(self):
        code, raw = self.call('/health', key='wrong')
        self.assertEqual(code, 200); self.assertEqual(json.loads(raw)['service'], 'context-cow')
        code, raw = self.call('/', key='wrong')
        self.assertEqual(code, 200); self.assertIn('web/workspace.js', raw)
        code, raw = self.call('/web/workspace.js', key='wrong')
        self.assertEqual(code, 200); self.assertIn('ContextWorkspace', raw)
        for path in ('/web/../cow_http.py', '/web/%2e%2e/cow_http.py',
                     '/web/%2f..%2fcow_http.py', '/web/absent.js'):
            self.assertEqual(self.call(path)[0], 404)

    def test_ordinary_workspace_mutations_are_not_exposed(self):
        ns = self.namespace()
        before = self.workspace.list_chats()
        before_branches = self.coordinator.store.branches()
        self.assertEqual(len(before_branches), 0)
        for path in ('/workspace/chats/create', '/workspace/generate',
                     '/workspace/branches/fork'):
            self.assertEqual(self.call(path, dict(chat_id=ns))[0], 404)
        self.assertEqual(self.workspace.list_chats(), before)
        self.assertEqual(self.coordinator.store.branches(), before_branches)

    def test_namespace_fork_and_rewind_publish_canonical_typed_receipts(self):
        ns = self.namespace()
        self.text.prepare('chat', [dict(role='user', content='hello')],
                          'prepare-one', namespace_id=ns).durable.result(5)
        prepared = self.coordinator.store.head('chat')
        self.text.answer('chat', prepared.revision, 'answer-op', 1).durable.result(5)
        answered = self.coordinator.store.head('chat')
        before = dict(self.models[0].stats)
        request = dict(name='copy', snapshot=answered.tip,
                       source_branch='chat', operation_id='fork-one')
        code, raw = self.call('/cow/fork', request)
        self.assertEqual(code, 200)
        self.assertEqual(self.coordinator.store.branch_namespace('copy'), ns)
        self.assertEqual(self.call('/cow/fork', request)[1], raw)
        self.assertEqual(self.workspace.history('copy')['messages'][-1]['text'], 'hi')
        code, raw = self.call('/cow/rewind', dict(branch='chat', revision=answered.revision,
            snapshot=prepared.tip, operation_id='rewind-one'))
        self.assertEqual(code, 200)
        self.assertEqual(self.coordinator.store.presentation_receipt('rewind-one')['presentation']['kind'], 'rewind')
        self.assertEqual(len(self.workspace.history('chat')['messages']), 1)
        self.assertEqual(self.workspace.history('copy')['messages'][-1]['text'], 'hi')
        self.assertEqual(self.models[0].stats, before)

    def test_failed_publication_can_be_finished_without_repeating_computation(self):
        ns = self.namespace()
        self.text.prepare('chat', [dict(role='user', content='hello')],
                          'prepare-one', namespace_id=ns).durable.result(5)
        old = self.coordinator.store.head('chat')
        model = self.models[0]
        original = model.save_prepared_branch
        def fail(*args, **kwargs): raise OSError('injected disk publication failure')
        model.save_prepared_branch = fail
        self.addCleanup(setattr, model, 'save_prepared_branch', original)
        request = self.text.answer('chat', old.revision, 'answer-op', 1)
        request.snapshot.result(5)
        with self.assertRaises(OSError): request.durable.result(5)
        code, raw = self.call('/cow/operation?operation=answer-op')
        self.assertEqual(code, 200)
        status = json.loads(raw)
        self.assertEqual(status['state'], 'failed')
        self.assertTrue(status['publication_retryable'])
        self.assertFalse(status['durable'])
        self.assertNotIn('receipt', status)
        code, raw = self.call('/cow/workspace/history?branch=chat')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['pending_request']['operation_id'], 'answer-op')
        self.assertEqual(self.coordinator.store.head('chat'), old)
        before = dict(model.stats)
        model.save_prepared_branch = original
        code, raw = self.call('/cow/publication/retry', dict(operation_id='answer-op'))
        self.assertEqual(code, 200)
        receipt = json.loads(raw)['receipt']
        self.assertEqual(receipt['branch']['revision'], old.revision + 1)
        self.assertEqual(model.stats, before)
        self.assertEqual(self.call('/cow/publication/retry', dict(operation_id='answer-op'))[1], raw)
        code, raw = self.call('/cow/operation?operation=answer-op')
        status = json.loads(raw)
        self.assertTrue(status['durable']); self.assertFalse(status['publication_retryable'])
        self.assertEqual(status['receipt'], receipt)
        self.assertEqual(self.workspace.history('chat')['messages'][-1]['text'], 'hi')

    def test_operation_controls_fail_before_mutation_for_unknown_or_unauthorized_ids(self):
        self.assertEqual(self.call('/cow/operation?operation=missing')[0], 404)
        self.assertEqual(self.call('/cow/operation?operation=a&operation=b')[0], 400)
        self.assertEqual(self.call('/cow/operation?operation=missing', key='wrong')[0], 401)
        self.assertEqual(self.call('/cow/publication/retry', dict(operation_id='missing'), key='wrong')[0], 401)
        self.assertEqual(self.call('/cow/publication/retry', dict(operation_id='missing'))[0], 409)
        self.assertEqual(self.coordinator.store.branches(), ())

    def test_retry_waits_for_original_failure_notification_to_finish(self):
        ns = self.namespace()
        self.text.prepare('chat', [dict(role='user', content='hello')],
                          'prepare-one', namespace_id=ns).durable.result(5)
        old = self.coordinator.store.head('chat')
        model = self.models[0]; original_save = model.save_prepared_branch
        original_fail = self.coordinator._fail
        entered = threading.Event(); release = threading.Event()
        def failed_save(*args, **kwargs): raise OSError('injected publication failure')
        def delayed_failure(*args, **kwargs):
            entered.set()
            if not release.wait(5): raise TimeoutError('test failure notification gate')
            return original_fail(*args, **kwargs)
        model.save_prepared_branch = failed_save
        self.coordinator._fail = delayed_failure
        self.addCleanup(setattr, model, 'save_prepared_branch', original_save)
        self.addCleanup(setattr, self.coordinator, '_fail', original_fail)
        self.addCleanup(release.set)
        request = self.text.answer('chat', old.revision, 'answer-op', 1)
        self.assertTrue(entered.wait(5))
        try:
            status = json.loads(self.call('/cow/operation?operation=answer-op')[1])
            self.assertFalse(status['publication_retryable'])
            self.assertEqual(self.call('/cow/publication/retry', dict(operation_id='answer-op'))[0], 409)
            self.assertFalse(request.durable.done())
            self.assertEqual(self.coordinator.store.head('chat'), old)
        finally: release.set()
        with self.assertRaises(OSError): request.durable.result(5)
        model.save_prepared_branch = original_save
        before = dict(model.stats)
        self.assertEqual(self.call('/cow/publication/retry', dict(operation_id='answer-op'))[0], 200)
        self.assertEqual(model.stats, before)


if __name__ == '__main__': unittest.main()
