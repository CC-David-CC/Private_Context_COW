"""Lost-ack replay across coordinator and HTTP server recreation.

This is CPU-only FlowModel coverage of the real Store/CAS/HTTP boundaries. It
does not model a native process crash or establish GPU recovery behavior.
"""
import json
import socket
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_workspace import CowWorkspace
from serve import test_cow_presentation_flow as flow


class CowPostcommitRestartTests(unittest.TestCase):
    make_text = staticmethod(flow.CowPresentationFlowTests.make_text)

    def setUp(self):
        flow.CowPresentationFlowTests.setUp(self)

    @staticmethod
    def _shutdown_http(server, thread):
        server.shutdown()
        server.server_close()
        thread.join(5)

    def _start_http(self):
        adapter = CowAdapter(self.coordinator)
        workspace = CowWorkspace(self.coordinator.store, self.text.tokenizer,
            root_mode_resolver=self.text.root_mode_for_projection)
        server = make_server(adapter, api_key='test-key', text_adapter=self.text,
                             workspace=workspace)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    @staticmethod
    def _read_headers(stream):
        while True:
            line = stream.readline()
            if not line:
                raise AssertionError('server closed before finishing response headers')
            if line in (b'\r\n', b'\n'):
                return

    def _send_until_snapshot(self, server, body):
        raw = json.dumps(body, separators=(',', ':')).encode()
        client = socket.create_connection(server.server_address, timeout=5)
        stream = client.makefile('rb')
        client.sendall(
            b'POST /cow/text/answer HTTP/1.1\r\n'
            b'Host: 127.0.0.1\r\n'
            b'Authorization: Bearer test-key\r\n'
            b'Content-Type: application/json\r\n'
            b'Connection: keep-alive\r\n'
            b'Content-Length: ' + str(len(raw)).encode('ascii') + b'\r\n\r\n' + raw)
        status_line = stream.readline()
        self.assertIn(b' 200 ', status_line)
        self._read_headers(stream)
        events = []
        while True:
            line = stream.readline()
            self.assertTrue(line, 'response ended before the snapshot event')
            event = json.loads(line)
            events.append(event)
            if event.get('event') == 'snapshot':
                return client, stream, events

    def _call(self, server, path, body=None):
        import urllib.error
        import urllib.request

        headers = {'Authorization': 'Bearer test-key'}
        payload = None
        if body is not None:
            headers['Content-Type'] = 'application/json'
            payload = json.dumps(body, separators=(',', ':')).encode()
        request = urllib.request.Request(
            f'http://127.0.0.1:{server.server_port}{path}', headers=headers, data=payload)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()

    def test_lost_ack_replay_recovers_exact_durable_answer_after_restart(self):
        namespace = self.namespace1
        prepared = self.text.prepare('restart-chat', [{'role': 'user', 'content': 'hello'}],
            'restart-root', namespace_id=namespace).durable.result(5)
        original_head = self.coordinator.store.head('restart-chat')
        self.assertEqual(original_head.revision, 1)
        first_model = self.models[0]
        first_model.plans['answer-after-move'] = (257,)

        server, thread = self._start_http()
        self.addCleanup(self._shutdown_http, server, thread)
        # FlowModel owns a separate Store instance for checkpoint publication;
        # patch that actual writer rather than the coordinator's read view.
        writer_store = first_model.store
        original_move = writer_store.move
        move_completed = threading.Event()
        allow_ack = threading.Event()

        def commit_then_pause(name, expected_revision, snapshot, operation_id, **kwargs):
            # Bind the same durable checkpoint metadata as the production writer.
            # The legacy FlowModel otherwise omits these optional annotations.
            checkpoint = first_model.cache[snapshot]
            kwargs['checkpoint_receipt'] = dict(schema=1, snapshot=snapshot,
                payload_sha256=checkpoint.checksum,
                model_contract_id=checkpoint.model_contract_id,
                execution_profile_id=checkpoint.execution_profile_id)
            result = original_move(name, expected_revision, snapshot, operation_id, **kwargs)
            if operation_id == 'answer-after-move':
                move_completed.set()
                if not allow_ack.wait(5):
                    raise TimeoutError('test did not release the post-commit gate')
            return result

        writer_store.move = commit_then_pause
        answer_body = {
            'branch': 'restart-chat', 'revision': original_head.revision,
            'operation_id': 'answer-after-move', 'max_tokens': 2,
        }
        try:
            client, stream, partial_events = self._send_until_snapshot(server, answer_body)
            self.assertTrue(move_completed.wait(5), 'Store.move did not reach the durable boundary')
            committed_head = self.coordinator.store.head('restart-chat')
            self.assertEqual(committed_head.revision, original_head.revision + 1)
            receipt = self.coordinator.store.presentation_receipt('answer-after-move')
            self.assertEqual(receipt['result']['tip'], committed_head.tip)
            self.assertNotIn('complete', [event.get('event') for event in partial_events])
            original_request = self.coordinator._requests['answer-after-move']
            self.assertFalse(original_request.durable.done())

            # The Store move is committed, but the coordinator has not yet
            # released the durable future and no HTTP complete frame was sent.
            client.shutdown(socket.SHUT_RDWR)
            stream.close()
            client.close()
        finally:
            allow_ack.set()
            writer_store.move = original_move

        old_request = self.coordinator._requests['answer-after-move']
        original_receipt = old_request.durable.result(5)
        self.assertEqual(original_receipt['branch']['revision'], 2)
        first_work_after_commit = first_model.work_counters()

        server.shutdown()
        server.server_close()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        flow.CowPresentationFlowTests.new_coordinator(self)
        self.assertIsNot(self.models[0], first_model)
        new_model = self.models[0]
        new_work_before_replay = new_model.work_counters()
        replay_server, replay_thread = self._start_http()
        self.addCleanup(self._shutdown_http, replay_server, replay_thread)

        # Canonical publication survives coordinator recreation. Both lookup
        # and the original POST recover the exact receipt without model work.
        status, raw = self._call(replay_server,
            '/cow/operation?operation=answer-after-move')
        self.assertEqual(status, 200)
        observed = json.loads(raw)
        self.assertTrue(observed['durable'])
        self.assertEqual(observed['scope'], 'canonical-presentation')
        self.assertEqual(observed['receipt'], original_receipt)
        self.assertEqual(new_model.work_counters(), new_work_before_replay)
        self.assertEqual(self.coordinator.store.head('restart-chat'), committed_head)
        status, raw = self._call(replay_server, '/cow/text/answer', answer_body)
        self.assertEqual(status, 200)
        replay_events = [json.loads(line) for line in raw.splitlines()]
        complete = [event for event in replay_events if event.get('event') == 'complete']
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]['receipt'], original_receipt)
        self.assertEqual(complete[0]['answer']['tokens'], [257])
        self.assertEqual(complete[0]['answer']['text'], 'hi')
        self.assertEqual(new_model.work_counters(), new_work_before_replay)
        self.assertEqual(self.coordinator.store.head('restart-chat'), committed_head)
        self.assertEqual(first_model.work_counters(), first_work_after_commit)
        self.assertEqual(prepared['snapshot'], original_head.tip)

        # Reusing the same operation ID with a different bound argument is a
        # invalid input under the existing HTTP contract; it cannot rerun work or move the already committed branch.
        status, raw = self._call(replay_server, '/cow/text/answer',
                                 dict(answer_body, max_tokens=3))
        self.assertEqual(status, 400)
        self.assertIn(b'operation id reused for different input', raw.lower())
        self.assertEqual(new_model.work_counters(), new_work_before_replay)
        self.assertEqual(self.coordinator.store.head('restart-chat'), committed_head)


if __name__ == '__main__':
    unittest.main()
