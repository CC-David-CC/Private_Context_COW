"""HTTP disconnect after snapshot creation must not lose durable COW history.

CPU-only integration fixture: inference is a deterministic FlowModel; the real
HTTP handler, coordinator, Store and publication path are used.
"""
import json
import socket
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_workspace import CowWorkspace
from serve import test_cow_presentation_flow as flow


class CowDisconnectDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gate = threading.Event()
        self.gate.set()
        self.models = {}

        def factory(gpu):
            model = flow.FlowModel(self.temp.name, gpu, (0,), [], self.gate)
            self.models[gpu] = model
            return model

        self.coordinator = ContextCoordinator(
            self.temp.name,
            PoolConfig(gpus=(0,), ram_bytes=1024**2, disk_bytes=8 * 1024**2,
                       min_available_bytes=0),
            factory,
        )
        self.coordinator.ready(5)
        self.addCleanup(self.coordinator.close)
        self.text = flow.CowPresentationFlowTests.make_text(self.coordinator)
        self.adapter = CowAdapter(self.coordinator)
        self.workspace = CowWorkspace(
            self.coordinator.store, self.text.tokenizer,
            root_mode_resolver=self.text.root_mode_for_projection,
        )
        self.server = make_server(self.adapter, api_key="test-key",
                                  text_adapter=self.text, workspace=self.workspace)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._close_server)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def _close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def _call(self, path, body=None):
        import urllib.error
        import urllib.request

        headers = {"Authorization": "Bearer test-key"}
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body, separators=(",", ":")).encode()
        request = urllib.request.Request(self.base + path, headers=headers, data=payload)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()

    def _request_with_disconnect(self, path, body):
        raw = json.dumps(body, separators=(",", ":")).encode()
        client = socket.create_connection(self.server.server_address, timeout=5)
        client.settimeout(5)
        stream = client.makefile("rb")
        client.sendall(
            b"POST " + path.encode() + b" HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Authorization: Bearer test-key\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: keep-alive\r\n"
            b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw
        )
        status_line = stream.readline()
        self.assertIn(b" 200 ", status_line)
        while True:
            header = stream.readline()
            self.assertNotEqual(header, b"", "HTTP headers ended prematurely")
            if header in (b"\r\n", b"\n"):
                break
        events = []
        while True:
            line = stream.readline()
            self.assertNotEqual(line, b"", "Connection closed before snapshot event")
            event = json.loads(line)
            events.append(event)
            if event.get("event") == "snapshot":
                break
        return client, stream, events

    def test_disconnect_after_snapshot_can_reconnect_and_retry_exact_operation(self):
        status, raw = self._call("/cow/namespace", {"title": "Disconnect", "operation_id": "ns-op"})
        self.assertEqual(status, 200)
        namespace = json.loads(raw)["namespace_id"]
        status, raw = self._call("/cow/text/prepare", {
            "branch": "chat", "messages": [{"role": "user", "content": "hello"}],
            "namespace_id": namespace, "operation_id": "prepare-op", "stream": False,
        })
        self.assertEqual(status, 200)
        prepared = json.loads(raw)
        self.assertEqual(prepared["receipt"]["branch"]["revision"], 1)

        model = self.models[0]
        entered_publication = threading.Event()
        allow_publication = threading.Event()
        self.addCleanup(allow_publication.set)
        original_save = model.save_prepared_branch

        def gated_save(name, revision, prepared_checkpoint, operation, **kwargs):
            if operation == "answer-op":
                entered_publication.set()
                if not allow_publication.wait(5):
                    raise TimeoutError("test did not release durable publication")
            return original_save(name, revision, prepared_checkpoint, operation, **kwargs)

        model.save_prepared_branch = gated_save
        self.addCleanup(setattr, model, "save_prepared_branch", original_save)
        answer_body = {
            "branch": "chat", "revision": 1, "operation_id": "answer-op",
            "max_tokens": 1,
        }
        client, stream, first_events = self._request_with_disconnect(
            "/cow/text/answer", answer_body)
        self.assertTrue(entered_publication.wait(5), "publication did not reach the test gate")
        self.assertEqual([event["event"] for event in first_events],
                         ["request_started", "text", "snapshot"])

        # Half-close the request side: the server observes a real peer FIN, while
        # this test can still verify that the old response stream has no complete.
        client.shutdown(socket.SHUT_WR)
        old_stream_events = list(first_events)
        while True:
            line = stream.readline()
            if not line:
                break
            old_stream_events.append(json.loads(line))
        stream.close()
        client.close()
        self.assertNotIn("complete", [event.get("event") for event in old_stream_events])

        allow_publication.set()
        request = self.coordinator._requests["answer-op"]
        receipt = request.durable.result(5)
        self.assertEqual(receipt["branch"]["revision"], 2)
        self.assertEqual(self.coordinator.store.head("chat").revision, 2)

        status, raw = self._call("/cow/operation?operation=answer-op")
        self.assertEqual(status, 200)
        operation = json.loads(raw)
        self.assertTrue(operation["durable"])
        self.assertEqual(operation["receipt"], receipt)
        status, raw = self._call("/cow/workspace/history?branch=chat")
        self.assertEqual(status, 200)
        history = json.loads(raw)
        self.assertEqual([message["text"] for message in history["messages"]], ["hello", "hi"])
        self.assertTrue(history["messages"][-1]["incomplete"])

        # Replaying the exact operation after reconnect returns its original
        # receipt and does not consume another token or move the branch again.
        stats_after_publication = dict(model.stats)
        status, raw = self._call("/cow/text/answer", answer_body)
        self.assertEqual(status, 200)
        replay_events = [json.loads(line) for line in raw.splitlines()]
        replay = replay_events[-1]
        self.assertEqual(replay["event"], "complete")
        self.assertEqual(replay["receipt"], receipt)
        self.assertEqual(model.stats, stats_after_publication)
        self.assertEqual(self.coordinator.store.head("chat").revision, 2)

        # An operation ID remains bound to its original source revision.
        status, _ = self._call("/cow/text/answer", dict(answer_body, revision=0))
        self.assertEqual(status, 409)
        self.assertEqual(model.stats, stats_after_publication)
        self.assertEqual(self.coordinator.store.head("chat").revision, 2)


if __name__ == "__main__":
    unittest.main()
