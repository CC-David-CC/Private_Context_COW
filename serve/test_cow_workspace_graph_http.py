"""Authenticated HTTP contract for the native message graph read endpoint."""

import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from context_time_travel import Store
from serve.cow_http import make_server
from serve.cow_workspace import CowWorkspace


class ByteTokenizer:
    def token_bytes(self, token):
        return bytes((token & 0xFF,))


class Adapter:
    def pending_request(self, branch):
        return None


class CowWorkspaceGraphHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cow-graph-http-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.workspace = CowWorkspace(self.store, ByteTokenizer())
        self.server = make_server(Adapter(), api_key="secret", workspace=self.workspace)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())

    def request(self, path, *, key="secret"):
        request = Request(self.base + path,
                          headers={"Authorization": "Bearer " + key})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_graph_route_requires_auth_and_exact_namespace_scope(self):
        self.assertEqual(self.request("/cow/workspace/graph?chat=x", key="wrong")[0], 401)
        self.assertEqual(self.request("/cow/workspace/graph")[0], 400)
        self.assertEqual(self.request("/cow/workspace/graph?chat=x&chat=y")[0], 400)
        self.assertEqual(self.request("/cow/workspace/graph?chat=x&other=y")[0], 400)
        self.assertEqual(self.request("/cow/workspace/graph?chat=" + "f" * 32)[0], 404)

        namespace = self.store.create_namespace("Empty", "graph-http-empty").namespace_id
        before = self.store.presentation_catalog(namespace)
        code, raw = self.request("/cow/workspace/graph?chat=" + namespace)
        self.assertEqual(code, 200)
        graph = json.loads(raw)
        self.assertEqual(graph["chat_id"], namespace)
        self.assertEqual(graph["branches"], [])
        self.assertEqual(graph["nodes"][0]["role"], "root")
        self.assertTrue(graph["previous_tips_supported"])
        self.assertEqual(self.store.presentation_catalog(namespace), before)
        code, previous_raw = self.request("/cow/workspace/graph?chat=" + namespace + "&previous=true")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(previous_raw)["previous_tips"], [])
        self.assertEqual(self.request("/cow/workspace/graph?chat=" + namespace + "&previous=maybe")[0], 400)
