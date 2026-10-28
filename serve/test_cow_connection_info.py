"""Authenticated discovery must never allocate a context or disclose local paths."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from serve.cow_http import make_server


class NoModelAdapter:
    def __getattr__(self, name):
        raise AssertionError("discovery tried to use adapter: " + name)


class ConnectionInfoTests(unittest.TestCase):
    def test_unauthenticated_peer_export_is_still_refused(self):
        class PeerAdapter:
            def cache_transfer_enabled(self): return True
        with self.assertRaisesRegex(ValueError, 'requires an API key'):
            make_server(PeerAdapter())

    def start(self, info=None, **kwargs):
        server = make_server(NoModelAdapter(), api_key="fixture-secret", connection_info=info, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return "http://127.0.0.1:" + str(server.server_port)

    @staticmethod
    def info():
        return dict(model_contract_id="a" * 64, execution_profile_id="b" * 64, eos_tokens=[10, 20])

    def get(self, base, path="/cow/info", key="fixture-secret"):
        req = Request(base + path, headers={"Authorization": "Bearer " + key})
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_reports_only_public_profile_fields_without_using_adapter(self):
        base = self.start(self.info(), text_adapter=object())
        status, info = self.get(base)
        self.assertEqual(status, 200)
        self.assertEqual(info, dict(schema=1, service="context-cow", **self.info(),
                                   capabilities=dict(text=True, inline_grammars=False)))

    def test_wrong_key_and_foreign_query_cannot_discover(self):
        base = self.start(self.info())
        self.assertEqual(self.get(base, key="wrong")[0], 401)
        self.assertEqual(self.get(base, "/cow/info?profile=other")[0], 400)

    def test_missing_profile_is_explicitly_unavailable(self):
        self.assertEqual(self.get(self.start())[0], 503)

    def test_input_is_detached_at_startup(self):
        info = self.info()
        base = self.start(info)
        info["eos_tokens"].append(30)
        info["execution_profile_id"] = "c" * 64
        self.assertEqual(self.get(base)[1]["eos_tokens"], [10, 20])
        self.assertEqual(self.get(base)[1]["execution_profile_id"], "b" * 64)

    def test_invalid_profile_rejected_before_listening(self):
        for change in [dict(model_path="/private/weights"), dict(model_contract_id="x"),
                       dict(eos_tokens=[True]), dict(eos_tokens=[]), dict(eos_tokens=[10, 10])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.start(dict(self.info(), **change))


if __name__ == "__main__":
    unittest.main()
