"""CLI reseed revision binding and real HTTP compare-and-swap controls."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import serve.test_cow_presentation_flow as flow
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from experimental.branching_context.tests.implementation.test_seed_presentation_flow import (
    _save_prepared_branch_with_checkpoint, _set_random_stream,
)
from context_time_travel.store import NotFound
from unittest.mock import patch


class CowCliReseedRevisionTests(unittest.TestCase):
    def setUp(self):
        self.stream_patch = patch.object(flow.FlowModel, "set_random_stream",
                                        _set_random_stream, create=True)
        self.save_patch = patch.object(flow.FlowModel, "save_prepared_branch",
                                       _save_prepared_branch_with_checkpoint)
        self.stream_patch.start()
        self.save_patch.start()
        self.addCleanup(self.save_patch.stop)
        self.addCleanup(self.stream_patch.stop)
        self.fixture = flow.CowPresentationFlowTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    @staticmethod
    def _stop(server, thread):
        server.shutdown()
        thread.join(5)
        server.server_close()

    def test_cli_reseed_revision_is_sent_unchanged_and_server_rejects_race(self):
        f = self.fixture
        f.models[0].plans["cow-seed-v1:0"] = (257,)
        adapter = CowAdapter(f.coordinator)
        api = make_server(adapter, host="127.0.0.1", port=0,
                          api_key="reseed-revision-key", text_adapter=f.text)
        api_thread = threading.Thread(target=api.serve_forever, daemon=True)
        api_thread.start()
        self.addCleanup(self._stop, api, api_thread)

        f.text.prepare("cli-chat", [{"role": "user", "content": "hello"}],
                       "cli-revision-root", namespace_id=f.namespace1).durable.result(5)
        prepared_head = f.coordinator.store.head("cli-chat")
        f.text.answer("cli-chat", prepared_head.revision, "cli-revision-answer", 1,
                      seed=0).durable.result(5)
        initial = f.coordinator.store.head("cli-chat")
        self.assertEqual(initial.revision, prepared_head.revision + 1)

        cli = Path(__file__).resolve().parents[4] / "tools" / "cow_cli.py"
        with tempfile.TemporaryDirectory(prefix="cow-cli-reseed-revision-") as state_dir:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["COW_API_KEY"] = "reseed-revision-key"
            base = f"http://127.0.0.1:{api.server_port}"

            def run(url, revision, operation):
                return subprocess.run(
                    [sys.executable, str(cli), "--url", url, "--state-dir", state_dir,
                     "reseed", "--branch", "cli-chat", "--seed", "9",
                     "--revision", str(revision), "--operation-id", operation],
                    capture_output=True, text=True, encoding="utf-8", timeout=15, env=env)

            applied = run(base, initial.revision, "cli-reseed-revision-ok")
            self.assertEqual(applied.returncode, 0,
                             f"CLI rejected valid explicit revision: {applied.stderr}")
            accepted = f.coordinator.store.head("cli-chat")
            self.assertEqual(accepted.revision, initial.revision + 1)
            accepted_receipt = f.coordinator.store.presentation_receipt(
                "cli-reseed-revision-ok")
            self.assertIsNotNone(accepted_receipt)

            stale_before_read = run(base, initial.revision, "cli-reseed-stale-before-read")
            self.assertNotEqual(stale_before_read.returncode, 0)
            self.assertIn("revision changed", stale_before_read.stderr)
            self.assertEqual(f.coordinator.store.head("cli-chat"), accepted)
            with self.assertRaises(NotFound):
                f.coordinator.store.presentation_receipt("cli-reseed-stale-before-read")

            raced = threading.Event()
            observed_body = {}
            race_receipt = {}
            upstream = base

            class RevisionRaceProxy(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, *_args):
                    return

                def _forward(self, method):
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length else None
                    request = Request(upstream + self.path, data=raw,
                                      headers={key: value for key, value in self.headers.items()
                                               if key.lower() not in ("host", "connection", "content-length")},
                                      method=method)
                    try:
                        with urlopen(request, timeout=5) as response:
                            status, body = response.status, response.read()
                            content_type = response.headers.get("Content-Type", "application/json")
                    except HTTPError as error:
                        status, body = error.code, error.read()
                        content_type = error.headers.get("Content-Type", "application/json")
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    self._forward("GET")
                    if urlsplit(self.path).path == "/cow/head":
                        # Do the competing real mutation after forwarding the
                        # observed head. The POST is held until this durable CAS
                        # completes, so it must reach the API with the old revision.
                        result = f.text.reseed("cli-chat", 11, accepted.revision,
                                               "revision-race-winner").durable.result(5)
                        race_receipt["branch"] = result
                        race_receipt["head"] = f.coordinator.store.head("cli-chat")
                        raced.set()

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    if urlsplit(self.path).path == "/cow/reseed":
                        observed_body.update(json.loads(raw.decode("utf-8")))
                        if not raced.wait(5):
                            self.send_error(504, "race mutation did not finish")
                            return
                    # Forward the exact CLI body to the actual COW HTTP API;
                    # its revision CAS must reject the now-stale request.
                    request = Request(upstream + self.path, data=raw,
                                      headers={key: value for key, value in self.headers.items()
                                               if key.lower() not in ("host", "connection", "content-length")},
                                      method="POST")
                    try:
                        with urlopen(request, timeout=5) as response:
                            status, body = response.status, response.read()
                            content_type = response.headers.get("Content-Type", "application/json")
                    except HTTPError as error:
                        status, body = error.code, error.read()
                        content_type = error.headers.get("Content-Type", "application/json")
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            proxy = ThreadingHTTPServer(("127.0.0.1", 0), RevisionRaceProxy)
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                race_revision = accepted.revision
                race = run(f"http://127.0.0.1:{proxy.server_port}", race_revision,
                           "cli-reseed-stale-race")
                self.assertNotEqual(race.returncode, 0)
                self.assertIn("HTTP 409", race.stderr)
                self.assertEqual(observed_body.get("revision"), race_revision,
                                 "CLI must send the caller's exact observed revision")
                final_head = f.coordinator.store.head("cli-chat")
                self.assertEqual(final_head.revision, race_revision + 1)
                self.assertEqual(final_head, race_receipt["head"])
                with self.assertRaises(NotFound):
                    f.coordinator.store.presentation_receipt("cli-reseed-stale-race")
            finally:
                proxy.shutdown()
                proxy_thread.join(5)
                proxy.server_close()

if __name__ == "__main__":
    unittest.main()
