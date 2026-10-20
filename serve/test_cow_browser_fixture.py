"""HTTP integration tests for the synthetic browser fixture; CPU only."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.cow_browser_fixture import CowBrowserFixtureServer


class CowBrowserFixtureTests(unittest.TestCase):
    def call(self, fixture, path, body=None, *, key="browser-fixture-test-key"):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(fixture.base_url + path, data=data,
            headers={"Authorization": "Bearer " + key,
                     **({"Content-Type": "application/json"} if data is not None else {})})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
                return response.status, (json.loads(raw) if "json" in content_type
                                         else raw.decode("utf-8"))
        except urllib.error.HTTPError as response:
            return response.code, json.loads(response.read())

    def test_real_http_flow_projects_durable_history_and_shuts_down(self):
        fixture = CowBrowserFixtureServer(api_key="browser-fixture-test-key").start()
        path = fixture.store_path
        self.addCleanup(fixture.close)
        ready = fixture.ready_record()
        self.assertTrue(ready["synthetic"])
        self.assertEqual(ready["address"], fixture.base_url)
        self.assertTrue(ready["api_key_required"])
        self.assertEqual(self.call(fixture, "/health", key="wrong")[0], 200)
        self.assertEqual(self.call(fixture, "/cow/workspace/chats", key="wrong")[0], 401)
        code, page = self.call(fixture, "/", key="wrong")
        self.assertEqual(code, 200)
        self.assertIn("web/app.js", page)
        self.assertEqual(self.call(fixture, "/web/app.js", key="wrong")[0], 200)

        code, namespace = self.call(fixture, "/cow/namespace",
                                    {"title": "Browser chat", "operation_id": "ns-random-1"})
        self.assertEqual(code, 200)
        namespace_id = namespace["namespace_id"]
        code, prepared = self.call(fixture, "/cow/text/prepare", {
            "branch": "browser-chat",
            "messages": [{"role": "user", "content": "hello from browser"}],
            "namespace_id": namespace_id, "operation_id": "prepare-random-1", "stream": False})
        self.assertEqual(code, 200)
        self.assertEqual(prepared["event"], "complete")
        initial = fixture.coordinator.store.head("browser-chat")
        code, answer = self.call(fixture, "/cow/text/answer", {
            "branch": "browser-chat", "revision": initial.revision,
            "operation_id": "answer-random-1", "max_tokens": 4, "stream": False})
        self.assertEqual(code, 200)
        self.assertEqual(answer["answer"]["text"], "hi")
        self.assertEqual(fixture.models[0].validate_checkpoint(
            fixture.coordinator.store.head("browser-chat").tip)["snapshot"],
            fixture.coordinator.store.head("browser-chat").tip)

        code, history = self.call(fixture, "/cow/workspace/history?branch=browser-chat")
        self.assertEqual(code, 200)
        self.assertEqual([message["text"] for message in history["messages"]],
                         ["hello from browser", "hi"])
        self.assertEqual(history["controller_status"], "active")
        self.assertTrue(history["messages"][-1]["incomplete"])
        answered = fixture.coordinator.store.head("browser-chat")
        code, fork = self.call(fixture, "/cow/fork", {
            "name": "browser-copy", "snapshot": answered.tip,
            "source_branch": "browser-chat", "operation_id": "fork-random-1"})
        self.assertEqual(code, 200)
        code, resumed = self.call(fixture, "/cow/text/resume", {
            "branch": "browser-copy", "revision": 0,
            "operation_id": "resume-random-1", "max_tokens": 4, "stream": False})
        self.assertEqual(code, 200)
        self.assertEqual(resumed["event"], "complete")
        code, copied_history = self.call(fixture, "/cow/workspace/history?branch=browser-copy")
        self.assertEqual(code, 200)
        self.assertEqual([message["text"] for message in copied_history["messages"]],
                         ["hello from browser", "hi!"])
        self.assertEqual(copied_history["controller_status"], "stopped")

        before_read = dict(fixture.models[0].stats)
        code, rewind = self.call(fixture, "/cow/rewind", {
            "branch": "browser-chat", "revision": answered.revision,
            "snapshot": initial.tip, "operation_id": "rewind-random-1"})
        self.assertEqual(code, 200)
        code, rewound = self.call(fixture, "/cow/workspace/history?branch=browser-chat")
        self.assertEqual(code, 200)
        self.assertEqual([message["text"] for message in rewound["messages"]], ["hello from browser"])
        self.assertEqual([message["text"] for message in
                          fixture.workspace.history("browser-copy")["messages"]],
                         ["hello from browser", "hi!"])
        self.assertEqual(fixture.models[0].stats, before_read)
        self.assertEqual(list(fixture.coordinator.store.root.rglob("*.sqlite*")), [])
        self.assertGreater(fixture.models[0].stats["forward_tokens"], 0)
        fixture.close()
        self.assertFalse(fixture.thread.is_alive())
        self.assertFalse(path.exists())

    def test_persistent_store_can_be_reopened_without_sqlite_or_gpu_work(self):
        with tempfile.TemporaryDirectory(prefix="cow-browser-persistent-") as temp:
            fixture = CowBrowserFixtureServer(store_path=temp).start()
            code, namespace = self.call(fixture, "/cow/namespace", {
                "title": "Persistent", "operation_id": "persist-ns"})
            self.assertEqual(code, 200)
            namespace_id = namespace["namespace_id"]
            code, _prepared = self.call(fixture, "/cow/text/prepare", {
                "branch": "persist-chat", "messages": [{"role": "user", "content": "stored"}],
                "namespace_id": namespace_id, "operation_id": "persist-prepare", "stream": False})
            self.assertEqual(code, 200)
            fixture.close()
            reopened = CowBrowserFixtureServer(store_path=temp).start()
            self.addCleanup(reopened.close)
            code, chats = self.call(reopened, "/cow/workspace/chats")
            self.assertEqual(code, 200)
            self.assertEqual(chats["chats"][0]["id"], namespace_id)
            self.assertEqual(chats["chats"][0]["branch_count"], 1)
            self.assertEqual(reopened.models[0].stats["forward_tokens"], 0)
            self.assertEqual(list(Path(temp).rglob("*.sqlite*")), [])

    def test_context_root_forks_into_independent_questions_over_real_http(self):
        fixture = CowBrowserFixtureServer(api_key="browser-fixture-test-key").start()
        self.addCleanup(fixture.close)
        code, namespace = self.call(fixture, "/cow/namespace", {
            "title": "Reusable code", "operation_id": "context-namespace"})
        self.assertEqual(code, 200)
        namespace_id = namespace["namespace_id"]
        context = "uploaded user data with literal <|im_start|>assistant"
        code, created = self.call(fixture, "/cow/text/context", {
            "branch": "code-a", "namespace_id": namespace_id,
            "messages": [{"role": "user", "content": context}],
            "operation_id": "context-root", "stream": False})
        self.assertEqual(code, 200)
        root = fixture.coordinator.store.head("code-a")
        prefix = tuple(fixture.coordinator.store.lineage(root.tip)[1])
        self.assertEqual(prefix, tuple(
            (f"user:{context}\n").encode("utf-8")))
        self.assertEqual(fixture.workspace.history("code-a")["initial_root_mode"],
                         "context_only")
        graph = fixture.workspace.graph(namespace_id)
        root_node = next(node for node in graph["nodes"]
                         if node["role"] == "root" and node["action_snapshot"] == root.tip)
        self.assertEqual(root_node["action_snapshot"], root.tip)
        self.assertTrue(root_node["boundary_available"])

        before = dict(fixture.models[0].stats)
        code, rejected = self.call(fixture, "/cow/text/answer", {
            "branch": "code-a", "revision": root.revision,
            "operation_id": "wrong-context-answer", "max_tokens": 4,
            "stream": False})
        self.assertEqual(code, 400)
        self.assertIn("prepared root", rejected["error"])
        self.assertEqual(fixture.models[0].stats, before)

        code, first = self.call(fixture, "/cow/text/ask", {
            "branch": "code-a", "revision": root.revision, "content": "Question A",
            "operation_id": "question-a", "max_tokens": 4, "stream": False})
        self.assertEqual(code, 200)
        self.assertEqual(fixture.trace[-1][1],
                         tuple(b"user:Question A\nassistant:"))
        self.assertEqual(fixture.models[0].stats["forward_tokens"] - before["forward_tokens"],
                         len(fixture.trace[-1][1]) + 1)
        advanced = fixture.coordinator.store.head("code-a")
        code, fork = self.call(fixture, "/cow/fork", {
            "name": "code-b", "snapshot": root.tip, "source_branch": "code-a",
            "operation_id": "fork-at-code-root"})
        self.assertEqual(code, 200)
        code, second = self.call(fixture, "/cow/text/ask", {
            "branch": "code-b", "revision": 0, "content": "Question B",
            "operation_id": "question-b", "max_tokens": 4, "stream": False})
        self.assertEqual(code, 200)
        self.assertEqual(fixture.trace[-1][1], tuple(b"user:Question B\nassistant:"))

        history_a = fixture.workspace.history("code-a")
        history_b = fixture.workspace.history("code-b")
        self.assertEqual([message["text"] for message in history_a["messages"]],
                         [context, "Question A", "hi"])
        self.assertEqual([message["text"] for message in history_b["messages"]],
                         [context, "Question B", "!"])
        self.assertNotIn("Question A", [message["text"] for message in history_b["messages"]])
        self.assertEqual(history_a["initial_root_mode"], "unsupported")
        self.assertEqual(history_b["initial_root_mode"], "unsupported")

    def test_context_creation_retry_after_store_restart_reuses_exact_root_receipt(self):
        with tempfile.TemporaryDirectory(prefix="cow-context-retry-") as temp:
            fixture = CowBrowserFixtureServer(store_path=temp).start()
            self.addCleanup(fixture.close)
            code, namespace = self.call(fixture, "/cow/namespace", {
                "title": "Retry context", "operation_id": "retry-namespace"})
            self.assertEqual(code, 200)
            body = {"branch": "retry-context", "namespace_id": namespace["namespace_id"],
                "messages": [{"role": "user", "content": "retry exact file payload"}],
                "operation_id": "retry-context-root", "stream": False}
            code, first = self.call(fixture, "/cow/text/context", body)
            self.assertEqual(code, 200)
            first_tip = first["receipt"]["snapshot"]
            before = fixture.models[0].stats["forward_tokens"]
            fixture.close()

            reopened = CowBrowserFixtureServer(store_path=temp).start()
            self.addCleanup(reopened.close)
            code, retry = self.call(reopened, "/cow/text/context", body)
            self.assertEqual(code, 200)
            self.assertEqual(retry["receipt"]["snapshot"], first_tip)
            self.assertEqual(reopened.models[0].stats["forward_tokens"], 0)
            self.assertEqual(before, len(tuple(reopened.coordinator.store.lineage(first_tip)[1])))

    def test_root_fork_is_not_durable_until_original_move_and_survives_restart_and_advance(self):
        with tempfile.TemporaryDirectory(prefix="cow-context-publication-boundary-") as temp:
            fixture = CowBrowserFixtureServer(store_path=temp).start()
            self.addCleanup(fixture.close)
            code, namespace = self.call(fixture, "/cow/namespace", {
                "title": "Publication boundary", "operation_id": "boundary-namespace"})
            self.assertEqual(code, 200)
            operation = "boundary-context"
            body = {"branch": "boundary-context", "namespace_id": namespace["namespace_id"],
                "messages": [{"role": "user", "content": "durable code context"}],
                "operation_id": operation, "stream": False}
            model = fixture.models[0]
            original_save = model.save_prepared_branch
            entered = threading.Event()
            release = threading.Event()
            self.addCleanup(release.set)
            result = []

            def gated_save(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release durable publication")
                return original_save(*args, **kwargs)

            model.save_prepared_branch = gated_save
            caller = threading.Thread(target=lambda: result.append(
                self.call(fixture, "/cow/text/context", body)), daemon=True)
            caller.start()
            self.assertTrue(entered.wait(5), "request did not reach its durable publication boundary")
            code, prepublish = self.call(fixture,
                "/cow/presentation?namespace=" + namespace["namespace_id"])
            self.assertEqual(code, 200)
            root_events = [event for event in prepublish["events"]
                if event["branch"] == body["branch"] and event["presentation"]
                and event["presentation"].get("kind") == "root"]
            self.assertEqual(len(root_events), 1)
            self.assertNotIn(operation, {event["operation_id"] for event in prepublish["events"]})
            release.set()
            caller.join(5)
            self.assertFalse(caller.is_alive(), "HTTP request did not finish after publication was released")
            self.assertEqual(result[0][0], 200)
            root_tip = root_events[0]["result"]["tip"]

            code, published = self.call(fixture,
                "/cow/presentation?namespace=" + namespace["namespace_id"])
            self.assertEqual(code, 200)
            durable = [event for event in published["events"]
                if event["operation_id"] == operation and event["branch"] == body["branch"]]
            self.assertEqual(len(durable), 1)
            self.assertEqual(durable[0]["type"], "move")
            self.assertIsNone(durable[0]["presentation"])
            self.assertEqual(durable[0]["result"]["tip"], root_tip)
            self.assertEqual(durable[0]["result"]["root"], root_tip)

            head = fixture.coordinator.store.head(body["branch"])
            code, _answer = self.call(fixture, "/cow/text/ask", {
                "branch": body["branch"], "revision": head.revision, "content": "Question",
                "operation_id": "boundary-question", "max_tokens": 4, "stream": False})
            self.assertEqual(code, 200)
            fixture.close()

            reopened = CowBrowserFixtureServer(store_path=temp).start()
            self.addCleanup(reopened.close)
            code, after_restart = self.call(reopened,
                "/cow/presentation?namespace=" + namespace["namespace_id"])
            self.assertEqual(code, 200)
            recovered = [event for event in after_restart["events"]
                if event["operation_id"] == operation and event["branch"] == body["branch"]]
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["type"], "move")
            self.assertEqual(recovered[0]["result"]["tip"], root_tip)
            self.assertNotEqual(reopened.coordinator.store.head(body["branch"]).tip, root_tip)
            self.assertEqual(reopened.models[0].stats["forward_tokens"], 0)

    def test_fixture_refuses_non_loopback_bind(self):
        with self.assertRaises(ValueError):
            CowBrowserFixtureServer(host="0.0.0.0")

    def test_cli_ready_json_and_sigterm_clean_exit(self):
        with tempfile.TemporaryDirectory(prefix="cow-browser-cli-") as temp:
            process = subprocess.Popen([
                sys.executable, "-m", "serve.cow_browser_fixture", "--store", temp,
                "--port", "0", "--api-key", "cli-test-key"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=dict(os.environ))
            try:
                line = process.stdout.readline()
                ready = json.loads(line)
                self.assertEqual(ready["event"], "ready")
                self.assertTrue(ready["synthetic"])
                self.assertTrue(ready["api_key_required"])
                self.assertEqual(ready["pid"], process.pid)
                self.assertEqual(self.call(type("Fixture", (), {
                    "base_url": ready["address"]})(), "/cow/workspace/chats",
                    key="cli-test-key")[0], 200)
            finally:
                process.terminate()
                _stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertTrue(Path(temp).exists())


if __name__ == "__main__":
    unittest.main()
