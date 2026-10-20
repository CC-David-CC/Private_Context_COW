"""CLI notice controls for canonical-history prefix reconstruction."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import tempfile
import threading
import unittest

from tools import cow_cli


SOURCE = "a" * 64
RECOMPUTATION = {
    "source_kind": "recomputed-history",
    "durable": False,
    "prior_computation_preserved": False,
    "prefix_history_tokens": 3,
    "prefix_forward_tokens": 3,
    "prefix_consumed_tokens": 3,
    "schedule_sha256": "b" * 64,
    "controller_sha256": "c" * 64,
    "seconds": 0.01,
}


class RecomputationHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "complete"
    posts = []

    def log_message(self, *_args):
        pass

    def _json(self, value):
        raw = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/cow/head?branch=b-recompute":
            return self._json({"name": "b-recompute", "revision": 4,
                               "tip": SOURCE, "root": SOURCE})
        return self._json({"error": "unexpected GET"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        type(self).posts.append((self.path, body))
        if self.path != "/cow/text/ask":
            return self._json({"error": "unexpected POST"})
        mode = type(self).mode
        events = []
        if mode in ("complete", "failure", "duplicate", "bad-snapshot", "bad-count",
                    "bad-completion", "recompute-failed-complete"):
            snapshot = "bad\x1b[31m" if mode == "bad-snapshot" else SOURCE
            count = True if mode == "bad-count" else 3
            events.append({"event": "source_recomputation_started",
                           "source_snapshot": snapshot, "history_tokens": count})
            if mode == "duplicate":
                events.append({"event": "source_recomputation_started",
                               "source_snapshot": SOURCE, "history_tokens": 3})
        elif mode == "ordinary":
            events.append({"event": "request_started", "gpu": 0, "source": SOURCE})
        if mode in ("complete", "duplicate", "bad-snapshot", "bad-count", "bad-completion",
                    "recompute-failed-complete"):
            forward = 2 if mode == "bad-completion" else 3
            if mode == "recompute-failed-complete":
                events.append({"event": "source_recomputation_failed", "source_snapshot": SOURCE,
                               "error": "failed\x1b[31m; must not echo"})
            events.extend([
                {"event": "source_recomputed", "source_snapshot": SOURCE,
                 "prefix_history_tokens": 3, "prefix_forward_tokens": forward,
                 "prefix_consumed_tokens": 3, "schedule_sha256": "b" * 64},
                {"event": "text", "text": "answer text"},
                {"event": "complete", "answer": {"text": "answer text", "tokens": [9],
                    "snapshot": "d" * 64, "source_recomputation": RECOMPUTATION},
                 "receipt": {"branch": {"name": "b-recompute", "revision": 5,
                    "tip": "d" * 64, "root": SOURCE}, "snapshot": "d" * 64}},
            ])
        elif mode == "failure":
            events.append({"event": "failure", "error": "rebuild failed"})
        elif mode == "ordinary":
            events.extend([
                {"event": "text", "text": "answer text"},
                {"event": "complete", "answer": {"text": "answer text", "tokens": [9],
                    "snapshot": "d" * 64},
                 "receipt": {"branch": {"name": "b-recompute", "revision": 5,
                    "tip": "d" * 64, "root": SOURCE}, "snapshot": "d" * 64}},
            ])
        raw_events = b"".join(
            f"{len(raw):X}\r\n".encode() + raw + b"\r\n"
            for raw in (json.dumps(event).encode("utf-8") + b"\n" for event in events))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(raw_events + b"0\r\n\r\n")
        self.wfile.flush()


class RecomputationNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RecomputationHandler.posts = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), RecomputationHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        RecomputationHandler.mode = "complete"
        RecomputationHandler.posts = []

    def invoke_ask(self, state_dir):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = cow_cli.main(["--url", self.url, "--state-dir", str(state_dir),
                "ask", "--branch", "b-recompute", "--text", "question",
                "--max-tokens", "1", "--operation-id", "op-recompute"])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_ask_notifies_once_and_keeps_answer_text_on_stdout(self):
        RecomputationHandler.mode = "duplicate"
        with tempfile.TemporaryDirectory() as directory:
            result, stdout, stderr = self.invoke_ask(directory)
        self.assertEqual(result, 0, stderr)
        self.assertEqual(stdout, "answer text\n")
        self.assertEqual(stderr.count("saved prefix is unavailable"), 1)
        self.assertIn("recomputing it from conversation history", stderr)
        self.assertIn("This operation is still running", stderr)
        self.assertEqual(stderr.count("durable receipt:"), 1)
        self.assertEqual(len(RecomputationHandler.posts), 1)

    def test_notice_does_not_claim_success_when_recomputation_fails(self):
        RecomputationHandler.mode = "failure"
        with tempfile.TemporaryDirectory() as directory:
            result, stdout, stderr = self.invoke_ask(directory)
        self.assertEqual(result, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr.count("saved prefix is unavailable"), 1)
        self.assertIn("rebuild failed", stderr)
        self.assertNotIn("durable receipt:", stderr)

    def test_recomputation_failure_event_stops_later_complete_without_echoing_error(self):
        RecomputationHandler.mode = "recompute-failed-complete"
        with tempfile.TemporaryDirectory() as directory:
            result, stdout, stderr = self.invoke_ask(directory)
        self.assertEqual(result, 2)
        self.assertEqual(stdout, "")
        self.assertIn("source recomputation failed", stderr)
        self.assertNotIn("must not echo", stderr)
        self.assertNotIn("\x1b", stderr)
        self.assertNotIn("durable receipt:", stderr)

    def test_invalid_recomputation_fields_are_rejected_without_echoing_controls(self):
        for mode in ("bad-snapshot", "bad-count"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                RecomputationHandler.mode = mode
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = cow_cli.main(["--url", self.url, "--state-dir", directory,
                        "ask", "--branch", "b-recompute", "--text", "question",
                        "--max-tokens", "1", "--operation-id", "op-recompute"])
                self.assertEqual(result, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertNotIn("saved prefix is unavailable", stderr.getvalue())
                self.assertNotIn("\x1b", stderr.getvalue())
                self.assertIn("invalid source recomputation progress", stderr.getvalue())
                self.assertNotIn("durable receipt:", stderr.getvalue())

    def test_inconsistent_recomputation_completion_is_rejected_before_text_or_receipt(self):
        RecomputationHandler.mode = "bad-completion"
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = cow_cli.main(["--url", self.url, "--state-dir", directory,
                    "ask", "--branch", "b-recompute", "--text", "question",
                    "--max-tokens", "1", "--operation-id", "op-recompute"])
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("invalid source recomputation completion", stderr.getvalue())
        self.assertNotIn("durable receipt:", stderr.getvalue())

    def test_ordinary_restore_progress_does_not_show_recomputation_notice(self):
        RecomputationHandler.mode = "ordinary"
        with tempfile.TemporaryDirectory() as directory:
            result, stdout, stderr = self.invoke_ask(directory)
        self.assertEqual(result, 0, stderr)
        self.assertEqual(stdout, "answer text\n")
        self.assertNotIn("saved prefix is unavailable", stderr)

    def test_retry_request_uses_the_same_recomputation_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            body = {"branch": "b-recompute", "operation_id": "op-recompute",
                    "stream": True, "content": "question", "revision": 4, "max_tokens": 1}
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            journal = cow_cli.RequestJournal(directory)
            journal.begin("op-recompute", self.url, "/cow/text/ask", raw)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = cow_cli.main(["--url", self.url, "--state-dir", directory,
                    "retry-request", "--operation-id", "op-recompute"])
        self.assertEqual(result, 0, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "answer text\n")
        self.assertEqual(stderr.getvalue().count("saved prefix is unavailable"), 1)
        self.assertEqual(len(RecomputationHandler.posts), 1)


if __name__ == "__main__":
    unittest.main()
