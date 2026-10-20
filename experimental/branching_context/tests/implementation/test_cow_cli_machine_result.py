from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import cow_cli


SNAPSHOT = "a" * 64
ROOT = "b" * 64


def _complete(operation_id: str, text: str) -> dict:
    receipt = {
        "operation_id": operation_id,
        "branch": {"name": "main", "revision": 10, "tip": SNAPSHOT, "root": ROOT},
        "snapshot": SNAPSHOT,
    }
    return {
        "event": "complete",
        "answer": {"text": text, "tokens": [1, 2], "snapshot": SNAPSHOT},
        "receipt": receipt,
    }


class _FakeClient:
    def __init__(self, _url, _api_key, _timeout, completion=None):
        self.base_url = "http://127.0.0.1:8099"
        self.completion = completion
        self.stream_calls = 0

    def request(self, method, path, _body=None):
        if method == "GET" and path.startswith("/cow/head?"):
            return {"name": "main", "revision": 9, "tip": "c" * 64, "root": ROOT}
        raise AssertionError((method, path))

    def stream(self, _path, _raw, on_provisional):
        self.stream_calls += 1
        answer_text = self.completion["answer"]["text"]
        if answer_text:
            on_provisional(answer_text)
        return self.completion


class CowCliMachineResultTests(unittest.TestCase):
    def _run_answer(self, directory: str, *, machine: bool, text: str):
        completion = _complete("op-machine", text)
        fake = _FakeClient(None, None, None, completion)
        argv = ["--state-dir", directory]
        if machine:
            argv.append("--machine-result")
        argv.extend(["answer", "--branch", "main", "--max-tokens", "4",
                     "--operation-id", "op-machine"])
        args = cow_cli.parser().parse_args(argv)
        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch.object(cow_cli, "CowClient", return_value=fake),
              patch.object(cow_cli, "_head", return_value={"revision": 9}),
              contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
            cow_cli.run(args)
        return fake, stdout.getvalue(), stderr.getvalue(), completion

    def test_machine_journal_answer_is_optional_for_non_text_stream_receipts(self):
        no_text = {"tokens": [1], "snapshot": SNAPSHOT}
        self.assertIs(cow_cli._canonical_journal_answer(
            {"receipt": {"snapshot": SNAPSHOT}, "answer": no_text}, streamed=True), no_text)
        exact = {"text": "answer", "tokens": [1], "snapshot": SNAPSHOT}
        self.assertIs(cow_cli._canonical_journal_answer(
            {"receipt": {"snapshot": SNAPSHOT}, "answer": exact}, streamed=True), exact)
        self.assertIsNone(cow_cli._canonical_journal_answer(
            {"answer": exact}, streamed=False))

    def test_machine_result_accepts_no_text_answer_and_rejects_non_string_text(self):
        answer = {"tokens": [1, 2], "snapshot": SNAPSHOT}
        receipt = {"snapshot": SNAPSHOT}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cow_cli._print_machine_result(answer, receipt)
        self.assertEqual(json.loads(output.getvalue()), {
            "schema": 1, "answer": answer, "receipt": receipt})
        with self.assertRaisesRegex(cow_cli.ClientError, "answer"):
            with contextlib.redirect_stdout(io.StringIO()):
                cow_cli._print_machine_result({"tokens": [1], "text": 7}, receipt)

    def test_journal_round_trips_exact_no_text_answer_and_rejects_bad_text_type(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = cow_cli.RequestJournal(directory)
            body = json.dumps({"operation_id": "op-no-text"}).encode()
            record = journal.begin("op-no-text", "http://127.0.0.1", "/cow/text/prepare", body)
            answer = {"tokens": [7, 8], "snapshot": SNAPSHOT}
            completed = journal.complete(record, {"snapshot": SNAPSHOT}, answer=answer)
            self.assertEqual(completed["answer"], answer)
            self.assertEqual(journal.get("op-no-text")["answer"], answer)

            bad = dict(completed, answer={"text": 3, "tokens": [7]})
            bad["record_sha256"] = journal._record_digest(bad)
            journal._atomic_write(journal._path("op-no-text"), bad)
            with self.assertRaisesRegex(cow_cli.ClientError, "canonical answer"):
                journal.get("op-no-text")

    def test_machine_result_preserves_trailing_lf_empty_and_unicode_exactly(self):
        for text in ("patch\n", "", "snowman \u2603\nsecond line\n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                _, stdout, _, completion = self._run_answer(directory, machine=True, text=text)
                envelope = json.loads(stdout)
                self.assertEqual(set(envelope), {"schema", "answer", "receipt"})
                self.assertEqual(envelope["schema"], 1)
                self.assertEqual(envelope["answer"], completion["answer"])
                self.assertEqual(envelope["answer"]["text"], text)
                self.assertEqual(envelope["receipt"], completion["receipt"])
                journal = cow_cli.RequestJournal(directory).get("op-machine")
                self.assertEqual(journal["answer"], completion["answer"])

    def test_machine_retry_request_replays_saved_exact_answer_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, completion = self._run_answer(directory, machine=True, text="exact\n")
            fake = _FakeClient(None, None, None, None)
            args = cow_cli.parser().parse_args([
                "--state-dir", directory, "--machine-result", "retry-request",
                "--operation-id", "op-machine",
            ])
            stdout, stderr = io.StringIO(), io.StringIO()
            with (patch.object(cow_cli, "CowClient", return_value=fake),
                  contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                cow_cli.run(args)
            envelope = json.loads(stdout.getvalue())
            self.assertEqual(envelope["answer"], completion["answer"])
            self.assertEqual(envelope["receipt"], completion["receipt"])
            self.assertEqual(fake.stream_calls, 0)

    def test_machine_mode_does_not_change_human_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            _, stdout, _, _ = self._run_answer(directory, machine=False, text="visible\n")
            self.assertEqual(stdout, "visible\n\n")

    def test_old_completed_journal_without_answer_fails_machine_replay_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            completion = _complete("op-machine", "prior")
            fake = _FakeClient(None, None, None, completion)
            _, _, _, _ = self._run_answer(directory, machine=True, text="prior")
            journal = cow_cli.RequestJournal(directory)
            record = journal.get("op-machine")
            self.assertIsNotNone(record)
            # Simulate a legitimate pre-machine-result journal schema-1 record.
            record.pop("answer", None)
            record["record_sha256"] = journal._record_digest(record)
            journal._atomic_write(journal._path("op-machine"), record)
            args = cow_cli.parser().parse_args([
                "--state-dir", directory, "--machine-result", "retry-request",
                "--operation-id", "op-machine",
            ])
            stdout = io.StringIO()
            with (patch.object(cow_cli, "CowClient", return_value=fake),
                  contextlib.redirect_stdout(stdout),
                  self.assertRaisesRegex(cow_cli.ClientError, "canonical answer")):
                cow_cli.run(args)
            self.assertEqual(stdout.getvalue(), "")

    def test_actual_cli_process_replays_exact_answer_from_completed_local_journal(self):
        from serve.cow_adapter import CowAdapter
        from serve.cow_http import make_server
        from serve.test_cow_presentation_flow import CowPresentationFlowTests

        fixture = CowPresentationFlowTests("runTest")
        fixture.setUp()
        try:
            model = fixture.models[0]
            old_plan = model.plans.get("machine-cli-answer")
            model.plans["machine-cli-answer"] = (257,)
            root = fixture.text.prepare(
                "machine-cli", [{"role": "user", "content": "start"}],
                "machine-cli-root", namespace_id=fixture.namespace1).durable.result(5)
            self.assertEqual(root["branch"]["name"], "machine-cli")

            key = "machine-result-cpu-key"
            server = make_server(CowAdapter(fixture.coordinator), host="127.0.0.1", port=0,
                                 api_key=key, text_adapter=fixture.text)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                cli = Path(__file__).resolve().parents[4] / "tools" / "cow_cli.py"
                with tempfile.TemporaryDirectory(prefix="cow-cli-machine-result-") as state_dir:
                    env = os.environ.copy()
                    env.pop("PYTHONPATH", None)
                    env["COW_API_KEY"] = key
                    command = [sys.executable, str(cli), "--machine-result", "--url",
                               f"http://127.0.0.1:{server.server_port}", "--state-dir", state_dir,
                               "answer", "--branch", "machine-cli", "--max-tokens", "1",
                               "--operation-id", "machine-cli-answer"]
                    first = subprocess.run(command, capture_output=True, text=True,
                                           encoding="utf-8", timeout=15, env=env)
                    self.assertEqual(first.returncode, 0, first.stderr)
                    first_envelope = json.loads(first.stdout)
                    self.assertEqual(first_envelope["answer"]["text"], "hi")
                    saved = json.loads((Path(state_dir) / "operations" /
                                        "machine-cli-answer.json").read_text(encoding="utf-8"))
                    self.assertEqual(saved["answer"], first_envelope["answer"])

                    server.shutdown()
                    thread.join(5)
                    replay = subprocess.run(
                        [sys.executable, str(cli), "--machine-result", "--url",
                         f"http://127.0.0.1:{server.server_port}", "--state-dir", state_dir,
                         "retry-request", "--operation-id", "machine-cli-answer"],
                        capture_output=True, text=True, encoding="utf-8", timeout=15, env=env)
                    self.assertEqual(replay.returncode, 0, replay.stderr)
                    self.assertEqual(json.loads(replay.stdout), first_envelope)
            finally:
                server.shutdown()
                thread.join(5)
                server.server_close()
        finally:
            if "model" in locals():
                if old_plan is None:
                    model.plans.pop("machine-cli-answer", None)
                else:
                    model.plans["machine-cli-answer"] = old_plan
            fixture.doCleanups()

    def test_actual_machine_cli_prepare_preserves_no_text_answer_on_journal_replay(self):
        from serve.cow_adapter import CowAdapter
        from serve.cow_http import make_server
        from serve.test_cow_presentation_flow import CowPresentationFlowTests

        fixture = CowPresentationFlowTests("runTest")
        fixture.setUp()
        server = None
        thread = None
        try:
            key = "machine-prepare-no-text-key"
            server = make_server(CowAdapter(fixture.coordinator), host="127.0.0.1", port=0,
                                 api_key=key, text_adapter=fixture.text)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            cli = Path(__file__).resolve().parents[4] / "tools" / "cow_cli.py"
            with tempfile.TemporaryDirectory(prefix="cow-cli-machine-no-text-") as state_dir:
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env["COW_API_KEY"] = key
                base = f"http://127.0.0.1:{server.server_port}"
                context_file = Path(state_dir) / "context.txt"
                context_file.write_text("start here", encoding="utf-8")
                scenarios = (
                    ("prepare", "machine-prepare", "machine-prepare-op", "--text", "start here"),
                    ("load", "machine-context", "machine-context-op", "--file", str(context_file)),
                )
                for command_name, branch, op, arg_name, arg_value in scenarios:
                    with self.subTest(command=command_name):
                        common = [sys.executable, str(cli), "--machine-result", "--url", base,
                                  "--state-dir", state_dir, command_name, "--branch", branch,
                                  "--namespace-id", fixture.namespace1, arg_name, arg_value,
                                  "--operation-id", op]
                        first = subprocess.run(common, capture_output=True, text=True,
                                               encoding="utf-8", timeout=15, env=env)
                        self.assertEqual(first.returncode, 0, first.stderr)
                        envelope = json.loads(first.stdout)
                        answer = envelope["answer"]
                        self.assertIsInstance(answer, dict)
                        self.assertNotIn("text", answer)
                        self.assertIn("snapshot", answer)
                        self.assertIn("tokens", answer)
                        self.assertEqual(envelope["receipt"]["snapshot"], answer["snapshot"])
                        saved = json.loads((Path(state_dir) / "operations" / f"{op}.json")
                                           .read_text(encoding="utf-8"))
                        self.assertEqual(saved["answer"], answer)

                        replay = subprocess.run(
                            [sys.executable, str(cli), "--machine-result", "--url", base,
                             "--state-dir", state_dir, "retry-request", "--operation-id", op],
                            capture_output=True, text=True, encoding="utf-8", timeout=15, env=env)
                        self.assertEqual(replay.returncode, 0, replay.stderr)
                        self.assertEqual(json.loads(replay.stdout), envelope)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
                if thread is not None:
                    thread.join(5)
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
