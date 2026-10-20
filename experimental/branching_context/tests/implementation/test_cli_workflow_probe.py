"""Focused integrity checks for the real CLI workflow evidence harness."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace

from run_cli_workflow_probe import (
    _Campaign,
    WorkflowLedger,
    canonical_sha256,
    validate_adoption_ack,
    validate_evaluation_record,
    validate_zero_work,
    redact_argv,
    validate_retry_result,
    parse_durable_receipt,
)
from serve.cow_http import make_server
from serve.cow_adapter import CowAdapter
from serve.cow_workspace import CowWorkspace
import serve.test_cow_presentation_flow as _presentation_fixture


def _repo_root():
    import serve.cow_http
    return Path(serve.cow_http.__file__).resolve().parents[1]


class CliWorkflowProbeTests(unittest.TestCase):
    def test_campaign_publish_redacts_nested_service_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = _Campaign(SimpleNamespace(
                run_dir=root,
                runtime_root=root / "runtime",
                profile=root / "profile.json",
            ))
            secret = campaign.api_key
            campaign.record = {
                "services": [{
                    "stop": {
                        "command": [sys.executable, "cow-http", "--api-key", secret],
                        "exit_code": 0,
                        "message": "stopped " + secret,
                    },
                }],
                "failures": [{"message": "failure context " + secret}],
            }

            campaign.publish()

            raw = campaign.result_path.read_bytes()
            self.assertNotIn(secret.encode("utf-8"), raw)
            persisted = json.loads(raw)
            stop = persisted["services"][0]["stop"]
            self.assertEqual(stop["command"], [sys.executable, "cow-http", "--api-key", "<REDACTED>"])
            self.assertEqual(stop["exit_code"], 0)
            self.assertEqual(stop["message"], "stopped <REDACTED>")
            self.assertEqual(persisted["failures"][0]["message"], "failure context <REDACTED>")
            self.assertEqual(campaign.record["services"][0]["stop"]["command"][-1], secret)

    def test_evaluation_record_is_content_addressed_and_snapshot_bound(self):
        record = {
            "schema": 1,
            "kind": "evaluation",
            "snapshot_id": "snap-a",
            "task_root": "root-a",
            "evaluator": {"name": "fixture", "version": "1", "source_sha256": "a" * 64},
            "fixtures": {},
            "metric": {"name": "score", "order": "maximize"},
            "seed": 1,
            "environment": {},
            "effect_policy": {"mode": "isolated"},
            "outcomes": {"mandatory": {"tests": "PASS"}, "score": 7, "unresolved": []},
            "resources": {"wall_seconds": 0.1, "cpu_seconds": 0.1, "peak_rss_bytes": 10},
            "artifacts": {},
        }
        envelope = {"evaluation_id": canonical_sha256(record), "record": record}
        self.assertEqual(validate_evaluation_record(envelope, "snap-a", ("tests",)), record)

        altered = json.loads(json.dumps(envelope))
        altered["record"]["outcomes"]["score"] += 1
        with self.assertRaisesRegex(ValueError, "content hash"):
            validate_evaluation_record(altered, "snap-a")
        other_record = json.loads(json.dumps(record))
        other_record["snapshot_id"] = "snap-b"
        wrong_snapshot = {"evaluation_id": canonical_sha256(other_record), "record": other_record}
        with self.assertRaisesRegex(ValueError, "snapshot"):
            validate_evaluation_record(wrong_snapshot, "snap-a")
        with self.assertRaisesRegex(ValueError, "mandatory"):
            validate_evaluation_record(envelope, "snap-a", ("different-check",))

    def test_adoption_ack_must_bind_operation_evaluation_and_cas_head(self):
        ack = {
            "operation_id": "adopt-1",
            "evaluation_id": "eval-1",
            "snapshot_id": "snap-a",
            "branch": {"name": "main", "revision": 9, "tip": "snap-a", "root": "root-a"},
        }
        validate_adoption_ack(ack, operation_id="adopt-1", evaluation_id="eval-1",
                              snapshot_id="snap-a", branch="main", expected_revision=8,
                              root="root-a")
        for key, value in (("operation_id", "other"), ("snapshot_id", "other")):
            changed = json.loads(json.dumps(ack))
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_adoption_ack(changed, operation_id="adopt-1", evaluation_id="eval-1",
                                      snapshot_id="snap-a", branch="main", expected_revision=8,
                                      root="root-a")
        changed = json.loads(json.dumps(ack))
        changed["branch"]["revision"] += 1
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_adoption_ack(changed, operation_id="adopt-1", evaluation_id="eval-1",
                                  snapshot_id="snap-a", branch="main", expected_revision=8,
                                  root="root-a")

    def test_zero_work_retry_requires_every_counter_to_remain_identical(self):
        before = {"forward_tokens": 128, "prefill_tokens": 64, "disk_loads": 1,
                  "native_calls": 20, "loaded_bytes": 900}
        after = dict(before)
        validate_zero_work(before, after)
        after["forward_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "forward_tokens"):
            validate_zero_work(before, after)

    def test_ledger_cannot_mark_incomplete_workflow_pass(self):
        ledger = WorkflowLedger(("setup", "fork", "answer", "evaluate", "adopt", "retry"))
        ledger.complete("setup", {"exit_code": 0})
        ledger.complete("fork", {"exit_code": 0})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ledger.finish()

    def test_ledger_rejects_duplicate_or_failed_stage(self):
        ledger = WorkflowLedger(("setup", "ask"))
        ledger.complete("setup", {"exit_code": 0})
        with self.assertRaisesRegex(ValueError, "already completed"):
            ledger.complete("setup", {"exit_code": 0})
        with self.assertRaisesRegex(ValueError, "exit"):
            ledger.complete("ask", {"exit_code": 3})

    def test_cli_argv_redaction_never_records_authentication_key(self):
        key = "secret-value-never-log"
        command = ["python", "cow_cli.py", "--api-key", key, "ask"]
        redacted = redact_argv(command, (key,))
        self.assertNotIn(key, json.dumps(redacted))
        self.assertEqual(redacted[3], "<REDACTED>")

    def test_retry_replay_and_new_operation_have_distinct_receipt_identity(self):
        tokens = [11, 12, 2]
        def response(replayed, operation):
            return {
                "event": "complete", "replayed": replayed,
                "answer": {"snapshot": "snap-a", "tokens": tokens, "text": "answer"},
                "receipt": {"snapshot": "snap-a", "operation_id": operation,
                            "branch": {"name": "main", "revision": 9,
                                       "tip": "snap-a", "root": "root-a"}},
            }
        validate_retry_result(response(True, "ask-a"), original_operation="ask-a",
                              expected_branch="main", expected_snapshot="snap-a",
                              expected_tokens=tokens, replay=True)
        validate_retry_result(response(False, "retry-b"), original_operation="ask-a",
                              new_operation="retry-b", expected_branch="main",
                              expected_snapshot="snap-a", expected_tokens=tokens, replay=False)
        with self.assertRaisesRegex(ValueError, "operation"):
            validate_retry_result(response(False, "ask-a"), original_operation="ask-a",
                                  new_operation="retry-b", expected_branch="main",
                                  expected_snapshot="snap-a", expected_tokens=tokens, replay=False)


class CliWorkflowSubprocessTests(unittest.TestCase):
    """Real cow_cli subprocesses against the CPU FlowModel HTTP service."""

    def setUp(self):
        fixture = _presentation_fixture.CowPresentationFlowTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        for name in ("coordinator", "text", "trace", "models", "namespace1"):
            setattr(self, name, getattr(fixture, name))
        self.api_key = "cli-flow-test-key"
        workspace = CowWorkspace(self.coordinator.store, self.text.tokenizer,
            root_mode_resolver=self.text.root_mode_for_projection)
        self.server = make_server(CowAdapter(self.coordinator), api_key=self.api_key,
                                  text_adapter=self.text, workspace=workspace)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.temp_cli = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_cli.cleanup)
        self.cli_path = _repo_root() / "tools" / "cow_cli.py"
        self.state_dir = Path(self.temp_cli.name) / "state"
        self.state_dir.mkdir()
        self.env = os.environ.copy()
        self.env["COW_API_KEY"] = self.api_key
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def cli(self, *args, receipt=False, json_output=False):
        command = [sys.executable, "-B", str(self.cli_path), "--url",
                   f"http://127.0.0.1:{self.server.server_port}", "--state-dir",
                   str(self.state_dir), *args]
        result = subprocess.run(command, cwd=_repo_root(), env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False)
        self.assertEqual(result.returncode, 0,
            ("CLI exited nonzero", command, result.stdout.decode(errors="replace"),
             result.stderr.decode(errors="replace")))
        value = parse_durable_receipt(result.stderr) if receipt else None
        if json_output:
            value = json.loads(result.stdout)
        return result, value

    def test_cli_prepare_fork_answer_and_g4_evaluate_select_adopt_history(self):
        result, namespace = self.cli("namespace", "--title", "CLI workflow",
            "--operation-id", "cli-ns", receipt=True)
        ns_id = namespace["namespace_id"]
        self.cli("prepare", "--namespace-id", ns_id, "--branch", "main", "--text",
            "Uploaded source context", "--operation-id", "cli-prepare", receipt=True)
        source = self.coordinator.store.head("main")
        self.assertEqual(source.tip, source.root)

        for name in ("candidate-a", "candidate-b"):
            self.cli("fork", "--source-branch", "main", "--name", name,
                     "--operation-id", "fork-" + name, receipt=True)
        answers = {}
        for name in ("candidate-a", "candidate-b"):
            operation = "alt-answer" if name.endswith("a") else "resume-op"
            result, receipt = self.cli("answer", "--branch", name,
                "--max-tokens", "1", "--operation-id", operation, receipt=True)
            self.assertTrue(result.stdout.decode("utf-8").strip())
            answer = self.coordinator.store.presentation_receipt(operation)["presentation"]
            span = answer["output_span"]
            _, lineage = self.coordinator.store.lineage(receipt["snapshot"])
            tokens = lineage[span["start"]:span["start"] + span["count"]]
            self.assertEqual(tokens, [257] if name.endswith("a") else [258])
            answers[name] = receipt["snapshot"]

        fixture_path = Path(self.temp_cli.name) / "sample.fixture"
        fixture_path.write_bytes(b"PASS")
        spec_path = Path(self.temp_cli.name) / "evaluation.json"
        spec = {
            "task_root": source.root,
            "evaluator": {"name": "fixture-check", "version": "1",
                "source": "result = {'mandatory': {'fixture_ok': 'PASS' if fixtures.get('sample') == b'PASS' else 'FAIL'}, 'score': int(candidate['tokens'][-1] == 258), 'unresolved': []}"},
            "metric": {"name": "candidate_is_bang", "order": "maximize"}, "seed": 1,
            "mandatory": ["fixture_ok"], "timeout_seconds": 5,
        }
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        evaluation_ids = []
        for name in ("candidate-a", "candidate-b"):
            result, envelope = self.cli("evaluate", "--snapshot-id", answers[name],
                "--spec", str(spec_path), "--fixture", f"sample={fixture_path}",
                "--operation-id", "eval-" + name, json_output=True)
            record = validate_evaluation_record(envelope, answers[name], ("fixture_ok",))
            evaluation_ids.append(envelope["evaluation_id"])
            _, inspected = self.cli("inspect-evaluation", "--evaluation-id",
                envelope["evaluation_id"], json_output=True)
            self.assertEqual(validate_evaluation_record(inspected, answers[name]), record)
        _, selected = self.cli("select-evaluations", "--evaluation-ids", *evaluation_ids,
            "--mandatory", "fixture_ok", json_output=True)
        self.assertIn(selected["evaluation_id"], evaluation_ids)
        before = self.coordinator.store.head("main")
        _, adopted = self.cli("adopt-evaluation", "--branch", "main", "--revision",
            str(before.revision), "--evaluation-ids", *evaluation_ids,
            "--mandatory", "fixture_ok", "--operation-id", "adopt-main", json_output=True)
        validate_adoption_ack(adopted, operation_id="adopt-main",
            evaluation_id=selected["evaluation_id"], snapshot_id=selected["snapshot_id"],
            branch="main", expected_revision=before.revision, root=source.root)
        _, history = self.cli("history", "--branch", "main", json_output=True)
        self.assertEqual(history["tip"], adopted["snapshot_id"])
        loser = "candidate-b" if adopted["snapshot_id"] == answers["candidate-a"] else "candidate-a"
        self.assertEqual(self.coordinator.store.head(loser).tip, answers[loser])
        workspace = CowWorkspace(self.coordinator.store, self.text.tokenizer,
            root_mode_resolver=self.text.root_mode_for_projection)
        adopted_history = workspace.history("main")
        loser_history = workspace.history(loser)
        self.assertTrue(any(message.get("text") == "!" for message in adopted_history["messages"]),
                        repr(adopted_history["messages"]))
        self.assertTrue(any(message.get("text") == "hi" for message in loser_history["messages"]),
                        repr(loser_history["messages"]))
        self.cli("continue", "--branch", "main", "--max-tokens", "1",
                 "--operation-id", "copy-resume", receipt=True)
        self.assertEqual(self.coordinator.store.head("main").revision,
                         adopted["branch"]["revision"] + 1)


if __name__ == "__main__":
    unittest.main()
