"""CPU contract tests for the real CLI retry probe's evidence validators."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from run_cli_retry_probe import (
    RetryProbe,
    _journaled_retry_body,
    validate_changed_retry,
    validate_exact_replay,
    validate_stale_retry,
)


def _receipt(operation="answer-1", snapshot="snap-answer"):
    return {
        "operation_id": operation,
        "snapshot": snapshot,
        "branch": {"name": "chat", "revision": 3, "tip": snapshot, "root": "root-1"},
    }


def _replay_observation():
    counters = {"forward_tokens": 16, "prefill_tokens": 12, "native_calls": 7}
    head = {"name": "chat", "revision": 3, "tip": "snap-answer", "root": "root-1"}
    sibling = {"name": "sibling", "revision": 1, "tip": "snap-source", "root": "root-1"}
    response = {
        "event": "complete", "replayed": True,
        "answer": {"snapshot": "snap-answer", "tokens": [70, 71], "text": "ok"},
        "receipt": _receipt(),
    }
    return {
        "exit_code": 0, "cli_stdout": b"ok\n", "cli_receipt": _receipt(),
        "response": response,
        "head_before": head, "head_after": dict(head),
        "sibling_before": sibling, "sibling_after": dict(sibling),
        "native_trace_bytes_before": 4096, "native_trace_bytes_after": 4096,
        "counters_before": counters, "counters_after": dict(counters),
    }


def _changed_observation():
    before = {"forward_tokens": 16, "prefill_tokens": 12, "native_calls": 7}
    after = {"forward_tokens": 18, "prefill_tokens": 12, "native_calls": 10}
    sibling = {"name": "sibling", "revision": 1, "tip": "snap-source", "root": "root-1"}
    response = {
        "event": "complete", "replayed": False,
        "answer": {"snapshot": "snap-retry", "tokens": [70, 71], "text": "ok"},
        "receipt": _receipt("retry-1", "snap-retry"),
    }
    return {
        "exit_code": 0, "response": response, "cli_receipt": _receipt("retry-1", "snap-retry"),
        "cli_stdout": b"ok\n",
        "operation_id": "retry-1", "original_operation": "answer-1",
        "source_snapshot": "snap-source",
        "expected_source_snapshot": "snap-source",
        "output_tokens": [70, 71], "expected_output_tokens": [70, 71],
        "branch_before": {"name": "retry-answer", "revision": 1,
                           "tip": "snap-answer", "root": "root-1"},
        "branch_after": {"name": "retry-answer", "revision": 2,
                          "tip": "snap-retry", "root": "root-1"},
        "sibling_before": sibling, "sibling_after": dict(sibling),
        "source_stream_key": "stream-original", "final_stream_key": "stream-original",
        "source_accepted": 4, "final_accepted": 6,
        "native_trace_bytes_before": 4096, "native_trace_bytes_after": 8192,
        "counters_before": before, "counters_after": after,
        "original_receipt_before": _receipt(), "original_receipt_after": _receipt(),
    }


class CliRetryNativeProbeTests(unittest.TestCase):
    def test_exact_replay_requires_same_text_receipt_head_sibling_and_zero_work(self):
        original = {"stdout": b"ok\n", "answer": {
            "snapshot": "snap-answer", "tokens": [70, 71]}, "receipt": _receipt()}
        validate_exact_replay(original, _replay_observation())

    def test_exact_replay_rejects_receipt_or_native_work_change(self):
        original = {"stdout": b"ok\n", "answer": {
            "snapshot": "snap-answer", "tokens": [70, 71]}, "receipt": _receipt()}
        changed = _replay_observation()
        changed["response"]["receipt"]["snapshot"] = "other"
        with self.assertRaisesRegex(ValueError, "receipt"):
            validate_exact_replay(original, changed)
        changed = _replay_observation()
        changed["counters_after"]["forward_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "forward_tokens"):
            validate_exact_replay(original, changed)

    def test_exact_replay_rejects_branch_change_and_native_rpc(self):
        original = {"stdout": b"ok\n", "answer": {
            "snapshot": "snap-answer", "tokens": [70, 71]}, "receipt": _receipt()}
        changed = _replay_observation()
        changed["head_after"]["revision"] += 1
        with self.assertRaisesRegex(ValueError, "branch head"):
            validate_exact_replay(original, changed)
        changed = _replay_observation()
        changed["native_trace_bytes_after"] += 1
        with self.assertRaisesRegex(ValueError, "native RPC"):
            validate_exact_replay(original, changed)

    def test_changed_retry_requires_new_operation_and_original_source_stream(self):
        validate_changed_retry(_changed_observation())
        changed = _changed_observation()
        changed["operation_id"] = "answer-1"
        changed["response"]["receipt"]["operation_id"] = "answer-1"
        with self.assertRaisesRegex(ValueError, "new operation"):
            validate_changed_retry(changed)
        changed = _changed_observation()
        changed["final_stream_key"] = "new-stream"
        with self.assertRaisesRegex(ValueError, "stream"):
            validate_changed_retry(changed)

    def test_changed_retry_confirmation_may_omit_replayed_flag(self):
        changed = _changed_observation()
        changed["response"].pop("replayed")
        validate_changed_retry(changed)

    def test_changed_retry_confirmation_rejects_non_boolean_replayed_flag(self):
        changed = _changed_observation()
        changed["response"]["replayed"] = "false"
        with self.assertRaisesRegex(ValueError, "invalid replay status"):
            validate_changed_retry(changed)

    def test_changed_retry_reposts_exact_completed_cli_journal_body(self):
        body = {"from_operation": "answer-1", "operation_id": "retry-1",
                "branch": "retry-answer", "revision": 4, "stream": True,
                "settings": {"temperature": 0.7, "top_k": 1, "min_p": 0.05}}
        state_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(state_dir, ignore_errors=True))
        operations = state_dir / "operations"
        operations.mkdir()
        body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        record = {"schema": 1, "operation_id": "retry-1", "url": "http://127.0.0.1:1",
                  "path": "/cow/text/retry", "state": "complete", "body_json": body_json,
                  "receipt": _receipt("retry-1", "snap-retry")}
        intent = json.dumps(["retry-1", record["url"], record["path"], body_json],
                            ensure_ascii=False, separators=(",", ":")).encode()
        record["intent_sha256"] = hashlib.sha256(intent).hexdigest()
        unsigned = json.dumps(record, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")).encode()
        record["record_sha256"] = hashlib.sha256(unsigned).hexdigest()
        path = operations / "retry-1.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        raw, parsed = _journaled_retry_body(
            {"client_state_dir": str(state_dir)}, "retry-1", "answer-1")
        self.assertEqual(raw, record["body_json"].encode("utf-8"))
        self.assertEqual(parsed, body)
        self.assertEqual(parsed["revision"], 4)

    def test_changed_retry_rejects_wrong_output_or_reset_continuation_cursor(self):
        changed = _changed_observation()
        changed["output_tokens"] = [70, 72]
        with self.assertRaisesRegex(ValueError, "output tokens"):
            validate_changed_retry(changed)
        changed = _changed_observation()
        changed["source_accepted"] = 0
        with self.assertRaisesRegex(ValueError, "accepted-token cursor"):
            validate_changed_retry(changed)

    def test_stale_revision_requires_conflict_before_any_native_or_branch_change(self):
        before = {"forward_tokens": 8, "prefill_tokens": 4, "native_calls": 3}
        after = dict(before)
        head = {"name": "chat", "revision": 9, "tip": "current", "root": "root-1"}
        validate_stale_retry(409, {"error": "Requested branch revision changed"},
            head, dict(head), before, after, 1000, 1000)

    def test_stale_revision_rejects_work_ack_or_unrelated_failure(self):
        before = {"forward_tokens": 8, "prefill_tokens": 4}
        head = {"name": "chat", "revision": 9, "tip": "current", "root": "root-1"}
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            validate_stale_retry(500, {"error": "storage failure"}, head, dict(head),
                before, dict(before), 1000, 1000)
        with self.assertRaisesRegex(ValueError, "native RPC"):
            validate_stale_retry(409, {"error": "Requested branch revision changed"},
                head, dict(head), before, dict(before), 1000, 1001)


class _FlowRetryCampaign:
    """Test-only injection around the actual CLI workflow and CPU HTTP fixture."""
    def __init__(self, fixture, root):
        from run_cli_workflow_probe import parse_durable_receipt
        self._parse_receipt = parse_durable_receipt
        self.fixture = fixture
        self.root = Path(root)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.run_id = "cpu-retry-flow"
        self.cpu_fixture = True
        self.api_key = "retry-test-key"
        self.store = fixture.coordinator.store
        self.ordinal = 1
        self.call_number = 0
        self.trace_offset = len(fixture.trace)
        self.trace_path = self.run_dir / "native-1.jsonl"
        self.trace_path.touch()
        self.service = SimpleNamespace(ready=fixture.base)
        self._ready_url = lambda ready: ready
        self.record = {"commands": [], "services": [
            {"ordinal": 1, "pid": 91001, "cpu_fixture": True}],
            "retry_observations": []}
        self._cli_path = Path(__file__).resolve().parents[3].parent / "tools" / "cow_cli.py"

    def publish(self):
        return None

    def phase(self, name):
        self.record["phase"] = name

    def _native_stats(self):
        model = next(iter(self.fixture.models.values()))
        # FlowModel only exposes aggregate CPU counters; its native trace is a
        # per-request fixture witness, not hardware instrumentation.
        return {"counters": {
            "forward_tokens": model.stats["forward_tokens"],
            "prefill_tokens": 0,
            "disk_loads": 0,
            "native_calls": len(self.fixture.trace),
            "loaded_bytes": model.stats.get("loaded_file_bytes", 0),
            "sampling_calls": len(model.selection_calls),
            "capture_bytes": 0,
            "hash_bytes": 0,
        }}

    def _sync_trace(self):
        rows = self.fixture.trace[self.trace_offset:]
        if rows:
            with self.trace_path.open("ab") as stream:
                for row in rows:
                    stream.write((json.dumps(row, separators=(",", ":"), default=list) + "\n").encode())
            self.trace_offset = len(self.fixture.trace)

    def cli(self, label, client_name, *arguments, expect=0, want_receipt=False,
            output_json=False, state_suffix=None):
        from serve.test_cow_answer_retry import RetryFlowModel
        self.call_number += 1
        model = next(iter(self.fixture.models.values()))
        if "--operation-id" in arguments:
            op = arguments[arguments.index("--operation-id") + 1]
            command = arguments[0]
            if command == "answer":
                model.plans[op] = (258,) * 8
            elif command == "continue":
                model.plans[op] = (258,) * 8
        state = self.run_dir / ("journal-" + client_name + ("-" + state_suffix if state_suffix else ""))
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["COW_API_KEY"] = self.api_key
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-B", str(self._cli_path), "--url", self.fixture.base,
                   "--state-dir", str(state), "--timeout", "30", *arguments]
        result = subprocess.run(command, cwd=self._cli_path.parents[1], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False)
        self._sync_trace()
        stdout_path = self.run_dir / f"{self.call_number:03d}-{label}.stdout"
        stderr_path = self.run_dir / f"{self.call_number:03d}-{label}.stderr"
        stdout_path.write_bytes(result.stdout)
        stderr_path.write_bytes(result.stderr)
        row = {"label": label, "exit_code": result.returncode,
               "stdout": str(stdout_path), "stderr": str(stderr_path)}
        self.record["commands"].append(row)
        if result.returncode != expect:
            raise AssertionError((label, result.returncode, result.stdout, result.stderr))
        value = self._parse_receipt(result.stderr) if want_receipt and expect == 0 else None
        if output_json and expect == 0:
            value = json.loads(result.stdout)
        return row, value

    def stop_service(self):
        self.fixture._stop_server()
        self.fixture.coordinator.close()
        self.fixture.coordinator = None
        self.record["services"][-1]["stop"] = {"exit_code": 0, "cpu_fixture": True}
        self.record["services"][-1]["worker_shutdown"] = {"exit_code": 0, "cpu_fixture": True}
        self.service = None

    def start_service(self):
        self.ordinal += 1
        self.fixture.coordinator = self.fixture._new_coordinator()
        self.fixture.coordinator.ready(5)
        self.fixture._start_server()
        self.store = self.fixture.coordinator.store
        self.service = SimpleNamespace(ready=self.fixture.base)
        self.trace_offset = len(self.fixture.trace)
        self.trace_path = self.run_dir / f"native-{self.ordinal}.jsonl"
        self.trace_path.touch()
        self.record["services"].append({"ordinal": self.ordinal,
            "pid": 91000 + self.ordinal, "cpu_fixture": True})


class CliRetryNativeProbeFlowTests(unittest.TestCase):
    """Exercise probe orchestration over real CLI+HTTP with the CPU FlowModel."""
    def test_complete_retry_workflow_against_canonical_retry_fixture(self):
        from serve.test_cow_answer_retry import CowAnswerRetryTests
        fixture = CowAnswerRetryTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = _FlowRetryCampaign(fixture, root)
            args = SimpleNamespace(
                run_dir=campaign.run_dir, store=fixture.root,
                branch="probe-chat", sibling_branch="probe-sibling",
                question="Explain the supplied context briefly.")
            probe = RetryProbe(args, campaign=campaign)
            probe._workflow()
            self.assertEqual(campaign.record["workflow_status"], "PIPELINE_PASS")
            exact = [row for row in campaign.record["retry_observations"]
                     if row["kind"] == "exact_replay"]
            self.assertEqual(len(exact), 3)
            changed = [row for row in campaign.record["retry_observations"]
                       if row["kind"] == "changed_settings"]
            self.assertEqual(len(changed), 2)
            self.assertEqual(changed[0]["observation"]["output_tokens"], [258] * 8)
            self.assertEqual(changed[1]["observation"]["source_accepted"], 8)
            self.assertEqual(changed[1]["observation"]["final_accepted"], 16)
            self.assertEqual(campaign.record["stale_revision_control"]["status"], 409)


if __name__ == "__main__":
    unittest.main()
