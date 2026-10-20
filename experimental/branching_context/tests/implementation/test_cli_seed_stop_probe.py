"""Fail-closed checks for the real CLI seed/STOP campaign harness."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from run_cli_seed_stop_probe import (
    COMPLETED_WORKFLOW,
    READ_STOP,
    SeedStopProbe,
    TEST_TEXT,
    head_record,
    mark_workflow_pass,
    redact_process_receipt,
    seed_registry_payload,
    validate_exact_replay,
    validate_reseed_transition,
    validate_sampled_continuation,
    validate_sampled_budget,
    validate_stopped_generation,
)


class SeedStopProbeValidatorTests(unittest.TestCase):
    def controller(self):
        return {
            "schema": 7,
            "status": "active",
            "stream_key": "0" * 64,
            "next_draw": 3,
            "accepted_in_scope": 1,
            "constraint": {"schema": 1, "agent_state": "SAMPLED", "scope_start": 9},
            "text_stop": {"stop_sequences": ["read_file"], "pending_hex": ""},
            "output_pending_hex": "",
        }

    def test_registry_is_server_owned_and_contains_three_named_gbnf_states(self):
        registry = seed_registry_payload()
        self.assertEqual(registry["schema"], 1)
        self.assertEqual(set(registry["states"]), {"SAMPLED", "READ", "TEST"})
        self.assertTrue(all(item["format"] == "gbnf" for item in registry["states"].values()))
        self.assertIn("read_file", registry["states"]["READ"]["source"])
        self.assertIn("run_tests", registry["states"]["TEST"]["source"])

    def test_reseed_preserves_every_controller_field_except_rng_key_and_cursor(self):
        before = self.controller()
        expected = copy.deepcopy(before)
        expected["stream_key"] = hashlib.sha256(b"cow-seed-v1:73").hexdigest()
        expected["next_draw"] = 0
        validate_reseed_transition(before, expected, 73)
        for field, value in (("accepted_in_scope", 2), ("status", "stopped"),
                             ("next_draw", 4), ("stream_key", "1" * 64)):
            changed = copy.deepcopy(expected)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reseed_transition(before, changed, 73)
        for bad_seed in (True, -1, 1 << 64, 1.0):
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                validate_reseed_transition(before, expected, bad_seed)

    def test_replay_requires_identical_output_receipt_head_and_zero_native_trace(self):
        receipt = {"snapshot": "snap-a", "branch": {"name": "A", "revision": 2}}
        head = {"name": "A", "revision": 2, "tip": "snap-a", "root": "root-a"}
        validate_exact_replay(stdout_before=b"A\n", stdout_after=b"A\n",
            receipt_before=receipt, receipt_after=copy.deepcopy(receipt),
            head_before=head, head_after=dict(head), trace_before=b"rpc\n", trace_after=b"rpc\n")
        changes = [
            {"stdout_after": b"B\n"},
            {"receipt_after": {"snapshot": "other"}},
            {"head_after": dict(head, revision=3)},
            {"trace_after": b"rpc\nrpc\n"},
        ]
        for changeset in changes:
            kwargs = dict(stdout_before=b"A\n", stdout_after=b"A\n",
                receipt_before=receipt, receipt_after=copy.deepcopy(receipt),
                head_before=head, head_after=dict(head), trace_before=b"rpc\n", trace_after=b"rpc\n")
            kwargs.update(changeset)
            with self.subTest(changeset=changeset), self.assertRaises(ValueError):
                validate_exact_replay(**kwargs)

    def test_stop_requires_consumed_marker_hidden_from_display_and_active_grammar_identity(self):
        controller = self.controller()
        controller["status"] = "stopped"
        controller["text_stop"]["matched_stop"] = READ_STOP
        display = '{"tool":"'
        validate_stopped_generation(display=display,
            canonical_bytes=display.encode() + READ_STOP.encode() + b'","path":"x"}',
            controller=controller, marker=READ_STOP, grammar_state="SAMPLED")
        for display_bad, raw_bad, changed in (
            (display + READ_STOP, display.encode() + READ_STOP.encode(), {}),
            (display, display.encode(), {}),
            (display, display.encode() + READ_STOP.encode(), {"grammar_complete": True}),
        ):
            candidate = copy.deepcopy(controller)
            candidate["constraint"].update(changed)
            with self.subTest(display=display_bad, candidate=candidate), self.assertRaises(ValueError):
                validate_stopped_generation(display=display_bad, canonical_bytes=raw_bad,
                    controller=candidate, marker=READ_STOP, grammar_state="SAMPLED")
        with self.assertRaisesRegex(ValueError, "grammar state"):
            validate_stopped_generation(display=display, canonical_bytes=display.encode() + READ_STOP.encode(),
                controller=controller, marker=READ_STOP, grammar_state="TEST")

    def test_workflow_cannot_publish_pass_before_all_required_stages_in_order(self):
        record = {"status": "RUNNING"}
        with self.assertRaisesRegex(ValueError, "every required"):
            mark_workflow_pass(record, list(COMPLETED_WORKFLOW[:-1]))
        mark_workflow_pass(record, list(COMPLETED_WORKFLOW))
        self.assertEqual(record["status"], "CLI_SEED_STOP_FLOW_PASS_NUMERICAL_AUDIT_PENDING")

    def test_branch_head_validation_rejects_boolean_revision(self):
        head_record({"name": "A", "revision": 1, "tip": "snap", "root": "root"})
        with self.assertRaisesRegex(ValueError, "head is malformed"):
            head_record({"name": "A", "revision": True, "tip": "snap", "root": "root"})

    def test_sampled_budget_does_not_assume_grammar_characters_equal_tokens(self):
        validate_sampled_budget(1, 16)
        validate_sampled_budget(8, 16)
        validate_sampled_budget(8, 64)
        for initial, continuation in ((0, 1), (1, 0), (True, 1), (8.0, 17), (1, False)):
            with self.subTest(initial=initial, continuation=continuation), \
                    self.assertRaisesRegex(ValueError, "positive exact integers"):
                validate_sampled_budget(initial, continuation)

    def test_restart_continuation_binds_actual_count_and_eos_outcome(self):
        before = self.controller()
        budget = 4
        active_after = copy.deepcopy(before)
        active_after["next_draw"] += budget
        active_after["accepted_in_scope"] += budget
        self.assertEqual(validate_sampled_continuation(
            before=before, after=active_after, tokens=[11, 12, 13, 14],
            output_span_count=4, budget=budget, eos_tokens={99}, grammar_complete=False),
            {"consumed_tokens": 4, "terminal_eos": False, "status": "active",
             "grammar_complete": False, "draw_delta": 4, "accepted_delta": 4})

        completed_text_pending_eos = validate_sampled_continuation(
            before=before, after=active_after, tokens=[11, 12, 13, 14],
            output_span_count=4, budget=budget, eos_tokens={99}, grammar_complete=True)
        self.assertTrue(completed_text_pending_eos["grammar_complete"])
        self.assertEqual(completed_text_pending_eos["status"], "active")

        eos_after = copy.deepcopy(before)
        eos_after["next_draw"] += 2
        eos_after["accepted_in_scope"] += 2
        eos_after["status"] = "stopped"
        self.assertEqual(validate_sampled_continuation(
            before=before, after=eos_after, tokens=[11, 99], output_span_count=2,
            budget=budget, eos_tokens={99}, grammar_complete=True),
            {"consumed_tokens": 2, "terminal_eos": True, "status": "stopped",
             "grammar_complete": True, "draw_delta": 2, "accepted_delta": 2})

        invalid = [
            dict(after=eos_after, tokens=[11, 99, 12], output_span_count=3),
            dict(after=eos_after, tokens=[11, 99], output_span_count=1),
            dict(after=dict(eos_after, next_draw=eos_after["next_draw"] + 1),
                 tokens=[11, 99], output_span_count=2),
            dict(after=dict(eos_after, status="active"), tokens=[11, 99], output_span_count=2),
            dict(after=active_after, tokens=[11, 12], output_span_count=2),
            dict(after=eos_after, tokens=[11, 99], output_span_count=2,
                 grammar_complete=False),
            dict(after=eos_after, tokens=[11, 99], output_span_count=2,
                 grammar_complete=1),
        ]
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                arguments = {"before": before, "budget": budget, "eos_tokens": {99},
                             "grammar_complete": True}
                arguments.update(case)
                validate_sampled_continuation(**arguments)

    def test_eos_stopped_continuation_requires_grammar_complete(self):
        before = self.controller()
        after = copy.deepcopy(before)
        after["next_draw"] += 1
        after["accepted_in_scope"] += 1
        after["status"] = "stopped"
        with self.assertRaisesRegex(ValueError, "EOS-terminated grammar continuation"):
            validate_sampled_continuation(before=before, after=after, tokens=[99],
                output_span_count=1, budget=4, eos_tokens={99}, grammar_complete=False)

    def test_journal_answer_uses_observed_answer_shape(self):
        answer = {"tokens": [11, 99], "grammar_complete": True}
        operation_id = "operation-1"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "journal"
            (state / "operations").mkdir(parents=True)
            path = state / "operations" / f"{operation_id}.json"
            path.write_text(json.dumps({"operation_id": operation_id,
                "state": "complete", "answer": answer}), encoding="utf-8")
            probe = SeedStopProbe.__new__(SeedStopProbe)
            observed = probe._operation_answer({"journal_dir": str(state)}, operation_id)
            self.assertEqual(observed, answer)
            self.assertIs(observed["grammar_complete"], True)

    def test_nested_service_shutdown_receipt_never_persists_api_key(self):
        secret = "service-key-keep-private"
        source = {"command": ["cow-http", "--api-key", secret],
                  "nested": {"stderr": "failed with " + secret}}
        clean = redact_process_receipt(source, secret)
        encoded = repr(clean)
        self.assertNotIn(secret, encoded)
        self.assertEqual(clean["command"][-1], "<REDACTED>")
        self.assertEqual(clean["nested"]["stderr"], "failed with <REDACTED>")

    def test_reseed_probe_reads_durable_receipt_from_stderr(self):
        receipt = {"operation_id": "reseed-op", "branch": {"name": "A", "revision": 2},
                   "snapshot": "snapshot-2"}
        stderr = b"durable receipt:\n" + json.dumps(receipt).encode() + b"\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = SeedStopProbe.__new__(SeedStopProbe)
            probe.args = SimpleNamespace(dependency_root=None)
            probe.run_dir = root / "run"
            probe.run_dir.mkdir()
            probe.run_id = "test"
            probe.api_key = "test-api-key"
            probe.runtime = root
            probe.cli_python = os.environ.get("COW_PYTHON", "python")
            probe.service = SimpleNamespace(ready={"address": ["127.0.0.1", 9000]})
            probe._ready_url = lambda ready: "http://127.0.0.1:9000"
            probe.call_id = 0
            probe.record = {"commands": []}
            probe.phase = lambda _name: None
            probe.publish = lambda: None
            probe._trace_bytes = lambda: b""
            completed = subprocess.CompletedProcess([], 0, b"", stderr)
            with mock.patch("run_cli_seed_stop_probe.subprocess.run", return_value=completed) as run:
                row, parsed = probe.reseed_cli("A", 0, 1, "reseed-op")

        self.assertEqual(parsed, receipt)
        self.assertEqual(row["stdout_text"], "")
        command = run.call_args.args[0]
        self.assertIn("--revision", command)
        self.assertEqual(command[command.index("--revision") + 1], "1")


if __name__ == "__main__":
    unittest.main()
