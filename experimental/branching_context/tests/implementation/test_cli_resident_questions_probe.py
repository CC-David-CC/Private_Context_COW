"""CPU checks for the bounded modern resident CLI probe."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_cli_resident_questions_probe.py"
SPEC = importlib.util.spec_from_file_location("cli_resident_questions_probe_test", SCRIPT) if SCRIPT.is_file() else None
PROBE = importlib.util.module_from_spec(SPEC) if SPEC and SPEC.loader else None
if SPEC is not None and SPEC.loader is not None:
    SPEC.loader.exec_module(PROBE)


def probe():
    if PROBE is None:
        raise AssertionError("modern resident CLI probe is not implemented yet (baseline RED)")
    return PROBE


class ResidentProbeTests(unittest.TestCase):
    def test_accepts_8087_token_root_and_three_independent_answers(self):
        p = probe()
        result = p.validate_campaign_shape(
            root_tokens=8087,
            question_sources=("P", "P", "P"),
            question_snapshots=("A1", "B1", "C1"),
            actual_output_counts=(128, 128, 128),
            max_tokens=128,
        )
        self.assertEqual(result["root_tokens"], 8087)
        self.assertEqual(result["independent_questions"], 3)
        self.assertEqual(result["actual_output_counts"], [128, 128, 128])

    def test_rejects_existing_run_directory_before_mutation(self):
        p = probe()
        with self.assertRaises(p.ResidentProbeError):
            p.require_fresh_output(True)
        self.assertIsNone(p.require_fresh_output(False))

    def test_input_hash_pin_must_be_well_formed_and_match(self):
        p = probe()
        self.assertEqual(len(p.verify_file_sha256(SCRIPT, __import__("hashlib").sha256(SCRIPT.read_bytes()).hexdigest())), 64)
        with self.assertRaises(p.ResidentProbeError):
            p.verify_file_sha256(SCRIPT, "not-a-hash")
        with self.assertRaises(p.ResidentProbeError):
            p.verify_file_sha256(SCRIPT, "0" * 64)

    def test_work_counters_require_all_fields_and_zero_saved_prefix_work(self):
        p = probe()
        before = {"forward_tokens": 8087, "prefill_tokens": 8087,
                  "loaded_file_bytes": 0, "integrity_mismatches": 0}
        after = {"forward_tokens": 8087 + 43 + 128,
                 "prefill_tokens": 8087 + 43, "loaded_file_bytes": 0,
                 "integrity_mismatches": 0}
        delta = p.validate_question_work(before, after, question_token_ids=tuple(range(43)),
                                         output_token_ids=tuple(range(128)))
        self.assertEqual(delta["old_prefix_forward_tokens"], 0)
        missing = dict(after)
        del missing["loaded_file_bytes"]
        with self.assertRaises(p.ResidentProbeError):
            p.validate_question_work(before, missing, question_token_ids=tuple(range(43)),
                                     output_token_ids=tuple(range(128)))

    def test_machine_answer_text_and_token_ids_are_exact(self):
        p = probe()
        envelope = {"schema": 1,
                    "answer": {"snapshot": "A1", "text": "code\n", "tokens": [1, 2],
                               "generation_seconds": 1.25, "request_seconds": 1.5},
                    "receipt": {"snapshot": "A1"}}
        answer = p.validate_machine_answer(envelope, expected_snapshot="A1",
                                           expected_output_ids=(1, 2), expected_text="code\n")
        self.assertEqual(answer["text"], "code\n")
        with self.assertRaises(p.ResidentProbeError):
            p.validate_machine_answer(envelope, expected_snapshot="A1",
                                      expected_output_ids=(1, 2), expected_text="code")

    def test_observer_requires_valid_timing_and_exact_answer_tokens(self):
        p = probe()
        summary = {"validation": {"status": "valid", "errors": []},
                   "token_ids": [1, 2], "metrics_ns": {"request_to_first_token": 10,
                   "request_to_durable": 100}}
        self.assertEqual(p.validate_observation_summary(summary, expected_tokens=(1, 2))["ttft_ns"], 10)
        bad = dict(summary, metrics_ns={"request_to_durable": 100})
        with self.assertRaises(p.ResidentProbeError):
            p.validate_observation_summary(bad, expected_tokens=(1, 2))

    def test_observed_completion_is_bound_to_question_request_and_receipt(self):
        import base64
        import json
        p = probe()
        receipt = {"snapshot": "A1"}
        events = [{"event": "complete", "answer": {"snapshot": "A1", "tokens": [1, 2]},
                   "receipt": receipt}]
        trace = {"cli_exit_code": 0, "cli_exception": None, "requests": [{
            "method": "POST", "request_path": "/cow/text/ask", "branch": "Question1",
            "operation_id": "resident-question-1", "events": [
                {"raw_line_b64": base64.b64encode(json.dumps(events[0]).encode()).decode()}],
            "summary": {"validation": {"status": "valid", "errors": []}, "token_ids": [1, 2],
                        "metrics_ns": {"request_to_first_token": 10, "request_to_durable": 100}}}]}
        result = p.validate_observation_trace(trace, branch="Question1",
            operation_id="resident-question-1", receipt=receipt, expected_tokens=(1, 2))
        self.assertEqual(result["complete_snapshot"], "A1")
        trace["requests"][0]["operation_id"] = "wrong-operation"
        with self.assertRaises(p.ResidentProbeError):
            p.validate_observation_trace(trace, branch="Question1",
                operation_id="resident-question-1", receipt=receipt, expected_tokens=(1, 2))

    def test_cli_auth_secret_is_only_in_child_environment_and_redacted_from_artifacts(self):
        import subprocess
        import tempfile
        from unittest.mock import patch
        from run_cli_workflow_probe import redact_sensitive_value

        p = probe()
        secret = "fixture-secret-never-persist"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=(
                    b'durable receipt:\n{"snapshot":"S","diagnostic":"'
                    + secret.encode("utf-8") + b'"}\n'))
            with patch.object(p.subprocess, "run", return_value=completed) as run:
                _, row = p._invoke(output, output, output / "cow_cli.py", "http://127.0.0.1",
                    "auth-test", "status", api_key=secret)
            called = run.call_args
            self.assertNotIn(secret, called.args[0])
            self.assertEqual(called.kwargs["env"]["COW_API_KEY"], secret)
            self.assertNotIn(secret.encode(), (output / "auth-test.stderr").read_bytes())
            self.assertEqual(row["receipt"]["diagnostic"], "[REDACTED]")
            shutdown = {"stop": {"command": ["cow-http", "--api-key", secret]},
                        "cleanup": {"nested": ["prefix-" + secret]}, secret: "as-a-key"}
            redacted = p._redact_campaign_value(shutdown, secret, redact_sensitive_value)
            self.assertNotIn(secret, __import__("json").dumps(redacted))
            self.assertIn("<REDACTED>", __import__("json").dumps(redacted))


if __name__ == "__main__":
    unittest.main()
