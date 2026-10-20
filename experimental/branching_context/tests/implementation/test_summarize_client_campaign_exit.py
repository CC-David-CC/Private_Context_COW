"""The client campaign summarizer requires a successful native exit code."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_client_campaign


class SummarizeNativeExitTests(unittest.TestCase):
    def make_campaign(self, directory, exit_codes_marker):
        labels = [f"{branch}-{index}" for branch in ("A", "B") for index in range(1, 4)]
        worker = {
            "status": "PASS",
            "native_exit_observed": True,
            "profile_id": "profile-test",
            "startup_seconds": 0.0,
            "requests": [],
        }
        if exit_codes_marker is not None:
            worker.update(exit_codes_marker)
        client = {"status": "PASS", "scope": "synthetic test", "requests": {}}
        for label in labels:
            worker["requests"].append({
                "request": label,
                "warmup": False,
                "mode": "resident",
                "outputs": 1,
                "counter_deltas": {"loaded_file_bytes": 0, "forward_tokens": 1},
                "disk_restore": None,
                "durable_save_seconds": 0.1,
            })
            client["requests"][label] = {
                "warmup": False,
                "mode": "resident",
                "outputs": 1,
                "durable": True,
                "time_to_first_token_seconds": 0.1,
                "steady_received_tps": 1.0,
                "median_tpot_seconds": 0.1,
                "p95_tpot_seconds": 0.1,
                "request_tps": 1.0,
                "durable_request_tps": 1.0,
            }
        worker_path = directory / "worker.json"
        client_path = directory / "client.json"
        worker_path.write_text(json.dumps(worker), encoding="utf-8")
        client["worker_sha256"] = hashlib.sha256(worker_path.read_bytes()).hexdigest()
        client_path.write_text(json.dumps(client), encoding="utf-8")
        return directory

    def summarize(self, marker):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_campaign(Path(temporary), marker)
            return summarize_client_campaign.summarize(directory)

    def test_nonzero_native_exit_is_rejected_even_when_observed(self):
        with self.assertRaisesRegex(ValueError, "numeric native exit"):
            self.summarize({"native_exit_codes": {"worker": 7}, "native_exit_success": False})

    def test_missing_numeric_native_exit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "numeric native exit"):
            self.summarize(None)

    def test_zero_native_exit_and_explicit_success_are_accepted(self):
        result = self.summarize({"native_exit_codes": {"worker": 0}, "native_exit_success": True})
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["native_exit_codes"], {"worker": 0})
        self.assertIs(result["native_exit_success"], True)

    def test_boolean_is_not_an_exact_integer_exit_code(self):
        with self.assertRaisesRegex(ValueError, "numeric native exit"):
            self.summarize({"native_exit_codes": {"worker": False}, "native_exit_success": True})


if __name__ == "__main__":
    unittest.main()
