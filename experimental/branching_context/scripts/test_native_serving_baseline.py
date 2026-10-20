"""CPU-only tests for exact fixture loading and normal-serving exit status."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import native_serving_baseline as baseline


class FakeProcess:
    def __init__(self, code):
        self.pid = 1234
        self.returncode = code

    def poll(self):
        return self.returncode


class NativeServingBaselineTests(unittest.TestCase):
    def test_fixture_json_preserves_exact_token_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.json"
            fixture = {"prefix": [1, 8, 9], "A": [4, 5], "B": [6],
                       "model_contract_id": "a" * 64, "scope": "code fixture"}
            path.write_text(json.dumps(fixture))
            actual, metadata = baseline.load_fixture(path, 99, 77)
            self.assertEqual(actual, {"prefix": [1, 8, 9], "A": [4, 5], "B": [6]})
            self.assertEqual(metadata["model_contract_id"], "a" * 64)
            self.assertEqual(metadata["source_fixture_sha256"], baseline.sha(path))

    def test_fixture_json_rejects_missing_or_noninteger_token_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.json"
            path.write_text(json.dumps({"prefix": [1], "A": [2]}))
            with self.assertRaisesRegex(ValueError, "B"):
                baseline.load_fixture(path, 1, 1)
            path.write_text(json.dumps({"prefix": [1, True], "A": [2], "B": [3]}))
            with self.assertRaisesRegex(ValueError, "integer token IDs"):
                baseline.load_fixture(path, 1, 1)

    def test_synthetic_fixture_remains_available_and_exit_details_are_numeric(self):
        fixture, metadata = baseline.load_fixture(None, 3, 2)
        self.assertEqual(fixture["prefix"], [100, 101, 102])
        self.assertEqual(fixture["A"], [117, 118])
        self.assertEqual(fixture["B"], [150, 151])
        self.assertEqual(metadata["fixture_kind"], "synthetic")
        self.assertEqual(baseline.native_exit_record(FakeProcess(0)),
                         {"pid": 1234, "exit_code": 0, "observed": True, "success": True})
        self.assertEqual(baseline.native_exit_record(FakeProcess(-9))["success"], False)
        self.assertEqual(baseline.native_exit_record(FakeProcess(False))["success"], False)
        self.assertEqual(baseline.native_exit_record(FakeProcess(None))["observed"], False)

    def _run_main(self, native_code, mode="fresh", resident_reuse=(3, 3),
                  repetitions=1, warmup=False):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        serve_root = root / "serve-root"
        (serve_root / "serve").mkdir(parents=True)
        (serve_root / "serve/server.py").write_text("# fake imported API module\n")
        executable = root / "strata"
        executable.write_bytes(b"fixture executable")
        config = root / "config.json"
        config.write_text(json.dumps({"exe": str(executable), "args": [
            "--max-context", "32", "--prefill", "8", "--kv", "fp16",
            "--spec", "2", "--suffix-draft", "0",
            "--prompt-cache", "0", "--conversation-cache-mib", "0"]}))
        fixture_path = root / "q8.json"
        input_fixture = {"prefix": [1, 2, 3], "A": [4, 5], "B": [6, 7],
                         "model_contract_id": "b" * 64}
        fixture_path.write_text(json.dumps(input_fixture))
        output = root / "result"
        captured = []
        sampling_captured = []
        actual_arguments = []

        class Engine:
            def __init__(self, *args, **kwargs):
                actual_arguments.append(list(args[1]))
                self.proc = FakeProcess(native_code)
                self.info = {"test": True}
                self.last = {}
                self.reused = 0
                self.calls = 0

            def generate(self, prompt, limit, sampling, cancel):
                captured.append(tuple(prompt))
                sampling_captured.append(dict(sampling))
                self.calls += 1
                self.last = {"generated": 2}
                if "strata_prefix" in sampling and self.calls >= 2:
                    self.reused = resident_reuse[(self.calls - 2) % len(resident_reuse)]
                else:
                    self.reused = 0
                yield 91
                yield 92

            def close(self):
                pass

        package = types.ModuleType("serve")
        package.__path__ = []
        server = types.ModuleType("serve.server")
        server.StrataEngine = Engine
        server.engine_args = lambda config: list(config["args"])
        with patch.dict(sys.modules, {"serve": package, "serve.server": server}):
            status_error = None
            try:
                baseline.main(["--serve-root", str(serve_root), "--config", str(config),
                               "--output", str(output), "--fixture-json", str(fixture_path),
                               "--capacity", "32", "--chunk", "8", "--generated", "2",
                               "--repetitions", str(repetitions), "--modes", mode,
                               *( ["--warmup"] if warmup else [])])
            except RuntimeError as error:
                status_error = error
        return temp, root, captured, sampling_captured, actual_arguments, status_error

    def test_main_uses_fixture_arrays_unchanged_and_requires_zero_native_exit(self):
        temp, root, captured, sampling, arguments, error = self._run_main(0)
        self.addCleanup(temp.cleanup)
        self.assertIsNone(error)
        self.assertEqual(captured, [(1, 2, 3, 4, 5), (1, 2, 3, 6, 7), (1, 2, 3, 4, 5)])
        result = json.loads((root / "result/result.json").read_text())
        self.assertEqual(result["status"], "MEASURED")
        self.assertEqual(result["exits"][0]["exit_code"], 0)
        self.assertEqual(result["fixture_kind"], "supplied-code-fixture")

        temp2, root2, captured2, sampling2, arguments2, error2 = self._run_main(-9)
        self.addCleanup(temp2.cleanup)
        self.assertIsNotNone(error2)
        result2 = json.loads((root2 / "result/result.json").read_text())
        self.assertEqual(result2["status"], "FAIL")
        self.assertEqual(result2["exits"][0]["exit_code"], -9)
        self.assertFalse(result2["exits"][0]["success"])

    def test_resident_prefix_pin_uses_fixture_length_not_cli_default(self):
        temp, root, prompts, sampling, arguments, error = self._run_main(0, mode="resident")
        self.addCleanup(temp.cleanup)
        self.assertIsNone(error)
        self.assertEqual([len(prompt) for prompt in prompts], [5, 5, 5])
        self.assertEqual([item["strata_prefix"]["tokens"] for item in sampling], [3, 3, 3])
        self.assertEqual(arguments[0][arguments[0].index("--prompt-cache") + 1], "3")
        result = json.loads((root / "result/result.json").read_text())
        self.assertEqual(result["prefix"], 3)
        self.assertTrue(result["runs"][1]["shared_prefix_reuse_verified"])
        self.assertTrue(result["runs"][2]["shared_prefix_reuse_verified"])

    def test_resident_mode_enables_and_observes_complete_prefix_reuse(self):
        temp, root, _prompts, _sampling, arguments, error = self._run_main(0, mode="resident")
        self.addCleanup(temp.cleanup)
        self.assertIsNone(error)
        args = arguments[0]
        self.assertGreaterEqual(int(args[args.index("--prompt-cache") + 1]), 3)
        result = json.loads((root / "result/result.json").read_text())
        self.assertGreaterEqual(result["runs"][1]["reused_prompt_tokens"], 3)

    def test_resident_missing_reuse_on_b_or_return_a_fails(self):
        for reuse in ((0, 3), (3, 0)):
            with self.subTest(reuse=reuse):
                temp, root, _prompts, _sampling, _arguments, error = self._run_main(
                    0, mode="resident", resident_reuse=reuse)
                self.addCleanup(temp.cleanup)
                self.assertRegex(str(error), "did not reuse the complete pinned prefix")
                result = json.loads((root / "result/result.json").read_text())
                self.assertEqual(result["status"], "FAIL")

    def test_one_native_process_covers_warmup_and_all_resident_repetitions(self):
        temp, root, _prompts, _sampling, arguments, error = self._run_main(
            0, mode="resident", repetitions=2, warmup=True)
        self.addCleanup(temp.cleanup)
        self.assertIsNone(error)
        self.assertEqual(len(arguments), 1)
        result = json.loads((root / "result/result.json").read_text())
        self.assertEqual(len(result["runs"]), 9)
        self.assertEqual(sum(row["warmup"] for row in result["runs"]), 3)
        self.assertIsNone(result["runs"][0]["shared_prefix_reuse_verified"])
        self.assertTrue(all(row["shared_prefix_reuse_verified"] for row in result["runs"][1:]))
        self.assertEqual(len(result["exits"]), 1)
        self.assertEqual(len(result["engine_startups"]), 1)


if __name__ == "__main__":
    unittest.main()
