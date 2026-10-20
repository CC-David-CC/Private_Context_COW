"""CPU FlowModel acceptance tests for the actual cow_cli grammar-state path."""
from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

from run_cli_grammar_probe import (READ_TEXT, TEST_TEXT, controller_binding,
                                   grammar_registry_payload, parse_durable_receipt)
import run_cli_grammar_probe as probe
from serve.cow_adapter import CowAdapter
from serve.cow_grammar import GrammarRegistry
from serve.cow_http import make_server
from serve.cow_workspace import CowWorkspace
import serve.test_cow_grammar_canonical_flow as flow_fixture


class CliGrammarProbeTests(unittest.TestCase):
    def test_cli_stdout_join_removes_only_one_cli_added_newline(self):
        self.assertEqual(probe.join_cli_stdout("{\n", ' "tool": "read_file"}\n'),
                         '{ "tool": "read_file"}')
        self.assertEqual(probe.join_cli_stdout("model whitespace \n", "next\n"),
                         "model whitespace next")

    def test_eos_check_accepts_any_configured_eos_id(self):
        self.assertTrue(probe.output_ends_with_eos([17, 29], [11, 29]))
        self.assertFalse(probe.output_ends_with_eos([17, 29], [11, 31]))

    def test_registry_contains_two_fixed_named_json_grammars(self):
        payload = grammar_registry_payload()
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(set(payload["states"]), {"READ", "TEST"})
        self.assertEqual(payload["states"]["READ"]["format"], "gbnf")
        self.assertEqual(payload["states"]["TEST"]["format"], "gbnf")
        self.assertEqual(json.loads(payload["states"]["READ"]["source"][len("root ::= "):]), READ_TEXT)
        self.assertEqual(json.loads(payload["states"]["TEST"]["source"][len("root ::= "):]), TEST_TEXT)

    def test_controller_binding_rejects_wrong_state_or_terminal_status(self):
        value = {"controller": {"status": "active", "accepted_in_scope": 1,
                  "constraint": {"agent_state": "READ", "engine_id": "a" * 64}}}
        self.assertEqual(controller_binding(value, "READ", "active"), value["controller"])
        with self.assertRaisesRegex(ValueError, "grammar state"):
            controller_binding(value, "TEST", "active")
        with self.assertRaisesRegex(ValueError, "status"):
            controller_binding(value, "READ", "stopped")

    def test_real_cli_partial_read_state_switch_rejection_and_independent_test_branch(self):
        fixture = flow_fixture.CowCanonicalGrammarFlowTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        registry_file = Path(fixture.tmp.name) / "cli-grammar-registry.json"
        registry_file.write_text(json.dumps(grammar_registry_payload()), encoding="utf-8")
        registry = GrammarRegistry.load(registry_file, fixture.tokenizer,
                                        [fixture.tokenizer.EOS])
        fixture.registry = registry
        fixture.token_adapter.grammar_registry = registry
        fixture.text.grammar_registry = registry

        key = "fixture-cli-grammar-key"
        workspace = CowWorkspace(fixture.coordinator.store, fixture.tokenizer,
            root_mode_resolver=fixture.text.root_mode_for_projection)
        server = make_server(CowAdapter(fixture.coordinator, grammar_registry=registry),
            api_key=key, text_adapter=fixture.text, workspace=workspace)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.addCleanup(stop_server)

        cli = Path(flow_fixture.__file__).resolve().parents[1] / "tools" / "cow_cli.py"
        state_dir = Path(fixture.tmp.name) / "cli-client-state"
        state_dir.mkdir()
        env = os.environ.copy()
        env.update({"COW_API_KEY": key, "PYTHONDONTWRITEBYTECODE": "1"})
        def call(*args, expected=0, receipt=True):
            command = [sys.executable, "-B", str(cli), "--url",
                f"http://127.0.0.1:{server.server_port}", "--state-dir", str(state_dir), *args]
            result = subprocess.run(command, cwd=cli.parents[1], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False)
            self.assertEqual(result.returncode, expected,
                (command, result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")))
            if expected:
                self.assertNotIn(b"durable receipt:\n", result.stderr)
                return result, None
            parsed = parse_durable_receipt(result.stderr) if receipt else None
            return result, parsed

        ns_text, ns_receipt = call("namespace", "--title", "CLI grammar fixture",
            "--operation-id", "grammar-cli-ns")
        namespace = ns_receipt.get("namespace_id") or ns_receipt.get("result", {}).get("namespace_id")
        self.assertIsInstance(namespace, str)
        _, root_receipt = call("prepare", "--namespace-id", namespace, "--branch", "main",
            "--text", "Use the constrained tool response requested by the host.",
            "--operation-id", "grammar-cli-root")
        store = fixture.coordinator.store
        root_head = store.head("main")
        self.assertEqual(root_receipt["snapshot"], root_head.tip)
        self.assertEqual(root_head.tip, root_head.root)
        for branch in ("read", "test"):
            call("fork", "--source-branch", "main", "--name", branch,
                 "--operation-id", "grammar-cli-fork-" + branch)
        self.assertEqual(store.head("read").tip, root_head.tip)
        self.assertEqual(store.head("test").tip, root_head.tip)

        first, first_receipt = call("answer", "--branch", "read", "--max-tokens", "1",
            "--operation-id", "grammar-cli-read-first", "--grammar-state", "READ")
        first_text = first.stdout.decode("utf-8").strip()
        self.assertTrue(first_text)
        first_snapshot = first_receipt["snapshot"]
        first_saved = store.get_snapshot(first_snapshot)
        first_controller = controller_binding(first_saved, "READ", "active")
        self.assertEqual(first_controller["accepted_in_scope"], 1)
        self.assertEqual(store.head("read").tip, first_snapshot)

        before_stats = dict(fixture.backends[0].stats)
        before_head = store.head("read")
        rejected, _ = call("continue", "--branch", "read", "--max-tokens", "64",
            "--operation-id", "grammar-cli-switch-invalid", "--grammar-state", "TEST",
            expected=2, receipt=False)
        error_text = (rejected.stdout + rejected.stderr).decode("utf-8", errors="replace").casefold()
        self.assertTrue("grammar" in error_text or "state" in error_text)
        self.assertEqual(fixture.backends[0].stats, before_stats)
        self.assertEqual(store.head("read").tip, before_head.tip)
        self.assertEqual(store.head("read").revision, before_head.revision)

        resumed, resume_receipt = call("continue", "--branch", "read", "--max-tokens", "64",
            "--operation-id", "grammar-cli-read-resume")
        read_text = first_text + resumed.stdout.decode("utf-8").strip()
        self.assertEqual(read_text, READ_TEXT)
        final_read = store.get_snapshot(resume_receipt["snapshot"])
        read_controller = controller_binding(final_read, "READ", "stopped")
        self.assertGreater(read_controller["accepted_in_scope"], 1)
        self.assertEqual(store.head("read").tip, resume_receipt["snapshot"])

        test_out, test_receipt = call("answer", "--branch", "test", "--max-tokens", "64",
            "--operation-id", "grammar-cli-test", "--grammar-state", "TEST")
        self.assertEqual(test_out.stdout.decode("utf-8").strip(), TEST_TEXT)
        final_test = store.get_snapshot(test_receipt["snapshot"])
        test_controller = controller_binding(final_test, "TEST", "stopped")
        self.assertGreater(test_controller["accepted_in_scope"], 1)
        self.assertEqual(store.head("test").tip, test_receipt["snapshot"])
        self.assertEqual(store.head("read").tip, resume_receipt["snapshot"])
        self.assertEqual(store.head("main").tip, root_head.tip)

        for operation, snapshot_id in (("grammar-cli-read-first", first_snapshot),
                                       ("grammar-cli-read-resume", resume_receipt["snapshot"]),
                                       ("grammar-cli-test", test_receipt["snapshot"])):
            envelope = store.presentation_receipt(operation)
            self.assertEqual(envelope["result"]["tip"], snapshot_id)
        first_input_tokens = store.presentation_receipt("grammar-cli-read-first")["presentation"]["output_span"]["count"]
        self.assertEqual(first_input_tokens, 1)

    def test_full_probe_execute_against_cpu_flowmodel_fixture(self):
        fixture = flow_fixture.CowCanonicalGrammarFlowTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        registry_file = Path(fixture.tmp.name) / "cli-grammar-runner-registry.json"
        registry_file.write_text(json.dumps(grammar_registry_payload()), encoding="utf-8")
        registry = GrammarRegistry.load(registry_file, fixture.tokenizer,
                                        [fixture.tokenizer.EOS])
        fixture.registry = registry
        fixture.token_adapter.grammar_registry = registry
        fixture.text.grammar_registry = registry
        workspace = CowWorkspace(fixture.coordinator.store, fixture.tokenizer,
            root_mode_resolver=fixture.text.root_mode_for_projection)
        server = make_server(CowAdapter(fixture.coordinator, grammar_registry=registry),
            api_key="runner-fixture-key", text_adapter=fixture.text, workspace=workspace)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        class FixtureService:
            def __init__(self, _command, **_kwargs):
                self.ready = server.server_port
                self.process = SimpleNamespace(pid=os.getpid())

            def start(self):
                return self.ready

            def stop(self):
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("CPU FlowModel fixture service did not stop")
                return {"exit_code": 0}

        class FixtureTracker:
            def __init__(self, *_args):
                pass

            def start(self):
                return self

            def phase(self, _name):
                pass

            def finish(self, *_args):
                pass

            def close(self):
                pass

        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5))
            if thread.is_alive() else None)
        work = Path(fixture.tmp.name)
        runtime = Path(flow_fixture.__file__).resolve().parents[1]
        profile = work / "cpu-fixture-profile.json"
        profile.write_text(json.dumps({"execution": {"eos_tokens": [fixture.tokenizer.EOS + 1,
                                                                          fixture.tokenizer.EOS]},
            "execution_profile_id": "cpu-fixture-only", "model_contract_id": "flow-model"}),
            encoding="utf-8")
        args = Namespace(run_dir=work / "probe-run", runtime_root=runtime,
            profile=profile, store=work / "unused-empty-store", question=
            "Use the constrained tool response requested by the host.",
            dependency_root=None, max_continue=64, ram_gib=96, disk_gib=128)
        probe_instance = probe.GrammarProbe(args, _test_dependencies={
            "runtime_closure": lambda *_args: None,
            "ProgressTracker": FixtureTracker,
            "ServiceProcess": FixtureService,
            "ready_url": lambda port: f"http://127.0.0.1:{port}",
            "Store": lambda _path: fixture.coordinator.store,
            "api_key": "runner-fixture-key",
            "source_imports": {"execution_kind": "FlowModel CPU fixture; no native process"},
        })
        execute_result = probe_instance.execute()
        command_details = []
        for row in probe_instance.record["commands"]:
            stderr_path = Path(row["stderr"])
            if stderr_path.is_file():
                command_details.append(stderr_path.read_text(encoding="utf-8", errors="replace"))
        self.assertEqual(execute_result, 0,
            json.dumps({"record": probe_instance.record, "cli_stderr": command_details}, sort_keys=True))
        result = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "CLI_GRAMMAR_FLOW_PASS_NUMERICAL_AUDIT_PENDING")
        self.assertEqual(result["execution_kind"], "NON_NATIVE_CPU_FLOWMODEL_FIXTURE")
        self.assertEqual(result["native_worker_shutdown"], "NOT_APPLICABLE")
        self.assertEqual(result["read_complete"]["combined_text"], READ_TEXT)
        self.assertEqual(result["test_complete"]["text"], TEST_TEXT)
        self.assertEqual(result["state_switch_rejection"]["status"], "PASS_NO_WORK")
        self.assertTrue(all("runner-fixture-key" not in path.read_text(encoding="utf-8")
            for path in args.run_dir.glob("*.stderr")))


if __name__ == "__main__":
    unittest.main()
