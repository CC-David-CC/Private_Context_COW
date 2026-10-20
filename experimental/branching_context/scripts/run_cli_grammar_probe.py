"""Drive the named-GBNF grammar workflow through cow_cli and the real HTTP service.

This is a bounded acceptance probe. It never writes Store history directly;
Store access is read-only and only binds CLI receipts to persisted snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit

READ_TEXT = '{"tool":"read_file","path":"src/main.py"}'
TEST_TEXT = '{"tool":"run_tests","suite":"unit"}'


def grammar_registry_payload() -> dict:
    return {"schema": 1, "states": {
        "READ": {"format": "gbnf", "source": "root ::= " + json.dumps(READ_TEXT, ensure_ascii=True)},
        "TEST": {"format": "gbnf", "source": "root ::= " + json.dumps(TEST_TEXT, ensure_ascii=True)},
    }}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact_argv(argv, secret: str) -> list[str]:
    result = []
    redact_next = False
    for value in argv:
        value = str(value)
        if redact_next:
            result.append("<REDACTED>")
            redact_next = False
        elif value in ("--api-key", "--authorization"):
            result.append(value)
            redact_next = True
        elif secret and secret in value:
            result.append(value.replace(secret, "<REDACTED>"))
        else:
            result.append(value)
    return result


def redact_tree(value, secret: str):
    if isinstance(value, dict):
        return {str(key).replace(secret, "<REDACTED>") if secret else str(key):
                redact_tree(child, secret) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_tree(child, secret) for child in value]
    if isinstance(value, tuple):
        return [redact_tree(child, secret) for child in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<REDACTED>")
    return value


def parse_durable_receipt(stderr: bytes) -> dict:
    marker = b"durable receipt:\n"
    if stderr.count(marker) != 1:
        raise ValueError("CLI stderr must contain exactly one durable receipt marker")
    try:
        value = json.loads(stderr.split(marker, 1)[1].strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CLI durable receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("CLI durable receipt must be a JSON object")
    return value


def strip_cli_added_newline(stdout: str) -> str:
    """Remove exactly the one LF cow_cli appends after each streamed answer."""
    if not isinstance(stdout, str) or not stdout.endswith("\n"):
        raise ValueError("CLI answer stdout must end with its framing newline")
    return stdout[:-1]


def join_cli_stdout(*fragments: str) -> str:
    """Join streamed CLI calls without changing model-provided whitespace."""
    if not fragments:
        return ""
    return "".join(strip_cli_added_newline(fragment) for fragment in fragments)


def output_ends_with_eos(token_ids, eos_tokens) -> bool:
    """Return whether the last output token is any configured EOS token."""
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        return False
    if not isinstance(eos_tokens, (list, tuple)) or not eos_tokens:
        raise ValueError("profile must configure at least one EOS token")
    if any(type(token) is not int for token in eos_tokens):
        raise ValueError("configured EOS token IDs must be exact integers")
    return type(token_ids[-1]) is int and token_ids[-1] in eos_tokens


def controller_binding(snapshot: dict, state: str, expected_status: str) -> dict:
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot is not an object")
    controller = snapshot.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("snapshot omitted persisted controller")
    if controller.get("status") != expected_status:
        raise ValueError(f"controller status is not {expected_status}")
    constraint = controller.get("constraint")
    if not isinstance(constraint, dict) or constraint.get("agent_state") != state:
        raise ValueError(f"persisted grammar state is not {state}")
    if not isinstance(constraint.get("engine_id"), str) or len(constraint["engine_id"]) != 64:
        raise ValueError("persisted grammar engine identity is malformed")
    accepted = controller.get("accepted_in_scope")
    if type(accepted) is not int or accepted < 1:
        raise ValueError("persisted parser progress is missing")
    return controller


def _atomic_json(path: Path, value, secret: str = "") -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    clean = redact_tree(value, secret)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean, stream, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class GrammarProbe:
    def __init__(self, args, *, _test_dependencies=None):
        self.args = args
        self._test_dependencies = _test_dependencies
        self.run_dir = args.run_dir.absolute()
        self.runtime = args.runtime_root.absolute()
        self.profile_path = args.profile.absolute()
        self.store_path = args.store.absolute()
        self.secret = (_test_dependencies.get("api_key") if _test_dependencies is not None else None) or secrets.token_urlsafe(32)
        self.run_id = uuid.uuid4().hex[:12]
        self.ordinal = 0
        self.call_id = 0
        self.service = None
        self.store = None
        self.tracker = None
        self.record = {"schema": "cow-cli-grammar-probe-v1", "status": "RUNNING",
                       "run_id": self.run_id, "commands": [], "services": [],
                       "grammar_states": {}, "failures": [],
                       "numerical_qualification": "PENDING_INDEPENDENT_REFERENCE"}

    def publish(self):
        _atomic_json(self.run_dir / "result.json", self.record, self.secret)

    def phase(self, name):
        self.record["phase"] = name
        if self.tracker:
            self.tracker.phase(name)
        self.publish()

    def _imports(self):
        if self._test_dependencies is not None:
            deps = self._test_dependencies
            self.runtime_closure = deps["runtime_closure"]
            self.ProgressTracker = deps["ProgressTracker"]
            self.ServiceProcess = deps["ServiceProcess"]
            self.ready_url = deps["ready_url"]
            self.trace_worker = deps.get("trace_worker")
            self.Store = deps["Store"]
            self.record["execution_kind"] = "NON_NATIVE_CPU_FLOWMODEL_FIXTURE"
            self.record["native_worker_shutdown"] = "NOT_APPLICABLE"
            return
        extra = [str(self.args.dependency_root.resolve())] if self.args.dependency_root else []
        sys.path[:0] = extra + [str(self.runtime), str(self.runtime / "tools"),
                        str(self.runtime / "experimental/branching_context/src"),
                        str(self.runtime / "experimental/branching_context/scripts")]
        from run_q4_publication_fault import _runtime_closure
        from run_cow_client_recovery import ProgressTracker, ServiceProcess, _ready_url, _trace_worker
        from context_time_travel.store import Store
        self.runtime_closure = _runtime_closure
        self.ProgressTracker = ProgressTracker
        self.ServiceProcess = ServiceProcess
        self.ready_url = _ready_url
        self.trace_worker = _trace_worker
        self.Store = Store

    def _bootstrap(self, registry_path: Path) -> Path:
        path = self.run_dir / f"server_bootstrap-{self.ordinal}.py"
        body = (
            "import hashlib,importlib,json,sys\n"
            "from pathlib import Path\n"
            "profile=json.loads(Path(sys.argv[1]).read_bytes())\n"
            "runtime=Path(sys.argv[2]).resolve()\n"
            "registry=Path(sys.argv[3]).resolve()\n"
            "dependency=Path(sys.argv[4]).resolve() if sys.argv[4] else None\n"
            "sys.path[:0]=([str(dependency)] if dependency else [])+[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src')]\n"
            "rows={}\n"
            "for name in ('context_time_travel.coordinator','context_time_travel.strata_backend','context_time_travel.real','context_time_travel.store','context_time_travel.grammar_first','serve.cow_grammar','serve.cow_http'):\n"
            " m=importlib.import_module(name);p=Path(m.__file__).resolve();rows[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}\n"
            "print(json.dumps({'event':'runtime_imports','modules':rows}),flush=True)\n"
            "sys.argv=['cow-http',*sys.argv[5:]]\n"
            "importlib.import_module('serve.cow_http').main()\n"
        )
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(body)
        return path

    def start_service(self, registry_path: Path):
        self.ordinal += 1
        self.phase(f"service-start-{self.ordinal}")
        trace = self.run_dir / f"native-{self.ordinal}.jsonl"
        trace.touch(mode=0o600, exist_ok=False)
        pool_trace = self.run_dir / f"pool-{self.ordinal}.json"
        bootstrap = self._bootstrap(registry_path)
        env = os.environ.copy()
        extra = [str(self.args.dependency_root.resolve())] if self.args.dependency_root else []
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "CTT_NATIVE_TRACE": str(trace),
                    "PYTHONPATH": os.pathsep.join(extra + [str(self.runtime), str(self.runtime / "tools"),
                    str(self.runtime / "experimental/branching_context/src")])})
        python = os.environ.get("COW_PYTHON", sys.executable)
        dependency = str(self.args.dependency_root.resolve()) if self.args.dependency_root else ""
        command = [python, "-B", str(bootstrap), str(self.profile_path), str(self.runtime),
                   str(registry_path), dependency, "--profile", str(self.profile_path), "--store", str(self.store_path),
                   "--host", "127.0.0.1", "--port", "0", "--text", "--grammar-registry",
                   str(registry_path), "--api-key=" + self.secret, "--ram-gib", str(self.args.ram_gib),
                   "--disk-gib", str(self.args.disk_gib), "--pool-trace-output", str(pool_trace)]
        self.service = self.ServiceProcess(command, cwd=self.runtime, env=env,
            stdout_path=self.run_dir / f"server-{self.ordinal}.stdout",
            stderr_path=self.run_dir / f"server-{self.ordinal}.stderr", timeout=180,
            progress=self.tracker)
        started = time.monotonic()
        ready = self.service.start()
        self.store = self.Store(self.store_path)
        if self._test_dependencies is not None:
            imports = [self._test_dependencies["source_imports"]]
        else:
            stdout = (self.run_dir / f"server-{self.ordinal}.stdout").read_text(encoding="utf-8")
            rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
            imports = [row.get("modules") for row in rows if row.get("event") == "runtime_imports"]
            if len(imports) != 1:
                raise RuntimeError("service did not report its source import closure")
        self.record["services"].append({"ordinal": self.ordinal, "pid": self.service.process.pid,
            "ready": ready, "startup_seconds": time.monotonic() - started,
            "argv_redacted": redact_argv(command, self.secret), "source_imports": imports[0],
            "native_trace": str(trace), "pool_trace": str(pool_trace),
            "grammar_registry": str(registry_path), "grammar_registry_sha256": sha256_file(registry_path)})
        self.publish()

    def _trace_bytes(self):
        path = self.run_dir / f"native-{self.ordinal}.jsonl"
        return path.read_bytes()

    def cli(self, label, *args, expect=0, receipt=False):
        self.call_id += 1
        self.phase(label)
        client_dir = self.run_dir / "cli-state"
        client_dir.mkdir(mode=0o700, exist_ok=True)
        command = [os.environ.get("COW_PYTHON", sys.executable), "-B",
                   str(self.runtime / "tools/cow_cli.py"), "--url", self.ready_url(self.service.ready),
                   "--state-dir", str(client_dir), "--timeout", "180", *args]
        env = os.environ.copy()
        extra = [str(self.args.dependency_root.resolve())] if self.args.dependency_root else []
        env.update({"COW_API_KEY": self.secret, "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join(extra + [str(self.runtime), str(self.runtime / "tools"),
                    str(self.runtime / "experimental/branching_context/src")])})
        before = self._trace_bytes()
        start = time.monotonic()
        completed = subprocess.run(command, cwd=self.runtime, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=240, check=False)
        elapsed = time.monotonic() - start
        label_path = f"{self.call_id:03d}-{label}"
        stdout_path = self.run_dir / f"{label_path}.stdout"
        stderr_path = self.run_dir / f"{label_path}.stderr"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        after = self._trace_bytes()
        row = {"label": label, "argv_redacted": redact_argv(command, self.secret),
               "exit_code": completed.returncode, "expected_exit_code": expect,
               "elapsed_seconds": elapsed, "stdout": str(stdout_path), "stderr": str(stderr_path),
               "stdout_sha256": sha256_bytes(completed.stdout), "stderr_sha256": sha256_bytes(completed.stderr),
               "trace_bytes_before": len(before), "trace_bytes_after": len(after),
               "trace_delta_sha256": sha256_bytes(after[len(before):])}
        self.record["commands"].append(row)
        self.publish()
        if completed.returncode != expect:
            raise RuntimeError(f"CLI {label} returned {completed.returncode}, expected {expect}")
        if expect:
            content = (completed.stdout + b"\n" + completed.stderr).decode("utf-8", errors="replace").casefold()
            if "grammar" not in content and "state" not in content:
                raise RuntimeError("state-switch rejection did not identify grammar/state conflict")
            row["error_text_sha256"] = sha256_bytes(content.encode("utf-8"))
            row["native_trace_unchanged"] = before == after
            self.publish()
            return row, None
        row["stdout_text"] = completed.stdout.decode("utf-8", errors="strict")
        value = parse_durable_receipt(completed.stderr) if receipt else None
        if receipt:
            row["receipt"] = value
        self.publish()
        return row, value

    @staticmethod
    def _head_map(head):
        return {"name": head.name, "revision": head.revision, "tip": head.tip, "root": head.root}

    def _assert_saved(self, snapshot_id, state, expected_status):
        value = self.store.get_snapshot(snapshot_id)
        controller = controller_binding(value, state, expected_status)
        return value, controller

    def _output_tokens(self, operation, snapshot_id):
        envelope = self.store.presentation_receipt(operation)
        presentation = envelope.get("presentation") if isinstance(envelope, dict) else None
        span = presentation.get("output_span") if isinstance(presentation, dict) else None
        if not isinstance(span, dict):
            raise RuntimeError("durable CLI operation lacks an output span")
        start, count = span.get("start"), span.get("count")
        if type(start) is not int or type(count) is not int or start < 0 or count < 0:
            raise RuntimeError("durable output span is malformed")
        root, lineage = self.store.lineage(snapshot_id)
        ids = lineage[start:start + count]
        if len(ids) != count:
            raise RuntimeError("durable output span exceeds canonical inputs")
        return ids

    def stop_service(self):
        if self.service is None:
            return
        self.phase(f"service-stop-{self.ordinal}")
        pid = self.service.process.pid
        receipt = self.service.stop()
        if type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0:
            raise RuntimeError("HTTP service shutdown lacks numeric zero")
        self.record["services"][-1]["stop"] = redact_tree(receipt, self.secret)
        if self._test_dependencies is None:
            worker = self.trace_worker(self.run_dir / f"pool-{self.ordinal}.json", pid)
            if type(worker.get("native_exit_code")) is not int or worker["native_exit_code"] != 0:
                raise RuntimeError("native worker shutdown lacks numeric zero")
            self.record["services"][-1]["worker_shutdown"] = worker
        else:
            self.record["services"][-1]["worker_shutdown"] = "NOT_APPLICABLE_CPU_FIXTURE"
        self.service = None
        self.publish()

    def execute(self):
        if self.run_dir.exists():
            raise SystemExit("run directory exists; use a unique immutable run path")
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.record.update({"runtime": str(self.runtime), "profile": str(self.profile_path),
            "store": str(self.store_path), "question": self.args.question,
            "question_sha256": sha256_bytes(self.args.question.encode("utf-8")),
            "max_continue": self.args.max_continue,
            "state_outputs": {"READ": READ_TEXT, "TEST": TEST_TEXT}})
        self.publish()
        error = None
        try:
            self.runtime = self.runtime.resolve(strict=True)
            self.profile_path = self.profile_path.resolve(strict=True)
            if not (self.runtime / "tools/cow_cli.py").is_file():
                raise RuntimeError("runtime does not contain cow_cli.py")
            if self.store_path.exists() and any(self.store_path.iterdir()):
                raise RuntimeError("store path must be absent or empty")
            self._imports()
            profile = json.loads(self.profile_path.read_bytes())
            self.runtime_closure(profile, self.runtime)
            registry = grammar_registry_payload()
            registry_path = self.run_dir / "grammar-registry.json"
            _atomic_json(registry_path, registry)
            self.record["registry"] = {"path": str(registry_path), "sha256": sha256_file(registry_path),
                "states": {name: {"format": item["format"], "source_sha256": sha256_bytes(item["source"].encode("utf-8")),
                                  "expected_text": READ_TEXT if name == "READ" else TEST_TEXT}
                           for name, item in registry["states"].items()}}
            self.record["profile_identity"] = {key: profile.get(key) for key in
                ("execution_profile_id", "model_contract_id", "model_weights_sha256")}
            self.record["runtime_pins"] = {"cow_cli_sha256": sha256_file(self.runtime / "tools/cow_cli.py"),
                "cow_http_sha256": sha256_file(self.runtime / "serve/cow_http.py"),
                "grammar_first_sha256": sha256_file(self.runtime / "experimental/branching_context/src/context_time_travel/grammar_first.py")}
            self.publish()
        except BaseException as exc:
            self.record["status"] = "PREFLIGHT_FAIL"
            self.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
            self.publish()
            return 1
        (self.run_dir / "calls").mkdir(mode=0o700)
        self.tracker = self.ProgressTracker(self.run_dir, "cli-grammar-probe").start()
        try:
            self.start_service(registry_path)
            suffix = self.run_id
            main = "main"
            ns_row, ns = self.cli("namespace", "namespace", "--title", "GBNF probe " + suffix,
                "--operation-id", "grammar-ns-" + suffix, receipt=True)
            ns_id = ns.get("namespace_id") or ns.get("result", {}).get("namespace_id")
            if not isinstance(ns_id, str):
                raise RuntimeError("namespace receipt omitted namespace_id")
            prep_row, prep = self.cli("prepare", "prepare", "--namespace-id", ns_id,
                "--branch", main, "--text", self.args.question,
                "--operation-id", "grammar-root-" + suffix, receipt=True)
            root = self.store.head(main)
            if prep.get("snapshot") != root.tip or root.tip != root.root:
                raise RuntimeError("prepared CLI receipt is not the canonical root")
            root_snapshot = root.tip
            root_head = self._head_map(root)
            self.record["root"] = {"snapshot": root_snapshot, "head": root_head,
                "namespace_id": ns_id, "prepare_operation": "grammar-root-" + suffix,
                "receipt": prep}
            self.publish()
            branch_a, branch_b = "grammar-read-" + suffix, "grammar-test-" + suffix
            for label, branch in (("fork-read", branch_a), ("fork-test", branch_b)):
                self.cli(label, "fork", "--source-branch", main, "--name", branch,
                         "--operation-id", label + "-" + suffix, receipt=True)
            if self.store.head(main).tip != root_snapshot:
                raise RuntimeError("fork changed the source branch head")
            if self.store.head(branch_a).tip != root_snapshot or self.store.head(branch_b).tip != root_snapshot:
                raise RuntimeError("forks do not share the prepared root")

            first_row, first_receipt = self.cli("read-partial", "answer", "--branch", branch_a,
                "--max-tokens", "1", "--operation-id", "grammar-read-first-" + suffix,
                "--grammar-state", "READ", receipt=True)
            if not first_row["stdout_text"]:
                raise RuntimeError("READ partial produced no CLI text")
            first_snapshot = first_receipt.get("snapshot")
            first_value, first_controller = self._assert_saved(first_snapshot, "READ", "active")
            if first_controller.get("accepted_in_scope") != 1:
                raise RuntimeError("READ one-token partial did not persist exactly one parser step")
            first_head = self.store.head(branch_a)
            if first_head.tip != first_snapshot:
                raise RuntimeError("READ partial receipt did not move branch A")
            self.record["read_partial"] = {"text": first_row["stdout_text"], "receipt": first_receipt,
                "head": self._head_map(first_head), "controller": first_controller}
            self.publish()

            before_head = self.store.head(branch_a)
            before_trace = self._trace_bytes()
            switched, _ = self.cli("reject-state-switch", "continue", "--branch", branch_a,
                "--max-tokens", str(self.args.max_continue), "--operation-id", "grammar-invalid-switch-" + suffix,
                "--grammar-state", "TEST", expect=2)
            after_head = self.store.head(branch_a)
            if self._head_map(before_head) != self._head_map(after_head):
                raise RuntimeError("rejected grammar-state switch changed branch head")
            if before_trace != self._trace_bytes() or not switched.get("native_trace_unchanged"):
                raise RuntimeError("rejected grammar-state switch reached native worker")
            self.record["state_switch_rejection"] = {"status": "PASS_NO_WORK", "command": switched}
            self.publish()

            finish_row, finish_receipt = self.cli("read-finish", "continue", "--branch", branch_a,
                "--max-tokens", str(self.args.max_continue), "--operation-id", "grammar-read-finish-" + suffix,
                receipt=True)
            read_text = join_cli_stdout(first_row["stdout_text"], finish_row["stdout_text"])
            if read_text != READ_TEXT:
                raise RuntimeError("READ CLI output differed from the exact registry JSON")
            final_read, read_controller = self._assert_saved(finish_receipt.get("snapshot"), "READ", "stopped")
            read_ids_first = self._output_tokens("grammar-read-first-" + suffix, first_snapshot)
            read_ids_finish = self._output_tokens("grammar-read-finish-" + suffix, finish_receipt["snapshot"])
            if len(read_ids_first) != 1 or not output_ends_with_eos(read_ids_finish, profile["execution"]["eos_tokens"]):
                raise RuntimeError("READ durable output spans do not show one partial token followed by EOS")
            if self.store.head(branch_a).tip != finish_receipt.get("snapshot"):
                raise RuntimeError("READ final durable receipt does not match branch A head")
            self.record["read_complete"] = {"combined_text": read_text,
                "partial_output_ids": read_ids_first, "finish_output_ids": read_ids_finish,
                "receipt": finish_receipt, "controller": read_controller,
                "status": read_controller["status"], "accepted_in_scope": read_controller["accepted_in_scope"]}
            self.publish()

            test_row, test_receipt = self.cli("test-answer", "answer", "--branch", branch_b,
                "--max-tokens", str(self.args.max_continue), "--operation-id", "grammar-test-answer-" + suffix,
                "--grammar-state", "TEST", receipt=True)
            test_text = strip_cli_added_newline(test_row["stdout_text"])
            if test_text != TEST_TEXT:
                raise RuntimeError("TEST CLI output differed from the exact registry JSON")
            _, test_controller = self._assert_saved(test_receipt.get("snapshot"), "TEST", "stopped")
            test_ids = self._output_tokens("grammar-test-answer-" + suffix, test_receipt["snapshot"])
            if not output_ends_with_eos(test_ids, profile["execution"]["eos_tokens"]):
                raise RuntimeError("TEST durable answer did not consume EOS")
            if self.store.head(branch_b).tip != test_receipt.get("snapshot"):
                raise RuntimeError("TEST receipt does not match branch B head")
            if self.store.head(main).tip != root_snapshot or self.store.head(branch_a).tip != finish_receipt["snapshot"]:
                raise RuntimeError("sibling branch or canonical source changed during TEST")
            self.record["test_complete"] = {"text": test_text, "output_ids": test_ids,
                "receipt": test_receipt, "controller": test_controller,
                "status": test_controller["status"], "accepted_in_scope": test_controller["accepted_in_scope"]}
            self.record["status"] = "CLI_GRAMMAR_FLOW_PASS_NUMERICAL_AUDIT_PENDING"
            self.record["completed_workflow"] = ["namespace", "prepare", "fork-read", "fork-test",
                "READ partial", "reject active state switch before native work", "READ continuation to EOS",
                "TEST generation to EOS", "canonical head and sibling preservation"]
        except BaseException as exc:
            error = exc
            self.record["status"] = "FAIL"
            self.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        finally:
            if self.service is not None:
                try:
                    self.stop_service()
                except BaseException as exc:
                    error = error or exc
                    self.record["status"] = "FAIL"
                    self.record["failures"].append({"type": type(exc).__name__, "message": "cleanup: " + str(exc)})
            if self.tracker:
                self.tracker.finish(self.record.get("status", "FAIL"), str(error) if error else None)
                self.tracker.close()
            self.publish()
        return 1 if error else 0


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--max-continue", type=int, default=64)
    parser.add_argument("--ram-gib", type=int, default=96)
    parser.add_argument("--disk-gib", type=int, default=128)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if not 1 <= args.max_continue <= 96 or not 16 <= args.ram_gib <= 128 or not 16 <= args.disk_gib <= 512:
        raise SystemExit("bounded options: max-continue 1..96; RAM 16..128 GiB; disk 16..512 GiB")
    return GrammarProbe(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
