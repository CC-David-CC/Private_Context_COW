"""Exercise a complete user workflow through the COW CLI and record evidence.

The harness is intentionally a client of the existing HTTP service. It does not
write canonical history directly; Store reads are used only for observation.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import secrets
from pathlib import Path
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit


def canonical_sha256(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_evaluation_record(envelope, expected_snapshot: str, mandatory=()) -> dict:
    if not isinstance(envelope, dict):
        raise ValueError("evaluation envelope must be an object")
    record = envelope.get("record")
    if not isinstance(record, dict):
        raise ValueError("evaluation record is missing")
    record_id = envelope.get("evaluation_id")
    if not isinstance(record_id, str) or record_id != canonical_sha256(record):
        raise ValueError("evaluation record content hash mismatch")
    if record.get("kind") != "evaluation" or record.get("schema") != 1:
        raise ValueError("unsupported evaluation record kind/schema")
    if not isinstance(expected_snapshot, str) or not expected_snapshot:
        raise ValueError("expected evaluation snapshot is missing")
    if record.get("snapshot_id") != expected_snapshot:
        raise ValueError("evaluation snapshot binding mismatch")
    outcomes = record.get("outcomes")
    checks = outcomes.get("mandatory") if isinstance(outcomes, dict) else None
    if not isinstance(checks, dict):
        raise ValueError("evaluation record lacks mandatory outcomes")
    if any(checks.get(name) != "PASS" for name in mandatory):
        raise ValueError("evaluation mandatory check did not pass")
    if outcomes.get("unresolved"):
        raise ValueError("evaluation contains unresolved outcomes")
    return record


def validate_adoption_ack(ack, *, operation_id: str, evaluation_id: str,
                          snapshot_id: str, branch: str,
                          expected_revision: int, root: str) -> None:
    if not isinstance(ack, dict):
        raise ValueError("adoption acknowledgement must be an object")
    if ack.get("operation_id") != operation_id:
        raise ValueError("adoption operation binding mismatch")
    if ack.get("evaluation_id") != evaluation_id:
        raise ValueError("adoption evaluation binding mismatch")
    if ack.get("snapshot_id") != snapshot_id:
        raise ValueError("adoption snapshot binding mismatch")
    head = ack.get("branch")
    if not isinstance(head, dict) or head.get("name") != branch:
        raise ValueError("adoption branch binding mismatch")
    revision = head.get("revision")
    if type(revision) is not int or revision != expected_revision + 1:
        raise ValueError("adoption revision is not the expected CAS successor")
    if head.get("tip") != snapshot_id or head.get("root") != root:
        raise ValueError("adoption head tip/root binding mismatch")


def validate_zero_work(before, after) -> None:
    """Require exact equality for all work/resource counters present on both sides."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("counter samples must be objects")
    keys = ("forward_tokens", "prefill_tokens", "disk_loads", "native_calls",
            "loaded_bytes", "sampling_calls")
    required = ("forward_tokens", "prefill_tokens")
    missing = [key for key in required if key not in before or key not in after]
    if missing:
        raise ValueError("missing required work counters: " + ", ".join(missing))
    compared = 0
    for key in keys:
        if key not in before or key not in after:
            continue
        left, right = before[key], after[key]
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError(f"counter {key} is not numeric")
        if not (math.isfinite(left) and math.isfinite(right)):
            raise ValueError(f"counter {key} is not finite")
        if left != right:
            raise ValueError(f"counter {key} changed: {left!r} -> {right!r}")
        compared += 1
    if not compared:
        raise ValueError("no recognized counters available for zero-work check")


class WorkflowLedger:
    """Fail-closed phase ledger; an unrecorded phase can never be called PASS."""

    def __init__(self, required):
        if not required or len(set(required)) != len(required):
            raise ValueError("workflow phases must be nonempty and unique")
        self.required = tuple(required)
        self.completed = {}

    def complete(self, name, receipt):
        if name not in self.required:
            raise ValueError(f"unknown workflow phase: {name}")
        if name in self.completed:
            raise ValueError(f"phase already completed: {name}")
        if not isinstance(receipt, dict) or type(receipt.get("exit_code")) is not int:
            raise ValueError("phase receipt needs an exact integer exit code")
        if receipt["exit_code"] != 0:
            raise ValueError(f"phase exit code is not zero: {name}")
        if receipt.get("status") in ("FAIL", "FAILED", "ERROR"):
            raise ValueError(f"phase terminal status is failure: {name}")
        self.completed[name] = receipt

    def finish(self):
        missing = [name for name in self.required if name not in self.completed]
        if missing:
            raise ValueError("incomplete workflow: " + ", ".join(missing))
        return {"status": "PASS", "completed": list(self.required)}

    def reject(self, name, receipt):
        """Record a required negative control only when it really failed."""
        if name not in self.required or name in self.completed:
            raise ValueError(f"invalid rejection phase: {name}")
        if not isinstance(receipt, dict) or type(receipt.get("exit_code")) is not int:
            raise ValueError("rejection receipt needs an exact integer exit code")
        if receipt["exit_code"] == 0:
            raise ValueError(f"negative control unexpectedly succeeded: {name}")
        self.completed[name] = {**receipt, "expected_rejection": True}


def validate_retry_result(response, *, original_operation: str, expected_branch: str,
                          expected_snapshot: str, expected_tokens, replay: bool,
                          new_operation: str | None = None) -> dict:
    """Validate the stable completion fields shared by retry modes.

    This function intentionally requires explicit replay/new-operation identity;
    callers must not infer success from printed text alone.
    """
    if not isinstance(response, dict) or response.get("event") != "complete":
        raise ValueError("retry did not return a complete event")
    if response.get("replayed") is not replay:
        raise ValueError("retry replay marker mismatch")
    answer = response.get("answer")
    receipt = response.get("receipt")
    if not isinstance(answer, dict) or not isinstance(receipt, dict):
        raise ValueError("retry answer/receipt is missing")
    if answer.get("snapshot") != expected_snapshot or receipt.get("snapshot") != expected_snapshot:
        raise ValueError("retry snapshot binding mismatch")
    if answer.get("tokens") != list(expected_tokens):
        raise ValueError("retry token sequence mismatch")
    head = receipt.get("branch")
    if not isinstance(head, dict) or head.get("name") != expected_branch:
        raise ValueError("retry branch binding mismatch")
    expected_id = original_operation if replay else new_operation
    # Some receipt schema versions bind operation identity in the completion
    # event/journal rather than duplicating it into the branch receipt.
    if "operation_id" in receipt and receipt.get("operation_id") != expected_id:
        raise ValueError("retry receipt operation binding mismatch")
    return response


def validate_no_native_rpc_delta(trace_path: Path, start_offset: int, end_offset: int) -> None:
    """A replay/rejection may not issue any native RPC after its request starts."""
    if type(start_offset) is not int or type(end_offset) is not int or start_offset < 0 or end_offset < start_offset:
        raise ValueError("invalid native trace offsets")
    with Path(trace_path).open("rb") as stream:
        stream.seek(start_offset)
        delta = stream.read(end_offset - start_offset)
    if delta:
        raise ValueError("native trace contains RPC records during zero-work action")


def redact_argv(argv, secrets_to_redact=()) -> list[str]:
    secrets_set = {value for value in secrets_to_redact if isinstance(value, str) and value}
    redacted = []
    redact_next = False
    for item in argv:
        text = str(item)
        if redact_next:
            redacted.append("<REDACTED>")
            redact_next = False
        elif text in ("--api-key", "--authorization"):
            redacted.append(text)
            redact_next = True
        elif text in secrets_set:
            redacted.append("<REDACTED>")
        else:
            redacted.append(text)
    return redacted


def redact_sensitive_value(value, secrets_to_redact=()):
    """Return a recursively detached JSON-safe value with secrets removed."""
    secrets_set = tuple(secret for secret in secrets_to_redact
                        if isinstance(secret, str) and secret)
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            clean_key = key
            if isinstance(key, str):
                for secret in secrets_set:
                    clean_key = clean_key.replace(secret, "<REDACTED>")
            result[clean_key] = redact_sensitive_value(child, secrets_set)
        return result
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            value = redact_argv(value, secrets_set)
        return [redact_sensitive_value(item, secrets_set) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_value(item, secrets_set) for item in value]
    if isinstance(value, str):
        for secret in secrets_set:
            value = value.replace(secret, "<REDACTED>")
    return value


def parse_durable_receipt(stderr: bytes) -> dict:
    marker = b"durable receipt:\n"
    if stderr.count(marker) != 1:
        raise ValueError("CLI stderr must contain exactly one durable receipt marker")
    tail = stderr.split(marker, 1)[1].strip()
    try:
        receipt = json.loads(tail)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CLI durable receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("CLI durable receipt must be an object")
    return receipt


def validate_rejection(stdout: bytes, stderr: bytes, exit_code: int, *, reason_terms) -> None:
    if type(exit_code) is not int or exit_code == 0:
        raise ValueError("negative control did not fail with a numeric nonzero exit")
    if b"durable receipt:\n" in stderr:
        raise ValueError("rejected command emitted a durable receipt")
    text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace").casefold()
    if not any(term.casefold() in text for term in reason_terms):
        raise ValueError("rejection did not identify the expected conflict")


def _json_output(stdout: bytes, description: str) -> dict:
    try:
        value = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} stdout is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} stdout must be an object")
    return value


def _atomic_write(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _Campaign:
    """Own one HTTP process and run each workflow mutation through cow_cli."""

    def __init__(self, args):
        self.args = args
        self.run_dir = args.run_dir.absolute()
        self.run_id = uuid.uuid4().hex[:12]
        self.api_key = secrets.token_urlsafe(32)
        self.runtime = args.runtime_root.absolute()
        self.profile_path = args.profile.absolute()
        self.profile = None
        self.cli_path = self.runtime / "tools" / "cow_cli.py"
        self.result_path = self.run_dir / "result.json"
        self.service = None
        self.client = None
        self.tracker = None
        self.store = None
        self.ordinal = 0
        self.call_number = 0
        self.record = {}
        self._store_module = None

    def publish(self):
        _atomic_write(self.result_path, redact_sensitive_value(self.record, (self.api_key,)))

    def phase(self, name):
        if self.tracker:
            self.tracker.phase(name)
        self.record["phase"] = name
        self.publish()

    def _runtime_imports(self):
        sys.path[:0] = [str(self.runtime), str(self.runtime / "tools"),
                        str(self.runtime / "experimental/branching_context/src"),
                        str(self.runtime / "experimental/branching_context/scripts")]
        from run_q4_publication_fault import _runtime_closure
        from context_time_travel.store import Store
        from run_cow_client_recovery import (HttpClient, ProgressTracker, ServiceProcess,
                                             _ready_url, _trace_worker)
        self._runtime_closure = _runtime_closure
        self._Store = Store
        self._HttpClient = HttpClient
        self._ProgressTracker = ProgressTracker
        self._ServiceProcess = ServiceProcess
        self._ready_url = _ready_url
        self._trace_worker = _trace_worker

    def _bootstrap(self, ordinal):
        bootstrap = self.run_dir / f"server_bootstrap-{ordinal}.py"
        text = (
            "import hashlib,importlib,json,sys\n"
            "from pathlib import Path\n"
            "profile=json.loads(Path(sys.argv[1]).read_bytes())\n"
            "runtime=Path(sys.argv[2]).resolve()\n"
            "sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src')]\n"
            "rows={}\n"
            "for name in ('context_time_travel.coordinator','context_time_travel.strata_backend','context_time_travel.real','context_time_travel.store','serve.cow_http'):\n"
            " m=importlib.import_module(name);p=Path(m.__file__).resolve();rows[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}\n"
            "print(json.dumps({'event':'runtime_imports','modules':rows}),flush=True)\n"
            "sys.argv=['cow-http',*sys.argv[3:]]\n"
            "importlib.import_module('serve.cow_http').main()\n"
        )
        with bootstrap.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return bootstrap

    def start_service(self):
        self.ordinal += 1
        self.phase(f"start-service-{self.ordinal}")
        profile = self.profile
        store_path = self.args.store.absolute()
        trace = self.run_dir / f"native-{self.ordinal}.jsonl"
        trace.touch(mode=0o600, exist_ok=False)
        pool_trace = self.run_dir / f"pool-{self.ordinal}.json"
        bootstrap = self._bootstrap(self.ordinal)
        env = os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "CTT_NATIVE_TRACE": str(trace),
                    "PYTHONPATH": os.pathsep.join((str(self.runtime), str(self.runtime / "tools"),
                    str(self.runtime / "experimental/branching_context/src")))})
        python = os.environ.get("COW_PYTHON", sys.executable)
        command = [python, "-B", str(bootstrap), str(self.profile_path), str(self.runtime),
                   "--profile", str(self.profile_path), "--store", str(store_path),
                   "--host", "127.0.0.1", "--port", "0", "--text", "--api-key", self.api_key,
                   "--ram-gib", "96", "--disk-gib", "128", "--pool-trace-output", str(pool_trace)]
        command_start = time.monotonic()
        self.service = self._ServiceProcess(command, cwd=self.runtime, env=env,
            stdout_path=self.run_dir / f"server-{self.ordinal}.stdout",
            stderr_path=self.run_dir / f"server-{self.ordinal}.stderr",
            timeout=180, progress=self.tracker)
        ready = self.service.start()
        self.client = self._HttpClient(self._ready_url(ready), timeout=180, progress=self.tracker)
        self.store = self._Store(store_path)
        rows = [json.loads(line) for line in (self.run_dir / f"server-{self.ordinal}.stdout").read_text(encoding="utf-8").splitlines() if line.startswith("{")]
        imports = [row.get("modules") for row in rows if row.get("event") == "runtime_imports"]
        if len(imports) != 1:
            raise RuntimeError("service did not report exactly one source import closure")
        self.record["services"].append({"ordinal": self.ordinal, "pid": self.service.process.pid,
            "ready": ready, "startup_seconds": time.monotonic() - command_start,
            "argv_redacted": redact_argv(command, (self.api_key,)),
            "source_imports": imports[0], "native_trace": str(trace), "pool_trace": str(pool_trace)})
        self.phase(f"service-{self.ordinal}-ready")

    def _status(self):
        parsed = urlsplit(self._ready_url(self.service.ready))
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
        try:
            connection.request("GET", "/cow/status", headers={"Authorization": "Bearer " + self.api_key})
            response = connection.getresponse()
            raw = response.read()
            code = response.status
        finally:
            connection.close()
        if code != 200:
            raise RuntimeError("COW status observation failed")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("COW status observation is not an object")
        return value

    @staticmethod
    def _counter_view(status):
        names = {"forward_tokens", "prefill_tokens", "disk_loads", "restore_loads",
                 "native_calls", "loaded_bytes", "sampling_calls", "capture_bytes",
                 "hash_bytes", "publication_count"}
        result = {}
        def walk(value, prefix=""):
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key in names and type(child) in (int, float):
                    result[path] = child
                elif isinstance(child, dict):
                    walk(child, path)
        walk(status)
        return result

    def _native_stats(self):
        path = self.run_dir / f"native-{self.ordinal}.jsonl"
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            if item.get("command") == "stats" and isinstance(item.get("response"), dict):
                values.append(item["response"])
        return values[-1] if values else {}

    def cli(self, label, client_name, *arguments, expect=0, want_receipt=False,
            output_json=False, state_suffix=None):
        self.call_number += 1
        self.phase(label)
        before_status = self._status()
        before_stats = self._native_stats()
        client_state = self.run_dir / f"journal-{client_name}{('-'+state_suffix) if state_suffix else ''}"
        client_state.mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [os.environ.get("COW_PYTHON", sys.executable), "-B", str(self.cli_path),
                   "--url", self._ready_url(self.service.ready), "--state-dir", str(client_state),
                   "--timeout", "180", *arguments]
        env = os.environ.copy()
        env.update({"COW_API_KEY": self.api_key, "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join((str(self.runtime), str(self.runtime / "tools"),
                    str(self.runtime / "experimental/branching_context/src")))})
        started = time.monotonic()
        completed = subprocess.run(command, cwd=self.runtime, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=240, check=False)
        elapsed = time.monotonic() - started
        name = f"{self.call_number:03d}-{label}"
        stdout_path = self.run_dir / f"{name}.stdout"
        stderr_path = self.run_dir / f"{name}.stderr"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        after_status = self._status()
        after_stats = self._native_stats()
        row = {"label": label, "client": client_name, "service": self.ordinal,
               "argv_redacted": redact_argv(command, (self.api_key,)),
               "exit_code": completed.returncode, "elapsed_seconds": elapsed,
               "stdout": str(stdout_path), "stderr": str(stderr_path),
               "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
               "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
               "counter_before": before_stats, "counter_after": after_stats,
               "status_before": before_status, "status_after": after_status,
               "expected_exit_code": expect}
        self.record["commands"].append(row)
        self.publish()
        if completed.returncode != expect:
            raise RuntimeError(f"CLI command {label} exited {completed.returncode}; expected {expect}")
        if expect != 0:
            return row, None
        if want_receipt:
            row["receipt"] = parse_durable_receipt(completed.stderr)
            self.publish()
        if output_json:
            row["json"] = _json_output(completed.stdout, label)
            self.publish()
        return row, row.get("json", row.get("receipt"))

    def stop_service(self):
        if self.service is None:
            return
        self.phase(f"stop-service-{self.ordinal}")
        pid = self.service.process.pid
        receipt = self.service.stop()
        if receipt.get("exit_code") != 0:
            raise RuntimeError("HTTP service did not exit numerically with zero")
        worker = self._trace_worker(self.run_dir / f"pool-{self.ordinal}.json", pid)
        if worker.get("native_exit_code") != 0:
            raise RuntimeError("native worker did not exit numerically with zero")
        service_record = self.record["services"][-1]
        service_record["stop"] = receipt
        service_record["worker_shutdown"] = worker
        self.service = None
        self.client = None
        self.publish()

    def execute(self):
        if self.run_dir.exists():
            raise SystemExit("run directory already exists; use a new immutable attempt path")
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.record = {"schema": "cow-cli-workflow-v1", "status": "RUNNING",
            "phase": "preflight", "run_id": self.run_id,
            "runtime": str(self.runtime), "profile": str(self.profile_path),
            "store": str(self.args.store.absolute()), "commands": [], "services": [],
            "workflow": [], "failures": [], "numerical_oracle": "PENDING_SEPARATE_FRESH_REFERENCE",
            "chat_server": "NOT_STARTED"}
        self.publish()
        try:
            self.runtime = self.args.runtime_root.resolve(strict=True)
            self.profile_path = self.args.profile.resolve(strict=True)
            self.profile = json.loads(self.profile_path.read_bytes())
            self.cli_path = self.runtime / "tools" / "cow_cli.py"
            if not self.cli_path.is_file():
                raise RuntimeError("runtime lacks tools/cow_cli.py")
            self._runtime_imports()
            self._runtime_closure(self.profile, self.runtime)
            if not self.args.evaluation_spec.is_file():
                raise RuntimeError("evaluation specification file is missing")
            fixture_names = set()
            for item in self.args.fixture:
                if "=" not in item:
                    raise RuntimeError("fixture must use NAME=PATH")
                name, path = item.split("=", 1)
                if not name or not Path(path).is_file():
                    raise RuntimeError("fixture path missing or name empty")
                if name in fixture_names:
                    raise RuntimeError("duplicate fixture name")
                fixture_names.add(name)
            if self.args.expected_fixture not in fixture_names:
                raise RuntimeError("expected fixture name was not supplied")
            if self.args.store.exists() and any(self.args.store.iterdir()):
                raise RuntimeError("store path must be a fresh empty campaign store")
        except BaseException as exc:
            self.record["status"] = "PREFLIGHT_FAIL"
            self.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
            self.publish()
            return 1
        (self.run_dir / "calls").mkdir(mode=0o700)
        self.tracker = self._ProgressTracker(self.run_dir, "cli-workflow").start()
        self.record.update({"schema": "cow-cli-workflow-v1", "status": "RUNNING",
            "phase": "preflight", "source": {"runtime": str(self.runtime),
            "cli_sha256": sha256_file(self.cli_path), "profile": str(self.profile_path),
            "profile_sha256": sha256_file(self.profile_path), "execution_profile_id": self.profile.get("execution_profile_id"),
            "model_contract_id": self.profile.get("model_contract_id"),
            "native_sha256": sha256_file(Path(self.profile["execution"]["sut_command"][0]))},
            "store": str(self.args.store.absolute()), "outputs": self.args.outputs,
            "commands": [], "services": [], "workflow": [], "failures": [],
            "fixture_hashes": {item.split("=", 1)[0]: sha256_file(Path(item.split("=", 1)[1]))
                               for item in self.args.fixture},
            "evaluation_spec_sha256": sha256_file(self.args.evaluation_spec),
            "numerical_oracle": "PENDING_SEPARATE_AUDIT", "chat_server": "NOT_STARTED"})
        self.publish()
        ledger = WorkflowLedger(("namespace", "prepare", "fork-a", "fork-b", "ask-a", "ask-b",
            "evaluate-a", "evaluate-b", "inspect-a", "inspect-b", "select", "adopt",
            "history", "continue"))
        error = None
        try:
            self.start_service()
            suffix = self.run_id
            branch_a, branch_b = "candidate-a-" + suffix, "candidate-b-" + suffix
            root_branch = "main"
            ns_row, ns_receipt = self.cli("namespace", "primary", "namespace", "--title",
                "CLI workflow " + suffix, "--operation-id", "ns-" + suffix, want_receipt=True)
            ns_id = ns_receipt.get("namespace_id")
            if not isinstance(ns_id, str):
                ns_id = ns_receipt.get("result", {}).get("namespace_id")
            if not isinstance(ns_id, str):
                raise RuntimeError("namespace receipt omitted namespace_id")
            ledger.complete("namespace", {"exit_code": 0})
            prepare_row, prepare_receipt = self.cli("prepare", "primary", "prepare",
                "--namespace-id", ns_id, "--branch", root_branch,
                "--text", self.args.question, "--operation-id", "prepare-" + suffix,
                want_receipt=True)
            root_head = self.store.head(root_branch)
            if prepare_receipt.get("snapshot") != root_head.tip or root_head.tip != root_head.root:
                raise RuntimeError("prepared context receipt is not the canonical root head")
            ledger.complete("prepare", {"exit_code": 0})
            source_snapshot = root_head.tip
            source_root = root_head.root

            for label, branch in (("fork-a", branch_a), ("fork-b", branch_b)):
                self.cli(label, "primary", "fork", "--source-branch", root_branch,
                    "--name", branch, "--operation-id", label + "-" + suffix,
                    want_receipt=True)
                ledger.complete(label, {"exit_code": 0})
            asks = {}
            for key, branch in (("a", branch_a), ("b", branch_b)):
                op = "alt-answer" if key == "a" else "resume-op"
                row, receipt = self.cli("answer-" + key, "primary", "answer", "--branch", branch,
                    "--max-tokens", str(self.args.outputs),
                    "--operation-id", op, want_receipt=True)
                snapshot = self.store.get_snapshot(receipt["snapshot"])
                presentation = self.store.presentation_receipt(op)["presentation"]
                output_span = presentation.get("output_span")
                root_id, all_inputs = self.store.lineage(receipt["snapshot"])
                if not isinstance(output_span, dict):
                    raise RuntimeError("ask presentation lacks its committed output span")
                start, count = output_span.get("start"), output_span.get("count")
                if type(start) is not int or type(count) is not int or start < 0 or count < 0:
                    raise RuntimeError("ask output span is malformed")
                output_tokens = all_inputs[start:start + count]
                if len(output_tokens) != count:
                    raise RuntimeError("ask output span exceeds canonical lineage")
                head = self.store.head(branch)
                if head.tip != receipt["snapshot"] or snapshot.get("parent") != source_snapshot:
                    raise RuntimeError(f"{key} answer receipt/source binding mismatch")
                if not Path(row["stdout"]).read_text(encoding="utf-8").strip():
                    raise RuntimeError(f"{key} answer emitted no CLI text")
                asks[key] = {"operation": op, "branch": branch, "source": source_snapshot,
                             "snapshot": receipt["snapshot"], "receipt": receipt,
                             "tokens": output_tokens, "output_span": output_span,
                             "root": root_id, "answer_stdout": row["stdout"]}
                self.record.setdefault("candidates", {})[key] = {
                    "operation_id": op, "branch": branch, "source_snapshot": source_snapshot,
                    "snapshot": receipt["snapshot"], "receipt": receipt,
                    "output_span": output_span, "output_token_ids": output_tokens,
                    "answer_stdout_sha256": row["stdout_sha256"],
                    "answer_stderr_sha256": row["stderr_sha256"]}
                self.publish()
                ledger.complete("ask-" + key, {"exit_code": 0})

            spec = json.loads(self.args.evaluation_spec.read_bytes())
            if spec.get("task_root") not in (None, source_snapshot):
                raise RuntimeError("evaluation template task_root must be empty or match prepared root")
            spec["task_root"] = source_snapshot
            effective_spec = self.run_dir / "effective-evaluation-spec.json"
            effective_spec.write_text(json.dumps(spec, ensure_ascii=False, sort_keys=True,
                indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            spec_path = str(effective_spec)
            self.record["effective_evaluation_spec"] = {
                "path": spec_path, "sha256": sha256_file(effective_spec),
                "task_root": source_snapshot, "template_sha256": sha256_file(self.args.evaluation_spec)}
            self.publish()
            fixture_args = []
            for fixture in self.args.fixture:
                name, path = fixture.split("=", 1)
                fixture_args.extend(("--fixture", f"{name}={Path(path).resolve()}"))
            evaluations = {}
            for key in ("a", "b"):
                row, envelope = self.cli("evaluate-" + key, "primary", "evaluate",
                    "--snapshot-id", asks[key]["snapshot"], "--spec", spec_path,
                    *fixture_args, "--operation-id", "eval-" + key + "-" + suffix,
                    output_json=True)
                record = validate_evaluation_record(envelope, asks[key]["snapshot"], self.args.mandatory)
                evaluations[key] = {"id": envelope["evaluation_id"], "record": record}
                self.record.setdefault("evaluations", {})[key] = {
                    "evaluation_id": envelope["evaluation_id"],
                    "snapshot": asks[key]["snapshot"], "record": record}
                self.publish()
                ledger.complete("evaluate-" + key, {"exit_code": 0})
            for key in ("a", "b"):
                row, inspected = self.cli("inspect-" + key, "primary", "inspect-evaluation",
                    "--evaluation-id", evaluations[key]["id"], output_json=True)
                inspected_record = validate_evaluation_record(inspected, asks[key]["snapshot"],
                    self.args.mandatory)
                if inspected_record != evaluations[key]["record"]:
                    raise RuntimeError("inspection returned a different immutable evaluation record")
                ledger.complete("inspect-" + key, {"exit_code": 0})
            mandatory = self.args.mandatory
            row, selection = self.cli("select", "primary", "select-evaluations",
                "--evaluation-ids", evaluations["a"]["id"], evaluations["b"]["id"],
                "--mandatory", *mandatory, output_json=True)
            selected_snapshot = selection.get("snapshot_id")
            selected_eval_id = selection.get("evaluation_id")
            if selected_eval_id not in {evaluations["a"]["id"], evaluations["b"]["id"]}:
                raise RuntimeError("selection chose an unsubmitted evaluation")
            selected_record = validate_evaluation_record(selection, selected_snapshot, mandatory)
            if selected_record != evaluations["a"]["record"] and selected_record != evaluations["b"]["record"]:
                raise RuntimeError("selection returned a record outside the submitted evaluations")
            self.record["selection"] = {"evaluation_id": selected_eval_id,
                "snapshot": selected_snapshot, "record": selected_record}
            self.publish()
            ledger.complete("select", {"exit_code": 0})
            main_before = self.store.head(root_branch)
            row, adoption = self.cli("adopt", "primary", "adopt-evaluation",
                "--branch", root_branch, "--revision", str(main_before.revision),
                "--evaluation-ids", evaluations["a"]["id"], evaluations["b"]["id"],
                "--mandatory", *mandatory, "--operation-id", "adopt-" + suffix,
                output_json=True)
            validate_adoption_ack(adoption, operation_id="adopt-" + suffix,
                evaluation_id=selected_eval_id, snapshot_id=selected_snapshot,
                branch=root_branch, expected_revision=main_before.revision, root=source_root)
            main_after = self.store.head(root_branch)
            if main_after.tip != selected_snapshot or main_after.revision != main_before.revision + 1:
                raise RuntimeError("adoption did not CAS-publish selected candidate")
            self.record["adoption"] = {"ack": adoption,
                "before": {"name": main_before.name, "revision": main_before.revision,
                           "tip": main_before.tip, "root": main_before.root},
                "after": {"name": main_after.name, "revision": main_after.revision,
                          "tip": main_after.tip, "root": main_after.root}}
            self.publish()
            ledger.complete("adopt", {"exit_code": 0})
            winner = "a" if selected_snapshot == asks["a"]["snapshot"] else "b"
            loser = "b" if winner == "a" else "a"
            loser_head = self.store.head(asks[loser]["branch"])
            if loser_head.tip != asks[loser]["snapshot"]:
                raise RuntimeError("losing candidate branch was not retained")

            _, history = self.cli("history", "primary", "history", "--branch", root_branch,
                                  output_json=True)
            if not isinstance(history, dict):
                raise RuntimeError("history response is not an object")
            winning_snapshot = self.store.get_snapshot(selected_snapshot)
            controller_status = winning_snapshot.get("controller", {}).get("status")
            if controller_status not in ("active", "stopped"):
                raise RuntimeError("adopted winner has no supported continuation state")
            self.record["adopted_history"] = {"tip": selected_snapshot,
                "branch_revision": main_after.revision, "controller_status": controller_status,
                "history": history}
            self.publish()
            ledger.complete("history", {"exit_code": 0})
            if controller_status == "active":
                continuation_args = ("continue", "--branch", root_branch,
                    "--max-tokens", str(self.args.max_continue), "--operation-id", "copy-resume")
            else:
                continuation_args = ("append-user", "--branch", root_branch,
                    "--text", "follow-up " + suffix, "--max-tokens", str(self.args.max_continue),
                    "--operation-id", "follow-up-" + suffix)
            continuation_row, continuation = self.cli("continue", "primary", *continuation_args,
                    want_receipt=True)
            self.record["continuation"] = continuation
            self.record["continuation_action"] = continuation_args[0]
            if not Path(continuation_row["stdout"]).read_text(encoding="utf-8").strip():
                raise RuntimeError("continuation from adopted state emitted no text")
            ledger.complete("continue", {"exit_code": 0})

            final = ledger.finish()
            self.record["workflow"] = final["completed"]
            self.record["retry_workflow"] = "PENDING_RETRY_API_STAGE"
            self.record["status"] = "CLI_FLOW_PASS_NUMERICAL_AUDIT_PENDING"
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
            if self.tracker is not None:
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
    parser.add_argument("--evaluation-spec", type=Path, required=True)
    parser.add_argument("--fixture", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--expected-fixture", required=True,
                        help="fixture name whose bytes are expected by the sample evaluator")
    parser.add_argument("--outputs", type=int, default=16)
    parser.add_argument("--max-continue", type=int, default=8)
    parser.add_argument("--mandatory", action="append", default=None,
                        help="mandatory scorer check; repeat for multiple checks")
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    """Campaign entry point.

    Runtime execution is deliberately separated from unit-testable validators;
    the evidence record is published after every completed subprocess action.
    """
    args = _parser().parse_args(argv)
    if not 1 <= args.outputs <= 64 or not 1 <= args.max_continue <= 32:
        raise SystemExit("bounded limits: outputs 1..64, max-continue 1..32")
    args.mandatory = args.mandatory or ["fixture_ok"]
    if args.run_dir.exists():
        raise SystemExit("run directory already exists; use a new immutable attempt path")
    return _Campaign(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
