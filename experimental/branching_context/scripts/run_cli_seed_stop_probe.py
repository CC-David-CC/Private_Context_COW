"""Exercise seeded continuation and grammar/text-stop CLI flows on the real service.

The probe submits mutations only through cow_cli. Store reads bind receipts,
controller state, branch heads, and canonical token spans for later review. A
CPU FlowModel can exercise orchestration, but only the normal service/native
campaign is eligible for real serving evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_cli_workflow_probe import (
    _Campaign as WorkflowCampaign,
    _atomic_write as _atomic_json,
    parse_durable_receipt,
    redact_argv,
    redact_sensitive_value,
    sha256_file,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def output_ends_with_eos(tokens, eos_tokens) -> bool:
    return bool(tokens) and tokens[-1] in eos_tokens


def redact_process_receipt(value, secret: str):
    if not isinstance(secret, str) or not secret:
        raise ValueError("service secret is required for receipt redaction")
    clean = redact_sensitive_value(value, (secret,))
    if _contains_secret(clean, secret):
        raise ValueError("service receipt still contains an unredacted secret")
    return clean


def _contains_secret(value, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(key, secret) or _contains_secret(child, secret)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(child, secret) for child in value)
    return False


SAMPLED_REMAINING_CHARACTERS = 40
SAMPLED_GRAMMAR = ('root ::= ("A" | "B" | "C") ' +
                   ' '.join(['("X" | "Y")'] * SAMPLED_REMAINING_CHARACTERS))
READ_TEXT = '{"tool":"read_file","path":"src/main.py"}'
TEST_TEXT = '{"tool":"run_tests","suite":"unit"}'
READ_STOP = 'read_file'
COMPLETED_WORKFLOW = (
    "namespace", "prepare", "fork A/B", "seeded sampled partial", "typed reseed",
    "continue", "exact replay", "service restart", "replay after restart",
    "continue active controller in the new service process",
    "READ grammar text stop", "TEST grammar append-user", "workspace history and branch isolation",
)


def seed_registry_payload() -> dict:
    return {"schema": 1, "states": {
        "SAMPLED": {"format": "gbnf", "source": SAMPLED_GRAMMAR},
        "READ": {"format": "gbnf", "source": "root ::= " + json.dumps(READ_TEXT, ensure_ascii=True)},
        "TEST": {"format": "gbnf", "source": "root ::= " + json.dumps(TEST_TEXT, ensure_ascii=True)},
    }}


def validate_sampled_budget(initial_tokens: int, continuation_tokens: int) -> None:
    if (type(initial_tokens) is not int or type(continuation_tokens) is not int
            or initial_tokens < 1 or continuation_tokens < 1):
        raise ValueError("sampled token budgets must be positive exact integers")


def validate_sampled_continuation(*, before: dict, after: dict, tokens: list,
                                  output_span_count: int, budget: int,
                                  eos_tokens, grammar_complete: bool,
                                  require_active: bool = False) -> dict:
    """Bind continuation progress to actual canonical tokens and its terminal reason."""
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or before.get("status") != "active"):
        raise ValueError("continuation must start from an active sampled controller")
    if type(budget) is not int or budget < 1:
        raise ValueError("continuation budget must be a positive exact integer")
    if type(grammar_complete) is not bool:
        raise ValueError("continuation grammar completion flag is malformed")
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= budget
            or type(output_span_count) is not int or output_span_count != len(tokens)):
        raise ValueError("continuation token count differs from its canonical output span or budget")
    eos = set(eos_tokens)
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("continuation contains an invalid token ID")
    if any(token in eos for token in tokens[:-1]):
        raise ValueError("continuation continued after consuming EOS")
    ended_eos = tokens[-1] in eos
    if type(before.get("next_draw")) is not int or type(after.get("next_draw")) is not int:
        raise ValueError("continuation RNG cursors are malformed")
    if after["next_draw"] - before["next_draw"] != len(tokens):
        raise ValueError("continuation RNG draw delta differs from consumed token count")
    if (type(before.get("accepted_in_scope")) is not int
            or type(after.get("accepted_in_scope")) is not int
            or after["accepted_in_scope"] - before["accepted_in_scope"] != len(tokens)):
        raise ValueError("continuation controller progress differs from consumed token count")
    if after.get("stream_key") != before.get("stream_key"):
        raise ValueError("continuation changed the sampled RNG stream")
    if require_active:
        if len(tokens) != budget or ended_eos or after.get("status") != "active":
            raise ValueError("pre-restart continuation must consume its full budget and remain active")
    elif ended_eos:
        if after.get("status") != "stopped" or not grammar_complete:
            raise ValueError("EOS-terminated grammar continuation lacks a completed stopped result")
    elif after.get("status") == "stopped":
        if not grammar_complete:
            raise ValueError("grammar-complete continuation did not persist stopped status")
    elif len(tokens) != budget or after.get("status") != "active":
        raise ValueError("short continuation without EOS is not a valid terminal result")
    return {"consumed_tokens": len(tokens), "terminal_eos": ended_eos,
            "grammar_complete": grammar_complete, "status": after["status"],
            "draw_delta": after["next_draw"] - before["next_draw"],
            "accepted_delta": after["accepted_in_scope"] - before["accepted_in_scope"]}


def head_record(value) -> dict:
    if isinstance(value, dict):
        row = value
    else:
        row = {name: getattr(value, name, None) for name in ("name", "revision", "tip", "root")}
    if (not isinstance(row.get("name"), str) or type(row.get("revision")) is not int
            or not isinstance(row.get("tip"), str) or not isinstance(row.get("root"), str)):
        raise ValueError("branch head is malformed")
    return {key: row[key] for key in ("name", "revision", "tip", "root")}


def validate_reseed_transition(before: dict, after: dict, seed: int) -> None:
    """Require a reseed to change only the existing RNG key and draw cursor."""
    if type(seed) is not int or not 0 <= seed < (1 << 64):
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("controller observations must be objects")
    if before.get("status") != "active":
        raise ValueError("probe requires the sampled source controller to remain active")
    if type(before.get("next_draw")) is not int or before["next_draw"] <= 0:
        raise ValueError("seeded partial answer did not consume an RNG draw")
    expected = json.loads(json.dumps(before))
    expected["stream_key"] = hashlib.sha256(f"cow-seed-v1:{seed}".encode("utf-8")).hexdigest()
    expected["next_draw"] = 0
    if after != expected:
        raise ValueError("reseed changed fields other than stream_key and next_draw")


def validate_exact_replay(*, stdout_before: bytes, stdout_after: bytes,
                          receipt_before: dict, receipt_after: dict,
                          head_before: dict, head_after: dict,
                          trace_before: bytes, trace_after: bytes) -> None:
    if stdout_after != stdout_before:
        raise ValueError("exact replay changed CLI output")
    if receipt_after != receipt_before:
        raise ValueError("exact replay changed its durable receipt")
    if head_after != head_before:
        raise ValueError("exact replay moved the branch head")
    if trace_after != trace_before:
        raise ValueError("exact replay emitted native worker RPCs")


def validate_stopped_generation(*, display: str, canonical_bytes: bytes,
                                controller: dict, marker: str,
                                grammar_state: str) -> None:
    marker_bytes = marker.encode("utf-8")
    if marker in display:
        raise ValueError("display text contains the configured stop marker")
    if marker_bytes not in canonical_bytes:
        raise ValueError("canonical consumed token span does not contain the stop marker")
    if not isinstance(controller, dict) or controller.get("status") != "stopped":
        raise ValueError("text stop did not persist a stopped controller")
    matcher = controller.get("text_stop")
    if not isinstance(matcher, dict) or matcher.get("matched_stop") != marker:
        raise ValueError("persisted controller does not identify the matched marker")
    constraint = controller.get("constraint")
    if not isinstance(constraint, dict) or constraint.get("agent_state") != grammar_state:
        raise ValueError("text stop did not preserve the selected grammar state")
    if "grammar_complete" in constraint:
        raise ValueError("grammar completion is not controller state and cannot be inferred from a text stop")


def mark_workflow_pass(record: dict, completed: list[str]) -> None:
    if not isinstance(record, dict) or not isinstance(completed, list):
        raise ValueError("workflow completion evidence is malformed")
    if tuple(completed) != COMPLETED_WORKFLOW:
        raise ValueError("workflow cannot pass before every required CLI stage is complete in order")
    record["completed_workflow"] = list(completed)
    record["status"] = "CLI_SEED_STOP_FLOW_PASS_NUMERICAL_AUDIT_PENDING"


class SeedStopProbe(WorkflowCampaign):
    """Use the proven service lifecycle and CLI receipt plumbing for seed/stop flows."""

    def __init__(self, args):
        super().__init__(args)
        self.record = {
            "schema": "cow-cli-seed-stop-probe-v1",
            "status": "RUNNING",
            "commands": [],
            "services": [],
            "failures": [],
            "numerical_qualification": "PENDING_SEPARATE_FRESH_REFERENCE",
        }
        self.call_id = 0
        self.cli_python = os.environ.get("COW_PYTHON", sys.executable)
        self.registry_path = None
        self.profile = None

    def _bootstrap(self, ordinal: int) -> Path:
        path = self.run_dir / f"seed_stop_server_bootstrap-{ordinal}.py"
        registry_path = self.registry_path.resolve(strict=True)
        dependency_path = (self.args.dependency_root.resolve(strict=True)
                           if self.args.dependency_root else None)
        body = (
            "import hashlib,importlib,json,sys\n"
            "from pathlib import Path\n"
            "profile=json.loads(Path(sys.argv[1]).read_bytes())\n"
            "runtime=Path(sys.argv[2]).resolve()\n"
            f"registry=Path({str(registry_path)!r}).resolve()\n"
            f"dependency=Path({str(dependency_path)!r}).resolve() if {dependency_path is not None!r} else None\n"
            "sys.path[:0]=([str(dependency)] if dependency else [])+[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),str(runtime/'experimental/branching_context/scripts')]\n"
            "names=('context_time_travel.coordinator','context_time_travel.presentation_binding','context_time_travel.presentation_request','context_time_travel.seed_controls','context_time_travel.text_stops','context_time_travel.sampled_controller','context_time_travel.grammar_first','context_time_travel.real','context_time_travel.store','serve.cow_prompt_encoder','serve.cow_text_adapter','serve.cow_workspace','serve.cow_http')\n"
            "rows={}\n"
            "for name in names:\n"
            " m=importlib.import_module(name);p=Path(m.__file__).resolve();rows[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}\n"
            "print(json.dumps({'event':'runtime_imports','modules':rows}),flush=True)\n"
            f"sys.argv=['cow-http','--grammar-registry',{str(registry_path)!r},*sys.argv[3:]]\n"
            "importlib.import_module('serve.cow_http').main()\n"
        )
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(body)
        return path

    def start_service(self, registry_path: Path):
        """Start the normal service with only the probe registry added."""
        self.registry_path = registry_path.resolve(strict=True)
        self.ordinal += 1
        self.phase(f"start-service-{self.ordinal}")
        trace = self.run_dir / f"native-{self.ordinal}.jsonl"
        trace.touch(mode=0o600, exist_ok=False)
        pool_trace = self.run_dir / f"pool-{self.ordinal}.json"
        bootstrap = self._bootstrap(self.ordinal)
        env = os.environ.copy()
        pythonpath = [str(self.runtime), str(self.runtime / "tools"),
                      str(self.runtime / "experimental/branching_context/src"),
                      str(self.runtime / "experimental/branching_context/scripts")]
        if self.args.dependency_root:
            pythonpath.insert(0, str(self.args.dependency_root.resolve()))
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "CTT_NATIVE_TRACE": str(trace),
                    "PYTHONPATH": os.pathsep.join(pythonpath)})
        command = [self.cli_python, "-B", str(bootstrap), str(self.profile_path),
                   str(self.runtime), "--profile", str(self.profile_path),
                   "--store", str(self.args.store.absolute()), "--host", "127.0.0.1",
                   "--port", "0", "--text", "--api-key", self.api_key,
                   "--ram-gib", str(self.args.ram_gib), "--disk-gib", str(self.args.disk_gib),
                   "--pool-trace-output", str(pool_trace)]
        started = time.monotonic()
        self.service = self._ServiceProcess(command, cwd=self.runtime, env=env,
            stdout_path=self.run_dir / f"server-{self.ordinal}.stdout",
            stderr_path=self.run_dir / f"server-{self.ordinal}.stderr",
            timeout=180, progress=self.tracker)
        ready = self.service.start()
        self.client = self._HttpClient(self._ready_url(ready), timeout=180,
                                       progress=self.tracker)
        self.store = self._Store(self.args.store)
        output = (self.run_dir / f"server-{self.ordinal}.stdout").read_text(encoding="utf-8")
        imports = [json.loads(line).get("modules") for line in output.splitlines()
                   if line.startswith('{') and json.loads(line).get("event") == "runtime_imports"]
        if len(imports) != 1:
            raise RuntimeError("service did not report exactly one source import closure")
        self.record["services"].append({"ordinal": self.ordinal,
            "pid": self.service.process.pid, "ready": ready,
            "startup_seconds": time.monotonic() - started,
            "argv_redacted": redact_argv(command, (self.api_key,)),
            "source_imports": imports[0], "native_trace": str(trace),
            "pool_trace": str(pool_trace)})
        self.publish()

    def _trace_bytes(self):
        if not self.ordinal:
            return b""
        path = self.run_dir / f"native-{self.ordinal}.jsonl"
        return path.read_bytes() if path.exists() else b""

    def stop_service(self):
        """Stop both workers and redact nested process receipts before saving."""
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
        service_record["stop"] = redact_process_receipt(receipt, self.api_key)
        service_record["worker_shutdown"] = redact_process_receipt(worker, self.api_key)
        self.service = None
        self.client = None
        self.publish()

    def cli(self, label, *arguments, expect=0, receipt=False, json_output=False):
        """Run one CLI command with a fresh local journal and durable per-call evidence."""
        self.call_id += 1
        self.phase(label)
        state_dir = self.run_dir / f"journal-{self.call_id:03d}-{label}"
        state_dir.mkdir(mode=0o700, exist_ok=False)
        command = [self.cli_python, "-B", str(self.runtime / "tools/cow_cli.py"),
                   "--url", self._ready_url(self.service.ready), "--state-dir", str(state_dir),
                   "--timeout", "180", *arguments]
        env = os.environ.copy()
        extra = [str(self.args.dependency_root.resolve())] if self.args.dependency_root else []
        env.update({"COW_API_KEY": self.api_key, "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join(extra + [str(self.runtime), str(self.runtime / "tools"),
                    str(self.runtime / "experimental/branching_context/src"),
                    str(self.runtime / "experimental/branching_context/scripts")])})
        before = self._trace_bytes()
        started = time.monotonic()
        completed = subprocess.run(command, cwd=self.runtime, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240, check=False)
        elapsed = time.monotonic() - started
        stdout = completed.stdout.replace(self.api_key.encode("utf-8"), b"<REDACTED>")
        stderr = completed.stderr.replace(self.api_key.encode("utf-8"), b"<REDACTED>")
        after = self._trace_bytes()
        name = f"{self.call_id:03d}-{label}"
        stdout_path, stderr_path = self.run_dir / f"{name}.stdout", self.run_dir / f"{name}.stderr"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        row = {"label": label, "argv_redacted": redact_argv(command, (self.api_key,)),
               "exit_code": completed.returncode, "expected_exit_code": expect,
               "elapsed_seconds": elapsed, "stdout_path": str(stdout_path),
               "stderr_path": str(stderr_path), "stdout_sha256": sha256_bytes(stdout),
               "journal_dir": str(state_dir),
               "stderr_sha256": sha256_bytes(stderr), "native_trace_bytes_before": len(before),
               "native_trace_bytes_after": len(after),
               "native_trace_delta_sha256": sha256_bytes(after[len(before):])}
        self.record["commands"].append(row)
        self.publish()
        if completed.returncode != expect:
            raise RuntimeError(f"CLI {label} returned {completed.returncode}, expected {expect}")
        value = None
        if receipt:
            value = parse_durable_receipt(stderr)
            row["receipt"] = value
        elif json_output and expect == 0:
            value = json.loads(stdout.decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(f"CLI {label} JSON output is not an object")
        row["stdout_text"] = stdout.decode("utf-8", errors="strict")
        self.publish()
        return row, value

    def reseed_cli(self, branch, seed, revision, operation_id):
        """Run reseed and parse its durable acknowledgement from stderr."""
        return self.cli("reseed", "reseed", "--branch", branch,
                        "--seed", str(seed), "--revision", str(revision),
                        "--operation-id", operation_id, receipt=True)

    def _tokens(self, operation, snapshot):
        envelope = self.store.presentation_receipt(operation)
        presentation = envelope.get("presentation")
        span = presentation.get("output_span") if isinstance(presentation, dict) else None
        if not isinstance(span, dict):
            raise RuntimeError(f"{operation} has no canonical output span")
        start, count = span.get("start"), span.get("count")
        if type(start) is not int or type(count) is not int or start < 0 or count < 0:
            raise RuntimeError("canonical output span is malformed")
        root, ids = self.store.lineage(snapshot)
        output = tuple(ids[start:start + count])
        if len(output) != count:
            raise RuntimeError("canonical output span exceeds Store history")
        return list(output), envelope

    def _operation_answer(self, row, operation_id):
        journal_dir = row.get("journal_dir") if isinstance(row, dict) else None
        if not isinstance(journal_dir, str):
            raise RuntimeError("CLI command did not record its private journal directory")
        path = Path(journal_dir) / "operations" / f"{operation_id}.json"
        try:
            record = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("CLI command did not persist a readable operation answer") from exc
        if (not isinstance(record, dict) or record.get("operation_id") != operation_id
                or record.get("state") != "complete" or not isinstance(record.get("answer"), dict)):
            raise RuntimeError("CLI operation journal does not contain the expected complete answer")
        return record["answer"]

    def _controller(self, snapshot):
        value = self.store.get_snapshot(snapshot)
        controller = value.get("controller")
        if not isinstance(controller, dict):
            raise RuntimeError("snapshot lacks a persisted controller")
        return controller

    def _start_registry(self):
        registry = seed_registry_payload()
        path = self.run_dir / "grammar-registry.json"
        _atomic_json(path, registry)
        self.record["grammar_registry"] = {
            "path": str(path), "sha256": sha256_file(path),
            "states": {name: {"format": item["format"],
                               "source_sha256": sha256_bytes(item["source"].encode("utf-8"))}
                       for name, item in registry["states"].items()},
            "stop_marker": READ_STOP,
        }
        self.publish()
        return path

    def execute(self):
        if self.run_dir.exists():
            raise SystemExit("run directory exists; use a fresh immutable run path")
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.record.update({"runtime": str(self.runtime), "profile": str(self.profile_path),
                            "store": str(self.args.store), "question": self.args.question,
                            "question_sha256": sha256_bytes(self.args.question.encode("utf-8")),
                            "seed": self.args.seed, "reseed": self.args.reseed,
                            "initial_sampled_tokens": self.args.initial_tokens,
                            "continuation_tokens": self.args.continue_tokens})
        self.publish()
        error = None
        try:
            self.runtime = self.runtime.resolve(strict=True)
            self.profile_path = self.profile_path.resolve(strict=True)
            if not (self.runtime / "tools/cow_cli.py").is_file():
                raise RuntimeError("runtime stage is missing cow_cli.py")
            if self.args.store.exists() and any(self.args.store.iterdir()):
                raise RuntimeError("probe Store must be absent or empty")
            profile = json.loads(self.profile_path.read_bytes())
            self._runtime_imports()
            self.profile = profile
            self._runtime_closure(profile, self.runtime)
            self.store = self._Store(self.args.store)
            registry_path = self._start_registry()
            self.record["profile_identity"] = {key: profile.get(key) for key in
                ("execution_profile_id", "model_contract_id", "model_weights_sha256")}
            self.record["runtime_pins"] = {
                "cow_cli_sha256": sha256_file(self.runtime / "tools/cow_cli.py"),
                "cow_http_sha256": sha256_file(self.runtime / "serve/cow_http.py"),
                "grammar_first_sha256": sha256_file(self.runtime / "experimental/branching_context/src/context_time_travel/grammar_first.py"),
                "seed_controls_sha256": sha256_file(self.runtime / "experimental/branching_context/src/context_time_travel/seed_controls.py"),
                "text_stops_sha256": sha256_file(self.runtime / "experimental/branching_context/src/context_time_travel/text_stops.py"),
            }
            self.publish()
        except BaseException as exc:
            self.record["status"] = "PREFLIGHT_FAIL"
            self.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
            self.publish()
            return 1

        (self.run_dir / "calls").mkdir(mode=0o700)
        self.tracker = self._ProgressTracker(self.run_dir, "cli-seed-stop-probe").start()
        try:
            self.start_service(registry_path)
            suffix = self.run_id
            namespace_row, namespace_receipt = self.cli("namespace", "namespace", "--title",
                "Seed/STOP probe " + suffix, "--operation-id", "seed-stop-ns-" + suffix, receipt=True)
            namespace = namespace_receipt.get("namespace_id")
            if not isinstance(namespace, str) or not namespace:
                raise RuntimeError("namespace receipt omitted namespace_id")
            prep_row, prepare_receipt = self.cli("prepare", "prepare", "--namespace-id", namespace,
                "--branch", "main", "--text", self.args.question,
                "--operation-id", "seed-stop-root-" + suffix, receipt=True)
            root_head = head_record(self.store.head("main"))
            if prepare_receipt.get("snapshot") != root_head["tip"] or root_head["tip"] != root_head["root"]:
                raise RuntimeError("prepare receipt is not the canonical root")
            root_ids = self.store.lineage(root_head["tip"])[1]
            if not root_ids:
                raise RuntimeError("prepared root has empty canonical history")
            self.record["root"] = {"head": root_head, "snapshot": root_head["tip"],
                                   "namespace_id": namespace, "history_tokens": len(root_ids),
                                   "prepare_operation": "seed-stop-root-" + suffix,
                                   "receipt": prepare_receipt}
            self.publish()

            branch_a, branch_b = "seeded-" + suffix, "grammar-stop-" + suffix
            for label, branch in (("fork-sampled", branch_a), ("fork-grammar", branch_b)):
                self.cli(label, "fork", "--source-branch", "main", "--name", branch,
                         "--operation-id", label + "-" + suffix, receipt=True)
            if self.store.head("main").tip != root_head["tip"]:
                raise RuntimeError("forks changed the prepared main branch")
            grammar_head_at_fork = head_record(self.store.head(branch_b))

            partial_op = "sampled-partial-" + suffix
            sampled_row, sampled_receipt = self.cli("sampled-partial", "answer", "--branch", branch_a,
                "--max-tokens", str(self.args.initial_tokens), "--operation-id", partial_op,
                "--seed", str(self.args.seed), "--temperature", str(self.args.temperature),
                "--min-p", "0", "--top-k", "0", "--top-p", "1",
                "--grammar-state", "SAMPLED", receipt=True)
            sampled_snapshot = sampled_receipt.get("snapshot")
            sampled_controller = self._controller(sampled_snapshot)
            sampled_tokens, sampled_envelope = self._tokens(partial_op, sampled_snapshot)
            if sampled_controller.get("status") != "active" or len(sampled_tokens) != self.args.initial_tokens:
                raise RuntimeError("sampled prefix did not leave an active partial grammar controller")
            if sampled_controller.get("constraint", {}).get("agent_state") != "SAMPLED":
                raise RuntimeError("sampled prefix lost its grammar state")
            if type(sampled_controller.get("next_draw")) is not int or sampled_controller["next_draw"] <= 0:
                raise RuntimeError("seeded sampled answer did not consume an RNG draw")
            self.record["sampled_partial"] = {
                "operation": partial_op, "snapshot": sampled_snapshot,
                "head": head_record(self.store.head(branch_a)), "tokens": sampled_tokens,
                "controller": sampled_controller, "presentation": sampled_envelope,
                "cli_stdout_sha256": sampled_row["stdout_sha256"],
            }
            self.publish()
            if head_record(self.store.head(branch_b)) != grammar_head_at_fork:
                raise RuntimeError("sampled branch changed its sibling grammar branch")

            before_reseed = self._controller(sampled_snapshot)
            reseed_op = "reseed-active-" + suffix
            reseed_row, reseed_json = self.reseed_cli(
                branch_a, self.args.reseed, self.store.head(branch_a).revision, reseed_op)
            reseed_head = head_record(self.store.head(branch_a))
            reseed_controller = self._controller(reseed_head["tip"])
            validate_reseed_transition(before_reseed, reseed_controller, self.args.reseed)
            self.record["reseed"] = {"operation": reseed_op, "head": reseed_head,
                                     "controller": reseed_controller, "ack": reseed_json,
                                     "stdout_sha256": reseed_row["stdout_sha256"]}
            self.publish()
            if head_record(self.store.head(branch_b)) != grammar_head_at_fork:
                raise RuntimeError("reseed changed the sibling grammar branch")

            continue_row, continue_receipt = self.cli("sampled-continue", "continue", "--branch", branch_a,
                "--max-tokens", str(self.args.continue_tokens),
                "--operation-id", "sampled-continue-" + suffix, receipt=True)
            continued_snapshot = continue_receipt.get("snapshot")
            continued_controller = self._controller(continued_snapshot)
            continued_tokens, continued_envelope = self._tokens(
                "sampled-continue-" + suffix, continued_snapshot)
            continued_answer = self._operation_answer(continue_row, "sampled-continue-" + suffix)
            continued_progress = validate_sampled_continuation(
                before=reseed_controller, after=continued_controller, tokens=continued_tokens,
                output_span_count=continued_envelope["presentation"]["output_span"]["count"],
                budget=self.args.continue_tokens,
                eos_tokens=self.profile["execution"]["eos_tokens"],
                grammar_complete=continued_answer.get("grammar_complete"), require_active=True)
            self.record["sampled_continue"] = {"receipt": continue_receipt,
                "head": head_record(self.store.head(branch_a)), "tokens": continued_tokens,
                "controller": continued_controller, "progress": continued_progress,
                "cli_stdout_sha256": continue_row["stdout_sha256"]}
            self.publish()
            if head_record(self.store.head(branch_b)) != grammar_head_at_fork:
                raise RuntimeError("sampled continuation changed the sibling grammar branch")

            # A distinct journal forces a server-side receipt replay.
            original_out = sampled_row["stdout_text"].encode("utf-8")
            original_head = head_record(self.store.head(branch_a))
            original_trace = self._trace_bytes()
            replay_row, replay_receipt = self.cli("sampled-exact-retry", "retry", "--from-operation",
                partial_op, receipt=True)
            validate_exact_replay(stdout_before=original_out,
                stdout_after=replay_row["stdout_text"].encode("utf-8"),
                receipt_before=sampled_receipt, receipt_after=replay_receipt,
                head_before=original_head, head_after=head_record(self.store.head(branch_a)),
                trace_before=original_trace, trace_after=self._trace_bytes())
            self.record["sampled_exact_retry"] = {"receipt": replay_receipt,
                "status": "PASS_NO_BRANCH_MOVE_NO_NATIVE_RPC"}
            self.publish()

            # Restart the real HTTP/native worker against the same Store, then
            # prove the old receipt still replays before continuing current state.
            self.stop_service()
            self.start_service(registry_path)
            restart_head = head_record(self.store.head(branch_a))
            restart_trace = self._trace_bytes()
            restart_row, restart_receipt = self.cli("restart-exact-retry", "retry", "--from-operation",
                partial_op, receipt=True)
            validate_exact_replay(stdout_before=original_out,
                stdout_after=restart_row["stdout_text"].encode("utf-8"),
                receipt_before=sampled_receipt, receipt_after=restart_receipt,
                head_before=restart_head, head_after=head_record(self.store.head(branch_a)),
                trace_before=restart_trace, trace_after=self._trace_bytes())
            self.record["restart_replay"] = {"receipt": restart_receipt,
                "status": "PASS_NO_BRANCH_MOVE_NO_NATIVE_RPC"}
            self.publish()
            restart_source_controller = self._controller(restart_head["tip"])
            if restart_source_controller.get("status") != "active":
                raise RuntimeError("restart replay source controller is not active")
            final_row, final_receipt = self.cli("restart-continue", "continue", "--branch", branch_a,
                "--max-tokens", str(self.args.continue_tokens),
                "--operation-id", "sampled-restart-continue-" + suffix, receipt=True)
            final_snapshot = final_receipt.get("snapshot")
            final_controller = self._controller(final_snapshot)
            final_tokens, final_envelope = self._tokens(
                "sampled-restart-continue-" + suffix, final_snapshot)
            final_answer = self._operation_answer(
                final_row, "sampled-restart-continue-" + suffix)
            final_progress = validate_sampled_continuation(
                before=restart_source_controller, after=final_controller, tokens=final_tokens,
                output_span_count=final_envelope["presentation"]["output_span"]["count"],
                budget=self.args.continue_tokens,
                eos_tokens=self.profile["execution"]["eos_tokens"],
                grammar_complete=final_answer.get("grammar_complete"))
            self.record["restart_continue"] = {"receipt": final_receipt,
                "head": head_record(self.store.head(branch_a)), "tokens": final_tokens,
                "controller": final_controller, "progress": final_progress,
                "stdout_sha256": final_row["stdout_sha256"]}
            self.publish()

            read_op = "read-stop-" + suffix
            read_row, read_receipt = self.cli("read-stop", "answer", "--branch", branch_b,
                "--max-tokens", str(self.args.grammar_tokens), "--operation-id", read_op,
                "--grammar-state", "READ", "--stop", READ_STOP, receipt=True)
            read_snapshot = read_receipt.get("snapshot")
            read_controller = self._controller(read_snapshot)
            read_tokens, read_envelope = self._tokens(read_op, read_snapshot)
            profile = json.loads(self.profile_path.read_bytes())
            sys.path.insert(0, str(self.runtime / "tools"))
            try:
                from strata_tokenizer import Tokenizer
                tokenizer = Tokenizer.from_gguf(profile["model_path"])
            finally:
                if sys.path[0] == str(self.runtime / "tools"):
                    sys.path.pop(0)
            canonical_output = b"".join(tokenizer.token_bytes(token) for token in read_tokens)
            validate_stopped_generation(display=read_row["stdout_text"],
                canonical_bytes=canonical_output, controller=read_controller,
                marker=READ_STOP, grammar_state="READ")
            if read_controller.get("accepted_in_scope") != len(read_tokens):
                raise RuntimeError("READ STOP controller progress differs from consumed token span")
            self.record["read_stop"] = {"operation": read_op, "snapshot": read_snapshot,
                "head": head_record(self.store.head(branch_b)), "tokens": read_tokens,
                "canonical_output_bytes_sha256": sha256_bytes(canonical_output),
                "display_sha256": read_row["stdout_sha256"], "controller": read_controller,
                "presentation": read_envelope}
            self.publish()

            test_op = "append-test-" + suffix
            test_row, test_receipt = self.cli("append-test", "append-user", "--branch", branch_b,
                "--text", "run the unit tests", "--max-tokens", str(self.args.grammar_tokens),
                "--operation-id", test_op, "--grammar-state", "TEST", receipt=True)
            test_snapshot = test_receipt.get("snapshot")
            test_controller = self._controller(test_snapshot)
            test_tokens, test_envelope = self._tokens(test_op, test_snapshot)
            if test_row["stdout_text"] != TEST_TEXT + "\n":
                raise RuntimeError("append-user TEST grammar output differs from expected exact JSON")
            if test_controller.get("constraint", {}).get("agent_state") != "TEST":
                raise RuntimeError("append-user lost TEST grammar state")
            if (not test_tokens or test_controller.get("status") != "stopped"
                    or not output_ends_with_eos(test_tokens, profile["execution"]["eos_tokens"])):
                raise RuntimeError("TEST append-user turn did not finish at EOS")
            input_span = test_envelope["presentation"].get("input_span")
            if not isinstance(input_span, dict):
                raise RuntimeError("TEST append-user omitted its canonical input span")
            start, count = input_span.get("start"), input_span.get("count")
            if type(start) is not int or type(count) is not int or count <= 0:
                raise RuntimeError("TEST append-user input span is malformed")
            full_history = self.store.lineage(test_snapshot)[1]
            suffix = full_history[start:start + count]
            suffix_bytes = b"".join(tokenizer.token_bytes(token) for token in suffix)
            if len(suffix) != count or b"<|im_end|>" not in suffix_bytes:
                raise RuntimeError("TEST append-user canonical suffix omitted the STOP-to-user EOT")
            history_row, history = self.cli("workspace-history", "history", "--branch", branch_b,
                                            json_output=True)
            message_rows = history.get("messages", [])
            display = "\n".join(str(row.get("text", "")) for row in message_rows
                                 if isinstance(row, dict))
            if READ_STOP in display or READ_STOP in test_row["stdout_text"]:
                raise RuntimeError("workspace or TEST output exposed the hidden READ stop marker")
            if TEST_TEXT not in display:
                raise RuntimeError("workspace history omitted the TEST grammar answer")
            if self.store.head("main").tip != root_head["tip"] or self.store.head(branch_a).root != root_head["root"]:
                raise RuntimeError("grammar branch changed canonical source or sampled sibling root")
            self.record["append_test"] = {"operation": test_op, "receipt": test_receipt,
                "head": head_record(self.store.head(branch_b)), "tokens": test_tokens,
                "controller": test_controller, "stdout_sha256": test_row["stdout_sha256"],
                "input_span": input_span, "input_suffix_sha256": sha256_bytes(suffix_bytes),
                "history_sha256": history_row["stdout_sha256"], "history_message_count": len(message_rows),
                "history_display_sha256": sha256_bytes(display.encode("utf-8"))}
            mark_workflow_pass(self.record, list(COMPLETED_WORKFLOW))
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reseed", type=int, default=73)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--initial-tokens", type=int, default=1)
    parser.add_argument("--continue-tokens", type=int, default=16)
    parser.add_argument("--grammar-tokens", type=int, default=64)
    parser.add_argument("--ram-gib", type=int, default=96)
    parser.add_argument("--disk-gib", type=int, default=128)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if (not 1 <= args.initial_tokens <= 8 or not 1 <= args.continue_tokens <= 64
            or not 8 <= args.grammar_tokens <= 128 or not 16 <= args.ram_gib <= 128
            or not 16 <= args.disk_gib <= 512):
        raise SystemExit("bounded options: initial 1..8, continue 1..64, grammar 8..128, RAM 16..128 GiB, disk 16..512 GiB")
    try:
        validate_sampled_budget(args.initial_tokens, args.continue_tokens)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if (type(args.seed) is not int or not 0 <= args.seed < (1 << 64)
            or type(args.reseed) is not int or not 0 <= args.reseed < (1 << 64)
            or not 0 < args.temperature < 10):
        raise SystemExit("seed values must be uint64; temperature must be finite and between 0 and 10")
    return SeedStopProbe(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
