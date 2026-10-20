#!/usr/bin/env python3
"""Fresh-process native same-operator reference for one static GBNF branch.

This deliberately talks to the pinned native stdin protocol directly. It does
not import the COW model/backend or any checkpoint materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
import threading
import time


PROTOCOL_COMMANDS = frozenset(("feed", "decode", "logits", "stats", "quit"))
SAMPLE_COUNTS = (1, 16, 64)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(ids) -> str:
    values = tuple(ids)
    return sha256_bytes(struct.pack("<" + "q" * len(values), *values))


def canonical_controller_sha256(controller) -> str:
    raw = json.dumps(controller, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    return sha256_bytes(raw)


def validate_fixture(case: dict, branch: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate the exact standalone fixture relationship before model startup."""
    if branch not in ("A", "B") or not isinstance(case, dict):
        raise ValueError("fixture branch must be A or B")
    root = case.get("root_token_ids")
    branches = case.get("branches")
    item = branches.get(branch) if isinstance(branches, dict) else None
    if (not isinstance(root, list) or not root or
            any(type(token) is not int or token < 0 for token in root) or
            not isinstance(item, dict)):
        raise ValueError("fixture root or selected branch token IDs are invalid")
    suffix = item.get("suffix_token_ids")
    full = item.get("full_token_ids")
    if (not isinstance(suffix, list) or not suffix or
            any(type(token) is not int or token < 0 for token in suffix) or
            not isinstance(full, list) or
            any(type(token) is not int or token < 0 for token in full) or
            full != root + suffix):
        raise ValueError("fixture full_token_ids must equal root_token_ids plus suffix_token_ids")
    return tuple(root), tuple(suffix)


def record_grammar_outcome(record: dict, session, outputs: list[int]) -> None:
    """Store candidate-compatible terminal grammar state and final sample alias."""
    stopped = session.is_stopped()
    if type(stopped) is not bool:
        raise ValueError("grammar stopped state must be boolean")
    record["grammar_complete"] = session.is_complete()
    record["stopped"] = stopped
    record["grammar_stopped"] = stopped
    record["output_count"] = len(outputs)
    if record.get("final_sample_key"):
        record["samples"]["final"] = record["samples"][record["final_sample_key"]]


def validate_request(branch: str, stream: str, max_outputs: int) -> None:
    if branch not in ("A", "B") or stream != "review-" + branch:
        raise ValueError("branch and review stream must bind exactly (A/review-A or B/review-B)")
    if type(max_outputs) is not int or not 1 <= max_outputs <= 128:
        raise ValueError("max-outputs must be an integer from 1 through 128")


def expected_chunks(position: int, count: int, chunk_size: int) -> list[list[int]]:
    if (type(position) is not int or position < 0 or type(count) is not int or count <= 0 or
            type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("invalid prefill chunk inputs")
    chunks = []
    while count:
        amount = min(count, chunk_size)
        chunks.append([position, amount])
        position += amount
        count -= amount
    return chunks


def validate_feed_receipt(response: dict, *, position: int, count: int,
                          chunk_size: int) -> dict:
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise ValueError("native feed did not return an ok receipt")
    if (response.get("operator") != "batched-prefill" or
            type(response.get("position")) is not int or response["position"] != position or
            type(response.get("count")) is not int or response["count"] != count or
            type(response.get("consumed")) is not int or response["consumed"] != position + count or
            response.get("chunks") != expected_chunks(position, count, chunk_size)):
        raise ValueError("native feed receipt position/count/chunks mismatch")
    return response


def validate_decode_receipt(response: dict, *, position: int) -> dict:
    if (not isinstance(response, dict) or response.get("status") != "ok" or
            response.get("operator") != "normal-verifier-decode" or
            type(response.get("position")) is not int or response["position"] != position or
            type(response.get("count")) is not int or response["count"] != 1 or
            type(response.get("consumed")) is not int or response["consumed"] != position + 1 or
            response.get("chunks") != [[position, 1]]):
        raise ValueError("native decode receipt position/count/chunks mismatch")
    return response


def validate_command_inventory(commands) -> None:
    if not isinstance(commands, list) or any(
            not isinstance(row, dict) or row.get("command") not in PROTOCOL_COMMANDS
            for row in commands):
        raise ValueError("native reference used a command outside feed/decode/logits/stats/quit")


def logits_command(path: Path) -> str:
    # The pinned native driver parses this operand with `operator>> string`,
    # not std::quoted; pass a single whitespace-free path token.
    value = str(path)
    if not value or any(char.isspace() for char in value) or '"' in value:
        raise ValueError("native logits scratch path must be one unquoted token")
    return "logits " + value


def validate_counter_delta(before: dict, after: dict, *, forward: int,
                           prefill: int, decode: int) -> dict:
    fields = ("forward_tokens", "prefill_tokens", "decode_timed_tokens")
    expected = (forward, prefill, decode)
    deltas = {}
    for field, wanted in zip(fields, expected):
        left, right = before.get(field), after.get(field)
        if type(left) is not int or type(right) is not int or right - left != wanted:
            raise ValueError(f"native {field} counter delta mismatch")
        deltas[field] = right - left
    if type(after.get("consumed")) is not int or after["consumed"] != after.get("forward_tokens"):
        raise ValueError("native consumed and forward counters disagree")
    return deltas


def _publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Heartbeat:
    def __init__(self, output: Path, interval: float = 30.0):
        self.output = output
        self.interval = interval
        self.started = time.monotonic()
        self.phase = "preflight"
        self.native_pid = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="grammar-reference-heartbeat", daemon=True)

    def start(self):
        self._write()
        self._thread.start()
        return self

    def set(self, phase: str, native_pid=None):
        with self._lock:
            self.phase = phase
            if native_pid is not None:
                self.native_pid = native_pid
        self._write()

    def _write(self):
        with self._lock:
            row = {"phase": self.phase, "native_pid": self.native_pid,
                   "elapsed_seconds": time.monotonic() - self.started,
                   "updated_unix_ns": time.time_ns()}
            _publish(self.output, row)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._write()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval + 1.0))
        if self._thread.is_alive():
            raise RuntimeError("heartbeat thread did not stop")


class NativePipe:
    """Small direct stdin-protocol client; intentionally not the COW adapter."""
    def __init__(self, command, environment, output: Path, *, timeout: float = 18000):
        self.command = list(command)
        self._stderr_log = (output / "native.stderr.log").open("wb")
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr_log, text=True, bufsize=1, env=environment)
        except BaseException:
            self._stderr_log.close()
            raise
        self.timeout = timeout
        self.sequence = 0
        self.commands = []
        self.stdout_path = output / "native.stdout.jsonl"
        self._stdout_log = self.stdout_path.open("ab")
        self.closed = False
        try:
            self.ready = self._read_response("ready")
            self.startup_error = None
        except BaseException as exc:
            # Retain the exact process handle so the caller can record numeric
            # exit and absence even when readiness parsing itself failed.
            self.ready = None
            self.startup_error = exc

    def _terminate_owned(self):
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)

    def _readline(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("native protocol response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native protocol closed its stdout")
        self._stdout_log.write(line.encode("utf-8"))
        self._stdout_log.flush()
        return line

    def _read_response(self, expected=None):
        response = json.loads(self._readline())
        if not isinstance(response, dict):
            raise ValueError("native response must be a JSON object")
        if expected == "ready":
            if response.get("status") != "ready":
                raise RuntimeError("native driver did not become ready")
        return response

    def request(self, command: str, kind: str, details: dict | None = None) -> dict:
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("native process is not live")
        if kind not in PROTOCOL_COMMANDS or "\n" in command or "\r" in command:
            raise ValueError("invalid native command")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        row = {"command": kind, "details": details or {}, "response": response}
        self.commands.append(row)
        return response

    def close(self) -> tuple[int | None, bool, list[str]]:
        errors = []
        if not self.closed and self.process.poll() is None:
            try:
                response = self.request("quit", "quit")
                if response.get("status") != "closed":
                    errors.append("native quit did not return closed")
            except BaseException as exc:
                errors.append(f"native quit: {type(exc).__name__}: {exc}")
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            errors.append("native quit timed out; terminating owned process")
            try:
                self._terminate_owned()
            except subprocess.TimeoutExpired:
                errors.append("native process remained after SIGKILL")
        self.closed = True
        self.process.stdin.close()
        self.process.stdout.close()
        self._stdout_log.close()
        self._stderr_log.close()
        absent = not Path("/proc", str(self.process.pid)).exists()
        return self.process.returncode, absent, errors


def _read_fixture(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("fixture must be a JSON object")
    return value


def _expected_chunk_receipts(start: int, count: int, chunk_size: int) -> list[list[int]]:
    return expected_chunks(start, count, chunk_size)


def _sample_frontier(native: NativePipe, output: Path, name: str, consumed: int,
                     vocab_size: int) -> dict:
    native.sequence += 1
    scratch = output / f"frontier-read-{native.sequence}.f32"
    response = native.request(logits_command(scratch), "logits",
                             {"path": scratch.name, "consumed": consumed})
    if response.get("status") != "ok" or type(response.get("bytes")) is not int:
        raise ValueError("native logits request did not return an ok byte receipt")
    raw = scratch.read_bytes()
    scratch.unlink()
    if len(raw) != vocab_size * 4 or response["bytes"] != len(raw):
        raise ValueError("native full frontier extent differs from vocabulary")
    if sys.byteorder != "little":
        raise RuntimeError("pinned native frontier format requires a little-endian host")
    values = struct.unpack("<" + "f" * vocab_size, raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("native full frontier contains nonfinite logits")
    target = output / f"{name}.f32"
    target.write_bytes(raw)
    return {"path": target.name, "sha256": sha256_bytes(raw), "consumed": consumed,
            "bytes": len(raw)}


def _native_environment(profile: dict) -> dict:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            env.pop(key)
    env.update(profile["execution"]["environment"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("CTT_NATIVE_TRACE", None)
    return env


def _apply_profile_environment(profile: dict) -> None:
    for key in list(os.environ):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            os.environ.pop(key)
    os.environ.update(profile["execution"]["environment"])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("CTT_NATIVE_TRACE", None)


def run_reference(*, runtime_root: Path, profile_path: Path, fixture_path: Path,
                  branch: str, grammar_path: Path, run_dir: Path,
                  max_outputs: int, stream: str) -> dict:
    validate_request(branch, stream, max_outputs)
    runtime = runtime_root.resolve(strict=True)
    profile_path = profile_path.resolve(strict=True)
    fixture_path = fixture_path.resolve(strict=True)
    grammar_path = grammar_path.resolve(strict=True)
    output = run_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = {"schema": "grammar-long-native-reference-v1", "status": "RUNNING",
        "scope": "fresh raw native same-operator reference; no COW adapter/checkpoint; no tools",
        "branch": branch, "stream": stream, "max_outputs": max_outputs,
        "python_pid": os.getpid(), "runtime_root": str(runtime),
        "profile_path": str(profile_path), "profile_sha256": sha256_file(profile_path),
        "fixture_path": str(fixture_path), "fixture_sha256": sha256_file(fixture_path),
        "grammar_path": str(grammar_path), "grammar_sha256": sha256_file(grammar_path),
        "tokens": [], "samples": {}, "feed_receipts": [], "decode_receipts": [],
        "failures": [], "commands": [], "native_pid": None,
        "native_exit_code": None, "native_pid_absent": None}
    result_path = output / "result.json"
    _publish(result_path, record)
    heartbeat = Heartbeat(output / "heartbeat.json").start()
    native = None
    try:
        case = _read_fixture(fixture_path)
        root_ids, suffix_ids = validate_fixture(case, branch)
        if len(root_ids) != 7231 or len(suffix_ids) != {"A": 512, "B": 488}[branch]:
            raise ValueError("fixture does not match the pinned 7,231-token root and selected branch suffix")
        all_ids = root_ids + suffix_ids
        record["fixture_binding"] = {"root_tokens": len(root_ids), "suffix_tokens": len(suffix_ids),
            "history_tokens": len(all_ids), "root_token_ids_sha256": token_ids_sha256(root_ids),
            "suffix_token_ids_sha256": token_ids_sha256(suffix_ids),
            "history_token_ids_sha256": token_ids_sha256(all_ids)}

        profile = json.loads(profile_path.read_bytes())
        _apply_profile_environment(profile)
        sys.path[:0] = [str(runtime), str(runtime / "tools"),
            str(runtime / "experimental/branching_context/src"),
            str(runtime / "experimental/branching_context/examples"),
            str(runtime / "experimental/branching_context/scripts")]
        from context_time_travel.strata_backend import _validate_execution, _validate_model
        from context_time_travel.grammar_first import GrammarEngine
        from tools.strata_tokenizer import Tokenizer

        _validate_execution(profile)
        record["model_verification"] = _validate_model(profile)
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        vocab_size = len(tokenizer.tokens)
        if any(token >= vocab_size for token in all_ids):
            raise ValueError("fixture token ID is outside pinned tokenizer vocabulary")
        grammar_source = grammar_path.read_text(encoding="utf-8")
        engine = GrammarEngine(tokenizer, {"REVIEW": {"format": "gbnf", "source": grammar_source}},
                               eos_tokens=profile["execution"]["eos_tokens"])
        session = engine.restore(engine.start("REVIEW", stream, all_ids, vocab_size), all_ids)
        record["grammar_binding"] = {"state": "REVIEW", "stream": stream,
            "engine_id": engine.engine_id, "tokenizer_sha256": engine.tokenizer_sha256,
            "spec_sha256": engine.spec_sha256, "eos_tokens": list(engine.eos_tokens)}
        heartbeat.set("native-start")

        native = NativePipe(profile["execution"]["sut_command"], _native_environment(profile), output)
        record["native_pid"] = native.process.pid
        record["native_ready"] = native.ready
        if native.startup_error is not None:
            raise RuntimeError(f"native readiness failed: {type(native.startup_error).__name__}: {native.startup_error}")
        if (native.ready.get("vocab") != vocab_size or
                native.ready.get("capacity") != profile["execution"]["capacity"] or
                native.ready.get("grammar_selection_protocol") != "finite-mask-argmax-v1"):
            raise ValueError("native ready geometry/capability differs from pinned profile")
        record["native_executable"] = profile["execution"]["sut_command"][0]
        record["execution_profile_id"] = profile["execution_profile_id"]
        record["model_contract_id"] = profile["model_contract_id"]
        chunk_size = profile["execution"]["prefill_chunk"]
        # Keep the wire row and stats object separate; this reference never
        # instantiates StrataBackend or issues a checkpoint/cache command.
        stats_before = native.request("stats", "stats")
        if (stats_before.get("status") != "ok" or stats_before.get("consumed") != 0 or
                stats_before.get("forward_tokens") != 0):
            raise ValueError("fresh native process did not start at the empty frontier")
        record["stats_before"] = stats_before

        heartbeat.set("feed-prefix", native.process.pid)
        response = native.request("feed " + " ".join(map(str, root_ids)), "feed",
            {"position": 0, "count": len(root_ids), "tokens_sha256": token_ids_sha256(root_ids)})
        validate_feed_receipt(response, position=0, count=len(root_ids), chunk_size=chunk_size)
        record["feed_receipts"].append(response)
        after_root = native.request("stats", "stats")
        if (after_root.get("consumed") != len(root_ids) or
                after_root.get("forward_tokens") != len(root_ids) or
                after_root.get("prefill_tokens") != len(root_ids)):
            raise ValueError("native prefix batch did not consume exactly P")
        record["stats_after_prefix"] = after_root
        record["samples"]["prefix"] = _sample_frontier(native, output, "prefix", len(root_ids), vocab_size)

        heartbeat.set("feed-suffix", native.process.pid)
        response = native.request("feed " + " ".join(map(str, suffix_ids)), "feed",
            {"position": len(root_ids), "count": len(suffix_ids),
             "tokens_sha256": token_ids_sha256(suffix_ids)})
        validate_feed_receipt(response, position=len(root_ids), count=len(suffix_ids), chunk_size=chunk_size)
        record["feed_receipts"].append(response)
        after_suffix = native.request("stats", "stats")
        if (after_suffix.get("consumed") != len(all_ids) or
                after_suffix.get("forward_tokens") != len(all_ids) or
                after_suffix.get("prefill_tokens") != len(all_ids)):
            raise ValueError("native suffix batch did not consume exactly the requested history")
        record["stats_after_suffix"] = after_suffix
        record["samples"]["suffix"] = _sample_frontier(native, output, "suffix", len(all_ids), vocab_size)

        outputs = []
        record["generated_counts"] = []
        record["status"] = "RUNNING"
        _publish(result_path, record)
        heartbeat.set("grammar-generation", native.process.pid)
        eos = set(engine.eos_tokens)
        for _ in range(max_outputs):
            logits_path = output / "choice.f32"
            response = native.request(logits_command(logits_path), "logits",
                                      {"consumed": len(all_ids) + len(outputs)})
            raw = logits_path.read_bytes()
            logits_path.unlink()
            if response.get("status") != "ok" or response.get("bytes") != vocab_size * 4 or len(raw) != vocab_size * 4:
                raise ValueError("native choice frontier has invalid byte receipt")
            logits = struct.unpack("<" + "f" * vocab_size, raw)
            token = session.choose(logits)
            piece = tokenizer.token_bytes(token)
            next_position = len(all_ids) + len(outputs)
            decode = native.request("decode " + str(token), "decode",
                {"position": next_position, "token": token})
            validate_decode_receipt(decode, position=next_position)
            session._consume_selected(token, piece, token in eos)
            outputs.append(token)
            record["tokens"] = list(outputs)
            record["generated_counts"].append({"position": next_position, "token": token,
                "consumed": next_position + 1, "decode_receipt": decode})
            count = len(outputs)
            if count in SAMPLE_COUNTS or count == max_outputs or token in eos:
                sample_name = f"generated-{count}"
                record["samples"][sample_name] = _sample_frontier(native, output, sample_name,
                    len(all_ids) + count, vocab_size)
                record["final_sample_key"] = sample_name
            _publish(result_path, record)
            if session.is_stopped():
                break

        stats_after = native.request("stats", "stats")
        record["stats_after_generation"] = stats_after
        record["work_delta"] = validate_counter_delta(record["stats_before"], stats_after,
            forward=len(all_ids) + len(outputs), prefill=len(all_ids), decode=len(outputs))
        if (after_root.get("forward_tokens") - record["stats_before"]["forward_tokens"] != len(root_ids) or
                after_suffix.get("forward_tokens") - after_root.get("forward_tokens") != len(suffix_ids)):
            raise ValueError("Native prefix/suffix work boundaries are not exact")
        controller = session.current_controller
        record["controller"] = controller
        record["controller_sha256"] = canonical_controller_sha256(controller)
        record_grammar_outcome(record, session, outputs)
        record["tools_executed"] = 0
        record["command_counts"] = {name: sum(row["command"] == name for row in native.commands)
                                    for name in sorted(PROTOCOL_COMMANDS)}
        validate_command_inventory(native.commands)
        record["native_commands"] = native.commands
        record["status"] = "PASS"
    except BaseException as exc:
        record["status"] = "FAIL"
        record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if native is not None:
            code, absent, errors = native.close()
            record["native_exit_code"] = code
            record["native_pid_absent"] = absent
            record["native_cleanup_errors"] = errors
            record["native_commands"] = native.commands
            if code != 0 or not absent or errors:
                record["status"] = "FAIL"
        try:
            heartbeat.set("terminal", record.get("native_pid"))
            heartbeat.close()
        except BaseException as exc:
            record.setdefault("failures", []).append({"type": type(exc).__name__,
                                                       "message": "heartbeat cleanup: " + str(exc)})
            record["status"] = "FAIL"
        record["python_exit_expected"] = 0 if record.get("status") == "PASS" else 1
        record["python_elapsed_seconds"] = time.monotonic() - heartbeat.started
        _publish(result_path, record)
    if record.get("status") != "PASS":
        raise RuntimeError("reference did not reach a clean PASS terminal status")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--branch", choices=("A", "B"), required=True)
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-outputs", type=int, default=128)
    parser.add_argument("--stream", required=True)
    args = parser.parse_args(argv)
    try:
        run_reference(runtime_root=args.runtime_root, profile_path=args.profile,
            fixture_path=args.fixture, branch=args.branch, grammar_path=args.grammar,
            run_dir=args.run_dir, max_outputs=args.max_outputs, stream=args.stream)
    except BaseException as exc:
        print(f"grammar long reference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
