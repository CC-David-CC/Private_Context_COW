"""Strict contracts for an independent raw-native seed/stop numerical audit.

This module does not start the COW service and does not import its execution
adapter.  It validates the request schedule and receipts emitted by a separate
ordinary Prefill/Verifier process.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


@dataclass(frozen=True)
class InterleavedCase:
    root_tokens: int
    verifier_segments: tuple[int, ...]
    appended_tokens: int
    append_after_segment: int


def expected_operator_schedule(case: InterleavedCase) -> list[dict]:
    """Return the exact operator order and absolute positions for a case."""
    _validate_case(case)
    rows = [{"operator": "batched-prefill", "purpose": "root", "position": 0,
             "count": case.root_tokens}]
    position = case.root_tokens
    for index, count in enumerate(case.verifier_segments, start=1):
        rows.append({"operator": "normal-verifier-decode", "purpose": f"segment-{index}",
                     "position": position, "count": count})
        position += count
        if index == case.append_after_segment:
            rows.append({"operator": "batched-prefill", "purpose": "appended-user",
                         "position": position, "count": case.appended_tokens})
            position += case.appended_tokens
    return rows


def expected_plain_operator_schedule(prompt_tokens: int, segments: tuple[int, ...]) -> list[dict]:
    """Return exact positions for a fresh Prefill followed only by Verifier segments."""
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("prompt token count must be a positive exact integer")
    if (not isinstance(segments, tuple) or not segments
            or any(type(value) is not int or value <= 0 for value in segments)):
        raise ValueError("plain verifier segments must be positive exact integers")
    position = prompt_tokens
    rows = [{"operator": "batched-prefill", "purpose": "root",
             "position": 0, "count": prompt_tokens}]
    for index, count in enumerate(segments, start=1):
        rows.append({"operator": "normal-verifier-decode", "purpose": f"segment-{index}",
                     "position": position, "count": count})
        position += count
    return rows


def make_child_environment(profile: dict, *, prompt_count: int, segments: tuple[int, ...],
                           forced_ids: tuple[int, ...], append_ids: tuple[int, ...],
                           append_after_segment: int | None = None,
                           reset_verifier_segments: bool = True) -> dict[str, str]:
    """Build a scrubbed environment with only this case's native schedule controls."""
    for values, label in ((segments, "segments"), (forced_ids, "forced IDs"),
                          (append_ids, "appended IDs")):
        if not isinstance(values, tuple) or any(type(value) is not int for value in values):
            raise ValueError(f"{label} must be tuples of exact integers")
    if type(prompt_count) is not int or prompt_count <= 0:
        raise ValueError("prompt count must be a positive exact integer")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_"))}
    configured = profile.get("execution", {}).get("environment", {})
    if not isinstance(configured, dict) or any(type(k) is not str or type(v) is not str
                                               for k, v in configured.items()):
        raise ValueError("profile environment must be a string mapping")
    env.update(configured)
    # A profile may itself contain stale audit schedule controls. The active
    # case below is authoritative, including the no-append case.
    env.pop("STRATA_PREFILL_AFTER_SEGMENT", None)
    env.pop("STRATA_PREFILL_APPEND_TOKENS", None)
    env["STRATA_PREFILL_REQUESTS"] = str(prompt_count)
    env["STRATA_DECODE_SEGMENTS"] = ",".join(map(str, segments))
    env["STRATA_FORCE_GENERATED_TOKENS"] = ",".join(map(str, forced_ids))
    env["STRATA_RESET_VERIFIER_SEGMENTS"] = "1" if reset_verifier_segments else "0"
    if append_ids:
        if type(append_after_segment) is not int:
            raise ValueError("append boundary must be an exact integer")
        env["STRATA_PREFILL_AFTER_SEGMENT"] = str(append_after_segment)
        env["STRATA_PREFILL_APPEND_TOKENS"] = ",".join(map(str, append_ids))
    elif append_after_segment is not None:
        raise ValueError("append boundary supplied without appended IDs")
    return env


def _validate_case(case: InterleavedCase) -> None:
    if not isinstance(case, InterleavedCase):
        raise ValueError("case must be InterleavedCase")
    counts = (case.root_tokens, case.appended_tokens, case.append_after_segment,
              *case.verifier_segments)
    if any(type(value) is not int for value in counts):
        raise ValueError("schedule counts must be exact integers")
    if case.root_tokens <= 0 or case.appended_tokens <= 0 or not case.verifier_segments:
        raise ValueError("root, append, and verifier segments must be nonempty")
    if any(value <= 0 for value in case.verifier_segments):
        raise ValueError("verifier segments must be positive")
    if not 1 <= case.append_after_segment < len(case.verifier_segments):
        raise ValueError("append boundary must separate two verifier segments")


def validate_schedule(observed: object, expected: list[dict]) -> list[dict]:
    """Reject missing, reordered, extra, or position-shifted native operations."""
    if not isinstance(observed, list) or observed != expected:
        raise ValueError("raw-native operator schedule differs from exact interleaved plan")
    return observed


def validate_case_receipt(receipt: object, case: InterleavedCase,
                          observed_schedule: object, expected_tokens: list[int]) -> dict:
    """Validate work/count/identity receipts; no numerical comparison is inferred."""
    _validate_case(case)
    expected = expected_operator_schedule(case)
    validate_schedule(observed_schedule, expected)
    if not isinstance(receipt, dict):
        raise ValueError("raw-native receipt must be an object")
    if (receipt.get("status") != "ok" or receipt.get("exit_code") != 0
            or receipt.get("pid_absent") is not True):
        raise ValueError("raw-native process did not terminate cleanly")
    generated = sum(case.verifier_segments)
    prefilled = case.root_tokens + case.appended_tokens
    if (receipt.get("generated") != expected_tokens
            or len(expected_tokens) != generated
            or receipt.get("normal_forward_tokens") != generated
            or receipt.get("normal_decode_windows") != generated
            or receipt.get("initial_prefill_tokens") != case.root_tokens
            or receipt.get("appended_prefill_tokens") != case.appended_tokens
            or receipt.get("prefill_tokens") != prefilled
            or receipt.get("consumed") != prefilled + generated
            or receipt.get("operator_schedule") != expected):
        raise ValueError("raw-native work receipt differs from the requested schedule")
    return receipt


def build_raw_command(profile: dict, binary: Path, token_file: Path,
                      decode_count: int) -> list[str]:
    execution = profile.get("execution") if isinstance(profile, dict) else None
    template = execution.get("oracle_command") if isinstance(execution, dict) else None
    if not isinstance(template, list) or not template or any(type(item) is not str for item in template):
        raise ValueError("profile lacks a pinned oracle command")
    if "--max-new" not in template or "--tokens-file" in template:
        raise ValueError("oracle command lacks max-new or already binds a token file")
    for flag in ("--prompt-cache", "--conversation-cache-mib"):
        if flag not in template or template[template.index(flag) + 1:template.index(flag) + 2] != ["0"]:
            raise ValueError(f"raw oracle must disable {flag}")
    if {"--checkpoint", "--load-state", "--state-file", "--import-kv"}.intersection(template):
        raise ValueError("raw oracle command contains a checkpoint import option")
    command = list(template)
    command[0] = str(binary)
    command[command.index("--max-new") + 1] = str(decode_count)
    command.extend(["--tokens-file", str(token_file)])
    return command


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_receipt(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _read_frontier_rows(path: Path, count: int, vocab_size: int) -> list[bytes]:
    width = vocab_size * 4
    if type(count) is not int or count <= 0 or not path.is_file():
        raise ValueError(f"frontier file is missing or count invalid: {path}")
    raw = path.read_bytes()
    if len(raw) != width * count:
        raise ValueError(f"frontier bytes do not match count and vocab width: {path}")
    return [raw[index * width:(index + 1) * width] for index in range(count)]


def run_raw_case(*, profile: dict, binary: Path, run_dir: Path, case_name: str,
                 prompt_ids: list[int], forced_ids: list[int], segments: tuple[int, ...],
                 vocab_size: int, appended_ids: list[int] | None = None,
                 append_after_segment: int | None = None,
                 reset_verifier_segments: bool = True, timeout_seconds: int = 1200) -> dict:
    """Execute one fresh raw native case and retain every frontier/operator receipt.

    This function imports no COW backend or BranchSession. It consumes explicit
    token files in ordinary native Prefill and Verifier operators only.
    """
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocabulary size must be a positive exact integer")
    for label, ids in (("prompt", prompt_ids), ("forced", forced_ids),
                       ("appended", appended_ids or [])):
        if (not isinstance(ids, list) or any(type(token) is not int or not 0 <= token < vocab_size
                                             for token in ids)):
            raise ValueError(f"{label} IDs are invalid for the pinned vocabulary")
    if not segments or any(type(size) is not int or size <= 0 for size in segments):
        raise ValueError("decode segments must be positive exact integers")
    if sum(segments) != len(forced_ids):
        raise ValueError("forced output count differs from verifier segment sum")
    append_ids = list(appended_ids or [])
    if append_ids:
        if (type(append_after_segment) is not int
                or not 1 <= append_after_segment < len(segments)):
            raise ValueError("interleaved append boundary must divide verifier segments")
        case = InterleavedCase(len(prompt_ids), tuple(segments), len(append_ids),
                               append_after_segment)
    else:
        if append_after_segment is not None:
            raise ValueError("append boundary supplied without appended token IDs")
        case = None
    run_dir = Path(run_dir).resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    token_file = run_dir / "prompt.tokens"
    token_file.write_text(",".join(map(str, prompt_ids)), encoding="ascii", newline="")
    binary = Path(binary).resolve(strict=True)
    if not os.access(binary, os.X_OK):
        raise ValueError("raw oracle binary is not executable")
    command = build_raw_command(profile, binary, token_file, len(forced_ids))
    logits_path = run_dir / "raw-frontier.f32"
    env = make_child_environment(profile, prompt_count=len(prompt_ids), segments=tuple(segments),
                                 forced_ids=tuple(forced_ids), append_ids=tuple(append_ids),
                                 append_after_segment=append_after_segment,
                                 reset_verifier_segments=reset_verifier_segments)
    env["STRATA_DUMP_FIRST_LOGITS"] = str(logits_path)
    started = time.monotonic()
    with (run_dir / "stdout.jsonl").open("xb") as stdout, (run_dir / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                 start_new_session=True, cwd="/")
        pid = child.pid
        proc_record = {"status": "RUNNING", "pid": pid, "argv": command,
                       "case": case_name, "gpu_launched_by_audit_runner": True,
                       "cow_service_started": False, "checkpoint_imports": 0,
                       "branch_sessions": 0, "heartbeat_count": 0}
        _write_receipt(run_dir / "process.json", proc_record)
        deadline = started + timeout_seconds
        heartbeat = started + 30
        try:
            while True:
                code = child.poll()
                now = time.monotonic()
                if code is not None:
                    break
                if now >= deadline:
                    os.killpg(pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(pid, signal.SIGKILL)
                        code = child.wait(timeout=20)
                    raise TimeoutError(f"raw native case exceeded timeout: {case_name}")
                if now >= heartbeat:
                    proc_record.update({"elapsed_seconds": now - started,
                                        "heartbeat_count": proc_record["heartbeat_count"] + 1})
                    _write_receipt(run_dir / "process.json", proc_record)
                    heartbeat = now + 30
                time.sleep(min(1.0, max(0.01, heartbeat - now)))
        except BaseException as error:
            if child.poll() is None:
                os.killpg(pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL)
                    child.wait(timeout=20)
            proc_record.update({"status": "FAILED", "exit_code": child.poll(),
                                "error": f"{type(error).__name__}: {error}",
                                "elapsed_seconds": time.monotonic() - started})
            _write_receipt(run_dir / "process.json", proc_record)
            raise
    absent = not Path("/proc", str(pid)).exists()
    proc_record.update({"status": "EXITED", "exit_code": code,
                        "pid_absent": absent, "elapsed_seconds": time.monotonic() - started})
    _write_receipt(run_dir / "process.json", proc_record)
    if code != 0 or not absent:
        raise RuntimeError(f"raw native child failed cleanup: pid={pid} exit={code} absent={absent}")
    stdout_raw = (run_dir / "stdout.jsonl").read_bytes()
    if not stdout_raw.endswith(b"\n"):
        raise ValueError("raw native work receipt is not newline terminated")
    work = json.loads(stdout_raw)
    schedule = json.loads(Path(str(logits_path) + ".schedule.json").read_bytes())
    if append_ids:
        expected_schedule = expected_operator_schedule(case)
        validate_schedule(work.get("operator_schedule"), expected_schedule)
        if schedule.get("operator_schedule") != expected_schedule:
            raise ValueError("sidecar native schedule differs from stdout schedule")
        receipt = {**work, "status": "ok", "exit_code": code, "pid_absent": absent}
        validate_case_receipt(receipt, case, schedule.get("operator_schedule"), forced_ids)
    else:
        expected_schedule = expected_plain_operator_schedule(len(prompt_ids), tuple(segments))
        validate_schedule(work.get("operator_schedule"), expected_schedule)
        if schedule.get("operator_schedule") != expected_schedule:
            raise ValueError("stdout and sidecar plain operator schedules differ from requested positions")
        if work.get("generated") != forced_ids or work.get("decode_tokens") != len(forced_ids):
            raise ValueError("raw native output count or IDs differ from forced sequence")
        if work.get("normal_forward_tokens") != len(forced_ids):
            raise ValueError("raw native forward count differs from forced sequence")
        if (work.get("normal_decode_windows") != len(forced_ids)
                or work.get("initial_prefill_tokens") != len(prompt_ids)
                or work.get("appended_prefill_tokens") != 0
                or work.get("prefill_tokens") != len(prompt_ids)):
            raise ValueError("raw native plain operator work counts differ from requested schedule")
        if (work.get("prompt_consumed") != len(prompt_ids)
                or work.get("consumed") != len(prompt_ids) + len(forced_ids)):
            raise ValueError("raw native no-append position counts are inconsistent")
    result = {"case": case_name, "pid": pid, "exit_code": code, "pid_absent": absent,
              "status": "PASS_RECEIPTS", "argv": command,
              "work": work, "schedule": schedule,
              "prompt_frontiers": _read_frontier_rows(Path(str(logits_path) + ".requests.f32"), 1, vocab_size),
              "decode_frontiers": _read_frontier_rows(Path(str(logits_path) + ".decode.f32"), len(forced_ids), vocab_size),
              "segment_frontiers": _read_frontier_rows(Path(str(logits_path) + ".segments.f32"), len(segments), vocab_size),
              "frontier_sha256": {"prompt": _sha256_file(Path(str(logits_path) + ".requests.f32")),
                                  "decode": _sha256_file(Path(str(logits_path) + ".decode.f32")),
                                  "segments": _sha256_file(Path(str(logits_path) + ".segments.f32"))}}
    if append_ids:
        append_frontier = Path(str(logits_path) + ".interleaved-prefill.f32")
        result["append_prefill_frontier"] = append_frontier.read_bytes()
        if len(result["append_prefill_frontier"]) != vocab_size * 4:
            raise ValueError("interleaved Prefill frontier width differs from vocabulary")
        result["frontier_sha256"]["append_prefill"] = _sha256_file(append_frontier)
    return result
