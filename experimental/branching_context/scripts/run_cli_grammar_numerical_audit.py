"""Independent raw Prefill+Verifier audit for a completed grammar CLI flow.

The script never starts the COW service. It reads and validates the four SUT
checkpoint frontiers, then runs a separate no-checkpoint raw oracle with the
canonical output IDs forced through ordinary Verifier consumption. Grammar
legality and greedy argmax are checked separately in Python using the pinned
GrammarEngine; no native sampler or BranchSession is used for that decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import uuid


VOCAB_SIZE = 248320
FRONTIER_WIDTH = VOCAB_SIZE * 4
REQUIRED_BOUNDARIES = ("root", "read_partial", "read_final", "test_final")
LLGUIDANCE_NATIVE_SHA256 = "131282df4f51cc55e938a7501789a51d2b3a25ab2abc7c6676b951ba908c924a"
FORCED_TOKEN_SOURCE_FIXTURE = """int main() {
    std::vector<int32_t> generated;
        int32_t token = 0;
        for (int32_t t = 1; t < (int32_t)logits.size(); ++t)
            if (logits[t] > logits[token]) token = t;
}
"""
_GREEDY_ANCHOR = """        int32_t token = 0;
        for (int32_t t = 1; t < (int32_t)logits.size(); ++t)
            if (logits[t] > logits[token]) token = t;"""
_FORCED_SELECTION = """        if (forced_tokens.size() != static_cast<size_t>(decode_steps)) {
            std::cerr << "forced token count differs from decode schedule" << std::endl; return 2;
        }
        const int32_t token = forced_tokens[static_cast<size_t>(i)];"""
_GENERATED_ANCHOR = "    std::vector<int32_t> generated;"
_FORCED_PARSE = """    std::vector<int32_t> forced_tokens;
    const char* supplied_forced = std::getenv("STRATA_FORCE_GENERATED_TOKENS");
    if (!supplied_forced || !*supplied_forced) {
        std::cerr << "missing forced token list" << std::endl; return 2;
    }
    std::string forced_text(supplied_forced);
    std::replace(forced_text.begin(), forced_text.end(), ',', ' ');
    std::istringstream forced_stream(forced_text);
    int64_t forced_token = 0;
    while (forced_stream >> forced_token) {
        if (forced_token < 0 || forced_token >= static_cast<int64_t>(logits.size())) {
            std::cerr << "forced token outside vocabulary" << std::endl; return 2;
        }
        forced_tokens.push_back(static_cast<int32_t>(forced_token));
    }
    if (!forced_stream.eof() || forced_tokens.size() != static_cast<size_t>(decode_steps)) {
        std::cerr << "invalid forced token schedule" << std::endl; return 2;
    }
"""

_RAW_WORK_DETERMINISTIC_FIELDS = (
    "consumed", "prompt_consumed", "decode_tokens", "normal_forward_tokens",
    "normal_decode_windows", "stopping_reason", "prefill_tokens", "prefill_chunks",
    "generated",
)
_RAW_WORK_TIMING_FIELDS = ("prefill_seconds", "decode_seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned_grammar_registry(runtime_root, profile, registry_path, grammar_deps):
    runtime_root = Path(runtime_root).resolve(strict=True)
    registry_path = Path(registry_path).resolve(strict=True)
    grammar_deps = Path(grammar_deps).resolve(strict=True)
    native_path = grammar_deps / "llguidance" / "_lib.abi3.so"
    if sha256_file(native_path) != LLGUIDANCE_NATIVE_SHA256:
        raise ValueError("llguidance native extension differs from the pinned dependency")
    api_root = runtime_root / "experimental" / "branching_context"
    sys.path[:0] = [str(grammar_deps), str(runtime_root), str(runtime_root / "tools"),
                    str(api_root / "src"), str(api_root / "scripts")]
    import llguidance
    if (getattr(llguidance, "__version__", None) != "1.9.1"
            or not Path(llguidance.__file__).resolve(strict=True).is_relative_to(grammar_deps)):
        raise ValueError("llguidance import is not the pinned 1.9.1 dependency")
    from tools.strata_tokenizer import Tokenizer
    from serve.cow_grammar import GrammarRegistry
    tokenizer = Tokenizer.from_gguf(profile["model_path"])
    if len(tokenizer.tokens) != VOCAB_SIZE:
        raise ValueError("tokenizer vocabulary differs from the pinned campaign")
    registry = GrammarRegistry.load(registry_path, tokenizer,
                                    profile["execution"]["eos_tokens"])
    if sorted(registry.states) != ["READ", "TEST"]:
        raise ValueError("pinned campaign registry state names differ")
    return registry, {
        "llguidance_version": llguidance.__version__,
        "llguidance_python_path": str(Path(llguidance.__file__).resolve(strict=True)),
        "llguidance_native_path": str(native_path),
        "llguidance_native_sha256": sha256_file(native_path),
        "grammar_registry_path": str(registry_path),
        "grammar_engine_id": registry.engine_id,
        "grammar_spec_sha256": registry.spec_sha256,
        "tokenizer_sha256": registry.tokenizer_sha256,
        "states": list(registry.states),
    }


def validate_token_ids(tokens, *, expected_count, vocab_size=VOCAB_SIZE):
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("token expected count is invalid")
    if not isinstance(tokens, list) or len(tokens) != expected_count:
        raise ValueError("token count differs from the audited schedule")
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in tokens):
        raise ValueError("token ID is invalid or outside vocabulary")
    return list(tokens)


def validate_trajectories(*, root_token_ids, read_partial_ids, read_finish_ids,
                          test_ids, read_segments, test_segments, eos_tokens,
                          vocab_size=VOCAB_SIZE):
    root = validate_token_ids(root_token_ids, expected_count=len(root_token_ids), vocab_size=vocab_size)
    partial = validate_token_ids(read_partial_ids, expected_count=len(read_partial_ids), vocab_size=vocab_size)
    finish = validate_token_ids(read_finish_ids, expected_count=len(read_finish_ids), vocab_size=vocab_size)
    test = validate_token_ids(test_ids, expected_count=len(test_ids), vocab_size=vocab_size)
    eos = set(validate_token_ids(eos_tokens, expected_count=len(eos_tokens), vocab_size=vocab_size))
    if not root or not partial or not finish or not test or not eos:
        raise ValueError("grammar audit histories and EOS set must be nonempty")
    for name, ids, segments in (("READ", partial + finish, read_segments), ("TEST", test, test_segments)):
        if (not isinstance(segments, list) or not segments
                or any(type(count) is not int or count <= 0 for count in segments)
                or sum(segments) != len(ids)):
            raise ValueError(f"{name} segment schedule does not partition its outputs")
        if any(token in eos for token in ids[:-1]):
            raise ValueError(f"{name} contains EOS before the terminal output")
        if ids[-1] not in eos:
            raise ValueError(f"{name} output does not end in EOS")
    return {
        "root": {"input": root},
        "read": {"outputs": partial + finish, "segments": list(read_segments),
                 "boundaries": [len(partial), len(partial) + len(finish)]},
        "test": {"outputs": test, "segments": list(test_segments),
                 "boundaries": [len(test)]},
        "eos_tokens": sorted(eos),
        "vocab_size": vocab_size,
    }


def validate_frontier_bytes(raw, *, vocab_size=VOCAB_SIZE):
    if type(raw) is not bytes or type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("frontier bytes or vocabulary width is invalid")
    if len(raw) != vocab_size * 4:
        raise ValueError(f"frontier width must be exactly {vocab_size * 4} bytes")
    values = struct.unpack(f"<{vocab_size}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("frontier contains nonfinite values")
    return values


def normalize_grammar_mask(mask, vocab_size):
    """Trim llguidance's padded bitset to the requested token vocabulary.

    GrammarSession applies the same boundary: matcher bitsets may include
    whole bytes beyond the vocabulary, while only the first ceil(vocab/8)
    bytes address token IDs. The in-vocabulary bits are preserved; unused bits
    in the last partial byte are cleared.
    """
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("grammar vocabulary size is invalid")
    if type(mask) not in (bytes, bytearray):
        raise ValueError("grammar matcher returned an invalid token mask")
    width = (vocab_size + 7) // 8
    if len(mask) < width:
        raise ValueError("grammar mask is shorter than the vocabulary")
    normalized = bytearray(mask[:width])
    remainder = vocab_size % 8
    if remainder:
        normalized[-1] &= (1 << remainder) - 1
    return bytes(normalized)


def compare_frontiers(actual, expected, label):
    validate_frontier_bytes(actual, vocab_size=len(actual) // 4)
    validate_frontier_bytes(expected, vocab_size=len(expected) // 4)
    if len(actual) != len(expected):
        raise AssertionError(f"{label} frontier width differs")
    if actual == expected:
        return 0.0
    actual_values = struct.unpack(f"<{len(actual)//4}f", actual)
    expected_values = struct.unpack(f"<{len(expected)//4}f", expected)
    max_error = max(abs(a - b) for a, b in zip(actual_values, expected_values))
    raise AssertionError(f"{label} frontier differs; max_abs_error={max_error}")


def greedy_legal_argmax(logits, mask, *, special_ids, eos_tokens, accepting):
    """Independent ID-ordered greedy argmax over a GrammarEngine bitmask."""
    if not isinstance(logits, (list, tuple)) or not logits:
        raise ValueError("raw frontier must be a nonempty numeric sequence")
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in logits):
        raise ValueError("raw frontier contains a nonfinite or nonnumeric value")
    vocab = len(logits)
    mask = normalize_grammar_mask(mask, vocab)
    specials = set(special_ids)
    eos = set(eos_tokens)
    legal = []
    for token in range(vocab):
        if not mask[token >> 3] & (1 << (token & 7)):
            continue
        if token in specials and token not in eos:
            continue
        if token in eos and not accepting:
            continue
        legal.append(token)
    if not legal:
        raise ValueError("grammar has no legal token")
    return max(legal, key=lambda token: (float(logits[token]), -token)), legal


def controller_progress_projection(controller):
    """Project only grammar/parser progress, excluding request stream identity."""
    if not isinstance(controller, dict):
        raise ValueError("grammar controller is missing")
    constraint = controller.get("constraint")
    if not isinstance(constraint, dict):
        raise ValueError("grammar controller constraint is missing")
    return {
        "engine_id": constraint.get("engine_id"),
        "spec_sha256": constraint.get("spec_sha256"),
        "tokenizer_sha256": constraint.get("tokenizer_sha256"),
        "agent_state": constraint.get("agent_state"),
        "scope_start": constraint.get("scope_start"),
        "accepted_in_scope": controller.get("accepted_in_scope"),
        "status": controller.get("status"),
        "seen": controller.get("seen"),
        "output_pending_hex": controller.get("output_pending_hex"),
    }


def controller_audit_receipt(controller):
    """Keep controller provenance useful without copying the full seen-token map."""
    if not isinstance(controller, dict):
        raise ValueError("controller is missing")
    constraint = controller.get("constraint")
    if constraint is None:
        constraint_projection = None
    elif isinstance(constraint, dict):
        constraint_projection = {key: constraint.get(key) for key in
                                 ("engine_id", "spec_sha256", "tokenizer_sha256",
                                  "agent_state", "scope_start")}
    else:
        raise ValueError("controller constraint is malformed")
    encoded = json.dumps(controller, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {
        "controller_sha256": hashlib.sha256(encoded).hexdigest(),
        "progress": {
            "schema": controller.get("schema"),
            "constraint": constraint_projection,
            "accepted_in_scope": controller.get("accepted_in_scope"),
            "status": controller.get("status"),
            "output_pending_hex": controller.get("output_pending_hex"),
        },
        "seen_token_count": len(controller.get("seen", {})),
    }


def validate_controller_progress(actual, expected, label):
    if controller_progress_projection(actual) != controller_progress_projection(expected):
        raise AssertionError(f"{label} grammar parser/controller progress differs")


def validate_controller_equal(actual, expected, label):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError(f"{label} canonical controller is missing")
    actual_bytes = json.dumps(actual, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
    expected_bytes = json.dumps(expected, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode("utf-8")
    if actual_bytes != expected_bytes:
        raise AssertionError(f"{label} complete canonical controller differs")
    return hashlib.sha256(actual_bytes).hexdigest()


def campaign_operation_id(campaign, label):
    rows = [row for row in campaign.get("commands", [])
            if isinstance(row, dict) and row.get("label") == label]
    if len(rows) != 1:
        raise ValueError(f"campaign must contain exactly one {label} command")
    argv = rows[0].get("argv_redacted")
    if not isinstance(argv, list) or "--operation-id" not in argv:
        raise ValueError(f"campaign {label} command lacks its generation operation ID")
    index = argv.index("--operation-id")
    if index + 1 >= len(argv) or not isinstance(argv[index + 1], str) or not argv[index + 1]:
        raise ValueError(f"campaign {label} operation ID is invalid")
    return argv[index + 1]


def compare_raw_trajectories(first, second, label):
    comparisons = []
    for field in ("prompt_frontier", "final_prefill_frontier"):
        comparisons.append({"label": f"{label} {field}",
                            "max_abs_error": compare_frontiers(
                                first[field], second[field], f"{label} {field}")})
    for field in ("decode_frontiers", "segment_frontiers"):
        if len(first[field]) != len(second[field]):
            raise AssertionError(f"{label} {field} count differs")
        for index, (left, right) in enumerate(zip(first[field], second[field])):
            comparisons.append({"label": f"{label} {field}[{index}]",
                                "max_abs_error": compare_frontiers(
                                    left, right, f"{label} {field}[{index}]")})
    if first["schedule"] != second["schedule"]:
        raise AssertionError(f"{label} native schedule differs between fresh runs")
    try:
        first_work = validate_raw_work_receipt(first["work"])
        second_work = validate_raw_work_receipt(second["work"])
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{label} native work receipt is invalid: {error}") from error
    if first_work != second_work:
        raise AssertionError(f"{label} native work receipt differs between fresh runs")
    return comparisons


def validate_raw_work_receipt(work, *, expected_generated=None):
    """Validate the pinned work receipt and project deterministic fields only.

    Wall-clock measurements are validated and retained in a separate record;
    they are expected to differ between fresh executions.
    """
    expected_keys = set(_RAW_WORK_DETERMINISTIC_FIELDS + _RAW_WORK_TIMING_FIELDS)
    if type(work) is not dict or set(work) != expected_keys:
        raise ValueError("raw oracle work receipt has missing or unknown fields")
    integer_fields = (
        "consumed", "prompt_consumed", "decode_tokens", "normal_forward_tokens",
        "normal_decode_windows", "prefill_tokens", "prefill_chunks",
    )
    for field in integer_fields:
        value = work[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"raw oracle work receipt {field} is invalid")
    if work["consumed"] != work["prompt_consumed"] + work["decode_tokens"]:
        raise ValueError("raw oracle consumed count is inconsistent")
    if (work["normal_forward_tokens"] != work["decode_tokens"]
            or work["prefill_tokens"] != work["prompt_consumed"]):
        raise ValueError("raw oracle forward/prefill counts are inconsistent")
    if work["prefill_chunks"] <= 0:
        raise ValueError("raw oracle prefill chunk count is invalid")
    if not isinstance(work["stopping_reason"], str) or not work["stopping_reason"]:
        raise ValueError("raw oracle stopping reason is invalid")
    generated = work["generated"]
    validate_token_ids(generated, expected_count=len(generated), vocab_size=VOCAB_SIZE)
    if expected_generated is not None and generated != expected_generated:
        raise ValueError("raw oracle generated IDs differ from the forced sequence")
    for field in _RAW_WORK_TIMING_FIELDS:
        value = work[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"raw oracle timing {field} is invalid")
    return {field: work[field] for field in _RAW_WORK_DETERMINISTIC_FIELDS}


def audit_forced_grammar_trajectory(engine, agent_state, stream, root_history,
                                    token_ids, frontiers, *, boundary_counts=()):
    """Check legal greedy choice and parser consumption at every fresh frontier.

    ``frontiers[0]`` is raw Prefill output; each later frontier is the raw
    Verifier output after consuming the corresponding prior token. No SUT
    selector, saved COW state, or checkpoint bytes enter this routine.
    """
    validate_token_ids(root_history, expected_count=len(root_history), vocab_size=engine.vocab_size)
    validate_token_ids(token_ids, expected_count=len(token_ids), vocab_size=engine.vocab_size)
    if len(frontiers) != len(token_ids) + 1:
        raise ValueError("grammar audit needs one prefill frontier plus one frontier per consumed token")
    if any(type(count) is not int or count <= 0 for count in boundary_counts):
        raise ValueError("grammar boundary counts must be positive exact integers")
    if boundary_counts and sum(boundary_counts) != len(token_ids):
        raise ValueError("grammar boundary counts do not partition generated tokens")
    state = engine.start(agent_state, stream, root_history, engine.vocab_size)
    session = engine.restore(state, root_history)
    reports = []
    boundary_controllers = []
    boundary_set = set()
    running = 0
    for count in boundary_counts:
        running += count
        boundary_set.add(running)
    for index, token in enumerate(token_ids):
        logits = frontiers[index]
        if type(logits) is bytes:
            logits = validate_frontier_bytes(logits, vocab_size=engine.vocab_size)
        if len(logits) != engine.vocab_size:
            raise ValueError("grammar audit raw logits width differs from tokenizer vocabulary")
        mask = session._matcher.compute_bitmask()
        accepting = bool(session._matcher.is_accepting())
        selected, legal = greedy_legal_argmax(
            logits, bytes(mask), special_ids=engine._special_ids,
            eos_tokens=engine.eos_tokens, accepting=accepting)
        if selected != token:
            raise AssertionError(
                f"forced canonical token {token} is not grammar-legal greedy argmax {selected} at {index}")
        if token not in legal:
            raise AssertionError("forced canonical token is outside independently calculated legal support")
        current = session.current_controller
        after = session.consume(current, token, engine._adapter.tokens[token],
                                token in engine.eos_tokens)
        reports.append({"index": index, "token": token, "selected": selected,
                        "legal_count": len(legal), "eos": token in engine.eos_tokens})
        if index + 1 in boundary_set:
            boundary_controllers.append(after)
    final = session.current_controller
    if boundary_counts and len(boundary_controllers) != len(boundary_counts):
        raise AssertionError("grammar parser boundary receipt count differs")
    return {"token_checks": reports, "final_controller": final,
            "boundary_controllers": boundary_controllers,
            "stopped": session.is_stopped(), "complete": session.is_complete()}


def _read_frontier_sequence(path: Path, count: int):
    raw = Path(path).read_bytes()
    if type(count) is not int or count < 0 or len(raw) != count * FRONTIER_WIDTH:
        raise ValueError(f"frontier trace {path} has the wrong byte count")
    return [raw[offset:offset + FRONTIER_WIDTH]
            for offset in range(0, len(raw), FRONTIER_WIDTH)]


def read_final_prefill_frontier(logits_path: Path, prompted_frontier: bytes):
    """Read the file named by STRATA_DUMP_FIRST_LOGITS (already the .f32 path)."""
    path = Path(logits_path)
    if path.stat().st_size != FRONTIER_WIDTH:
        raise ValueError(f"raw oracle final prefill frontier is incomplete for {path}")
    final_prefill = path.read_bytes()
    if final_prefill != prompted_frontier:
        raise ValueError("raw oracle final/request Prefill frontier differs")
    return final_prefill


def _run_forced_raw_oracle(profile, binary: Path, output_dir: Path, name: str,
                           prompt_ids, forced_ids, segments):
    if sum(segments) != len(forced_ids) or not forced_ids:
        raise ValueError("forced raw-oracle segments do not partition the supplied IDs")
    output_dir.mkdir(parents=True, exist_ok=False)
    token_file = output_dir / f"{name}.tokens"
    token_file.write_text(",".join(map(str, prompt_ids)), encoding="ascii", newline="")
    logits_path = output_dir / f"{name}.f32"
    command = list(profile["execution"]["oracle_command"])
    if "--max-new" not in command:
        raise ValueError("pinned profile oracle command lacks --max-new")
    command[0] = str(binary)
    command[command.index("--max-new") + 1] = str(len(forced_ids))
    command += ["--tokens-file", str(token_file)]
    env = dict(os.environ)
    env.update(profile["execution"].get("environment", {}))
    env["STRATA_DUMP_FIRST_LOGITS"] = str(logits_path)
    env["STRATA_PREFILL_REQUESTS"] = str(len(prompt_ids))
    env["STRATA_DECODE_SEGMENTS"] = ",".join(map(str, segments))
    env["STRATA_FORCE_GENERATED_TOKENS"] = ",".join(map(str, forced_ids))
    with (output_dir / f"{name}.stdout").open("xb") as stdout, \
            (output_dir / f"{name}.stderr").open("xb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                 start_new_session=True)
        pid = child.pid
        started = time.monotonic()
        heartbeat_count = 0
        process_path = output_dir / f"{name}.process.json"
        process_receipt = {
            "status": "RUNNING", "phase": "raw-prefill-and-verifier",
            "pid": pid, "argv": command,
            "forced_token_count": len(forced_ids), "segments": segments,
            "checkpoint_imports": 0, "branch_sessions": 0,
        }
        try:
            _atomic_json(process_path, {**process_receipt, "elapsed_seconds": 0.0,
                                        "heartbeat_count": heartbeat_count})
            deadline = started + 900
            next_heartbeat = started + 30
            while True:
                code = child.poll()
                now = time.monotonic()
                if code is not None:
                    break
                if now >= deadline:
                    import signal
                    os.killpg(pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(pid, signal.SIGKILL)
                        code = child.wait(timeout=20)
                    _atomic_json(process_path, {**process_receipt, "status": "TIMEOUT",
                                                "exit_code": code,
                                                "elapsed_seconds": now - started,
                                                "heartbeat_count": heartbeat_count})
                    raise TimeoutError(f"raw forced oracle timed out: {name} pid={pid} exit={code}")
                if now >= next_heartbeat:
                    heartbeat_count += 1
                    _atomic_json(process_path, {**process_receipt,
                                                "elapsed_seconds": now - started,
                                                "heartbeat_count": heartbeat_count})
                    next_heartbeat = now + 30
                time.sleep(min(1.0, max(0.01, next_heartbeat - now)))
        except BaseException as error:
            if child.poll() is None:
                import signal
                os.killpg(pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL)
                    child.wait(timeout=20)
            _atomic_json(process_path, {**process_receipt,
                                        "status": "TIMEOUT" if isinstance(error, TimeoutError) else "FAILED",
                                        "exit_code": child.poll(),
                                        "elapsed_seconds": time.monotonic() - started,
                                        "heartbeat_count": heartbeat_count,
                                        "error": repr(error)})
            raise
    _atomic_json(process_path, {**process_receipt, "status": "EXITED", "exit_code": code,
                                "elapsed_seconds": time.monotonic() - started,
                                "heartbeat_count": heartbeat_count})
    if code != 0 or Path("/proc", str(pid)).exists():
        raise RuntimeError(f"raw forced oracle failed or left process alive: {name} pid={pid} exit={code}")
    work = json.loads((output_dir / f"{name}.stdout").read_text(encoding="utf-8"))
    validate_raw_work_receipt(work, expected_generated=forced_ids)
    schedule = json.loads(Path(str(logits_path) + ".schedule.json").read_text(encoding="utf-8"))
    generated = work.get("generated")
    if generated != forced_ids:
        raise ValueError(f"raw oracle did not consume the exact forced sequence for {name}")
    if (work.get("decode_tokens") != len(forced_ids)
            or work.get("normal_forward_tokens") != len(forced_ids)
            or work.get("prefill_tokens") != len(prompt_ids)
            or work.get("prompt_consumed") != len(prompt_ids)):
        raise ValueError(f"raw oracle work receipt differs for {name}")
    expected_segments = []
    generated_start = 0
    for count in segments:
        expected_segments.append({"generated_start": generated_start,
                                  "native_start_position": len(prompt_ids) + generated_start,
                                  "count": count})
        generated_start += count
    if (schedule.get("decode_operator") != "normal-verifier-decode"
            or schedule.get("requested_segments") != segments
            or schedule.get("completed_segments") != expected_segments):
        raise ValueError(f"raw oracle decode schedule differs for {name}")
    decoded = _read_frontier_sequence(Path(str(logits_path) + ".decode.f32"), len(forced_ids))
    prompted = _read_frontier_sequence(Path(str(logits_path) + ".requests.f32"), 1)
    final_prefill = read_final_prefill_frontier(logits_path, prompted[0])
    return {"pid": pid, "exit_code": code, "command": command, "work": work,
            "schedule": schedule, "prompt_frontier": prompted[0],
            "final_prefill_frontier": final_prefill,
            "decode_frontiers": decoded,
            "segment_frontiers": _read_frontier_sequence(
                Path(str(logits_path) + ".segments.f32"), len(segments))}


def _campaign_row(campaign, key):
    value = campaign.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("receipt"), dict):
        raise ValueError(f"campaign is missing durable {key} receipt")
    return value


def _compare_snapshot_checkpoint(store, snapshot_id, receipt, history, profile, label):
    from verify_cli_workflow_numerics import _snapshot_frontier
    if receipt.get("snapshot") != snapshot_id:
        raise ValueError(f"{label} campaign receipt snapshot differs")
    payload_hash = receipt.get("payload_sha256")
    if not isinstance(payload_hash, str):
        raise ValueError(f"{label} campaign payload digest is missing")
    manifest, frontier, actual_payload_hash = _snapshot_frontier(
        store.root, snapshot_id, payload_hash, history=history,
        model_contract_id=profile["model_contract_id"], profile=profile)
    snapshot = store.get_snapshot(snapshot_id)
    return {"manifest": manifest, "frontier": frontier, "payload_sha256": actual_payload_hash,
            "snapshot_id": snapshot_id, "snapshot": snapshot,
            "controller": snapshot.get("controller")}


def validate_lineage_roots(root_id, lineage_roots):
    if not isinstance(root_id, str) or not root_id:
        raise ValueError("canonical root ID is invalid")
    if not isinstance(lineage_roots, dict) or not lineage_roots:
        raise ValueError("lineage roots are missing")
    for label, lineage_root in lineage_roots.items():
        if lineage_root != root_id:
            raise ValueError(f"{label} lineage does not share the canonical source root")
    return True


def snapshot_audit_summary(row):
    snapshot_id = row.get("snapshot_id")
    snapshot = row.get("snapshot")
    if not isinstance(snapshot_id, str) or not isinstance(snapshot, dict):
        raise ValueError("snapshot audit row is incomplete")
    return {"snapshot": snapshot_id, "parent": snapshot.get("parent"),
            "payload_sha256": row.get("payload_sha256"),
            "controller": controller_audit_receipt(row.get("controller"))}


def _audit_store_campaign(campaign, profile):
    expected = {
        "root": ("f4253f99410b818b7016ae8277dc5ea0c1e46cd359503da6296488417ca4f464",
                 "62603a02956fc6e2e57b55cc4c3eaef3e2533703f4c9e33785428c6f59bc9fe8"),
        "read_partial": ("0849de3c27cd6bc952330d16e7425359729ea28bbf5e475d4eefe2956e1ef63c",
                         "f782c127c57fe34610e1ffe2a449b376689ae09446484f0668345ec7a9056e07"),
        "read_final": ("9bcdba47834b53c1559305d436df616b02660c8d4ae665f3bc3241fc84c4e39f",
                       "46e6ba22093d0d1eb34a4e87e08efb4cd36e0f426722341a32c0000adda730b6"),
        "test_final": ("1a4edbb3bb66c12114a57ed42b171a5d248721f04c0807c8f8c52e55cfc8ca9e",
                       "88ab3e2915dfac57ed509cc1063a92774d8d650a743f97c523735e7716ebfadc"),
    }
    if campaign.get("schema") != "cow-cli-grammar-probe-v1" or campaign.get("status") != "CLI_GRAMMAR_FLOW_PASS_NUMERICAL_AUDIT_PENDING":
        raise ValueError("campaign is not the completed GBNF CLI workflow awaiting numerical audit")
    if campaign.get("profile_identity", {}).get("model_contract_id") != profile.get("model_contract_id"):
        raise ValueError("campaign and profile model contracts differ")
    root_id = campaign.get("root", {}).get("snapshot")
    if root_id != expected["root"][0]:
        raise ValueError("campaign root snapshot differs from the pinned GBNF flow")
    read_partial = _campaign_row(campaign, "read_partial")
    read_final = _campaign_row(campaign, "read_complete")
    test_final = _campaign_row(campaign, "test_complete")
    rows = {"root": campaign["root"], "read_partial": read_partial,
            "read_final": read_final, "test_final": test_final}
    for key, row in rows.items():
        receipt = row.get("receipt")
        if not isinstance(receipt, dict):
            raise ValueError(f"campaign {key} receipt is missing")
        if (receipt.get("snapshot") != expected[key][0]
                or receipt.get("payload_sha256") != expected[key][1]):
            raise ValueError(f"campaign {key} snapshot or payload digest differs from pinned flow")
    from context_time_travel.store import Store
    store = Store(Path(campaign["store"]).resolve(strict=True))
    root_lineage_id, root_tokens = store.lineage(root_id)
    if root_lineage_id != root_id or store.get_snapshot(root_id).get("parent") is not None:
        raise ValueError("campaign source is not a canonical root")
    read_partial_id = read_partial["receipt"]["snapshot"]
    read_final_id = read_final["receipt"]["snapshot"]
    test_final_id = test_final["receipt"]["snapshot"]
    partial_lineage_id, partial_history = store.lineage(read_partial_id)
    read_lineage_id, read_history = store.lineage(read_final_id)
    test_lineage_id, test_history = store.lineage(test_final_id)
    validate_lineage_roots(root_id, {
        "READ partial": partial_lineage_id,
        "READ final": read_lineage_id,
        "TEST final": test_lineage_id,
    })
    partial_ids = partial_history[len(root_tokens):]
    read_ids = read_history[len(root_tokens):]
    test_ids = test_history[len(root_tokens):]
    if (read_ids[:len(partial_ids)] != partial_ids or test_history[:len(root_tokens)] != root_tokens
            or read_history[:len(root_tokens)] != root_tokens):
        raise ValueError("canonical output histories do not share the saved root prefix")
    if partial_history != root_tokens + [4754]:
        raise ValueError("READ partial canonical history differs from the campaign")
    expected_read = [4754, 13766, 3147, 851, 2378, 2129, 2267,
                     3147, 3431, 14905, 6971, 8934, 248046]
    expected_test = [4754, 13766, 3147, 5917, 31436, 2129,
                     48310, 3147, 3715, 8934, 248046]
    if read_ids != expected_read or test_ids != expected_test:
        raise ValueError("canonical READ/TEST output token IDs differ from pinned CLI receipt")
    if (store.get_snapshot(read_partial_id).get("parent") != root_id
            or store.get_snapshot(read_final_id).get("parent") != read_partial_id
            or store.get_snapshot(test_final_id).get("parent") != root_id):
        raise ValueError("grammar snapshots do not have the expected canonical parents")
    frontier_rows = {
        "root": _compare_snapshot_checkpoint(store, root_id, campaign["root"]["receipt"],
                                              root_tokens, profile, "root"),
        "read_partial": _compare_snapshot_checkpoint(store, read_partial_id,
                                                      read_partial["receipt"], partial_history,
                                                      profile, "READ partial"),
        "read_final": _compare_snapshot_checkpoint(store, read_final_id,
                                                   read_final["receipt"], read_history,
                                                   profile, "READ final"),
        "test_final": _compare_snapshot_checkpoint(store, test_final_id,
                                                   test_final["receipt"], test_history,
                                                   profile, "TEST final"),
    }
    return {"store": store, "root_tokens": root_tokens,
            "read_partial_ids": partial_ids, "read_ids": read_ids,
            "test_ids": test_ids, "frontier_rows": frontier_rows}


def run_store_preflight(campaign_path: Path, profile_path: Path, grammar_deps: Path):
    """Read-only real Store/checkpoint validation; this never opens a model."""
    campaign_path = Path(campaign_path).resolve(strict=True)
    profile_path = Path(profile_path).resolve(strict=True)
    campaign = json.loads(campaign_path.read_bytes())
    profile = json.loads(profile_path.read_bytes())
    if sha256_file(campaign_path) != "8111ea6fccef702e80c4c537400f25f3d82e489295ba676e44c73330ca801230":
        raise ValueError("campaign bytes differ from the independently reviewed CLI result")
    if sha256_file(profile_path) != "3551eeb132aeb7642cd3868acb9c940210ec0e9be97018bebff82a907f88b32a":
        raise ValueError("SUT profile bytes differ from the completed grammar campaign")
    registry_path = Path(campaign.get("registry", {}).get("path", "")).resolve(strict=True)
    if (sha256_file(registry_path) != "8443070884f6e35347d9720d585bcbb67fc95d0725c11fa30bfd1b1c886dfee6"
            or campaign.get("registry", {}).get("sha256") != "8443070884f6e35347d9720d585bcbb67fc95d0725c11fa30bfd1b1c886dfee6"):
        raise ValueError("grammar registry differs from the completed campaign")
    runtime_root = Path(campaign.get("runtime", "")).resolve(strict=True)
    registry, dependency_record = load_pinned_grammar_registry(
        runtime_root, profile, registry_path, grammar_deps)
    inspected = _audit_store_campaign(campaign, profile)
    frontier_rows = inspected["frontier_rows"]
    return {
        "schema": "cli-grammar-store-readonly-preflight-v1",
        "status": "PREFLIGHT_PASS",
        "model_oracle_executed": False,
        "gpu_launched": False,
        "campaign_sha256": sha256_file(campaign_path),
        "profile_sha256": sha256_file(profile_path),
        "grammar_dependency": dependency_record,
        "store_root_id": campaign["root"]["snapshot"],
        "lineage_root_ids": {name: inspected["store"].lineage(row["snapshot_id"])[0]
                              for name, row in frontier_rows.items()},
        "snapshots": {name: snapshot_audit_summary(row)
                      for name, row in frontier_rows.items()},
        "history_token_counts": {
            "root": len(inspected["root_tokens"]),
            "read_partial": len(inspected["root_tokens"]) + len(inspected["read_partial_ids"]),
            "read_final": len(inspected["root_tokens"]) + len(inspected["read_ids"]),
            "test_final": len(inspected["root_tokens"]) + len(inspected["test_ids"]),
        },
        "frontier_sha256": {name: hashlib.sha256(row["frontier"]).hexdigest()
                            for name, row in frontier_rows.items()},
    }


def _claim_audit_output(output_dir: Path, audit_run_id: str) -> Path:
    output_dir = Path(output_dir).resolve(strict=False)
    if type(audit_run_id) is not str or not audit_run_id:
        raise ValueError("audit run identity is invalid")
    output_dir.mkdir(parents=True, exist_ok=False)
    owner_path = output_dir / "audit-owner.json"
    with owner_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"audit_run_id": audit_run_id}, stream, sort_keys=True,
                  separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_json(output_dir / "audit-progress.json", {
        "audit_run_id": audit_run_id,
        "status": "RUNNING",
        "phase": "preflight",
        "completed_raw_processes": 0,
    })
    return output_dir


def run_audit(args, *, audit_run_id):
    output_dir = Path(args.output_dir).resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"audit output directory already exists: {output_dir}")
    output_dir = _claim_audit_output(output_dir, audit_run_id)
    campaign_path = args.campaign.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    registry_path = args.registry.resolve(strict=True)
    runtime_root = args.runtime_root.resolve(strict=True)
    binary = args.oracle_binary.resolve(strict=True)
    build_path = args.build_receipt.resolve(strict=True)
    campaign_sha = "8111ea6fccef702e80c4c537400f25f3d82e489295ba676e44c73330ca801230"
    profile_sha = "3551eeb132aeb7642cd3868acb9c940210ec0e9be97018bebff82a907f88b32a"
    registry_sha = "8443070884f6e35347d9720d585bcbb67fc95d0725c11fa30bfd1b1c886dfee6"
    if sha256_file(campaign_path) != campaign_sha:
        raise ValueError("campaign bytes differ from the independently reviewed CLI result")
    if sha256_file(profile_path) != profile_sha:
        raise ValueError("SUT profile bytes differ from the completed grammar campaign")
    if sha256_file(registry_path) != registry_sha:
        raise ValueError("grammar registry bytes differ from the completed campaign")
    build = json.loads(build_path.read_bytes())
    validate_build_provenance(build)
    if sha256_file(binary) != build.get("binary_sha256"):
        raise ValueError("forced raw-oracle binary does not match its build receipt")
    campaign = json.loads(campaign_path.read_bytes())
    profile = json.loads(profile_path.read_bytes())
    if Path(campaign.get("runtime", "")).resolve(strict=True) != runtime_root:
        raise ValueError("runtime source root differs from the exact campaign runtime")
    if campaign.get("profile") != str(profile_path):
        raise ValueError("campaign did not use the supplied profile path")
    if campaign.get("registry", {}).get("sha256") != registry_sha:
        raise ValueError("campaign registry identity differs from pinned bytes")
    if campaign.get("profile_identity", {}).get("execution_profile_id") != profile.get("execution_profile_id"):
        raise ValueError("campaign execution profile identity differs")
    labels = {row.get("label"): row for row in campaign.get("commands", [])
              if isinstance(row, dict)}
    for label, state in (("read-partial", "READ"), ("test-answer", "TEST")):
        row = labels.get(label)
        argv = row.get("argv_redacted") if isinstance(row, dict) else None
        if not isinstance(argv, list) or "--grammar-state" not in argv:
            raise ValueError(f"campaign {label} omitted its explicit grammar state")
        position = argv.index("--grammar-state")
        if position + 1 >= len(argv) or argv[position + 1] != state:
            raise ValueError(f"campaign {label} grammar state differs")
    finish = labels.get("read-finish")
    finish_argv = finish.get("argv_redacted") if isinstance(finish, dict) else None
    if (not isinstance(finish_argv, list) or "--branch" not in finish_argv
            or "grammar-read-" not in finish_argv[finish_argv.index("--branch") + 1]):
        raise ValueError("READ continuation is not bound to the partial READ branch")
    execution_imports = []
    for service in campaign.get("services", []):
        stop = service.get("stop", {})
        worker = service.get("worker_shutdown", {})
        if stop.get("exit_code") != 0 or worker.get("native_exit_code") != 0:
            raise ValueError("campaign service/native workers did not stop with numeric zero exits")
        for name, imported in service.get("source_imports", {}).items():
            path = Path(imported.get("path", "")).resolve(strict=True)
            if not path.is_relative_to(runtime_root):
                raise ValueError(f"campaign runtime import escapes its pinned source root: {name}")
            if sha256_file(path) != imported.get("sha256"):
                raise ValueError(f"campaign runtime source import changed: {name}")
            execution_imports.append({"module": name, "path": str(path),
                                     "sha256": imported["sha256"]})
    if not campaign.get("services"):
        raise ValueError("campaign has no terminal service receipts")

    api_root = runtime_root / "experimental" / "branching_context"
    registry, dependency_record = load_pinned_grammar_registry(
        runtime_root, profile, registry_path, args.grammar_deps)
    from verify_cli_workflow_numerics import (
        FRONTIER_WIDTH as VERIFIED_FRONTIER_WIDTH, validate_prefill_chunks,
        validate_oracle_profile,
    )
    if VERIFIED_FRONTIER_WIDTH != FRONTIER_WIDTH:
        raise ValueError("shared frontier validator width differs")
    validate_oracle_profile(profile, sut_path=profile["execution"]["sut_command"][0])
    if not os.access(binary, os.X_OK):
        raise ValueError("forced raw-oracle binary is not executable")
    eos_tokens = profile["execution"]["eos_tokens"]

    inspected = _audit_store_campaign(campaign, profile)
    frontier_rows = inspected["frontier_rows"]
    sut_frontiers = validate_frontier_map({
        "root": frontier_rows["root"]["frontier"],
        "read_partial": frontier_rows["read_partial"]["frontier"],
        "read_final": frontier_rows["read_final"]["frontier"],
        "test_final": frontier_rows["test_final"]["frontier"],
    })
    read_ids = inspected["read_ids"]
    test_ids = inspected["test_ids"]
    read_partial_count = len(inspected["read_partial_ids"])
    read_operation_id = campaign_operation_id(campaign, "read-partial")
    test_operation_id = campaign_operation_id(campaign, "test-answer")
    progress_path = output_dir / "audit-progress.json"
    raw_runs = {}
    grammar_audits = {}
    for name, state, stream, token_ids, segments, boundary_counts in (
        ("read-1", "READ", read_operation_id, read_ids,
         [read_partial_count, len(read_ids) - read_partial_count],
         [read_partial_count, len(read_ids) - read_partial_count]),
        ("read-2", "READ", read_operation_id, read_ids,
         [read_partial_count, len(read_ids) - read_partial_count],
         [read_partial_count, len(read_ids) - read_partial_count]),
        ("test-1", "TEST", test_operation_id, test_ids, [len(test_ids)], [len(test_ids)]),
        ("test-2", "TEST", test_operation_id, test_ids, [len(test_ids)], [len(test_ids)]),
    ):
        _atomic_json(progress_path, {"audit_run_id": audit_run_id,
                                    "status": "RUNNING", "phase": name,
                                    "completed_raw_processes": len(raw_runs)})
        run = _run_forced_raw_oracle(
            profile, binary, output_dir / name, name, inspected["root_tokens"],
            token_ids, segments)
        run["grammar_audit"] = audit_forced_grammar_trajectory(
            registry.engine, state, stream, inspected["root_tokens"], token_ids,
            [run["prompt_frontier"], *run["decode_frontiers"]],
            boundary_counts=boundary_counts)
        raw_runs[name] = run
        grammar_audits[name] = run["grammar_audit"]

    frontier_comparisons = []
    for left_name, right_name, state in (
        ("read-1", "read-2", "READ repeated fresh trajectory"),
        ("test-1", "test-2", "TEST repeated fresh trajectory"),
    ):
        frontier_comparisons.extend(compare_raw_trajectories(
            raw_runs[left_name], raw_runs[right_name], state))
    for name, state in (("read-1", "READ"), ("read-2", "READ"),
                        ("test-1", "TEST"), ("test-2", "TEST")):
        run = raw_runs[name]
        validate_prefill_chunks(frontier_rows["root"]["manifest"], run["schedule"])
        frontier_comparisons.append({
            "label": f"SUT root / {name} raw Prefill",
            "max_abs_error": compare_frontiers(
                sut_frontiers["root"], run["prompt_frontier"],
                f"SUT root / {name} raw Prefill"),
        })
    read_raw = raw_runs["read-1"]
    test_raw = raw_runs["test-1"]
    if len(read_raw["segment_frontiers"]) != 2 or read_raw["segment_frontiers"][0] != read_raw["decode_frontiers"][0]:
        raise AssertionError("READ partial segment frontier differs from first Verifier frontier")
    for actual, expected, label in (
        (sut_frontiers["read_partial"], read_raw["segment_frontiers"][0], "READ partial SUT / raw boundary"),
        (sut_frontiers["read_final"], read_raw["segment_frontiers"][1], "READ final SUT / raw boundary"),
        (sut_frontiers["test_final"], test_raw["segment_frontiers"][0], "TEST final SUT / raw boundary"),
    ):
        frontier_comparisons.append({"label": label,
                                     "max_abs_error": compare_frontiers(actual, expected, label)})

    read_audit = grammar_audits["read-1"]
    test_audit = grammar_audits["test-1"]
    raw_controller_hashes = {}
    for name, row, audit, boundary_index, label in (
        ("read_partial", frontier_rows["read_partial"], read_audit, 0, "READ partial"),
        ("read_final", frontier_rows["read_final"], read_audit, 1, "READ final"),
        ("test_final", frontier_rows["test_final"], test_audit, 0, "TEST final"),
    ):
        digest = validate_controller_equal(audit["boundary_controllers"][boundary_index],
                                           row["controller"], label)
        raw_controller_hashes[name] = digest
    validate_controller_equal(read_audit["final_controller"],
                              frontier_rows["read_final"]["controller"], "READ completion")
    validate_controller_equal(test_audit["final_controller"],
                              frontier_rows["test_final"]["controller"], "TEST completion")
    for label, row, state in (
        ("READ partial", frontier_rows["read_partial"], "READ"),
        ("READ final", frontier_rows["read_final"], "READ"),
        ("TEST final", frontier_rows["test_final"], "TEST"),
    ):
        constraint = row["controller"].get("constraint", {})
        if (constraint.get("agent_state") != state
                or constraint.get("engine_id") != registry.engine_id
                or constraint.get("spec_sha256") != registry.spec_sha256
                or constraint.get("tokenizer_sha256") != registry.tokenizer_sha256):
            raise AssertionError(f"{label} controller is not bound to the pinned grammar registry")
    record = {
        "schema": "independent-cli-grammar-raw-numerical-audit-v1",
        "status": "PASS", "model_oracle_executed": True,
        "campaign_sha256": sha256_file(campaign_path),
        "profile_sha256": sha256_file(profile_path),
        "registry_sha256": sha256_file(registry_path),
        "grammar_dependency": dependency_record,
        "oracle_binary_sha256": sha256_file(binary),
        "oracle_build_sha256": sha256_file(build_path),
        "runtime_imports": execution_imports,
        "sut_snapshots": {name: snapshot_audit_summary(row)
                          for name, row in frontier_rows.items()},
        "root_token_count": len(inspected["root_tokens"]),
        "read_output_ids": read_ids, "test_output_ids": test_ids,
        "read_segments": [read_partial_count, len(read_ids) - read_partial_count],
        "test_segments": [len(test_ids)],
        "raw_operation_ids": {"READ": read_operation_id, "TEST": test_operation_id},
        "raw_processes": {name: {"pid": run["pid"], "exit_code": run["exit_code"],
                                 "work": {field: run["work"][field]
                                          for field in _RAW_WORK_DETERMINISTIC_FIELDS},
                                 "timings": {field: run["work"][field]
                                             for field in _RAW_WORK_TIMING_FIELDS},
                                 "schedule": run["schedule"],
                                 "heartbeat_count": json.loads(
                                     (output_dir / name / f"{name}.process.json").read_text(
                                         encoding="utf-8")).get("heartbeat_count", 0)}
                           for name, run in raw_runs.items()},
        "raw_controller_sha256": {
            name: {"boundary": [controller_audit_receipt(controller)["controller_sha256"]
                                 for controller in grammar_audits[name]["boundary_controllers"]],
                  "final": controller_audit_receipt(
                      grammar_audits[name]["final_controller"])["controller_sha256"]}
            for name in raw_runs
        },
        "sut_controller_sha256": raw_controller_hashes,
        "read_greedy_legal_checks": read_audit["token_checks"],
        "test_greedy_legal_checks": test_audit["token_checks"],
        "frontier_comparisons": frontier_comparisons,
        "exact_frontier_comparisons": len(frontier_comparisons),
        "frontier_max_abs_error": max(
            (row["max_abs_error"] for row in frontier_comparisons), default=0.0),
        "oracle_checkpoint_imports": 0, "oracle_branch_sessions": 0,
        "scope": "Four fresh raw Prefill+Verifier trajectories (two each for READ and TEST); every forced canonical ID is independently checked as the grammar-legal greedy choice before ordinary Verifier consumption. Full controller state is regenerated with the original operation stream and compared at saved boundaries. No SUT checkpoint is passed into the oracle.",
    }
    _atomic_json(output_dir / "audit.json", record)
    _atomic_json(progress_path, {"audit_run_id": audit_run_id,
                                "status": "PASS", "phase": "complete"})
    return record


def validate_frontier_map(mapping):
    if not isinstance(mapping, dict) or set(mapping) != set(REQUIRED_BOUNDARIES):
        raise ValueError("frontier map must contain exactly the four required boundaries")
    if any(type(value) is not bytes for value in mapping.values()):
        raise ValueError("frontier map values must be immutable bytes")
    return dict(mapping)


def validate_input_rows(rows, *, expected_count):
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"verified input count must be exactly {expected_count}")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not {"path"}.issubset(row):
            raise ValueError("verified input row is malformed")
        path = Path(row["path"]).resolve(strict=True)
        if str(path) in seen:
            raise ValueError("verified input path is duplicated")
        seen.add(str(path))
        digest = row.get("sha256", row.get("expected_sha256"))
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or sha256_file(path) != digest):
            raise ValueError(f"verified input hash mismatch: {path}")
    return rows


def validate_build_provenance(record, expected_count=53):
    if not isinstance(record, dict) or record.get("status") != "BUILT_NOT_EXECUTED":
        raise ValueError("oracle build must be recorded as BUILT_NOT_EXECUTED")
    if record.get("runtime_executed") is not False or record.get("gpu_launched") is not False:
        raise ValueError("oracle build receipt claims runtime or GPU execution")
    if (record.get("verified_input_count") != expected_count
            or record.get("verified_input_mismatches") != []):
        raise ValueError("oracle build receipt input count/hash mismatches")
    return record


def instrument_forced_tokens(source):
    if not isinstance(source, str):
        raise ValueError("segmented source must be text")
    if source.count(_GREEDY_ANCHOR) != 1 or source.count(_GENERATED_ANCHOR) != 1:
        raise ValueError("segmented source anchor missing, duplicated, or already modified")
    transformed = source.replace(_GENERATED_ANCHOR, _FORCED_PARSE + _GENERATED_ANCHOR, 1)
    transformed = transformed.replace(_GREEDY_ANCHOR, _FORCED_SELECTION, 1)
    if transformed.count("STRATA_FORCE_GENERATED_TOKENS") != 1:
        raise ValueError("forced-token source transform did not apply exactly once")
    return transformed


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mark_audit_failure(output_dir: Path, error: BaseException, audit_run_id: str):
    output_dir = Path(output_dir)
    progress_path = output_dir / "audit-progress.json"
    owner_path = output_dir / "audit-owner.json"
    if not output_dir.is_dir() or not owner_path.is_file():
        return
    try:
        owner = json.loads(owner_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(owner, dict) or owner.get("audit_run_id") != audit_run_id:
        return
    progress = {}
    if progress_path.is_file():
        try:
            loaded = json.loads(progress_path.read_bytes())
            if isinstance(loaded, dict):
                progress = loaded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            progress = {}
    if progress and progress.get("audit_run_id") != audit_run_id:
        return
    failed_phase = progress.get("phase", "audit")
    if not isinstance(failed_phase, str) or not failed_phase:
        failed_phase = "audit"
    progress.update({
        "audit_run_id": audit_run_id,
        "status": "FAIL",
        "phase": "failed",
        "failed_phase": failed_phase,
        "failure_type": type(error).__name__,
        "failure": str(error),
    })
    _atomic_json(progress_path, progress)


def _rewrite_compile_argv(argv, *, source_path, object_path):
    result = list(argv)
    if result.count("-c") != 1 or "-o" not in result:
        raise ValueError("pinned compile command does not have a single object output")
    output_index = result.index("-o")
    if output_index + 1 >= len(result):
        raise ValueError("pinned compile command has a malformed output argument")
    source_candidates = [index for index, value in enumerate(result)
                         if value.endswith("fresh_prefill_segmented.cpp")]
    if len(source_candidates) != 1:
        raise ValueError("pinned compile command source anchor is missing or duplicated")
    result[source_candidates[0]] = str(source_path)
    result[output_index + 1] = str(object_path)
    return result


def _rewrite_link_argv(argv, *, object_path, binary_path):
    result = list(argv)
    if "-o" not in result:
        raise ValueError("pinned link command has no output argument")
    output_index = result.index("-o")
    if output_index + 1 >= len(result):
        raise ValueError("pinned link command has a malformed output argument")
    matches = [index for index, value in enumerate(result)
               if value.endswith("fresh_prefill_segmented.o")]
    if len(matches) != 1:
        raise ValueError("pinned link command object anchor is missing or duplicated")
    result[matches[0]] = str(object_path)
    result[output_index + 1] = str(binary_path)
    return result


def build_forced_oracle(*, source_path: Path, original_build_receipt: Path,
                        compile_link_receipt: Path, input_report: Path,
                        output_dir: Path) -> dict:
    """Compile a distinct forced-token TU against the exact verified 53 inputs.

    No model/runtime process is started. Existing objects and static libraries
    are referenced read-only; only the transformed TU, its object, and a new
    executable are written below ``output_dir``.
    """
    expected_receipt_sha = "d251cd11267d2e32a57144d25588a7f6450bd466df72f1cd828b21b36a2c7b0c"
    expected_compile_link_sha = "7b26945aef71ed8ad46e3c8c030b9eebe3381a0b5fe6dab9f152667c6ac6aa30"
    expected_inputs_sha = "9ec1bb03b22e8b18449dc62775c57b84658b84181adb9f8c2787ef75ce8af6e1"
    expected_source_sha = "85f9c1942f697ede539f977674db4ffe9af9f0800a30ace4dfe7baf718223b5c"
    if sha256_file(original_build_receipt) != expected_receipt_sha:
        raise ValueError("original compile recipe hash differs from the pinned receipt")
    if sha256_file(compile_link_receipt) != expected_compile_link_sha:
        raise ValueError("compile/link provenance hash differs from the reviewed build")
    if sha256_file(input_report) != expected_inputs_sha:
        raise ValueError("53-input verification report hash differs from reviewed artifact")
    if sha256_file(source_path) != expected_source_sha:
        raise ValueError("segmented oracle source differs from the pinned source")
    build_recipe = json.loads(compile_link_receipt.read_bytes())
    verification = json.loads(input_report.read_bytes())
    rows = verification.get("rows")
    if verification.get("receipt_sha256") != expected_receipt_sha:
        raise ValueError("input report is bound to a different original recipe")
    validate_input_rows([
        {"path": row.get("path"), "expected_sha256": row.get("expected_sha256")}
        for row in rows
    ], expected_count=53)
    if verification.get("checked_files") != 53 or verification.get("mismatches") != []:
        raise ValueError("input verification report is incomplete or contains mismatches")
    if (not isinstance(build_recipe.get("compiler_and_link_steps"), list)
            or len(build_recipe["compiler_and_link_steps"]) != 2):
        raise ValueError("pinned compile/link recipe must contain exactly two steps")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    transformed_source = instrument_forced_tokens(Path(source_path).read_text(encoding="utf-8"))
    generated_source = output_dir / "fresh_prefill_segmented_forced.cpp"
    generated_source.write_text(transformed_source, encoding="utf-8", newline="\n")
    object_path = output_dir / "fresh_prefill_segmented_forced.o"
    binary_path = output_dir / "strata-batched-fresh-segmented-forced-audit"
    original_steps = build_recipe["compiler_and_link_steps"]
    commands = [
        _rewrite_compile_argv(original_steps[0]["argv"], source_path=generated_source,
                              object_path=object_path),
        _rewrite_link_argv(original_steps[1]["argv"], object_path=object_path,
                            binary_path=binary_path),
    ]
    results = []
    for name, argv, cwd in zip(("compile", "link"), commands,
                               (original_steps[0]["cwd"], original_steps[1]["cwd"])):
        completed = subprocess.run(argv, cwd=cwd, check=False, capture_output=True, text=True)
        result = {"stage": name, "argv": argv, "cwd": cwd,
                  "exit_code": completed.returncode, "stdout": completed.stdout,
                  "stderr": completed.stderr}
        results.append(result)
        _atomic_json(output_dir / f"{name}.result.json", result)
        if completed.returncode != 0:
            raise RuntimeError(f"forced raw-oracle {name} step failed: {completed.returncode}")
    receipt = {
        "schema": "gbnf-forced-segmented-raw-oracle-build-v1",
        "status": "BUILT_NOT_EXECUTED", "runtime_executed": False, "gpu_launched": False,
        "original_build_receipt_sha256": expected_receipt_sha,
        "compile_link_receipt_sha256": expected_compile_link_sha,
        "input_verification_sha256": expected_inputs_sha,
        "verified_input_count": 53, "verified_input_mismatches": [],
        "source_original_sha256": expected_source_sha,
        "forced_source_sha256": sha256_file(generated_source),
        "output_object_sha256": sha256_file(object_path),
        "binary_sha256": sha256_file(binary_path),
        "compiler_and_link_steps": results,
        "output_source": str(generated_source), "output_object": str(object_path),
        "output_binary": str(binary_path),
    }
    validate_build_provenance(receipt)
    _atomic_json(output_dir / "build.json", receipt)
    return receipt


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("build-only", help="compile without executing the raw oracle")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--original-build-receipt", type=Path, required=True)
    build.add_argument("--compile-link-receipt", type=Path, required=True)
    build.add_argument("--input-report", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    audit = modes.add_parser("audit", help="run the separate raw oracle and independent grammar checks")
    audit.add_argument("--campaign", type=Path, required=True)
    audit.add_argument("--profile", type=Path, required=True)
    audit.add_argument("--registry", type=Path, required=True)
    audit.add_argument("--grammar-deps", type=Path, required=True,
                       help="pinned llguidance 1.9.1 dependency directory")
    audit.add_argument("--runtime-root", type=Path, required=True)
    audit.add_argument("--oracle-binary", type=Path, required=True)
    audit.add_argument("--build-receipt", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    store_check = modes.add_parser("store-preflight",
                                   help="read actual campaign Store/checkpoints without opening a model")
    store_check.add_argument("--campaign", type=Path, required=True)
    store_check.add_argument("--profile", type=Path, required=True)
    store_check.add_argument("--grammar-deps", type=Path, required=True,
                             help="pinned llguidance 1.9.1 dependency directory")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.mode == "build-only":
        record = build_forced_oracle(
            source_path=args.source.resolve(strict=True),
            original_build_receipt=args.original_build_receipt.resolve(strict=True),
            compile_link_receipt=args.compile_link_receipt.resolve(strict=True),
            input_report=args.input_report.resolve(strict=True),
            output_dir=args.output_dir.resolve(strict=False))
    else:
        if args.mode == "store-preflight":
            record = run_store_preflight(args.campaign, args.profile, args.grammar_deps)
            print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
            return 0
        else:
            audit_run_id = uuid.uuid4().hex
            try:
                record = run_audit(args, audit_run_id=audit_run_id)
            except BaseException as exc:
                _mark_audit_failure(args.output_dir, exc, audit_run_id)
                raise
    print(json.dumps({"status": record["status"],
                      "binary_sha256": record.get("binary_sha256", record.get("oracle_binary_sha256")),
                      "verified_input_count": record.get("verified_input_count", 53),
                      "exact_frontier_comparisons": record.get("exact_frontier_comparisons")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
