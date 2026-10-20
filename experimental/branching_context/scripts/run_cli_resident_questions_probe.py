"""Bounded v16 CLI probe for three questions on one resident code-context root."""
from __future__ import annotations

import argparse
import base64
import codecs
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import time

from run_cow_client_recovery import (
    ProgressTracker, _private_run_dir, checkpoint_fingerprint,
)
from context_time_travel.store import Store
from context_time_travel import controller
from serve.cow_http import model_text_adapter
from tools.strata_tokenizer import Tokenizer


class ResidentQuestionsError(RuntimeError):
    """The real request did not meet the bounded resident-prefix contract."""


ResidentProbeError = ResidentQuestionsError


def require_fresh_output(exists):
    if exists:
        raise ResidentProbeError("run output directory already exists")


def verify_file_sha256(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ResidentProbeError("input SHA-256 pin is malformed")
    actual = _sha(path)
    if actual != expected:
        raise ResidentProbeError("context file differs from its pinned SHA-256")
    return actual


def validate_campaign_shape(*, root_tokens, question_sources, question_snapshots,
                            actual_output_counts, max_tokens):
    if type(root_tokens) is not int or root_tokens < 8000:
        raise ResidentProbeError("prepared context is below the requested approximately 8K-token root")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 5000:
        raise ResidentProbeError("max_tokens must be in the supported range 1..5000")
    if len(question_sources) != 3 or len(question_snapshots) != 3 or len(actual_output_counts) != 3:
        raise ResidentProbeError("exactly three independent question results are required")
    if len(set(question_snapshots)) != 3:
        raise ResidentProbeError("question outputs must be distinct durable snapshots")
    if any(source != question_sources[0] for source in question_sources):
        raise ResidentProbeError("all questions must start at the same saved root")
    if any(type(count) is not int or not 1 <= count <= max_tokens for count in actual_output_counts):
        raise ResidentProbeError("actual output count is empty or exceeds the configured ceiling")
    return {"root_tokens": root_tokens, "independent_questions": 3,
            "actual_output_counts": list(actual_output_counts), "max_tokens": max_tokens}


def validate_machine_answer(envelope, *, expected_snapshot, expected_output_ids, expected_text):
    if not isinstance(envelope, dict) or envelope.get("schema") != 1:
        raise ResidentProbeError("CLI machine-result envelope schema is invalid")
    answer, receipt = envelope.get("answer"), envelope.get("receipt")
    if not isinstance(answer, dict) or not isinstance(receipt, dict):
        raise ResidentProbeError("CLI machine-result lacks canonical answer or receipt")
    if answer.get("snapshot") != expected_snapshot or receipt.get("snapshot") != expected_snapshot:
        raise ResidentProbeError("machine answer and durable receipt identify different snapshots")
    ids = answer.get("tokens")
    if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
            or tuple(ids) != tuple(expected_output_ids)):
        raise ResidentProbeError("machine answer token IDs differ from canonical stored lineage")
    if not isinstance(answer.get("text"), str) or answer["text"] != expected_text:
        raise ResidentProbeError("machine answer text differs from canonical token decoding")
    for key in ("generation_seconds", "request_seconds"):
        value = answer.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ResidentProbeError(f"machine answer is missing valid {key}")
    if answer["request_seconds"] < answer["generation_seconds"]:
        raise ResidentProbeError("request duration is shorter than its generation loop")
    return answer


def validate_observation_summary(summary, *, expected_tokens):
    if not isinstance(summary, dict) or summary.get("validation") != {"status": "valid", "errors": []}:
        raise ResidentProbeError("observed CLI NDJSON stream did not validate")
    if summary.get("token_ids") != list(expected_tokens):
        raise ResidentProbeError("observed token events differ from exact stored answer tokens")
    metrics = summary.get("metrics_ns")
    if not isinstance(metrics, dict):
        raise ResidentProbeError("observer timing metrics are missing")
    ttft = metrics.get("request_to_first_token")
    if type(ttft) is not int or ttft < 0:
        raise ResidentProbeError("observer first-token timing is missing or invalid")
    durable = metrics.get("request_to_durable")
    if type(durable) is not int or durable < ttft:
        raise ResidentProbeError("observer durable completion timing is missing or invalid")
    return {"ttft_ns": ttft, "request_to_durable_ns": durable}


def validate_observation_trace(trace, *, branch, operation_id, receipt, expected_tokens):
    requests = trace.get("requests") if isinstance(trace, dict) else None
    if (not isinstance(requests, list) or len(requests) != 1
            or trace.get("cli_exit_code") != 0 or trace.get("cli_exception") is not None):
        raise ResidentProbeError("observer did not capture exactly one request")
    request = requests[0]
    if (request.get("method") != "POST" or request.get("request_path") != "/cow/text/ask"
            or request.get("branch") != branch or request.get("operation_id") != operation_id
            or request.get("open_error") is not None or request.get("read_error") is not None
            or request.get("observer_error") is not None):
        raise ResidentProbeError("observed request binding differs from the submitted question")
    events = []
    for row in request.get("events", []):
        try:
            events.append(json.loads(base64.b64decode(row["raw_line_b64"], validate=True).decode("utf-8")))
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise ResidentProbeError("observer captured a malformed NDJSON event") from None
    complete = [event for event in events if event.get("event") == "complete"]
    if len(complete) != 1:
        raise ResidentProbeError("observer did not capture exactly one complete event")
    complete_answer = complete[0].get("answer")
    if (complete[0].get("receipt") != receipt or not isinstance(complete_answer, dict)
            or complete_answer.get("snapshot") != receipt.get("snapshot")
            or complete_answer.get("tokens") != list(expected_tokens)):
        raise ResidentProbeError("observed completion differs from the canonical CLI answer/receipt")
    summary = validate_observation_summary(request.get("summary"), expected_tokens=expected_tokens)
    return {**summary, "event_count": len(events), "complete_snapshot": receipt["snapshot"]}


def verify_selected_imports(profile, runtime_root, module_names):
    closure = profile.get("native_build", {}).get("selected_runtime_source_closure")
    declared = closure.get("files_sha256") if isinstance(closure, dict) else None
    if not isinstance(declared, dict) or not declared:
        raise ResidentProbeError("profile lacks a selected runtime source closure")
    root = Path(runtime_root).resolve(strict=True)
    expected_by_path = {}
    for path, digest in declared.items():
        parts = PurePosixPath(str(path).replace("\\", "/")).parts
        for marker in ("serve", "experimental", "tools"):
            if marker in parts:
                candidate = (root / Path(*parts[parts.index(marker):])).resolve()
                expected_by_path[str(candidate)] = digest
    imported = {}
    for name in module_names:
        module = importlib.import_module(name)
        source = getattr(module, "__file__", None)
        if not source:
            raise ResidentProbeError(f"selected module {name} has no source path")
        source_path = Path(source).resolve(strict=True)
        try:
            source_path.relative_to(root)
        except ValueError:
            raise ResidentProbeError(f"selected module {name} escaped the pinned runtime root") from None
        digest = _sha(source_path)
        expected = expected_by_path.get(str(source_path))
        if expected != digest:
            raise ResidentProbeError(f"selected module {name} is not bound by the profile source closure")
        imported[name] = {"path": str(source_path), "sha256": digest}
    return imported


QUESTIONS = (
    "Write three short paragraphs. Summarize the control flow and identify the main entry points.",
    "Write three short paragraphs. Which parts own persistent state, and how is that state updated?",
    "Write three short paragraphs. What invariants and failure cases does this code appear to handle?",
)
COUNTERS = ("forward_tokens", "prefill_tokens", "loaded_file_bytes", "integrity_mismatches")


def _integer_counter(mapping, key):
    value = mapping.get(key)
    if type(value) is not int or value < 0:
        raise ResidentQuestionsError(f"native counter {key!r} is missing or invalid")
    return value


def validate_question_work(before, after, *, question_token_ids, output_token_ids):
    """Require only this question suffix plus its actual answer to be computed."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ResidentQuestionsError("native work counters are not objects")
    q_count, output_count = len(question_token_ids), len(output_token_ids)
    deltas = {key: _integer_counter(after, key) - _integer_counter(before, key)
              for key in COUNTERS}
    if any(value < 0 for value in deltas.values()):
        raise ResidentQuestionsError("native work counters moved backwards")
    old_prefix = deltas["forward_tokens"] - q_count - output_count
    if old_prefix != 0:
        raise ResidentQuestionsError(f"old code-context prefix was recomputed ({old_prefix} tokens)")
    if deltas["forward_tokens"] != q_count + output_count:
        raise ResidentQuestionsError("forward work differs from question plus actual output tokens")
    if deltas["prefill_tokens"] != q_count:
        raise ResidentQuestionsError("prefill work differs from the new question suffix")
    if deltas["loaded_file_bytes"] != 0:
        raise ResidentQuestionsError("question request loaded checkpoint bytes from disk")
    if deltas["integrity_mismatches"] != 0:
        raise ResidentQuestionsError("native integrity mismatch counter increased")
    return {"question_tokens": q_count, "output_tokens": output_count,
            "old_prefix_forward_tokens": old_prefix, **deltas}


def validate_question_lineage(*, source_snapshot, expected_root_snapshot,
                              root_token_ids, branch_source_token_ids,
                              branch_lineage_ids, question_token_ids, output_token_ids):
    """Require a question branch to start at P and append exactly Q then answer."""
    root = tuple(root_token_ids)
    source = tuple(branch_source_token_ids)
    lineage = tuple(branch_lineage_ids)
    expected = root + tuple(question_token_ids) + tuple(output_token_ids)
    if source_snapshot != expected_root_snapshot:
        raise ResidentQuestionsError("question branch source is not the original code root")
    if source != root:
        raise ResidentQuestionsError("question branch source token history differs from the code root")
    if lineage != expected:
        raise ResidentQuestionsError("question lineage contains accumulated or unexpected tokens")
    return {"source_snapshot": source_snapshot, "root_tokens": len(root),
            "question_tokens": len(question_token_ids), "output_tokens": len(output_token_ids),
            "lineage_tokens": len(lineage)}


def validate_resident_prefix(*, root_handle, root_tokens, root_capture, question_traces):
    """Verify one native capture is reused/restored across all question requests."""
    if type(root_handle) is not int or root_handle < 0 or type(root_tokens) is not int or root_tokens <= 0:
        raise ResidentQuestionsError("native root handle or token count is invalid")
    if (not isinstance(root_capture, dict) or root_capture.get("command") != "capture"
            or not isinstance(root_capture.get("response"), dict)
            or root_capture["response"].get("status") != "ok"
            or root_capture["response"].get("handle") != root_handle
            or root_capture["response"].get("consumed") != root_tokens):
        raise ResidentQuestionsError("native root capture does not identify the full code prefix")
    if not isinstance(question_traces, (list, tuple)) or len(question_traces) != 3:
        raise ResidentQuestionsError("exactly three native question traces are required")
    for index, rows in enumerate(question_traces):
        if not isinstance(rows, (list, tuple)):
            raise ResidentQuestionsError("question native trace is malformed")
        dropped = False
        restored = False
        for row in rows:
            if not isinstance(row, dict):
                raise ResidentQuestionsError("native trace row is malformed")
            command = row.get("command", "")
            response = row.get("response", {})
            if not isinstance(command, str) or not isinstance(response, dict):
                raise ResidentQuestionsError("native trace command/response is malformed")
            if command == f"drop {root_handle}":
                dropped = True
            if command == f"restore {root_handle}":
                if response.get("status") != "ok" or response.get("consumed") != root_tokens:
                    raise ResidentQuestionsError("native root restore did not restore the code prefix")
                restored = True
        if dropped and index < 2:
            raise ResidentQuestionsError("native root handle was dropped before the final question")
        if index > 0 and not restored:
            raise ResidentQuestionsError("later question did not restore the original native root handle")
    return {"root_handle": root_handle, "root_tokens": root_tokens,
            "question_count": len(question_traces), "later_restores": 2,
            "handle_dropped_before_final_question": False}


def validate_answer_completion(*, output_token_ids, max_tokens, controller_status, eos_tokens):
    """Report the actual output count; EOS may stop before the requested ceiling."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ResidentQuestionsError("max_tokens must be a positive integer")
    output = tuple(output_token_ids)
    if not output or len(output) > max_tokens:
        raise ResidentQuestionsError("actual output count is empty or exceeds max_tokens")
    eos = set(eos_tokens)
    ended_by_eos = output[-1] in eos
    if controller_status == "stopped":
        if not ended_by_eos:
            raise ResidentQuestionsError("stopped answer does not end with an EOS token")
    elif controller_status == "active":
        if ended_by_eos or len(output) != max_tokens:
            raise ResidentQuestionsError("active answer ended early or emitted EOS without stopping")
    else:
        raise ResidentQuestionsError("answer controller has an unsupported terminal status")
    return {"actual_output_tokens": len(output), "max_tokens": max_tokens,
            "controller_status": controller_status, "ended_by_eos": ended_by_eos,
            "reached_max_tokens": len(output) == max_tokens}


def validate_root_stability(*, root_tip_before, root_tip_after, sibling_tips_before,
                            sibling_tips_after, root_payload_sha_before,
                            root_payload_sha_after):
    if root_tip_before != root_tip_after:
        raise ResidentQuestionsError("canonical code-root branch moved during questions")
    if sibling_tips_before != sibling_tips_after:
        raise ResidentQuestionsError("an unrelated question sibling branch moved")
    if (not isinstance(root_payload_sha_before, str) or len(root_payload_sha_before) != 64
            or root_payload_sha_before != root_payload_sha_after):
        raise ResidentQuestionsError("canonical code-root payload bytes changed")
    return {"root_tip_unchanged": True, "sibling_tips_unchanged": True,
            "root_payload_sha256_unchanged": True}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _latest_stats(trace_path):
    if not trace_path.is_file():
        raise ResidentQuestionsError("native request trace is missing")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    states = [row.get("response") for row in rows if row.get("command") == "stats"]
    if not states or not isinstance(states[-1], dict):
        raise ResidentQuestionsError("native trace has no successful stats observation")
    return states[-1], rows


def _redact_bytes(payload, secret):
    if secret is None:
        return payload
    encoded = secret.encode("utf-8")
    return payload.replace(encoded, b"[REDACTED]")


def _redact_campaign_value(value, secret, redactor):
    if not callable(redactor):
        raise ResidentProbeError("campaign secret redactor is unavailable")
    return redactor(value, (secret,) if secret else ())


def _invoke(output, repo, cli, url, label, *command, timeout=1800, mutation=True,
            machine_result=False, observer=None, trace_path=None, api_key=None):
    if observer is not None:
        if trace_path is None:
            raise ResidentProbeError("observed CLI invocation lacks its trace path")
        args = [sys.executable, "-B", str(observer), "--runtime-root", str(repo),
                "--trace-file", str(trace_path), "--", "--machine-result",
                "--url", url, "--timeout", str(timeout), *command]
    else:
        args = [sys.executable, "-B", str(cli)]
        if machine_result:
            args.append("--machine-result")
        args += ["--url", url, "--timeout", str(timeout), *command]
    child_env = os.environ.copy()
    if api_key is not None:
        child_env["COW_API_KEY"] = api_key
    started = time.monotonic()
    result = subprocess.run(args, cwd=repo, env=child_env, capture_output=True, timeout=timeout + 30)
    elapsed = time.monotonic() - started
    safe_stdout = _redact_bytes(result.stdout, api_key)
    safe_stderr = _redact_bytes(result.stderr, api_key)
    (output / f"{label}.stdout").write_bytes(safe_stdout)
    (output / f"{label}.stderr").write_bytes(safe_stderr)
    if result.returncode != 0:
        raise ResidentQuestionsError(
            f"CLI {label} exited {result.returncode}: {safe_stderr.decode('utf-8', 'replace')[-4000:]}")
    stdout = safe_stdout.decode("utf-8")
    stderr = safe_stderr.decode("utf-8")
    if machine_result:
        try:
            envelope = json.loads(stdout)
        except (ValueError, TypeError) as error:
            raise ResidentProbeError(f"CLI {label} did not emit a valid machine result: {error}") from None
        receipt = envelope.get("receipt") if isinstance(envelope, dict) else None
        row_answer = envelope.get("answer") if isinstance(envelope, dict) else None
        if not isinstance(receipt, dict) or not isinstance(row_answer, dict):
            raise ResidentProbeError(f"CLI {label} machine result lacks answer or durable receipt")
        row = {"label": label, "argv": args, "exit_code": result.returncode,
            "elapsed_seconds": elapsed, "stdout_sha256": hashlib.sha256(safe_stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(safe_stderr).hexdigest(), "stdout": stdout,
            "stderr_sha256_only": True, "receipt": receipt, "answer": row_answer,
            "machine_envelope": envelope}
        if observer is not None:
            try:
                observation = json.loads(Path(trace_path).read_bytes())
            except (OSError, ValueError) as error:
                raise ResidentProbeError(f"CLI {label} observer trace is missing or invalid: {error}") from None
            row["observation"] = observation
        return receipt, row
    if mutation:
        marker = "durable receipt:\n"
        if marker not in stderr:
            raise ResidentQuestionsError(f"CLI {label} did not return a durable receipt")
        receipt = json.loads(stderr.split(marker, 1)[1])
    else:
        receipt = json.loads(stdout)
    return receipt, {"label": label, "argv": args, "exit_code": result.returncode,
        "elapsed_seconds": elapsed, "stdout_sha256": hashlib.sha256(safe_stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(safe_stderr).hexdigest(), "stdout": stdout,
        "stderr_sha256_only": True, "receipt": receipt}


def _read_trace(trace_path):
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]


def _trace_slice(rows, start, end):
    return rows[start:end]


def _call_receipt(row):
    return {key: value for key, value in row.items() if key not in ("stdout", "stderr")}


def _expected_controller(operation_id, history, output_ids, tokenizer, profile):
    eos = list(profile["execution"]["eos_tokens"])
    state = controller.start({"temperature": 0, "eos_tokens": eos}, operation_id,
                             tuple(history), len(tokenizer.tokens))
    return controller.consume_many(state,
        ((token, tokenizer.token_bytes(token), token in eos) for token in output_ids))


def _decoded_answer(output_ids, source_controller, tokenizer, eos_tokens):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = bytes.fromhex(source_controller.get("output_pending_hex", ""))
    decoder.setstate((pending, 0))
    eos = set(eos_tokens)
    parts = []
    for token in output_ids:
        if token in eos:
            parts.append(decoder.decode(b"", final=True))
            break
        parts.append(decoder.decode(tokenizer.token_bytes(token), final=False))
    return "".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--expected-root-tokens", type=int, default=8087)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--observer", type=Path)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    profile_path = args.profile.resolve(strict=True)
    context_file_input = args.context_file.resolve(strict=True)
    pinned_context_sha256 = verify_file_sha256(context_file_input, args.context_sha256)
    profile = json.loads(profile_path.read_bytes())
    if profile.get("model_contract_id") is None or not 1 <= args.max_tokens <= 5000:
        raise ResidentProbeError("profile identity or output-token ceiling is invalid")
    runtime = args.runtime_root.resolve(strict=True)
    cli = runtime / "tools" / "cow_cli.py"
    if not cli.is_file():
        raise ResidentProbeError("frozen runtime has no cow_cli.py")
    observer = args.observer.resolve(strict=True) if args.observer else runtime / "experimental/branching_context/scripts/observe_cow_cli.py"
    if _sha(observer) != args.observer_sha256:
        raise ResidentProbeError("observer source hash differs from the pinned observer")
    if not observer.is_file():
        raise ResidentProbeError("pinned runtime has no observe_cow_cli.py")
    if not args.run_dir.exists():
        args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    require_fresh_output(args.run_dir.exists())
    output = _private_run_dir(args.run_dir)
    store_dir = output / "store"
    cli_state_dir = output / "client-state"
    cli_state_dir.mkdir(mode=0o700)
    prior_state_dir = os.environ.get("COW_CLI_STATE_DIR")
    os.environ["COW_CLI_STATE_DIR"] = str(cli_state_dir)
    progress = ProgressTracker(output, "resident-questions").start()
    service = None
    campaign = None
    campaign_runtime_helper = None
    store = None
    result = {"schema": "resident-code-root-questions-v2", "status": "RUNNING",
        "question_ceiling": args.max_tokens, "completed_questions": [], "limitations": [
            "This measures resident-prefix work, durable request completion, and native handle reuse; it is not a logits oracle.",
            "No explicit residency lock is used. The evidence is the native capture/restore handle and zero native file-load bytes.",
            "TTFT is CLI-side NDJSON token-event arrival, not a model-internal timestamp or kernel-only metric."]}
    ordinal = 1
    try:
        sys.path[:0] = [str(runtime), str(runtime / "tools"),
                        str(runtime / "experimental/branching_context/src"),
                        str(runtime / "experimental/branching_context/scripts")]
        from run_q4_publication_fault import _runtime_closure
        runtime_closure = _runtime_closure(profile, runtime)
        selected_imports = verify_selected_imports(profile, runtime, (
            "context_time_travel.coordinator", "context_time_travel.strata_backend",
            "context_time_travel.real", "context_time_travel.store",
            "context_time_travel.controller", "serve.cow_http", "tools.cow_cli",
            "tools.strata_tokenizer"))
        from run_cli_workflow_probe import _Campaign
        from types import SimpleNamespace
        campaign = _Campaign(SimpleNamespace(run_dir=output, runtime_root=runtime,
            profile=profile_path, store=store_dir))
        campaign.profile = profile
        campaign.tracker = progress
        campaign.result_path = output / "campaign-lifecycle.json"
        campaign.record = {"schema": "resident-probe-campaign-lifecycle-v1",
                           "status": "RUNNING", "services": [], "commands": []}
        campaign._runtime_imports()
        campaign_runtime_helper = importlib.import_module("run_cli_workflow_probe")
        campaign.start_service()
        service = campaign.service
        ready = service.ready
        if ready.get("execution_profile_id") != profile.get("execution_profile_id"):
            raise ResidentProbeError("ready service profile differs from the pinned profile")
        service_imports = campaign.record["services"][-1].get("source_imports")
        if not isinstance(service_imports, dict):
            raise ResidentProbeError("campaign service omitted its selected runtime import witness")
        for name, expected in selected_imports.items():
            if name in service_imports and service_imports[name] != expected:
                raise ResidentProbeError(f"service import {name} differs from the verified client runtime")
        required_service_imports = {"context_time_travel.coordinator", "context_time_travel.strata_backend",
            "context_time_travel.real", "context_time_travel.store", "serve.cow_http"}
        if not required_service_imports.issubset(service_imports):
            raise ResidentProbeError("campaign service import witness is incomplete")
        progress.phase("initialize_pinned_tokenizer_and_renderer")
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        adapter = model_text_adapter(None, profile["model_path"],
            template_kwargs=profile["execution"].get("text_template_kwargs"),
            model_contract_id=profile["model_contract_id"])
        root_text = context_file_input.read_text(encoding="utf-8")
        context_file = output / "code-context.txt"
        context_file.write_text(root_text, encoding="utf-8", newline="\n")
        context_file_sha = _sha(context_file)
        progress.phase("single_native_service_ready")
        url = campaign._ready_url(ready)
        store = campaign.store

        _, status = _invoke(output, runtime, cli, url, "01-status", "status",
                            timeout=args.timeout, mutation=False, api_key=campaign.api_key)
        namespace, nsrow = _invoke(output, runtime, cli, url, "02-namespace", "namespace",
            "--title", "Resident code-context questions", api_key=campaign.api_key)
        result["setup_calls"] = [_call_receipt(status), _call_receipt(nsrow)]
        namespace_id = namespace["namespace_id"]
        native_trace_path = output / "native-1.jsonl"
        before_load_stats, _ = _latest_stats(native_trace_path)
        root_receipt, root_row = _invoke(output, runtime, cli, url, "03-load-root", "load",
            "--namespace-id", namespace_id, "--branch", "CodeRoot", "--file", str(context_file),
            "--operation-id", "resident-root-load", api_key=campaign.api_key)
        after_load_stats, _ = _latest_stats(native_trace_path)
        root_branch = store.head("CodeRoot")
        if root_branch.tip != root_branch.root or root_branch.tip != root_receipt.get("snapshot"):
            raise ResidentQuestionsError("loaded context branch is not the single canonical code root")
        _, root_ids = store.lineage(root_branch.tip)
        root_ids = tuple(root_ids)
        if not root_ids:
            raise ResidentQuestionsError("code context tokenized to an empty root")
        if len(root_ids) != args.expected_root_tokens:
            raise ResidentProbeError(
                f"prepared context token count {len(root_ids)} differs from pinned expectation {args.expected_root_tokens}")
        root_record = store.get_snapshot(root_branch.tip)
        root_controller = root_record["controller"]
        if root_controller.get("status") != "idle":
            raise ResidentQuestionsError("code root controller is not the expected active context-only state")
        load_deltas = {key: _integer_counter(after_load_stats, key) - _integer_counter(before_load_stats, key)
                       for key in COUNTERS}
        if (load_deltas["forward_tokens"] != len(root_ids)
                or load_deltas["prefill_tokens"] != len(root_ids)
                or load_deltas["loaded_file_bytes"] != 0
                or load_deltas["integrity_mismatches"] != 0):
            raise ResidentQuestionsError("initial context root load did not perform one clean full-prefix prefill")
        original_fingerprint = checkpoint_fingerprint(store_dir, root_branch.tip)
        result.update(profile_sha256=_sha(profile_path), context_file_sha256_input=pinned_context_sha256,
            execution_profile_id=profile["execution_profile_id"],
            model_contract_id=profile["model_contract_id"], runtime_root=str(runtime),
            cli_sha256=_sha(cli), observer_path=str(observer), observer_sha256=_sha(observer),
            harness_sha256=_sha(Path(__file__)), fixture_root_text_sha256=hashlib.sha256(root_text.encode("utf-8")).hexdigest(),
            campaign_lifecycle_helper_path=str(Path(campaign_runtime_helper.__file__).resolve()),
            campaign_lifecycle_helper_sha256=_sha(campaign_runtime_helper.__file__),
            context_file_sha256=context_file_sha, root_snapshot=root_branch.tip,
            root_token_count=len(root_ids), root_payload_sha256=original_fingerprint["payload_sha256"],
            initial_root_load_call=_call_receipt(root_row),
            initial_root_load_counter_deltas=load_deltas,
            selected_runtime_source_closure=runtime_closure,
            selected_runtime_imports=selected_imports)
        branches = ("Question1", "Question2", "Question3")
        fork_calls = []
        for index, branch in enumerate(branches, 1):
            _, row = _invoke(output, runtime, cli, url, f"{3+index:02}-fork-{index}", "fork",
                "--source-branch", "CodeRoot", "--name", branch,
                "--operation-id", f"resident-fork-{index}", timeout=args.timeout,
                api_key=campaign.api_key)
            if store.head(branch).tip != root_branch.tip:
                raise ResidentQuestionsError(f"{branch} did not start from the exact code root")
            fork_calls.append(_call_receipt(row))
        result["fork_calls"] = fork_calls

        trace_rows = _read_trace(native_trace_path)
        captures = [row for row in trace_rows if row.get("command") == "capture"
                    and isinstance(row.get("response"), dict)
                    and row["response"].get("consumed") == len(root_ids)]
        if not captures:
            raise ResidentQuestionsError("native trace has no full-prefix capture for the code root")
        root_capture = captures[-1]
        root_handle = root_capture["response"].get("handle")
        if type(root_handle) is not int:
            raise ResidentQuestionsError("native code-root capture has no integer handle")

        question_rows = []
        question_traces = []
        root_tip_before = store.head("CodeRoot").tip
        for index, (branch, question) in enumerate(zip(branches, QUESTIONS), 1):
            progress.phase(f"question_{index}_ask", completed=f"question_{index-1}_durable")
            source_head = store.head(branch)
            if source_head.tip != root_branch.tip:
                raise ResidentQuestionsError(f"{branch} moved away from the original code root")
            sibling_tips_before = {name: store.head(name).tip for name in branches if name != branch}
            _, source_ids = store.lineage(source_head.tip)
            source_ids = tuple(source_ids)
            source_record = store.get_snapshot(source_head.tip)
            question_ids = tuple(adapter.append_initial_user_suffix(source_ids,
                source_record["controller"], question))
            before_stats, before_trace = _latest_stats(native_trace_path)
            trace_start = len(before_trace)
            observer_trace_path = output / f"question-{index}.observer.json"
            receipt, row = _invoke(output, runtime, cli, url, f"question-{index}", "ask",
                "--branch", branch, "--text", question, "--max-tokens", str(args.max_tokens),
                "--operation-id", f"resident-question-{index}", "--revision", str(source_head.revision),
                timeout=args.timeout, machine_result=True, observer=observer, trace_path=observer_trace_path,
                api_key=campaign.api_key)
            after_stats, after_trace = _latest_stats(native_trace_path)
            output_ids = tuple(store.lineage(receipt["snapshot"])[1][len(root_ids)+len(question_ids):])
            if not output_ids:
                raise ResidentQuestionsError(f"question {index} generated no answer tokens")
            target_head = store.head(branch)
            final_record = store.get_snapshot(receipt["snapshot"])
            if (target_head.tip != receipt["snapshot"]
                    or target_head.revision != source_head.revision + 1
                    or final_record.get("parent") != source_head.tip):
                raise ResidentQuestionsError(f"question {index} receipt is not the branch's single child of P")
            completion = validate_answer_completion(output_token_ids=output_ids,
                max_tokens=args.max_tokens, controller_status=final_record["controller"]["status"],
                eos_tokens=final_record["controller"]["policy"]["eos_tokens"])
            work = validate_question_work(before_stats, after_stats,
                question_token_ids=question_ids, output_token_ids=output_ids)
            lineage = validate_question_lineage(source_snapshot=source_head.tip,
                expected_root_snapshot=root_branch.tip, root_token_ids=root_ids,
                branch_source_token_ids=source_ids,
                branch_lineage_ids=store.lineage(receipt["snapshot"])[1],
                question_token_ids=question_ids, output_token_ids=output_ids)
            expected_controller = _expected_controller(f"resident-question-{index}",
                source_ids + question_ids, output_ids, tokenizer, profile)
            if final_record["controller"] != expected_controller:
                raise ResidentQuestionsError(f"question {index} controller differs from consumed token sequence")
            output_text = _decoded_answer(output_ids, source_record["controller"], tokenizer,
                profile["execution"]["eos_tokens"])
            answer = validate_machine_answer(row["machine_envelope"], expected_snapshot=receipt["snapshot"],
                expected_output_ids=output_ids, expected_text=output_text)
            observed_timing = validate_observation_trace(row["observation"], branch=branch,
                operation_id=f"resident-question-{index}", receipt=receipt,
                expected_tokens=output_ids)
            if store.head("CodeRoot").tip != root_branch.tip:
                raise ResidentQuestionsError("canonical code-root branch moved during an answer")
            sibling_tips_after = {name: store.head(name).tip for name in branches if name != branch}
            if sibling_tips_after != sibling_tips_before:
                raise ResidentQuestionsError(f"question {index} changed an inactive sibling branch")
            question_rows.append({"index": index, "branch": branch, "question": question,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "request": _call_receipt(row), "receipt": receipt, "source_snapshot": source_head.tip,
                "snapshot": receipt["snapshot"], "output_token_ids": list(output_ids),
                "output_count": len(output_ids), "completion": completion, "work": work,
                "lineage": lineage, "controller_exact": True,
                "inactive_sibling_tips_unchanged": sibling_tips_before == sibling_tips_after,
                "elapsed_cli_seconds": row["elapsed_seconds"],
                "generation_seconds": answer["generation_seconds"],
                "request_seconds": answer["request_seconds"],
                "ttft_seconds_client_observed": observed_timing["ttft_ns"] / 1_000_000_000,
                "request_to_durable_seconds_client_observed": observed_timing["request_to_durable_ns"] / 1_000_000_000,
                "machine_answer_sha256": hashlib.sha256(row["stdout"].encode("utf-8")).hexdigest(),
                "observer_trace_sha256": _sha(observer_trace_path),
                "counter_observation_scope": "native stats trace immediately before and after full CLI request"})
            question_traces.append(_trace_slice(after_trace, trace_start, len(after_trace)))
            result["completed_questions"] = question_rows
            result["status"] = "RUNNING"
            _publish(output / "progress-result.json", result)
            progress.phase(f"question_{index}_complete", completed=f"question_{index}_durable")

        residency = validate_resident_prefix(root_handle=root_handle, root_tokens=len(root_ids),
            root_capture=root_capture, question_traces=question_traces)
        final_root_fingerprint = checkpoint_fingerprint(store_dir, root_branch.tip)
        stable = validate_root_stability(root_tip_before=root_tip_before,
            root_tip_after=store.head("CodeRoot").tip, sibling_tips_before={},
            sibling_tips_after={},
            root_payload_sha_before=original_fingerprint["payload_sha256"],
            root_payload_sha_after=final_root_fingerprint["payload_sha256"])
        shape = validate_campaign_shape(root_tokens=len(root_ids),
            question_sources=[row["source_snapshot"] for row in question_rows],
            question_snapshots=[row["snapshot"] for row in question_rows],
            actual_output_counts=[row["output_count"] for row in question_rows],
            max_tokens=args.max_tokens)
        result.update(status="PASS", server_pid=service.process.pid, root_handle_evidence=residency,
            root_stability=stable, final_root_fingerprint=final_root_fingerprint,
            campaign_shape=shape,
            native_trace_sha256=_sha(native_trace_path), questions=question_rows,
            total_actual_output_tokens=sum(row["output_count"] for row in question_rows),
            numeric_qualification="NOT_PERFORMED_NO_INDEPENDENT_LOGITS_ORACLE",
            limitations=["The configured output value is a ceiling; each row reports actual generated tokens and EOS status.",
                "No explicit residency lock is used; the evidence covers the retained native KV image handle, not all auxiliary model state.",
                "TTFT is client-observed event arrival; it is not a model-internal timestamp or kernel-only measure.",
                "Initial root-load CLI duration includes tokenization, full prefix prefill, checkpoint publication, and native capture; capture-only duration is not exposed by the current trace.",
                "This is not an independent logits or quality evaluation."])
        _publish(output / "result.json", result)
        campaign.stop_service()
        service = None
        service_record = campaign.record["services"][-1]
        close = service_record.get("stop")
        worker = service_record.get("worker_shutdown")
        if (not isinstance(close, dict) or not isinstance(worker, dict)
                or close.get("exit_code") != 0 or worker.get("native_exit_code") != 0):
            raise ResidentQuestionsError("HTTP service or native worker did not exit cleanly")
        result["service_shutdown"] = _redact_campaign_value(
            service_record, campaign.api_key, campaign_runtime_helper.redact_sensitive_value)
        _publish(output / "result.json", result)
        progress.finish("PASS")
        print(json.dumps({"status": result["status"], "root_tokens": len(root_ids),
            "outputs": [row["output_count"] for row in question_rows],
            "native_pids": [worker["native_pid"]], "exit_codes": [close["exit_code"], worker["native_exit_code"]]}))
        return 0
    except BaseException as error:
        result["status"] = "FAIL"
        error_text = f"{type(error).__name__}: {error}"
        if campaign is not None and campaign_runtime_helper is not None:
            error_text = _redact_campaign_value(
                error_text, campaign.api_key, campaign_runtime_helper.redact_sensitive_value)
        result["error"] = error_text
        try:
            _publish(output / "result.json", result)
        except BaseException:
            pass
        progress.finish("FAIL", result["error"])
        raise
    finally:
        if service is not None:
            try:
                campaign.stop_service()
                result["cleanup_service"] = _redact_campaign_value(
                    campaign.record["services"][-1], campaign.api_key,
                    campaign_runtime_helper.redact_sensitive_value)
            except BaseException as cleanup_error:
                cleanup_text = f"{type(cleanup_error).__name__}: {cleanup_error}"
                if campaign is not None and campaign_runtime_helper is not None:
                    cleanup_text = _redact_campaign_value(
                        cleanup_text, campaign.api_key, campaign_runtime_helper.redact_sensitive_value)
                result["cleanup_error"] = cleanup_text
            try:
                _publish(output / "result.json", result)
            except BaseException:
                pass
        progress.close()
        if prior_state_dir is None:
            os.environ.pop("COW_CLI_STATE_DIR", None)
        else:
            os.environ["COW_CLI_STATE_DIR"] = prior_state_dir


if __name__ == "__main__":
    raise SystemExit(main())
