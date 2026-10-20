"""Bounded CLI test for automatic reuse of one resident code-context root.

This is an integration harness, not a general cache manager or a logits oracle.
It creates one code root and three independent question branches in one service
process, then audits native work, lineage, controller state, and root immutability.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_cow_client_recovery import (
    ProgressTracker, _private_run_dir, _ready_url, _start_service,
    _trace_worker, checkpoint_fingerprint,
)
from context_time_travel.store import Store
from context_time_travel import controller
from serve.cow_http import model_text_adapter
from tools.cow_cli import safe_text
from tools.strata_tokenizer import Tokenizer


class ResidentQuestionsError(RuntimeError):
    """The real request did not meet the bounded resident-prefix contract."""


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


def _invoke(output, repo, cli, url, label, *command, timeout=1800, mutation=True):
    args = [sys.executable, str(cli), "--url", url, "--timeout", str(timeout), *command]
    started = time.monotonic()
    result = subprocess.run(args, cwd=repo, capture_output=True, timeout=timeout + 30)
    elapsed = time.monotonic() - started
    (output / f"{label}.stdout").write_bytes(result.stdout)
    (output / f"{label}.stderr").write_bytes(result.stderr)
    if result.returncode != 0:
        raise ResidentQuestionsError(
            f"CLI {label} exited {result.returncode}: {result.stderr.decode('utf-8', 'replace')[-4000:]}")
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    if mutation:
        marker = "durable receipt:\n"
        if marker not in stderr:
            raise ResidentQuestionsError(f"CLI {label} did not return a durable receipt")
        receipt = json.loads(stderr.split(marker, 1)[1])
    else:
        receipt = json.loads(stdout)
    return receipt, {"label": label, "argv": args, "exit_code": result.returncode,
        "elapsed_seconds": elapsed, "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(), "stdout": stdout,
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
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.max_tokens != 5000:
        raise ResidentQuestionsError("this bounded test is pinned to the approved 5000-token ceiling")
    profile_path = args.profile.resolve(strict=True)
    fixture_path = args.fixture.resolve(strict=True)
    profile = json.loads(profile_path.read_bytes())
    fixture = json.loads(fixture_path.read_bytes())
    if (profile.get("model_contract_id") is None
            or not isinstance(fixture.get("root_messages"), list)
            or not fixture["root_messages"]
            or not isinstance(fixture["root_messages"][0].get("content"), str)):
        raise ResidentQuestionsError("profile or real code-context fixture has an unexpected schema")
    runtime = Path(os.environ["COW_RUNTIME_REPO"]).resolve(strict=True)
    cli = runtime / "tools" / "cow_cli.py"
    if not cli.is_file():
        raise ResidentQuestionsError("frozen runtime has no cow_cli.py")
    output = _private_run_dir(args.run_dir)
    store_dir = output / "store"
    cli_state_dir = output / "client-state"
    cli_state_dir.mkdir(mode=0o700)
    prior_state_dir = os.environ.get("COW_CLI_STATE_DIR")
    os.environ["COW_CLI_STATE_DIR"] = str(cli_state_dir)
    progress = ProgressTracker(output, "resident-questions").start()
    service = None
    store = None
    result = {"schema": "resident-code-root-questions-v1", "status": "RUNNING",
        "question_ceiling": args.max_tokens, "completed_questions": [], "limitations": [
            "This measures resident-prefix work, durable request completion, and native handle reuse; it is not a logits oracle.",
            "No explicit residency lock is used. The evidence is the native capture/restore handle and zero native file-load bytes.",
            "First-token latency is not measured by the CLI subprocess interface."]}
    ordinal = 1
    try:
        progress.phase("initialize_pinned_tokenizer_and_renderer")
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        adapter = model_text_adapter(None, profile["model_path"],
            template_kwargs=profile["execution"].get("text_template_kwargs"),
            model_contract_id=profile["model_contract_id"])
        root_text = fixture["root_messages"][0]["content"]
        context_file = output / "code-context.txt"
        context_file.write_text(root_text, encoding="utf-8", newline="\n")
        context_file_sha = _sha(context_file)
        progress.phase("start_single_native_service")
        service, ready, _http = _start_service("real", store_dir, output, ordinal,
            profile=profile_path, timeout=args.timeout, ram_gib=96, disk_gib=128,
            progress=progress)
        url = _ready_url(ready)
        store = Store(store_dir)

        _, status = _invoke(output, runtime, cli, url, "01-status", "status",
                            timeout=args.timeout, mutation=False)
        namespace, nsrow = _invoke(output, runtime, cli, url, "02-namespace", "namespace",
            "--title", "Resident code-context questions")
        result["setup_calls"] = [_call_receipt(status), _call_receipt(nsrow)]
        namespace_id = namespace["namespace_id"]
        before_load_stats, _ = _latest_stats(output / "native-trace-1.jsonl")
        root_receipt, root_row = _invoke(output, runtime, cli, url, "03-load-root", "load",
            "--namespace-id", namespace_id, "--branch", "CodeRoot", "--file", str(context_file),
            "--operation-id", "resident-root-load")
        after_load_stats, _ = _latest_stats(output / "native-trace-1.jsonl")
        root_branch = store.head("CodeRoot")
        if root_branch.tip != root_branch.root or root_branch.tip != root_receipt.get("snapshot"):
            raise ResidentQuestionsError("loaded context branch is not the single canonical code root")
        _, root_ids = store.lineage(root_branch.tip)
        root_ids = tuple(root_ids)
        if not root_ids:
            raise ResidentQuestionsError("code context tokenized to an empty root")
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
        result.update(profile_sha256=_sha(profile_path), fixture_sha256=_sha(fixture_path),
            execution_profile_id=profile["execution_profile_id"],
            model_contract_id=profile["model_contract_id"], runtime_root=str(runtime),
            cli_sha256=_sha(cli), fixture_root_text_sha256=hashlib.sha256(root_text.encode("utf-8")).hexdigest(),
            context_file_sha256=context_file_sha, root_snapshot=root_branch.tip,
            root_token_count=len(root_ids), root_payload_sha256=original_fingerprint["payload_sha256"],
            initial_root_load_call=_call_receipt(root_row),
            initial_root_load_counter_deltas=load_deltas)
        branches = ("Question1", "Question2", "Question3")
        fork_calls = []
        for index, branch in enumerate(branches, 1):
            _, row = _invoke(output, runtime, cli, url, f"{3+index:02}-fork-{index}", "fork",
                "--source-branch", "CodeRoot", "--name", branch,
                "--operation-id", f"resident-fork-{index}", timeout=args.timeout)
            if store.head(branch).tip != root_branch.tip:
                raise ResidentQuestionsError(f"{branch} did not start from the exact code root")
            fork_calls.append(_call_receipt(row))
        result["fork_calls"] = fork_calls

        trace_path = output / "native-trace-1.jsonl"
        trace_rows = _read_trace(trace_path)
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
            before_stats, before_trace = _latest_stats(trace_path)
            trace_start = len(before_trace)
            receipt, row = _invoke(output, runtime, cli, url, f"question-{index}", "ask",
                "--branch", branch, "--text", question, "--max-tokens", str(args.max_tokens),
                "--operation-id", f"resident-question-{index}", timeout=args.timeout)
            after_stats, after_trace = _latest_stats(trace_path)
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
            # CLI output is the decoded answer; this also catches a mismatch between
            # durable token lineage and the text presented to the user.
            output_text = _decoded_answer(output_ids, source_record["controller"], tokenizer,
                profile["execution"]["eos_tokens"])
            if row["stdout"] != safe_text(output_text) + "\n":
                raise ResidentQuestionsError(f"question {index} CLI text differs from durable answer tokens")
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
        result.update(status="PASS", server_pid=service.process.pid, root_handle_evidence=residency,
            root_stability=stable, final_root_fingerprint=final_root_fingerprint,
            native_trace_sha256=_sha(trace_path), questions=question_rows,
            total_actual_output_tokens=sum(row["output_count"] for row in question_rows),
            numeric_qualification="NOT_PERFORMED_NO_INDEPENDENT_LOGITS_ORACLE",
            limitations=["The 5000-token value is a ceiling; each row reports actual generated tokens and EOS status.",
                "No explicit residency lock is used; the evidence covers the retained native KV image handle, not all auxiliary model state.",
                "CLI subprocess duration includes client startup and durable publication; no first-token timing is claimed.",
                "This is not an independent logits or quality evaluation."])
        _publish(output / "result.json", result)
        close = service.stop()
        worker = _trace_worker(output / "pool-trace-1.json", close["pid"])
        progress.native_started(worker["native_pid"])
        if close["exit_code"] != 0 or worker["native_exit_code"] != 0:
            raise ResidentQuestionsError("HTTP service or native worker did not exit cleanly")
        result["service_shutdown"] = {"http": close, "native_worker": worker}
        _publish(output / "result.json", result)
        service = None
        progress.finish("PASS")
        print(json.dumps({"status": result["status"], "root_tokens": len(root_ids),
            "outputs": [row["output_count"] for row in question_rows],
            "native_pids": [worker["native_pid"]], "exit_codes": [close["exit_code"], worker["native_exit_code"]]}))
        return 0
    except BaseException as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        try:
            _publish(output / "result.json", result)
        except BaseException:
            pass
        progress.finish("FAIL", result["error"])
        raise
    finally:
        if service is not None:
            try:
                close = service.stop()
                result["cleanup_http"] = close
            except BaseException as cleanup_error:
                result["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
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
