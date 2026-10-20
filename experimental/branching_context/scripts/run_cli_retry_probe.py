"""Bounded real-service CLI retry probe; root owns any native/GPU launch.

The runner reuses the existing CLI workflow campaign's process, HTTP, progress,
and native-trace helpers. It observes canonical Store records but submits all
user mutations through the unmodified stdlib CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from run_cli_workflow_probe import (
    _Campaign,
    _atomic_write,
    parse_durable_receipt,
    validate_zero_work,
)


class ProbeFailure(RuntimeError):
    pass


class UnsupportedFixture(ProbeFailure):
    """The model stopped before the bounded answer/continue scenario existed."""


def _exact_int(value):
    return type(value) is int


def _error_text(value):
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value)


def validate_exact_replay(original, observed):
    """Validate an actual terminal HTTP event and the accompanying CLI replay."""
    if not isinstance(original, dict) or not isinstance(observed, dict):
        raise ValueError("replay observations must be objects")
    if observed.get("exit_code") != 0:
        raise ValueError("replay CLI did not exit successfully")
    if observed.get("cli_stdout") != original.get("stdout"):
        raise ValueError("replay stdout differs from original")
    response = observed.get("response")
    if not isinstance(response, dict):
        raise ValueError("actual replay completion response is missing")
    if response.get("event") != "complete" or response.get("replayed") is not True:
        raise ValueError("retry did not report an exact replay completion")
    old_answer = original.get("answer")
    answer = response.get("answer")
    if not isinstance(old_answer, dict) or not isinstance(answer, dict):
        raise ValueError("replay answer is missing")
    if answer.get("snapshot") != old_answer.get("snapshot"):
        raise ValueError("replay answer snapshot changed")
    if answer.get("tokens") != old_answer.get("tokens"):
        raise ValueError("replay answer tokens changed")
    text = answer.get("text")
    if not isinstance(text, str) or observed.get("cli_stdout") != text.encode("utf-8") + b"\n":
        raise ValueError("replay CLI text differs from the actual HTTP answer")
    if (response.get("receipt") != original.get("receipt")
            or observed.get("cli_receipt") != original.get("receipt")):
        raise ValueError("replay durable receipt differs from original")
    for label in ("head", "sibling"):
        if observed.get(label + "_before") != observed.get(label + "_after"):
            raise ValueError(f"replay changed {label} branch head")
    before_trace = observed.get("native_trace_bytes_before")
    after_trace = observed.get("native_trace_bytes_after")
    if not _exact_int(before_trace) or not _exact_int(after_trace) or before_trace != after_trace:
        raise ValueError("replay emitted a native RPC trace record")
    validate_zero_work(observed.get("counters_before"), observed.get("counters_after"))


def validate_changed_retry(observed):
    """Require a new operation computed from the original source/controller."""
    if not isinstance(observed, dict) or observed.get("exit_code") != 0:
        raise ValueError("changed retry did not complete successfully")
    response = observed.get("response")
    if not isinstance(response, dict) or response.get("event") != "complete":
        raise ValueError("changed retry did not return a complete event")
    if "replayed" in response and type(response["replayed"]) is not bool:
        raise ValueError("changed retry HTTP confirmation has invalid replay status")
    operation = observed.get("operation_id")
    original_operation = observed.get("original_operation")
    if not isinstance(operation, str) or not operation or operation == original_operation:
        raise ValueError("changed retry requires a new operation ID")
    receipt = response.get("receipt")
    answer = response.get("answer")
    if not isinstance(receipt, dict) or not isinstance(answer, dict):
        raise ValueError("changed retry answer/receipt is missing")
    if receipt.get("operation_id", operation) != operation:
        raise ValueError("changed retry receipt operation binding mismatch")
    if answer.get("tokens") != observed.get("output_tokens"):
        raise ValueError("changed retry output tokens differ from canonical output span")
    text = answer.get("text")
    if not isinstance(text, str) or observed.get("cli_stdout") != text.encode("utf-8") + b"\n":
        raise ValueError("changed retry CLI text differs from the actual HTTP answer")
    if answer.get("snapshot") != observed.get("branch_after", {}).get("tip"):
        raise ValueError("changed retry HTTP confirmation snapshot differs from branch head")
    if observed.get("cli_receipt") != receipt:
        raise ValueError("changed retry CLI receipt differs from actual HTTP confirmation")
    if observed.get("output_tokens") != observed.get("expected_output_tokens"):
        raise ValueError("changed retry output tokens differ from greedy baseline")
    if observed.get("source_snapshot") != observed.get("expected_source_snapshot"):
        raise ValueError("changed retry did not use original pre-answer source")
    if observed.get("source_stream_key") != observed.get("final_stream_key"):
        raise ValueError("changed retry reset the original RNG stream")
    source_accepted = observed.get("source_accepted")
    final_accepted = observed.get("final_accepted")
    if (not _exact_int(source_accepted) or not _exact_int(final_accepted)
            or source_accepted < 0 or final_accepted != source_accepted + len(observed["output_tokens"])):
        raise ValueError("changed retry reset or miscounted the accepted-token cursor")
    before = observed.get("branch_before")
    after = observed.get("branch_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("changed retry branch heads are missing")
    if (after.get("name") != before.get("name") or after.get("root") != before.get("root")
            or after.get("revision") != before.get("revision", -2) + 1
            or after.get("tip") != answer.get("snapshot") or after.get("tip") == before.get("tip")):
        raise ValueError("changed retry did not publish one new branch successor")
    if observed.get("sibling_before") != observed.get("sibling_after"):
        raise ValueError("changed retry moved the sibling branch")
    if observed.get("original_receipt_before") != observed.get("original_receipt_after"):
        raise ValueError("changed retry changed the original durable answer")
    before_trace = observed.get("native_trace_bytes_before")
    after_trace = observed.get("native_trace_bytes_after")
    if not _exact_int(before_trace) or not _exact_int(after_trace) or after_trace <= before_trace:
        raise ValueError("changed retry produced no native RPC work")
    before_counts = observed.get("counters_before")
    after_counts = observed.get("counters_after")
    validate_zero_work(before_counts, before_counts)
    if not isinstance(after_counts, dict):
        raise ValueError("changed retry counters are missing")
    for key in ("forward_tokens", "prefill_tokens"):
        if key not in before_counts or key not in after_counts:
            raise ValueError(f"changed retry is missing {key} counters")
    if after_counts["forward_tokens"] - before_counts["forward_tokens"] < len(observed["output_tokens"]):
        raise ValueError("changed retry lacks forward-token work for its output")


def validate_stale_retry(status, body, head_before, head_after,
                         counters_before, counters_after,
                         trace_bytes_before, trace_bytes_after):
    if status != 409 or "revision" not in _error_text(body).casefold():
        raise ValueError("stale retry was not rejected as a branch revision conflict")
    if head_before != head_after:
        raise ValueError("stale retry changed the branch head")
    if trace_bytes_before != trace_bytes_after:
        raise ValueError("stale retry emitted a native RPC")
    validate_zero_work(counters_before, counters_after)


def _head_record(head):
    return {"name": head.name, "revision": head.revision, "tip": head.tip, "root": head.root}


def _fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    return hashlib.sha256(raw).hexdigest()


def _counter_view(campaign):
    stats = campaign._native_stats()
    if not isinstance(stats, dict):
        raise ProbeFailure("native stats trace is not an object")
    # The native stats command is the authoritative cumulative worker counter.
    counters = stats.get("counters", stats)
    if not isinstance(counters, dict):
        raise ProbeFailure("native stats counters are missing")
    missing = [key for key in ("forward_tokens", "prefill_tokens") if key not in counters]
    if missing:
        raise ProbeFailure("native stats omit zero-work counters: " + ", ".join(missing))
    return {key: counters[key] for key in (
        "forward_tokens", "prefill_tokens", "disk_loads", "native_calls",
        "loaded_bytes", "sampling_calls", "capture_bytes", "hash_bytes",
    ) if key in counters}


def _trace_size(campaign):
    path = campaign.run_dir / f"native-{campaign.ordinal}.jsonl"
    return path.stat().st_size


def _terminal_stream(campaign, body=None, *, raw_body=None):
    """Read one real HTTP retry stream; retain the actual terminal event."""
    parsed = urlsplit(campaign._ready_url(campaign.service.ready))
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    if raw_body is None:
        raw_body = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if not isinstance(raw_body, bytes):
        raise TypeError("raw retry body must be bytes")
    try:
        connection.request("POST", "/cow/text/retry", body=raw_body, headers={
            "Authorization": "Bearer " + campaign.api_key,
            "Content-Type": "application/json",
            "Connection": "close",
        })
        response = connection.getresponse()
        status = response.status
        events = []
        if status == 200:
            while True:
                line = response.fp.readline()
                if not line:
                    break
                item = json.loads(line)
                if not isinstance(item, dict) or not isinstance(item.get("event"), str):
                    raise ProbeFailure("retry HTTP stream emitted a malformed event")
                events.append(item)
        else:
            detail = response.read()
            return status, detail.decode("utf-8", errors="replace"), events
    finally:
        connection.close()
    terminal = [item for item in events if item.get("event") in ("complete", "error")]
    if len(terminal) != 1:
        raise ProbeFailure(f"retry HTTP stream had {len(terminal)} terminal events")
    return status, terminal[0], events


def _stream_summary(event, events):
    answer = event.get("answer") if isinstance(event, dict) else None
    return {
        "event": event.get("event"), "replayed": event.get("replayed"),
        "answer": ({"snapshot": answer.get("snapshot"),
                    "tokens": answer.get("tokens"),
                    "text_sha256": hashlib.sha256(
                        answer.get("text", "").encode("utf-8")).hexdigest()}
                   if isinstance(answer, dict) else None),
        "receipt": event.get("receipt"),
        "stream_events": [row.get("event") for row in events],
    }


def _settings_from_flags(flags):
    names = {"--temperature": ("temperature", float), "--top-k": ("top_k", int),
             "--min-p": ("min_p", float), "--top-p": ("top_p", float)}
    settings = {}
    index = 0
    while index < len(flags):
        name = flags[index]
        if name not in names or index + 1 >= len(flags):
            raise ValueError(f"unsupported retry setting in probe: {name}")
        field, convert = names[name]
        if field in settings:
            raise ValueError(f"duplicate retry setting: {field}")
        settings[field] = convert(flags[index + 1])
        index += 2
    return settings or None


def _journaled_retry_body(row, operation_id, original_operation):
    """Read the exact successful CLI POST body from its private journal."""
    state_dir = row.get("client_state_dir")
    if not isinstance(state_dir, str):
        raise ProbeFailure("CLI observation omitted its request-journal directory")
    path = Path(state_dir) / "operations" / (operation_id + ".json")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeFailure(f"cannot read completed retry journal: {exc}") from None
    if (not isinstance(record, dict) or record.get("schema") != 1
            or record.get("operation_id") != operation_id or record.get("state") != "complete"
            or record.get("path") != "/cow/text/retry"):
        raise ProbeFailure("CLI retry journal is not a completed /cow/text/retry operation")
    body_json = record.get("body_json")
    if (not isinstance(body_json, str) or not isinstance(record.get("url"), str)
            or not isinstance(record.get("intent_sha256"), str)
            or not isinstance(record.get("record_sha256"), str)):
        raise ProbeFailure("CLI retry journal lacks its original serialized body")
    intent = json.dumps([operation_id, record["url"], record["path"], body_json],
                        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(intent).hexdigest() != record["intent_sha256"]:
        raise ProbeFailure("CLI retry journal intent checksum does not match")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    canonical_record = json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical_record).hexdigest() != record["record_sha256"]:
        raise ProbeFailure("CLI retry journal record checksum does not match")
    raw_body = body_json.encode("utf-8")
    body = json.loads(body_json)
    if (not isinstance(body, dict) or body.get("operation_id") != operation_id
            or body.get("from_operation") != original_operation):
        raise ProbeFailure("CLI retry journal body is bound to another operation")
    return raw_body, body


def _output(campaign, operation):
    envelope = campaign.store.presentation_receipt(operation)
    presentation = envelope.get("presentation")
    result = envelope.get("result")
    if not isinstance(presentation, dict) or not isinstance(result, dict):
        raise ProbeFailure(f"canonical presentation missing for {operation}")
    span = presentation.get("output_span")
    if not isinstance(span, dict):
        raise ProbeFailure(f"canonical output span missing for {operation}")
    start, count = span.get("start"), span.get("count")
    if not _exact_int(start) or not _exact_int(count) or start < 0 or count < 1:
        raise ProbeFailure(f"canonical output span invalid for {operation}")
    _, ids = campaign.store.lineage(result["tip"])
    tokens = ids[start:start + count]
    if len(tokens) != count:
        raise ProbeFailure(f"canonical output span exceeds lineage for {operation}")
    snapshot = campaign.store.get_snapshot(result["tip"])
    source = presentation.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("snapshot"), str):
        raise ProbeFailure(f"source snapshot missing for {operation}")
    return {
        "operation_id": operation,
        "snapshot": result["tip"],
        "source_snapshot": source["snapshot"],
        "tokens": list(tokens),
        "presentation": presentation,
        "receipt": envelope,
        "controller": snapshot.get("controller"),
        "source_controller": campaign.store.get_snapshot(source["snapshot"]).get("controller"),
        "checkpoint_receipt": campaign.store.checkpoint_receipt(operation),
    }


def _checkpoint_fingerprint(campaign, store, snapshot):
    from run_cow_client_recovery import checkpoint_fingerprint
    return checkpoint_fingerprint(store, snapshot,
                                  synthetic_fixture=bool(getattr(campaign, "cpu_fixture", False)))


class RetryProbe:
    def __init__(self, args, campaign=None):
        self.args = args
        self.campaign = campaign or _Campaign(SimpleNamespace(
            run_dir=args.run_dir, runtime_root=args.runtime_root,
            profile=args.profile, store=args.store))
        self.campaign.profile = getattr(self.campaign, "profile", None)

    def _cli(self, label, *args, expect=0, want_receipt=False,
             client_name="retry-probe", state_suffix=None):
        row, result = self.campaign.cli(label, client_name, *args,
                                        expect=expect, want_receipt=want_receipt,
                                        state_suffix=state_suffix)
        state_name = "journal-" + client_name + ("-" + state_suffix if state_suffix else "")
        row["client_state_dir"] = str(self.campaign.run_dir / state_name)
        value = None
        if result is not None:
            value = result
        return row, value

    def _replay(self, label, original, *settings):
        c = self.campaign
        branch = original["branch"]
        head_before = _head_record(c.store.head(branch))
        sibling_before = _head_record(c.store.head(self.args.sibling_branch))
        counts_before = _counter_view(c)
        trace_before = _trace_size(c)
        row, receipt = self._cli(label, "retry", "--from-operation", original["operation_id"],
                                 *settings, want_receipt=True)
        http_body = {"from_operation": original["operation_id"],
                     "operation_id": original["operation_id"], "branch": branch,
                     "revision": head_before["revision"], "stream": True}
        retry_settings = _settings_from_flags(settings)
        if retry_settings is not None:
            http_body["settings"] = retry_settings
        status, response, events = _terminal_stream(c, http_body)
        if status != 200 or not isinstance(response, dict):
            raise ProbeFailure(f"exact replay HTTP confirmation returned {status}")
        trace_after = _trace_size(c)
        counts_after = _counter_view(c)
        head_after = _head_record(c.store.head(branch))
        sibling_after = _head_record(c.store.head(self.args.sibling_branch))
        stdout = Path(row["stdout"]).read_bytes()
        observation = {
            "exit_code": row["exit_code"], "cli_stdout": stdout,
            "cli_receipt": receipt, "response": response,
            "head_before": head_before, "head_after": head_after,
            "sibling_before": sibling_before, "sibling_after": sibling_after,
            "native_trace_bytes_before": trace_before, "native_trace_bytes_after": trace_after,
            "counters_before": counts_before, "counters_after": counts_after,
        }
        # The real CLI's durable receipt is the response identity. It must match
        # the original even when the local journal or native process is new.
        validate_exact_replay({"stdout": original["stdout"],
                               "answer": {"snapshot": original["snapshot"],
                                          "tokens": original["tokens"]},
                               "receipt": original["cli_receipt"]}, observation)
        observation["cli_stdout_sha256"] = hashlib.sha256(stdout).hexdigest()
        observation["cli_stdout_bytes"] = len(stdout)
        del observation["cli_stdout"]
        observation["response"] = _stream_summary(response, events)
        return observation

    def _changed(self, label, original, destination_branch, operation, *settings):
        c = self.campaign
        before = _head_record(c.store.head(destination_branch))
        sibling_before = _head_record(c.store.head(self.args.sibling_branch))
        original_receipt_before = c.store.presentation_receipt(original["operation_id"])
        counts_before = _counter_view(c)
        trace_before = _trace_size(c)
        cli_args = ["retry", "--from-operation", original["operation_id"],
                    "--branch", destination_branch, "--operation-id", operation, *settings]
        row, receipt = self._cli(label, *cli_args, want_receipt=True)
        trace_after_cli = _trace_size(c)
        counts_after_cli = _counter_view(c)
        raw_body, http_body = _journaled_retry_body(
            row, operation, original["operation_id"])
        if (http_body.get("branch") != destination_branch
                or http_body.get("revision") != before["revision"]):
            raise ProbeFailure("CLI journal retry body does not preserve submitted branch CAS")
        status, response, events = _terminal_stream(c, raw_body=raw_body)
        if status != 200 or not isinstance(response, dict):
            detail = _error_text(response).replace("\n", " ")[:400]
            raise ProbeFailure(f"changed retry HTTP confirmation returned {status}: {detail}")
        trace_after = _trace_size(c)
        counts_after = _counter_view(c)
        if trace_after != trace_after_cli:
            raise ProbeFailure("idempotent HTTP confirmation issued native RPCs")
        validate_zero_work(counts_after_cli, counts_after)
        after = _head_record(c.store.head(destination_branch))
        sibling_after = _head_record(c.store.head(self.args.sibling_branch))
        result = _output(c, operation)
        original_receipt_after = c.store.presentation_receipt(original["operation_id"])
        source_controller = result["source_controller"]
        final_controller = result["controller"]
        source_accepted = (source_controller.get("accepted_in_scope", 0)
                           if isinstance(source_controller, dict) else 0)
        final_accepted = (final_controller.get("accepted_in_scope", 0)
                          if isinstance(final_controller, dict) else len(result["tokens"]))
        observation = {
            "exit_code": row["exit_code"], "response": response,
            "cli_receipt": receipt, "cli_stdout": Path(row["stdout"]).read_bytes(),
            "operation_id": operation, "original_operation": original["operation_id"],
            "source_snapshot": result["source_snapshot"],
            "expected_source_snapshot": original["source_snapshot"],
            "output_tokens": result["tokens"], "expected_output_tokens": original["tokens"],
            "branch_before": before, "branch_after": after,
            "sibling_before": sibling_before, "sibling_after": sibling_after,
            "source_stream_key": ((source_controller.get("stream_key")
                                   if isinstance(source_controller, dict) else None)
                                  or (original.get("controller", {}).get("stream_key")
                                      if isinstance(original.get("controller"), dict) else None)),
            "final_stream_key": (final_controller.get("stream_key")
                                 if isinstance(final_controller, dict) else None),
            "source_accepted": source_accepted, "final_accepted": final_accepted,
            "native_trace_bytes_before": trace_before, "native_trace_bytes_after": trace_after,
            "counters_before": counts_before, "counters_after": counts_after,
            "native_trace_bytes_after_cli": trace_after_cli,
            "counters_after_cli": counts_after_cli,
            "journal_body_sha256": hashlib.sha256(raw_body).hexdigest(),
            "journal_revision": http_body.get("revision"),
            "original_receipt_before": _fingerprint(original_receipt_before),
            "original_receipt_after": _fingerprint(original_receipt_after),
        }
        validate_changed_retry(observation)
        observation["cli_stdout_sha256"] = hashlib.sha256(observation["cli_stdout"]).hexdigest()
        observation["cli_stdout_bytes"] = len(observation["cli_stdout"])
        del observation["cli_stdout"]
        observation["response"] = _stream_summary(response, events)
        return {"observation": observation, "result": result, "row": row}

    def _stale_conflict(self, original, stale_revision):
        c = self.campaign
        branch = original["branch"]
        head_before = _head_record(c.store.head(branch))
        before_counts = _counter_view(c)
        before_trace = _trace_size(c)
        parsed = urlsplit(c._ready_url(c.service.ready))
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
        body = json.dumps({
            "from_operation": original["operation_id"],
            "operation_id": "stale-native-retry",
            "branch": branch,
            "revision": stale_revision,
            "settings": {"temperature": 0.7, "top_k": 1, "min_p": 0.05},
            "stream": False,
        }, separators=(",", ":")).encode("utf-8")
        try:
            connection.request("POST", "/cow/text/retry", body=body, headers={
                "Authorization": "Bearer " + c.api_key,
                "Content-Type": "application/json",
            })
            response = connection.getresponse()
            raw = response.read()
            status = response.status
        finally:
            connection.close()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = raw.decode("utf-8", errors="replace")
        after_trace = _trace_size(c)
        after_counts = _counter_view(c)
        head_after = _head_record(c.store.head(branch))
        validate_stale_retry(status, value, head_before, head_after,
                              before_counts, after_counts, before_trace, after_trace)
        return {"status": status, "body": value, "stale_revision": stale_revision,
                "current_head": head_after, "native_trace_bytes_before": before_trace,
                "native_trace_bytes_after": after_trace}

    def execute(self):
        run_dir = self.args.run_dir.absolute()
        if run_dir.exists():
            raise SystemExit("run directory already exists; use a new immutable attempt")
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        c = self.campaign
        c.record = {"schema": "cow-cli-retry-native-probe-v1", "status": "RUNNING",
                    "phase": "preflight", "commands": [], "services": [],
                    "retry_observations": [], "failures": [],
                    "numerical_oracle": "NOT_RUN_NO_INDEPENDENT_REFERENCE",
                    "chat_server": "NOT_STARTED"}
        c.publish()
        tracker = None
        error = None
        try:
            c.runtime = self.args.runtime_root.resolve(strict=True)
            c.profile_path = self.args.profile.resolve(strict=True)
            c.profile = json.loads(c.profile_path.read_bytes())
            c.cli_path = c.runtime / "tools" / "cow_cli.py"
            if not c.cli_path.is_file():
                raise ProbeFailure("runtime lacks tools/cow_cli.py")
            c._runtime_imports()
            c._runtime_closure(c.profile, c.runtime)
            if self.args.store.exists() and any(self.args.store.iterdir()):
                raise ProbeFailure("campaign Store must be fresh and empty")
            if not 1 <= len(self.args.question.split()) <= 256:
                raise ProbeFailure("prompt text must be bounded to at most 256 whitespace tokens")
            (run_dir / "calls").mkdir(mode=0o700)
            tracker = c._ProgressTracker(run_dir, "cli-retry-native-probe").start()
            c.tracker = tracker
            c.record.update({
                "phase": "preflight_pass",
                "runtime": str(c.runtime), "profile": str(c.profile_path),
                "profile_sha256": hashlib.sha256(c.profile_path.read_bytes()).hexdigest(),
                "execution_profile_id": c.profile.get("execution_profile_id"),
                "model_contract_id": c.profile.get("model_contract_id"),
                "cli_sha256": hashlib.sha256(c.cli_path.read_bytes()).hexdigest(),
                "native_sha256": hashlib.sha256(Path(c.profile["execution"]["sut_command"][0]).read_bytes()).hexdigest(),
                "store": str(self.args.store.absolute()),
                "question_sha256": hashlib.sha256(self.args.question.encode("utf-8")).hexdigest(),
                "question_whitespace_words": len(self.args.question.split()),
                "outputs_requested": 8,
            })
            c.publish()
            c.start_service()
            self._workflow()
            c.record["status"] = "PASS_PIPELINE_NO_INDEPENDENT_NUMERICAL_ORACLE"
        except UnsupportedFixture as exc:
            error = exc
            c.record["status"] = "UNSUPPORTED_EARLY_EOS"
            c.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        except BaseException as exc:
            error = exc
            c.record["status"] = "FAIL"
            c.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        finally:
            if c.service is not None:
                try:
                    c.stop_service()
                except BaseException as exc:
                    error = error or exc
                    c.record["status"] = "FAIL"
                    c.record["failures"].append({"type": type(exc).__name__, "message": "cleanup: " + str(exc)})
            if tracker is not None:
                tracker.finish(c.record.get("status", "FAIL"), str(error) if error else None)
                tracker.close()
            c.record["status_detail"] = str(error) if error else None
            c.publish()
        return 1 if error else 0

    def _workflow(self):
        c = self.campaign
        run_id = c.run_id
        self._phase("create-input")
        _, ns = self._cli("namespace", "namespace", "--title",
                           "CLI retry probe " + run_id,
                           "--operation-id", "retry-probe-ns-" + run_id,
                           want_receipt=True)
        namespace_id = ns.get("namespace_id") or ns.get("result", {}).get("namespace_id")
        if not isinstance(namespace_id, str):
            raise ProbeFailure("namespace result omitted its ID")
        _, prepared = self._cli("prepare-root", "prepare",
            "--namespace-id", namespace_id, "--branch", "main", "--text", self.args.question,
            "--operation-id", "retry-probe-prepare-" + run_id, want_receipt=True)
        main = c.store.head("main")
        if prepared.get("snapshot") != main.tip or main.tip != main.root:
            raise ProbeFailure("prepared source did not produce the canonical root head")
        source_snapshot = main.tip
        source_record = c.store.get_snapshot(source_snapshot)
        source_controller = source_record.get("controller")
        source_lineage = c.store.lineage(source_snapshot)[1]
        c.record["source"] = {
            "snapshot": source_snapshot, "root": main.root,
            "revision": main.revision, "input_token_count": len(source_lineage),
            "controller": source_controller,
        }
        try:
            from run_cow_client_recovery import checkpoint_fingerprint
            c.record["source"]["checkpoint_fingerprint"] = _checkpoint_fingerprint(
                c, self.args.store, source_snapshot)
        except Exception as exc:
            raise ProbeFailure(f"prepared source checkpoint is not fingerprintable: {exc}") from exc
        c.publish()
        self._cli("fork-chat", "fork", "--source-branch", "main",
                  "--name", self.args.branch, "--operation-id", "retry-probe-fork-chat-" + run_id,
                  want_receipt=True)
        self._cli("fork-sibling", "fork", "--source-branch", "main",
                  "--name", self.args.sibling_branch,
                  "--operation-id", "retry-probe-fork-sibling-" + run_id,
                  want_receipt=True)
        sibling_before = _head_record(c.store.head(self.args.sibling_branch))

        self._phase("answer-eight")
        answer_op = "retry-probe-answer-" + run_id
        row, answer_receipt = self._cli("answer-eight", "answer",
            "--branch", self.args.branch, "--max-tokens", "8", "--operation-id", answer_op,
            want_receipt=True)
        answer = _output(c, answer_op)
        answer["cli_receipt"] = answer_receipt
        answer["branch"] = self.args.branch
        answer["stdout"] = Path(row["stdout"]).read_bytes()
        if len(answer["tokens"]) != 8:
            raise UnsupportedFixture(f"greedy answer ended after {len(answer['tokens'])}/8 outputs")
        if not Path(row["stdout"]).read_bytes().strip():
            raise UnsupportedFixture("greedy answer produced no printable text")
        answer_head = _head_record(c.store.head(self.args.branch))
        c.record["answer"] = {**answer, "stdout": answer["stdout"].decode("utf-8", "replace"),
                              "head": answer_head}
        c.record["original_receipt"] = answer_receipt
        c.record["sibling_before"] = sibling_before
        c.publish()
        source_fp_before = c.record["source"]["checkpoint_fingerprint"]

        self._phase("exact-answer-replay")
        replay = self._replay("exact-answer-replay", answer)
        c.record["retry_observations"].append({"kind": "exact_replay", "operation": answer_op,
                                               "settings": None, "observation": replay})
        self._phase("explicit-t0-answer-replay")
        replay_t0 = self._replay("explicit-t0-answer-replay", answer, "--temperature", "0")
        c.record["retry_observations"].append({"kind": "exact_replay", "operation": answer_op,
                                               "settings": {"temperature": 0.0},
                                               "observation": replay_t0})

        self._phase("changed-answer-retry")
        retry_answer_branch = self.args.branch + "-changed-answer"
        self._cli("fork-retry-answer", "fork", "--source-branch", self.args.branch,
                  "--name", retry_answer_branch,
                  "--operation-id", "retry-probe-fork-answer-" + run_id,
                  want_receipt=True)
        changed_answer_op = "retry-probe-changed-answer-" + run_id
        changed_answer = self._changed("changed-answer-retry", answer,
            retry_answer_branch, changed_answer_op,
            "--temperature", "0.7", "--top-k", "1", "--min-p", "0.05")
        c.record["retry_observations"].append({"kind": "changed_settings", "source": answer_op,
                                               "destination": retry_answer_branch,
                                               **changed_answer})
        c.record["answer_branch_preserved"] = _head_record(c.store.head(self.args.branch))
        c.record["sibling_after_answer_retry"] = _head_record(c.store.head(self.args.sibling_branch))
        c.record["original_answer_receipt_after_changed_retry"] = c.store.presentation_receipt(answer_op)
        c.record["source_checkpoint_after_answer_retry"] = _checkpoint_fingerprint(
            c, self.args.store, source_snapshot)
        if c.record["source_checkpoint_after_answer_retry"] != source_fp_before:
            raise ProbeFailure("changed answer retry altered the original source checkpoint")
        c.publish()

        self._phase("continue-eight")
        continue_op = "retry-probe-continue-" + run_id
        row, continue_receipt = self._cli("continue-eight", "continue",
            "--branch", self.args.branch, "--max-tokens", "8", "--operation-id", continue_op,
            want_receipt=True)
        continued = _output(c, continue_op)
        continued["cli_receipt"] = continue_receipt
        continued["branch"] = self.args.branch
        continued["stdout"] = Path(row["stdout"]).read_bytes()
        if len(continued["tokens"]) != 8:
            raise UnsupportedFixture(f"greedy continuation ended after {len(continued['tokens'])}/8 outputs")
        if not isinstance(continued.get("controller"), dict) or continued["controller"].get("status") != "active":
            raise UnsupportedFixture("greedy continuation did not leave an active controller")
        continue_head = _head_record(c.store.head(self.args.branch))
        c.record["continuation"] = {**continued, "stdout": continued["stdout"].decode("utf-8", "replace"),
                                    "head": continue_head}
        c.publish()
        self._phase("exact-continue-replay")
        replay_continue = self._replay("exact-continue-replay", continued)
        c.record["retry_observations"].append({"kind": "exact_replay", "operation": continue_op,
                                               "settings": None, "observation": replay_continue})
        self._phase("changed-continue-retry")
        retry_continue_branch = self.args.branch + "-changed-continue"
        self._cli("fork-retry-continue", "fork", "--source-branch", self.args.branch,
                  "--name", retry_continue_branch,
                  "--operation-id", "retry-probe-fork-continue-" + run_id,
                  want_receipt=True)
        changed_continue_op = "retry-probe-changed-continue-" + run_id
        changed_continue = self._changed("changed-continue-retry", continued,
            retry_continue_branch, changed_continue_op,
            "--temperature", "0.7", "--top-k", "1", "--min-p", "0.05")
        c.record["retry_observations"].append({"kind": "changed_settings", "source": continue_op,
                                               "destination": retry_continue_branch,
                                               **changed_continue})
        c.publish()

        self._phase("stale-revision-conflict")
        stale = self._stale_conflict(answer, answer_head["revision"])
        c.record["stale_revision_control"] = stale
        c.publish()

        self._phase("stop-and-restart")
        first_service = c.record["services"][-1]
        first_pid = first_service["pid"]
        c.stop_service()
        first_service = c.record["services"][-1]
        self._phase("restart-service")
        c.start_service()
        second_service = c.record["services"][-1]
        if second_service["pid"] == first_pid:
            raise ProbeFailure("HTTP service PID did not change after restart")
        before_head = _head_record(c.store.head(self.args.branch))
        before_sibling = _head_record(c.store.head(self.args.sibling_branch))
        restart_before_stats = _counter_view(c)
        restart_trace_before = _trace_size(c)
        self._phase("restart-exact-answer-replay")
        row, receipt = self._cli("restart-exact-answer-replay",
            "retry", "--from-operation", answer_op, want_receipt=True,
            client_name="fresh-journal-after-restart")
        http_body = {"from_operation": answer_op, "operation_id": answer_op,
                     "branch": answer["branch"], "revision": before_head["revision"],
                     "stream": True}
        status, response, events = _terminal_stream(c, http_body)
        if status != 200 or not isinstance(response, dict):
            raise ProbeFailure(f"post-restart replay HTTP confirmation returned {status}")
        restart_trace_after = _trace_size(c)
        restart_after_stats = _counter_view(c)
        after_head = _head_record(c.store.head(self.args.branch))
        after_sibling = _head_record(c.store.head(self.args.sibling_branch))
        restart_obs = {"exit_code": row["exit_code"],
            "cli_stdout": Path(row["stdout"]).read_bytes(), "cli_receipt": receipt,
            "response": response,
            "head_before": before_head, "head_after": after_head,
            "sibling_before": before_sibling, "sibling_after": after_sibling,
            "native_trace_bytes_before": restart_trace_before,
            "native_trace_bytes_after": restart_trace_after,
            "counters_before": restart_before_stats, "counters_after": restart_after_stats}
        validate_exact_replay({"stdout": answer["stdout"],
            "answer": {"snapshot": answer["snapshot"], "tokens": answer["tokens"]},
            "receipt": answer["cli_receipt"]}, restart_obs)
        restart_stdout = restart_obs.pop("cli_stdout")
        restart_obs["cli_stdout_sha256"] = hashlib.sha256(restart_stdout).hexdigest()
        restart_obs["cli_stdout_bytes"] = len(restart_stdout)
        restart_obs["response"] = _stream_summary(response, events)
        c.record["retry_observations"].append({"kind": "restart_replay", "operation": answer_op,
                                               "observation": restart_obs})
        c.record["services_restarted"] = {"first_pid": first_pid,
            "second_pid": second_service["pid"], "changed": True,
            "first_stop": first_service.get("stop"),
            "first_worker_shutdown": first_service.get("worker_shutdown")}
        c.record["workflow_status"] = "PIPELINE_PASS"
        c.publish()

    def _phase(self, value):
        self.campaign.phase(value)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True,
                        help="fresh disposable Store; never point at source fixtures")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--question", required=True,
                        help="bounded Q4 prompt; record the exact measured token count")
    parser.add_argument("--branch", default="chat")
    parser.add_argument("--sibling-branch", default="sibling")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.branch == args.sibling_branch or not args.branch or not args.sibling_branch:
        raise SystemExit("branch and sibling names must be nonempty and distinct")
    return RetryProbe(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
