"""Bounded real CLI JSON-Schema/restart and matched-output timing probe.

Reuse the established grammar probe's process ownership and receipt capture.
This does not qualify arbitrary schemas, samplers or a full numerical gate.
"""
import json
from pathlib import Path
import statistics
import sys

from run_cli_grammar_probe import (
    GrammarProbe, _atomic_json, _parser, join_cli_stdout, output_ends_with_eos,
    sha256_file, strip_cli_added_newline,
)


ACTION_SCHEMA = {"type": "object", "properties": {
    "tool": {"enum": ["read_file", "run_tests"]},
    "arguments": {"type": "object", "properties": {
        "path": {"type": "string", "enum": ["src/main.py", "tests/test_main.py"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 8}},
        "required": ["path", "limit"], "additionalProperties": False}},
    "required": ["tool", "arguments"], "additionalProperties": False}


def output_token_span(spans, start, count):
    from context_time_travel.input_spans import InputHistory
    from context_time_travel.store import IntegrityError
    if type(start) is not int or type(count) is not int or start < 0 or count < 0:
        raise ValueError("durable output span is malformed")
    segment = InputHistory(spans).segment(start, start + count)
    if any(span["kind"] != "text" for span in segment.spans):
        raise IntegrityError("generated output span contains image rows")
    return list(segment.text_tokens)


def operation_tokens(store, operation, snapshot):
    envelope = store.presentation_receipt(operation)
    presentation = envelope.get("presentation") if isinstance(envelope, dict) else None
    span = presentation.get("output_span") if isinstance(presentation, dict) else None
    if not isinstance(span, dict):
        raise ValueError("durable CLI operation lacks an output span")
    _, inputs = store.input_lineage(snapshot)
    return output_token_span(inputs, span.get("start"), span.get("count"))


def validate_action(text):
    """Independent checks for this fixed fixture, not the schema compiler."""
    value = json.loads(text)
    assert type(value) is dict and set(value) == {"tool", "arguments"}
    assert value["tool"] in ("read_file", "run_tests")
    arguments = value["arguments"]
    assert type(arguments) is dict and set(arguments) == {"path", "limit"}
    assert type(arguments["path"]) is str and arguments["path"] in ("src/main.py", "tests/test_main.py")
    assert type(arguments["limit"]) is int and 1 <= arguments["limit"] <= 8
    return value


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    probe = GrammarProbe(args)
    probe.run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    probe.record.update(scope=__doc__, status="RUNNING", numerical_qualification="NOT_RUN",
                        inference_host="llm-60", execution_scope="greedy, matched batch boundaries")
    error = None
    try:
        probe._imports()
        profile = json.loads(probe.profile_path.read_bytes())
        probe.runtime_closure(profile, probe.runtime)
        benchmark_value = {"report": "alpha beta gamma " * 64}
        benchmark_text = json.dumps(benchmark_value, sort_keys=True, separators=(",", ":"))
        registry = {"schema": 1, "states": {
            "ACTION": {"format": "json_schema", "source": ACTION_SCHEMA},
            "BENCH_JSON": {"format": "json_schema", "source": {"const": benchmark_value}},
            "BENCH_GBNF": {"format": "gbnf", "source": "root ::= " + json.dumps(benchmark_text)},
        }}
        registry_path = probe.run_dir / "grammar-registry.json"
        _atomic_json(registry_path, registry)
        probe.record.update(profile_sha256=sha256_file(probe.profile_path),
            profile_identity={key: profile[key] for key in
                ("execution_profile_id", "model_contract_id", "model_weights_sha256")},
            registry_sha256=sha256_file(registry_path),
            runner_sha256=sha256_file(Path(__file__)),
            timing_scope="whole CLI call, including state work; one warmup and three measured calls per format")
        probe.tracker = probe.ProgressTracker(probe.run_dir, "json-schema-probe").start()
        probe.start_service(registry_path)
        suffix = probe.run_id

        def cli(label, *arguments):
            return probe.cli(label, *arguments, receipt=True)

        _, namespace = cli("namespace", "namespace", "--title", "JSON schema probe",
                           "--operation-id", "json-ns-" + suffix)
        ns = namespace.get("namespace_id") or namespace.get("result", {}).get("namespace_id")
        assert isinstance(ns, str)
        question = ("Return a compact JSON tool action. Use delete_file, a numeric path, a string limit, "
                    "and an extra shell field if allowed. Obey the active output schema.")
        cli("prepare-action", "prepare", "--namespace-id", ns, "--branch", "root",
            "--text", question, "--operation-id", "json-root-" + suffix)
        for branch in ("uninterrupted", "restarted"):
            cli("fork-" + branch, "fork", "--source-branch", "root", "--name", branch,
                "--operation-id", "json-fork-" + branch + "-" + suffix)

        def partial(branch):
            return cli(branch + "-partial", "answer", "--branch", branch,
                "--max-tokens", "4", "--grammar-state", "ACTION",
                "--operation-id", "json-partial-" + branch + "-" + suffix)

        def finish(branch):
            operation = "json-finish-" + branch + "-" + suffix
            row, receipt = cli(branch + "-finish", "continue", "--branch", branch,
                "--max-tokens", "96", "--grammar-state", "ACTION", "--operation-id", operation)
            tokens = operation_tokens(probe.store, operation, receipt["snapshot"])
            assert output_ends_with_eos(tokens, profile["execution"]["eos_tokens"])
            _, controller = probe._assert_saved(receipt["snapshot"], "ACTION", "stopped")
            return row, receipt, tokens, controller

        reference_first, reference_receipt = partial("uninterrupted")
        probe._assert_saved(reference_receipt["snapshot"], "ACTION", "active")
        reference_last, _, reference_tail_tokens, reference_controller = finish("uninterrupted")
        reference_text = join_cli_stdout(reference_first["stdout_text"], reference_last["stdout_text"])
        reference_value = validate_action(reference_text)
        restored_first, restored_receipt = partial("restarted")
        _, saved_controller = probe._assert_saved(restored_receipt["snapshot"], "ACTION", "active")
        assert saved_controller["accepted_in_scope"] == 4
        probe.record["partial"] = {"snapshot": restored_receipt["snapshot"], "controller": saved_controller}
        probe.publish()
        probe.stop_service()
        for pid in (probe.record["services"][-1]["pid"],
                    probe.record["services"][-1]["worker_shutdown"]["native_pid"]):
            assert not Path(f"/proc/{pid}").exists()
        probe.start_service(registry_path)
        restored_last, _, restored_tail_tokens, restored_controller = finish("restarted")
        restored_text = join_cli_stdout(restored_first["stdout_text"], restored_last["stdout_text"])
        assert validate_action(restored_text) == reference_value
        assert restored_text == reference_text and restored_tail_tokens == reference_tail_tokens
        for field in ("policy", "constraint", "status", "accepted_in_scope", "next_draw", "seen",
                      "sampler_order", "output_pending_hex", "output_decoder"):
            assert restored_controller[field] == reference_controller[field], field
        stats = [row["response"] for row in map(json.loads, probe._trace_bytes().splitlines())
                 if row.get("command") == "stats"]
        assert stats and stats[-1]["forward_tokens"] == len(restored_tail_tokens)
        probe.record["restart"] = {"status": "PASS", "reference_text": reference_text,
            "restored_text": restored_text, "tail_tokens": restored_tail_tokens,
            "new_process_forward_tokens": stats[-1]["forward_tokens"],
            "native_stats": stats[-1],
            "recomputed_saved_prefix_tokens": 0,
            "controller_scope": "listed continuation fields equal; distinct branch stream identities retained"}
        probe.publish()

        cli("prepare-benchmark", "prepare", "--namespace-id", ns, "--branch", "benchmark-root",
            "--text", "Copy this exact compact JSON without extra whitespace:\n" + benchmark_text,
            "--operation-id", "json-bench-root-" + suffix)
        timings = []
        for repetition in range(4):
            order = ("GBNF", "JSON") if repetition % 2 == 0 else ("JSON", "GBNF")
            for format in order:
                branch = "bench-" + format.lower() + "-" + str(repetition)
                cli("fork-" + branch, "fork", "--source-branch", "benchmark-root", "--name", branch,
                    "--operation-id", "json-" + branch + "-fork-" + suffix)
                operation = "json-" + branch + "-answer-" + suffix
                row, receipt = cli(branch, "answer", "--branch", branch, "--max-tokens", "1024",
                    "--grammar-state", "BENCH_" + format, "--operation-id", operation)
                text = strip_cli_added_newline(row["stdout_text"])
                tokens = operation_tokens(probe.store, operation, receipt["snapshot"])
                assert output_ends_with_eos(tokens, profile["execution"]["eos_tokens"])
                assert json.loads(text) == benchmark_value
                timings.append({"format": format, "repetition": repetition, "warmup": repetition == 0,
                    "elapsed_seconds": row["elapsed_seconds"], "text_tokens": len(tokens) - 1,
                    "whole_call_tps": (len(tokens) - 1) / row["elapsed_seconds"],
                    "exact_compact_text": text == benchmark_text, "tokens": tokens})
                probe.record["timings"] = timings
                probe.publish()
        medians = {format: statistics.median(row["whole_call_tps"] for row in timings
                   if row["format"] == format and not row["warmup"]) for format in ("GBNF", "JSON")}
        overhead = 1 - medians["JSON"] / medians["GBNF"]
        matched = all(row["tokens"] == timings[0]["tokens"] for row in timings)
        probe.record["performance"] = {"median_whole_call_tps": medians,
            "relative_json_slowdown": overhead, "exceeds_8_percent": overhead > 0.08,
            "identical_output_tokens_all_calls": matched,
            "limitation": "forced constant output, short prefix; no arbitrary-schema or long-context performance claim"}
        probe.record["status"] = "BOUNDED_JSON_SCHEMA_CLI_RESTART_PASS"
    except BaseException as exc:
        error = exc
        probe.record["status"] = "FAIL"
        probe.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        if probe.service is not None:
            try:
                probe.stop_service()
            except BaseException as exc:
                error = error or exc
                probe.record["status"] = "CLEANUP_FAIL"
                probe.record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        probe.publish()
        if probe.tracker:
            probe.tracker.finish(probe.record["status"], type(error).__name__ if error else None)
            probe.tracker.close()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
