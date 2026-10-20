"""Independent fresh-process numerical audit for a completed G4 CLI workflow.

The SUT is never started here. Fresh oracle processes consume canonical token
histories and explicit request boundaries; saved SUT checkpoint bytes are read
only for integrity/frontier comparison and are never passed to the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
from pathlib import Path
import struct
import sys


VOCAB_SIZE = 248320
PINNED_FRESH_EOS_TOKENS = (248044, 248046)
SEGMENTED_ORACLE_SHA256 = "130866087ea5c19e1a6b2c556e77a9537ffa4854d19bb637696af8c612ffad5a"
SEGMENTED_ORACLE_SOURCE_SHA256 = "85f9c1942f697ede539f977674db4ffe9af9f0800a30ace4dfe7baf718223b5c"
SEGMENTED_ORACLE_BUILD_RECEIPT_SHA256 = "d251cd11267d2e32a57144d25588a7f6450bd466df72f1cd828b21b36a2c7b0c"
FRONTIER_WIDTH = VOCAB_SIZE * 4
COMMAND_LABELS = (
    "namespace", "prepare", "fork-a", "fork-b", "answer-a", "answer-b",
    "evaluate-a", "evaluate-b", "inspect-a", "inspect-b", "select", "adopt",
    "history", "continue",
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_segmented_oracle_build(binary: Path, original_receipt: Path,
                                    build_result: Path, input_report: Path) -> dict:
    if sha256_file(original_receipt) != SEGMENTED_ORACLE_BUILD_RECEIPT_SHA256:
        raise ValueError("original fresh-oracle build receipt hash differs")
    build = json.loads(build_result.read_bytes())
    inputs = json.loads(input_report.read_bytes())
    binary_hash = sha256_file(binary)
    if (build.get("status") != "BUILT_NOT_EXECUTED" or build.get("runtime_executed") is not False
            or build.get("gpu_launched") is not False):
        raise ValueError("segmented oracle build receipt is not compile-only")
    if (build.get("segmented_source_sha256") != SEGMENTED_ORACLE_SOURCE_SHA256
            or build.get("binary_sha256") != SEGMENTED_ORACLE_SHA256
            or binary_hash != SEGMENTED_ORACLE_SHA256):
        raise ValueError("segmented oracle build receipt does not bind source and ELF")
    if (build.get("original_build_receipt_sha256") != SEGMENTED_ORACLE_BUILD_RECEIPT_SHA256
            or build.get("verified_input_count") != 53
            or build.get("verified_input_mismatches") != []):
        raise ValueError("segmented oracle build did not use the fully verified pinned inputs")
    if (inputs.get("receipt_sha256") != SEGMENTED_ORACLE_BUILD_RECEIPT_SHA256
            or inputs.get("checked_files") != 53 or inputs.get("mismatches") != []):
        raise ValueError("segmented oracle input-verification report is incomplete")
    if build.get("verified_input_report_sha256") != sha256_file(input_report):
        raise ValueError("segmented oracle build receipt points to a different input report")
    return {"binary_sha256": binary_hash,
            "original_build_receipt_sha256": sha256_file(original_receipt),
            "segmented_build_receipt_sha256": sha256_file(build_result),
            "input_verification_sha256": sha256_file(input_report),
            "verified_input_count": inputs["checked_files"]}


def validate_frontier_bytes(raw: bytes) -> int:
    """Validate one complete little-endian FP32 vocabulary frontier."""
    if not isinstance(raw, bytes) or len(raw) != FRONTIER_WIDTH:
        raise ValueError(f"frontier width must be exactly {FRONTIER_WIDTH} bytes")
    values = struct.unpack(f"<{VOCAB_SIZE}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("frontier contains nonfinite FP32 values")
    return VOCAB_SIZE


def validate_campaign_schedule(campaign: dict, *, expected_outputs: int,
                               expected_continue: int) -> None:
    """Reject incomplete, reordered, or differently sized CLI trajectories."""
    if not isinstance(campaign, dict) or campaign.get("status") != "CLI_FLOW_PASS_NUMERICAL_AUDIT_PENDING":
        raise ValueError("campaign is not a completed pending-numerical CLI flow")
    commands = campaign.get("commands")
    if not isinstance(commands, list) or [row.get("label") for row in commands] != list(COMMAND_LABELS):
        raise ValueError("campaign command schedule differs from the required workflow")
    if any(type(row.get("exit_code")) is not int or row["exit_code"] != 0 for row in commands):
        raise ValueError("campaign contains a failed or nonnumeric CLI command exit")
    if type(expected_outputs) is not int or expected_outputs <= 0:
        raise ValueError("expected candidate output count is invalid")
    if type(expected_continue) is not int or expected_continue <= 0:
        raise ValueError("expected continuation output count is invalid")
    if campaign.get("outputs") != expected_outputs:
        raise ValueError("campaign candidate output limit differs")
    candidates = campaign.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {"a", "b"}:
        raise ValueError("campaign does not contain exactly two candidate answers")
    for key, operation in (("a", "alt-answer"), ("b", "resume-op")):
        row = candidates[key]
        tokens = row.get("output_token_ids")
        span = row.get("output_span")
        if row.get("operation_id") != operation:
            raise ValueError(f"candidate {key} operation binding differs")
        if (not isinstance(row.get("source_snapshot"), str) or not row["source_snapshot"]
                or not isinstance(row.get("snapshot"), str) or not row["snapshot"]):
            raise ValueError(f"candidate {key} snapshot identity is missing")
        if not isinstance(tokens, list) or len(tokens) != expected_outputs:
            raise ValueError(f"candidate {key} answer count differs")
        if not all(type(token) is int and 0 <= token < VOCAB_SIZE for token in tokens):
            raise ValueError(f"candidate {key} token IDs are invalid")
        if not isinstance(span, dict) or type(span.get("count")) is not int or span["count"] != expected_outputs:
            raise ValueError(f"candidate {key} presentation output span differs")
        receipt = row.get("receipt")
        if (not isinstance(receipt, dict) or receipt.get("snapshot") != row["snapshot"]
                or receipt.get("branch", {}).get("tip") != row["snapshot"]):
            raise ValueError(f"candidate {key} durable receipt differs from its target")
    if candidates["a"]["snapshot"] == candidates["b"]["snapshot"]:
        raise ValueError("independent answer branches unexpectedly share one snapshot identity")
    if candidates["a"]["source_snapshot"] != candidates["b"]["source_snapshot"]:
        raise ValueError("candidate answers do not share the prepared source snapshot")
    selection = campaign.get("selection")
    if not isinstance(selection, dict) or selection.get("snapshot") not in {
            candidates["a"].get("snapshot"), candidates["b"].get("snapshot")}:
        raise ValueError("selection did not choose one of the two answered candidates")
    if campaign.get("continuation_action") != "continue":
        raise ValueError("continuation action is not a same-turn resume")
    adoption = campaign.get("adoption")
    before = adoption.get("before") if isinstance(adoption, dict) else None
    after = adoption.get("after") if isinstance(adoption, dict) else None
    ack = adoption.get("ack") if isinstance(adoption, dict) else None
    if not all(isinstance(value, dict) for value in (before, after, ack)):
        raise ValueError("adoption CAS evidence is incomplete")
    selected_snapshot = campaign["selection"]["snapshot"]
    if (before.get("name") != "main" or after.get("name") != "main"
            or type(before.get("revision")) is not int
            or after.get("revision") != before["revision"] + 1
            or after.get("tip") != selected_snapshot or after.get("root") != before.get("root")
            or ack.get("snapshot_id") != selected_snapshot
            or ack.get("branch") != after):
        raise ValueError("adoption receipt is not the selected snapshot's single CAS move")
    continuation = campaign.get("continuation")
    branch = continuation.get("branch") if isinstance(continuation, dict) else None
    if not isinstance(branch, dict) or branch.get("tip") != continuation.get("snapshot"):
        raise ValueError("continuation receipt is not bound to its branch tip")
    if (branch.get("name") != after.get("name") or type(branch.get("revision")) is not int
            or branch["revision"] != after["revision"] + 1 or branch.get("root") != after.get("root")):
        raise ValueError("continuation does not advance the adopted branch exactly once")
    continuation_row = commands[-1]
    argv = continuation_row.get("argv_redacted")
    if not isinstance(argv, list) or "--max-tokens" not in argv:
        raise ValueError("continuation command omitted its explicit output limit")
    position = argv.index("--max-tokens")
    if position + 1 >= len(argv) or argv[position + 1] != str(expected_continue):
        raise ValueError("continuation output limit differs from required schedule")
    services = campaign.get("services")
    if not isinstance(services, list) or not services:
        raise ValueError("campaign has no service lifecycle receipts")
    for service in services:
        stop = service.get("stop")
        worker = service.get("worker_shutdown")
        if not isinstance(stop, dict) or type(stop.get("exit_code")) is not int or stop["exit_code"] != 0:
            raise ValueError("service has no numeric zero shutdown receipt")
        if not isinstance(worker, dict) or type(worker.get("native_exit_code")) is not int or worker["native_exit_code"] != 0:
            raise ValueError("native worker has no numeric zero shutdown receipt")


def validate_operation_binding(operation_id: str, envelope: dict, source_id: str,
                               target_id: str, target_snapshot: dict,
                               expected_output_span: dict, *, expected_kind: str) -> None:
    """Bind a Store lookup, presentation source/span, and canonical child."""
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError("operation ID is missing")
    if not isinstance(envelope, dict) or envelope.get("operation_id") != operation_id:
        raise ValueError("presentation operation receipt does not bind the requested operation")
    if envelope.get("result", {}).get("tip") != target_id:
        raise ValueError("operation receipt does not bind the target snapshot")
    presentation = envelope.get("presentation")
    if not isinstance(presentation, dict):
        raise ValueError("presentation annotation is missing")
    if presentation.get("kind") != expected_kind:
        raise ValueError("presentation action kind differs from requested boundary")
    source = presentation.get("source")
    if not isinstance(source, dict) or source.get("snapshot") != source_id:
        raise ValueError("presentation source snapshot differs")
    input_span = presentation.get("input_span")
    if not isinstance(input_span, dict) or input_span.get("count") != 0:
        raise ValueError("resume/answer operation unexpectedly consumed a new input span")
    if presentation.get("output_span") != expected_output_span:
        raise ValueError("presentation output span differs from campaign record")
    if not isinstance(target_snapshot, dict) or target_snapshot.get("parent") != source_id:
        raise ValueError("canonical target is not a direct child of the bound source")
    if target_snapshot.get("id") not in (None, target_id):
        raise ValueError("canonical target snapshot identity differs")


def validate_root_presentation(annotations: list[dict], *, namespace_id: str,
                               token_count: int, input_identity: str) -> dict:
    """Bind a prepared canonical root to its namespace root annotation."""
    if not isinstance(annotations, list) or len(annotations) != 1:
        raise ValueError("prepared source must have exactly one verified root presentation")
    annotation = annotations[0]
    if not isinstance(annotation, dict) or annotation.get("kind") != "root":
        raise ValueError("prepared source annotation is not a root presentation")
    if annotation.get("namespace_id") != namespace_id:
        raise ValueError("root presentation namespace differs from the campaign chat")
    source = annotation.get("source")
    if (not isinstance(source, dict) or source.get("snapshot") is not None
            or source.get("revision") != 0 or source.get("token_count") != 0):
        raise ValueError("root presentation source is not the empty canonical source")
    span = annotation.get("input_span")
    if (not isinstance(span, dict) or span.get("start") != 0
            or type(span.get("count")) is not int or span["count"] != token_count
            or span.get("token_ids_sha256") != input_identity):
        raise ValueError("root presentation input span differs from prepared source history")
    if annotation.get("output_span") is not None:
        raise ValueError("root presentation unexpectedly contains generated output")
    return annotation


def validate_controller_exact(actual: dict, expected: dict, *, label: str) -> None:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError(f"{label} controller is missing")
    if actual != expected:
        raise ValueError(f"{label} complete canonical controller differs")


def validate_oracle_profile(profile: dict, *, sut_path: str) -> list[str]:
    execution = profile.get("execution") if isinstance(profile, dict) else None
    if not isinstance(execution, dict):
        raise ValueError("profile execution is missing")
    sut = execution.get("sut_command")
    oracle = execution.get("oracle_command")
    if not isinstance(sut, list) or not sut or not isinstance(oracle, list) or not oracle:
        raise ValueError("SUT/oracle command is missing")
    oracle_path = str(Path(oracle[0]).resolve(strict=True))
    if oracle_path == str(Path(sut_path).resolve(strict=True)):
        raise ValueError("oracle executable must be distinct from SUT executable")
    for flag in ("--prompt-cache", "--conversation-cache-mib"):
        if flag not in oracle:
            raise ValueError(f"oracle command is missing {flag}")
        index = oracle.index(flag)
        if index + 1 >= len(oracle) or oracle[index + 1] != "0":
            raise ValueError(f"oracle {flag} must be disabled")
    forbidden = {"--checkpoint", "--load-state", "--state-file", "--import-kv"}
    if forbidden.intersection(oracle):
        raise ValueError("oracle command attempts to import saved checkpoint state")
    return [str(value) for value in oracle]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _snapshot_frontier(store_path: Path, snapshot_id: str, expected_payload_sha256: str,
                       *, history: list[int], model_contract_id: str,
                       profile: dict) -> tuple[dict, bytes, str]:
    manifest_path = store_path / "checkpoints" / (snapshot_id + ".json")
    envelope = json.loads(manifest_path.read_bytes())
    from context_time_travel.checkpoint_cache import validate_manifest
    from context_time_travel.real import BranchingModel
    expected = {
        "schema": 1,
        "snapshot": snapshot_id,
        "input_identity": BranchingModel._input_identity(history),
        "model_contract_id": model_contract_id,
        "execution_profile_id": profile["execution_profile_id"],
        "consumed": len(history),
        "positions": "absolute-zero-based",
        "logits_count": VOCAB_SIZE,
    }
    resume_contract = profile.get("execution", {}).get("resume_contract")
    if resume_contract == "identical-native-batches-v1":
        expected["schema"] = 2
    elif resume_contract == "identical-native-operators-v1":
        expected["schema"] = 3
    manifest = validate_manifest(envelope, expected)
    if expected["schema"] in (2, 3):
        from context_time_travel.execution_receipt import (
            validate_operator_schedule, validate_schedule, schedule_digest,
        )
        validator = validate_schedule if expected["schema"] == 2 else validate_operator_schedule
        schedule_digest_value = validator(
            manifest["execution_schedule"], history,
            profile["execution"]["prefill_chunk"],
        )
        if schedule_digest_value != manifest["schedule_sha256"]:
            raise ValueError("checkpoint execution schedule digest differs")
    payload_hash = manifest.get("payload_sha256")
    if payload_hash != expected_payload_sha256:
        raise ValueError("checkpoint payload digest differs from durable campaign receipt")
    payload = store_path / "checkpoints" / (payload_hash + ".state")
    actual_hash = sha256_file(payload)
    if actual_hash != payload_hash:
        raise ValueError("checkpoint payload bytes fail their content hash")
    state_bytes = manifest.get("state_bytes")
    if type(state_bytes) is not int or state_bytes < 0:
        raise ValueError("checkpoint state offset is invalid")
    with payload.open("rb") as stream:
        stream.seek(state_bytes)
        frontier = stream.read(FRONTIER_WIDTH)
        if stream.read(1):
            raise ValueError("checkpoint payload has unexpected trailing bytes")
    validate_frontier_bytes(frontier)
    return manifest, frontier, actual_hash


def continuation_reference_plan(root_tokens: list[int], answer_tokens: list[int],
                                continuation_outputs: int) -> dict:
    """Describe exact answer and continuation decode segments for the raw oracle.

    The SUT submits answer and continuation as separate requests, but the second
    request supplies no new tokens. A segmented raw Prefill+Verifier oracle
    therefore prefills only the canonical root, decodes the same two output
    segments without resetting verifier state, and records the boundary frontier.
    """
    if not root_tokens or not answer_tokens:
        raise ValueError("continuation oracle requires nonempty root and answer histories")
    if type(continuation_outputs) is not int or continuation_outputs <= 0:
        raise ValueError("continuation output count must be a positive exact integer")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE
           for token in root_tokens + answer_tokens):
        raise ValueError("continuation oracle history contains invalid token IDs")
    return {
        "prompt_ids": list(root_tokens),
        "request_sizes": [len(root_tokens)],
        "generated": len(answer_tokens) + continuation_outputs,
        "decode_segments": [len(answer_tokens), continuation_outputs],
        "answer_boundary": len(answer_tokens),
        "continuation_outputs": continuation_outputs,
        "answer_tokens": list(answer_tokens),
    }


def validate_continuation_reference_plan(*, prompt_ids: list[int], request_sizes: list[int],
                                         generated: int, decode_segments: list[int],
                                         root_tokens: list[int],
                                         answer_tokens: list[int], continuation_outputs: int) -> None:
    """Fail closed if answer tokens are incorrectly substituted as prompt input."""
    expected = continuation_reference_plan(root_tokens, answer_tokens, continuation_outputs)
    if request_sizes != expected["request_sizes"]:
        raise ValueError("continuation reference request schedule cannot prefill answer tokens")
    if prompt_ids != expected["prompt_ids"]:
        raise ValueError("continuation reference must prefill only the canonical root")
    if generated != expected["generated"]:
        raise ValueError("continuation reference must decode answer and continuation in one stream")
    if decode_segments != expected["decode_segments"]:
        raise ValueError("continuation oracle must execute the two exact generation segments")


def validate_continuation_plan(plan: dict, *, root_tokens: list[int],
                               answer_tokens: list[int], continuation_outputs: int) -> None:
    """Validate every field of the plan object that run_audit passes to the oracle."""
    if not isinstance(plan, dict):
        raise ValueError("continuation oracle plan is missing")
    validate_continuation_reference_plan(
        prompt_ids=plan.get("prompt_ids"),
        request_sizes=plan.get("request_sizes"),
        generated=plan.get("generated"),
        decode_segments=plan.get("decode_segments"),
        root_tokens=root_tokens,
        answer_tokens=answer_tokens,
        continuation_outputs=continuation_outputs,
    )


def validate_fresh_oracle_eos(profile_eos: list[int]) -> None:
    """The pinned raw fresh-prefill oracle has these two literal EOS IDs."""
    if (not isinstance(profile_eos, list) or any(type(token) is not int for token in profile_eos)
            or len(profile_eos) != len(PINNED_FRESH_EOS_TOKENS)
            or set(profile_eos) != set(PINNED_FRESH_EOS_TOKENS)):
        raise ValueError("profile EOS IDs are incompatible with the pinned fresh oracle")


def validate_segment_eos(tokens: list[int], eos_tokens: list[int], *, must_continue: bool) -> None:
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("oracle output token sequence is invalid")
    eos = set(eos_tokens)
    if any(token in eos for token in tokens[:-1]):
        raise ValueError("oracle segment contains EOS before its final token")
    if must_continue and tokens[-1] in eos:
        raise ValueError("oracle answer ended with EOS before its continuation request")


def validate_segmented_decode_receipt(work: dict, schedule: dict, *, segments: list[int],
                                      prompt_tokens: int) -> None:
    """Require the raw oracle to have completed the exact normal-decode segments."""
    if not isinstance(segments, list) or not segments or any(type(n) is not int or n <= 0 for n in segments):
        raise ValueError("oracle decode segment request is invalid")
    expected = [{"generated_start": sum(segments[:i]),
                 "native_start_position": prompt_tokens + sum(segments[:i]), "count": count}
                for i, count in enumerate(segments)]
    if schedule.get("decode_operator") != "normal-verifier-decode":
        raise ValueError("oracle did not use normal verifier decode")
    if schedule.get("requested_segments") != segments or schedule.get("completed_segments") != expected:
        raise ValueError("oracle completed decode segments differ from requested positions/counts")
    total = sum(segments)
    if (work.get("decode_tokens") != total or work.get("normal_forward_tokens") != total
            or work.get("consumed") != prompt_tokens + total
            or work.get("prefill_tokens") != prompt_tokens):
        raise ValueError("oracle normal forward count differs from segmented decode total")


def validate_prefill_chunks(root_manifest: dict, oracle_schedule: dict) -> None:
    """Require a fresh oracle to execute the root's exact recorded prefill chunks."""
    execution = root_manifest.get("execution_schedule") if isinstance(root_manifest, dict) else None
    if not isinstance(execution, list) or not execution:
        raise ValueError("SUT root manifest has no execution schedule")
    prefill = execution[0]
    if not isinstance(prefill, dict) or prefill.get("operator") != "batched-prefill":
        raise ValueError("SUT root schedule does not begin with batched prefill")
    chunks = prefill.get("chunks")
    count = prefill.get("count")
    position = prefill.get("position")
    if (not isinstance(chunks, list) or not chunks
            or type(count) is not int or count <= 0
            or type(position) is not int or position != 0
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(type(value) is not int for value in pair) for pair in chunks)):
        raise ValueError("SUT root prefill chunk receipt is malformed")
    if sum(pair[1] for pair in chunks) != count:
        raise ValueError("SUT root prefill chunks do not sum to the recorded count")
    if not isinstance(oracle_schedule, dict):
        raise ValueError("oracle prefill schedule is missing")
    if (oracle_schedule.get("prefill_operator") != "batched-prefill"
            or oracle_schedule.get("prefill_chunks") != chunks
            or oracle_schedule.get("prefill_requests") != [count]):
        raise ValueError("oracle prefill chunks differ from the SUT root execution receipt")


def reference_segmented(profile, outdir: Path, name: str, ids: list[int],
                        prefill_requests: list[int], segments: list[int],
                        oracle_binary: Path) -> dict:
    """Run raw Prefill+Verifier with request boundaries but no COW/Store session."""
    from run_root_reuse_probe import FRONTIER_WIDTH, publish

    if not segments or any(type(n) is not int or n <= 0 for n in segments):
        raise ValueError("segmented oracle schedule is invalid")
    outdir.mkdir(parents=True, exist_ok=False)
    token_path = outdir / f"{name}.tokens"
    logits_path = outdir / f"{name}.f32"
    token_path.write_text(",".join(map(str, ids)), encoding="ascii", newline="")
    command = list(profile["execution"]["oracle_command"])
    if "--max-new" not in command:
        raise ValueError("pinned oracle command is missing --max-new")
    command[0] = str(oracle_binary)
    command[command.index("--max-new") + 1] = str(sum(segments))
    command += ["--tokens-file", str(token_path)]
    env = dict(os.environ)
    env.update(profile["execution"].get("environment", {}))
    env["STRATA_DUMP_FIRST_LOGITS"] = str(logits_path)
    env["STRATA_PREFILL_REQUESTS"] = ",".join(map(str, prefill_requests))
    env["STRATA_DECODE_SEGMENTS"] = ",".join(map(str, segments))
    with (outdir / f"{name}.stdout").open("xb") as stdout, (outdir / f"{name}.stderr").open("xb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        pid = child.pid
        process = {"name": name, "pid": pid, "command": command, "status": "RUNNING"}
        publish(outdir / f"{name}.process.json", process)
        try:
            code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                code = child.wait(timeout=20)
            process.update(status="TIMEOUT", exit_code=code)
            publish(outdir / f"{name}.process.json", process)
            raise TimeoutError(f"segmented oracle {name} timed out; pid={pid}, exit={code}")
        process.update(status="EXITED", exit_code=code)
        publish(outdir / f"{name}.process.json", process)
    if code != 0 or Path("/proc", str(pid)).exists():
        raise RuntimeError(f"segmented oracle {name} did not exit cleanly: code={code}, pid={pid}")
    work = json.loads((outdir / f"{name}.stdout").read_text(encoding="utf-8"))
    schedule = json.loads(Path(str(logits_path) + ".schedule.json").read_text(encoding="utf-8"))
    validate_segmented_decode_receipt(work, schedule, segments=segments, prompt_tokens=len(ids))
    if schedule.get("prefill_requests") != prefill_requests:
        raise ValueError("segmented oracle prompt prefill schedule differs")
    if work.get("prefill_tokens") != len(ids) or work.get("normal_forward_tokens") != sum(segments):
        raise ValueError("segmented oracle prefill or normal forward count differs")
    tokens = work.get("generated")
    if not isinstance(tokens, list) or len(tokens) != sum(segments):
        raise ValueError("segmented oracle did not complete all requested outputs")
    raw = Path(str(logits_path) + ".decode.f32").read_bytes()
    prompt = Path(str(logits_path) + ".requests.f32").read_bytes()
    segment_raw = Path(str(logits_path) + ".segments.f32").read_bytes()
    if len(raw) != FRONTIER_WIDTH * len(tokens) or len(prompt) != FRONTIER_WIDTH * len(prefill_requests):
        raise ValueError("segmented oracle per-token/prompt frontiers are incomplete")
    if len(segment_raw) != FRONTIER_WIDTH * len(segments):
        raise ValueError("segmented oracle boundary frontier file is incomplete")
    return {"name": name, "pid": pid, "exit_code": code, "command": command,
            "work": work, "schedule": schedule, "tokens": tokens,
            "generated_frontiers": [raw[i:i + FRONTIER_WIDTH] for i in range(0, len(raw), FRONTIER_WIDTH)],
            "prompt_frontiers": [prompt[i:i + FRONTIER_WIDTH] for i in range(0, len(prompt), FRONTIER_WIDTH)],
            "segment_frontiers": [segment_raw[i:i + FRONTIER_WIDTH] for i in range(0, len(segment_raw), FRONTIER_WIDTH)]}


def _ids_for_receipt(store, operation_id: str, campaign_row: dict, source_id: str,
                     source_history: list[int], expected_count: int) -> tuple[dict, dict, list[int]]:
    envelope = store.presentation_receipt(operation_id)
    target_id = campaign_row.get("snapshot")
    target = store.get_snapshot(target_id)
    span = campaign_row.get("output_span")
    validate_operation_binding(operation_id, envelope, source_id, target_id, target, span,
                               expected_kind="answer")
    start, count = span.get("start"), span.get("count")
    if type(start) is not int or type(count) is not int or start != len(source_history) or count != expected_count:
        raise ValueError("presentation span is outside the exact source history boundary")
    root_id, lineage = store.lineage(target_id)
    if root_id != store.lineage(source_id)[0] or lineage[:len(source_history)] != source_history:
        raise ValueError("target lineage does not preserve its source history")
    output_tokens = lineage[start:start + count]
    if len(output_tokens) != count:
        raise ValueError("target lineage truncates the presentation output span")
    return envelope, target, output_tokens


def _expected_resume_controller(controller_module, source_controller: dict,
                                tokens: list[int], tokenizer, eos_tokens: list[int]) -> dict:
    return controller_module.consume_many(
        source_controller,
        ((token, tokenizer.token_bytes(token), token in eos_tokens) for token in tokens),
    )


def run_audit(args) -> dict:
    runtime = args.runtime_root.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    campaign_path = args.campaign.resolve(strict=True)
    profile = json.loads(profile_path.read_bytes())
    campaign = json.loads(campaign_path.read_bytes())
    oracle_binary = args.oracle_binary.resolve(strict=True)
    oracle_build_proof = validate_segmented_oracle_build(
        oracle_binary, args.original_build_receipt.resolve(strict=True),
        args.build_result.resolve(strict=True), args.input_verification.resolve(strict=True))
    validate_campaign_schedule(campaign, expected_outputs=args.outputs,
                               expected_continue=args.continue_outputs)
    if sha256_file(profile_path) != campaign["source"]["profile_sha256"]:
        raise ValueError("campaign profile hash differs from the supplied profile")
    if profile.get("execution_profile_id") != campaign["source"].get("execution_profile_id"):
        raise ValueError("campaign execution profile identity differs")
    if profile.get("model_contract_id") != campaign["source"].get("model_contract_id"):
        raise ValueError("campaign model contract identity differs")
    sut_path = str(Path(profile["execution"]["sut_command"][0]).resolve(strict=True))
    if sha256_file(Path(sut_path)) != campaign["source"].get("native_sha256"):
        raise ValueError("campaign SUT binary hash differs from profile binary")
    oracle_command = validate_oracle_profile(profile, sut_path=sut_path)
    if not os.access(oracle_binary, os.X_OK):
        raise ValueError("segmented oracle binary is not executable")
    oracle_sha256 = sha256_file(oracle_binary)
    if oracle_sha256 != SEGMENTED_ORACLE_SHA256:
        raise ValueError("segmented oracle binary hash differs from the reviewed audit build")
    if oracle_binary == Path(sut_path).resolve(strict=True):
        raise ValueError("segmented oracle binary must differ from the SUT")
    if oracle_binary == Path(oracle_command[0]).resolve(strict=True):
        raise ValueError("segmented oracle binary must differ from the profile oracle")
    for service in campaign["services"]:
        for pid in (service.get("pid"), service.get("worker_shutdown", {}).get("native_pid")):
            if type(pid) is not int or Path("/proc", str(pid)).exists():
                raise ValueError("campaign producer process is not absent")

    for key in list(os.environ):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            os.environ.pop(key, None)
    os.environ.update(profile["execution"].get("environment", {}))
    sys.path[:0] = [str(runtime), str(runtime / "tools"),
                    str(runtime / "experimental/branching_context/src"),
                    str(runtime / "experimental/branching_context/scripts")]
    from run_root_reuse_probe import compare_vectors, expected_controller
    from run_cow_client_recovery import ProgressTracker
    from context_time_travel import controller as controller_module
    from context_time_travel.store import Store
    from tools.strata_tokenizer import Tokenizer
    from context_time_travel.strata_backend import _validate_execution, _validate_model

    _validate_execution(profile)
    model_verification = _validate_model(profile)
    store_path = Path(campaign["store"]).resolve(strict=True)
    store = Store(store_path)
    tokenizer = Tokenizer.from_gguf(profile["model_path"])
    eos_tokens = profile["execution"]["eos_tokens"]
    if len(tokenizer.tokens) != VOCAB_SIZE:
        raise ValueError("tokenizer vocabulary differs from pinned frontier width")
    validate_fresh_oracle_eos(eos_tokens)

    candidates = campaign["candidates"]
    selected_snapshot = campaign["selection"]["snapshot"]
    selected_key = next(key for key in ("a", "b") if candidates[key]["snapshot"] == selected_snapshot)
    selected = candidates[selected_key]
    root_id = selected["source_snapshot"]
    root_snapshot = store.get_snapshot(root_id)
    root_id_from_lineage, root_tokens = store.lineage(root_id)
    if root_id_from_lineage != root_id or root_snapshot.get("parent") is not None:
        raise ValueError("prepared source is not a canonical root snapshot")
    prepare_commands = [row for row in campaign["commands"] if row["label"] == "prepare"]
    prepare_argv = prepare_commands[0].get("argv_redacted", [])
    if "--operation-id" not in prepare_argv:
        raise ValueError("prepare command omitted its operation ID")
    prepare_op = prepare_argv[prepare_argv.index("--operation-id") + 1]
    namespace_id = campaign.get("adopted_history", {}).get("history", {}).get("chat_id")
    if not isinstance(namespace_id, str) or not namespace_id:
        raise ValueError("campaign does not identify the prepared source namespace")
    root_annotations = store.root_presentations(namespace_id, root_id)
    from context_time_travel.real import BranchingModel
    validate_root_presentation(root_annotations, namespace_id=namespace_id,
                               token_count=len(root_tokens),
                               input_identity=BranchingModel._input_identity(root_tokens))
    root_checkpoint_receipt = store.checkpoint_receipt(prepare_op)
    if (root_checkpoint_receipt.get("operation_id") != prepare_op
            or root_checkpoint_receipt.get("checkpoint_receipt", {}).get("snapshot") != root_id
            or root_checkpoint_receipt.get("result", {}).get("tip") != root_id):
        raise ValueError("prepare checkpoint receipt is not bound to the canonical root")
    root_manifest, root_frontier, root_payload_hash = _snapshot_frontier(
        store_path, root_id, root_checkpoint_receipt["checkpoint_receipt"]["payload_sha256"],
        history=root_tokens, model_contract_id=root_snapshot["model_contract_id"], profile=profile)

    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    tracker = ProgressTracker(output, "independent-g4-cli-numerics").start()
    record = {
        "schema": "g4-cli-independent-numerics-v1", "status": "RUNNING",
        "campaign_sha256": sha256_file(campaign_path), "profile_sha256": sha256_file(profile_path),
        "execution_profile_id": profile["execution_profile_id"],
        "model_contract_id": profile["model_contract_id"],
        "sut_sha256": sha256_file(Path(sut_path)), "oracle_sha256": sha256_file(Path(oracle_command[0])),
        "segmented_oracle_binary": str(oracle_binary),
        "segmented_oracle_sha256": oracle_sha256,
        "segmented_oracle_build_proof": oracle_build_proof,
        "segmented_oracle_source_sha256": SEGMENTED_ORACLE_SOURCE_SHA256,
        "segmented_oracle_original_build_receipt_sha256": SEGMENTED_ORACLE_BUILD_RECEIPT_SHA256,
        "oracle_checkpoint_imports": 0, "model_verification": model_verification,
        "source_snapshot": root_id, "source_token_count": len(root_tokens),
        "source_payload_sha256": root_payload_hash,
        "reference_repetitions": 2, "comparisons": [], "references": [],
        "trajectories": [], "failures": [],
        "scope": "Two fresh-process references for each 8-token candidate and the selected 8-token continuation; saved payloads are read-only audit inputs, never oracle inputs.",
    }
    result_path = output / "result.json"
    _write_json(result_path, record)
    try:
        def exact(actual: bytes, expected: bytes, label: str):
            maximum = compare_vectors(actual, expected, label, output)
            record["comparisons"].append({"label": label, "exact": True,
                                          "max_abs_error": maximum,
                                          "actual_sha256": hashlib.sha256(actual).hexdigest(),
                                          "expected_sha256": hashlib.sha256(expected).hexdigest()})
            _write_json(result_path, record)

        def run_two(name: str, ids: list[int], request_sizes: list[int], generated: int,
                    decode_segments: list[int]):
            if sum(request_sizes) != len(ids) or any(type(x) is not int or x <= 0 for x in request_sizes):
                raise ValueError("oracle request schedule does not partition its exact history")
            repeated = []
            for repeat in range(2):
                case_name = f"{name}-fresh-{repeat + 1}"
                tracker.phase(case_name)
                if sum(decode_segments) != generated:
                    raise ValueError("oracle decode segments do not sum to requested outputs")
                fresh = reference_segmented(profile, output / case_name, case_name, ids,
                                            request_sizes, decode_segments, oracle_binary)
                if fresh["exit_code"] != 0 or Path("/proc", str(fresh["pid"])).exists():
                    raise ValueError(f"oracle {case_name} did not exit cleanly")
                if fresh["schedule"].get("prefill_requests") != request_sizes:
                    raise ValueError(f"oracle {case_name} executed a different request schedule")
                validate_prefill_chunks(root_manifest, fresh["schedule"])
                validate_segmented_decode_receipt(fresh["work"], fresh["schedule"],
                                                   segments=decode_segments,
                                                   prompt_tokens=len(ids))
                if fresh["work"].get("prefill_tokens") != len(ids):
                    raise ValueError(f"oracle {case_name} prefetched a different history length")
                compact = {key: fresh[key] for key in ("name", "pid", "exit_code", "command", "work", "schedule", "tokens")}
                compact["prompt_frontier_sha256"] = [hashlib.sha256(value).hexdigest() for value in fresh["prompt_frontiers"]]
                compact["generated_frontier_sha256"] = [hashlib.sha256(value).hexdigest() for value in fresh["generated_frontiers"]]
                compact["segment_frontier_sha256"] = [hashlib.sha256(value).hexdigest()
                                                        for value in fresh["segment_frontiers"]]
                record["references"].append(compact)
                _write_json(result_path, record)
                repeated.append(fresh)
            if repeated[0]["tokens"] != repeated[1]["tokens"]:
                raise ValueError(f"independent {name} reference tokens differ")
            for field in ("prompt_frontiers", "generated_frontiers", "segment_frontiers"):
                if len(repeated[0][field]) != len(repeated[1][field]):
                    raise ValueError(f"independent {name} references have different {field} lengths")
                for index, (left, right) in enumerate(zip(repeated[0][field], repeated[1][field])):
                    exact(left, right, f"{name}-repeat-{field}-{index}")
            return repeated[0], repeated[1]

        eos_set = set(eos_tokens)
        for key in ("a", "b"):
            item = candidates[key]
            op = item["operation_id"]
            source_id = item["source_snapshot"]
            source_record = store.get_snapshot(source_id)
            source_root, source_history = store.lineage(source_id)
            receipt, target, tokens = _ids_for_receipt(store, op, item, source_id,
                                                       source_history, args.outputs)
            if tokens != item["output_token_ids"]:
                raise ValueError(f"candidate {key} campaign token list differs from canonical lineage")
            if source_root != root_id or source_history != root_tokens:
                raise ValueError(f"candidate {key} source is not the prepared canonical root")
            validate_segment_eos(tokens, eos_tokens, must_continue=(key == selected_key))
            expected = expected_controller(op, root_tokens, tokens, tokenizer, eos_tokens, VOCAB_SIZE)
            validate_controller_exact(target.get("controller"), expected, label=f"candidate {key}")
            checkpoint_receipt = store.checkpoint_receipt(op)
            manifest, final_frontier, payload_hash = _snapshot_frontier(
                store_path, item["snapshot"], checkpoint_receipt["checkpoint_receipt"]["payload_sha256"],
                history=store.lineage(item["snapshot"])[1],
                model_contract_id=target["model_contract_id"], profile=profile)
            first, second = run_two("candidate-" + key, root_tokens, [len(root_tokens)],
                                    args.outputs, decode_segments=[args.outputs])
            if first["tokens"] != tokens:
                raise ValueError(f"candidate {key} tokens differ from independent fresh oracle")
            exact(root_frontier, first["prompt_frontiers"][0], f"candidate-{key}-prepared-prefix")
            exact(final_frontier, first["generated_frontiers"][-1], f"candidate-{key}-final")
            record["trajectories"].append({"name": "candidate-" + key, "operation_id": op,
                "source_snapshot": source_id, "target_snapshot": item["snapshot"],
                "stream_key": expected.get("stream_key"), "actual_output_tokens": len(tokens),
                "tokens_exact": True, "controller_exact": True, "source_frontier_sha256": hashlib.sha256(root_frontier).hexdigest(),
                "final_frontier_sha256": hashlib.sha256(final_frontier).hexdigest(),
                "payload_sha256": payload_hash,
                "request_schedule": first["schedule"].get("prefill_requests"),
                "request_sizes": [len(root_tokens)]})
            tracker.phase("candidate-" + key + "-audited", completed="candidate-" + key)
            _write_json(result_path, record)

        continuation = campaign["continuation"]
        continuation_row = next(row for row in campaign["commands"] if row["label"] == "continue")
        argv = continuation_row["argv_redacted"]
        operation_id = argv[argv.index("--operation-id") + 1]
        selected_snapshot_record = store.get_snapshot(selected_snapshot)
        selected_root, selected_history = store.lineage(selected_snapshot)
        if selected_root != root_id:
            raise ValueError("selected candidate changed canonical computational root")
        selected_controller = expected_controller(selected["operation_id"], root_tokens,
            selected["output_token_ids"], tokenizer, eos_tokens, VOCAB_SIZE)
        validate_controller_exact(selected_snapshot_record.get("controller"), selected_controller,
                                  label="selected candidate before continuation")
        cont_item = {"snapshot": continuation.get("snapshot"),
                     "output_span": None}
        cont_envelope = store.presentation_receipt(operation_id)
        presentation = cont_envelope.get("presentation")
        cont_target = store.get_snapshot(continuation["snapshot"])
        span = presentation.get("output_span") if isinstance(presentation, dict) else None
        if not isinstance(span, dict):
            raise ValueError("continuation presentation has no generated output span")
        validate_operation_binding(operation_id, cont_envelope, selected_snapshot,
            continuation["snapshot"], cont_target, span, expected_kind="resume")
        if span.get("start") != len(selected_history) or span.get("count") != args.continue_outputs:
            raise ValueError("continuation span differs from exact selected-history boundary")
        cont_root, cont_history = store.lineage(continuation["snapshot"])
        if cont_root != root_id or cont_history[:len(selected_history)] != selected_history:
            raise ValueError("continuation history does not extend the selected answer")
        continuation_tokens = cont_history[span["start"]:span["start"] + span["count"]]
        if len(continuation_tokens) != args.continue_outputs:
            raise ValueError("continuation output span is truncated")
        expected_cont_controller = _expected_resume_controller(controller_module,
            selected_controller, continuation_tokens, tokenizer, eos_set)
        validate_controller_exact(cont_target.get("controller"), expected_cont_controller,
                                  label="continuation after winner controller")
        cont_checkpoint = store.checkpoint_receipt(operation_id)
        _, cont_frontier, cont_payload_hash = _snapshot_frontier(store_path,
            continuation["snapshot"], cont_checkpoint["checkpoint_receipt"]["payload_sha256"],
            history=cont_history, model_contract_id=cont_target["model_contract_id"], profile=profile)
        # Continuation adds no new prompt tokens. Use the ordinary verifier
        # decode path for all answer+continuation tokens, then compare the
        # independently saved SUT answer frontier at the exact request edge.
        plan = continuation_reference_plan(root_tokens, selected["output_token_ids"],
                                           args.continue_outputs)
        validate_continuation_plan(
            plan, root_tokens=root_tokens,
            answer_tokens=selected["output_token_ids"],
            continuation_outputs=args.continue_outputs)
        validate_segment_eos(selected["output_token_ids"], eos_tokens, must_continue=True)
        first, second = run_two("selected-continuation", plan["prompt_ids"],
            plan["request_sizes"], plan["generated"], decode_segments=plan["decode_segments"])
        answer_boundary = plan["answer_boundary"]
        reference_answer = first["tokens"][:answer_boundary]
        reference_continuation = first["tokens"][answer_boundary:]
        if reference_answer != selected["output_token_ids"]:
            raise ValueError("continuation reference answer segment differs from selected CLI answer")
        if reference_continuation != continuation_tokens:
            raise ValueError("continuation tokens differ from independent fresh oracle")
        validate_segment_eos(continuation_tokens, eos_tokens, must_continue=False)
        exact(root_frontier, first["prompt_frontiers"][0], "continuation-prepared-prefix")
        selected_checkpoint_receipt = store.checkpoint_receipt(selected["operation_id"])
        _, selected_frontier, selected_payload_hash = _snapshot_frontier(store_path,
            selected_snapshot, selected_checkpoint_receipt["checkpoint_receipt"]["payload_sha256"],
            history=selected_history, model_contract_id=selected_snapshot_record["model_contract_id"],
            profile=profile)
        exact(selected_frontier, first["generated_frontiers"][answer_boundary - 1],
              "continuation-selected-answer-boundary")
        exact(selected_frontier, first["segment_frontiers"][0],
              "continuation-first-request-boundary")
        exact(cont_frontier, first["generated_frontiers"][-1], "continuation-final")
        record["trajectories"].append({"name": "selected-continuation", "operation_id": operation_id,
            "source_snapshot": selected_snapshot, "target_snapshot": continuation["snapshot"],
            "stream_key": selected_controller.get("stream_key"), "actual_output_tokens": len(continuation_tokens),
            "tokens_exact": True, "controller_exact": True,
            "source_frontier_sha256": hashlib.sha256(selected_frontier).hexdigest(),
            "final_frontier_sha256": hashlib.sha256(cont_frontier).hexdigest(),
            "source_payload_sha256": selected_payload_hash, "payload_sha256": cont_payload_hash,
            "request_schedule": first["schedule"].get("prefill_requests"),
            "request_sizes": plan["request_sizes"],
            "oracle_generated_total": plan["generated"],
            "answer_boundary_decode_index": answer_boundary,
            "separate_sut_generation_boundaries": True,
            "oracle_schedule": "root-prefill; verifier-decode answer segment; exact SUT answer checkpoint frontier; verifier-decode continuation segment"})
        record["status"] = "PASS"
        record["all_reference_pids_absent"] = all(
            not Path("/proc", str(reference_row["pid"])).exists() for reference_row in record["references"])
        if record["all_reference_pids_absent"] is not True:
            raise ValueError("an independent reference process remains alive")
        record["exact_comparisons"] = len(record["comparisons"])
        _write_json(result_path, record)
    except BaseException as exc:
        record["status"] = "FAIL"
        record["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_json(result_path, record)
        raise
    finally:
        tracker.finish(record["status"], str(record["failures"]) if record["failures"] else None)
        tracker.close()
    return record


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--oracle-binary", type=Path, required=True)
    parser.add_argument("--original-build-receipt", type=Path, required=True)
    parser.add_argument("--build-result", type=Path, required=True)
    parser.add_argument("--input-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", type=int, default=8)
    parser.add_argument("--continue-outputs", type=int, default=8)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    record = run_audit(args)
    print(json.dumps({"status": record["status"],
                      "exact_comparisons": record["exact_comparisons"],
                      "trajectories": len(record["trajectories"]),
                      "references": len(record["references"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
