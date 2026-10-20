"""Compare CLI seed/stop durable frontiers with fresh raw native references.

The oracle children receive token IDs and ordinary Prefill/Verifier schedules
only. They never receive a COW checkpoint, adapter object, or server process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

from run_cli_seed_stop_raw_audit import run_raw_case


VOCAB_SIZE = 248320
FRONTIER_BYTES = VOCAB_SIZE * 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _frontier_bytes(raw: bytes, label: str) -> bytes:
    if len(raw) != FRONTIER_BYTES:
        raise ValueError(f"{label} is not exactly one pinned FP32 vocabulary vector")
    if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", raw)):
        raise ValueError(f"{label} contains a nonfinite logit")
    return raw


def compare_frontiers(actual: bytes, expected: bytes, label: str) -> dict:
    actual = _frontier_bytes(actual, label + " actual")
    expected = _frontier_bytes(expected, label + " expected")
    if actual != expected:
        max_abs = max(abs(a - b) for (a,), (b,) in zip(
            struct.iter_unpack("<f", actual), struct.iter_unpack("<f", expected)))
        raise ValueError(f"{label} full-frontier mismatch: max_abs_error={max_abs!r}")
    return {"label": label, "exact": True, "max_abs_error": 0.0,
            "actual_sha256": hashlib.sha256(actual).hexdigest(),
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "logits_count": VOCAB_SIZE}


def _exact_ids(value: object, label: str) -> list[int]:
    if (not isinstance(value, list)
            or any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in value)):
        raise ValueError(f"{label} must be a list of exact in-vocabulary token IDs")
    return value


def _receipt_snapshot(store, operation_id: str, snapshot_id: str,
                      profile: dict, model_contract_id: str, root_id: str) -> tuple[dict, bytes, list[int]]:
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError("durable checkpoint operation ID is missing")
    receipt = store.checkpoint_receipt(operation_id)
    durable = receipt.get("checkpoint_receipt") if isinstance(receipt, dict) else None
    if (not isinstance(durable, dict) or durable.get("snapshot") != snapshot_id
            or durable.get("model_contract_id") != model_contract_id
            or durable.get("execution_profile_id") != profile.get("execution_profile_id")):
        raise ValueError(f"checkpoint receipt does not bind snapshot {snapshot_id}")
    actual_root, history = store.lineage(snapshot_id)
    if actual_root != root_id:
        raise ValueError(f"snapshot {snapshot_id} has a different canonical root")
    target = store.get_snapshot(snapshot_id)
    if target.get("model_contract_id") != model_contract_id:
        raise ValueError(f"snapshot {snapshot_id} has a different model contract")
    manifest_path = store.root / "checkpoints" / (snapshot_id + ".json")
    envelope = json.loads(manifest_path.read_bytes())
    from context_time_travel.checkpoint_cache import validate_manifest
    from context_time_travel.real import BranchingModel
    expected = {
        "schema": 1, "snapshot": snapshot_id,
        "input_identity": BranchingModel._input_identity(history),
        "model_contract_id": model_contract_id,
        "execution_profile_id": profile["execution_profile_id"],
        "consumed": len(history), "positions": "absolute-zero-based",
        "logits_count": VOCAB_SIZE,
    }
    resume_contract = profile.get("execution", {}).get("resume_contract")
    if resume_contract == "identical-native-batches-v1":
        expected["schema"] = 2
    elif resume_contract == "identical-native-operators-v1":
        expected["schema"] = 3
    manifest = validate_manifest(envelope, expected)
    if manifest.get("payload_sha256") != durable.get("payload_sha256"):
        raise ValueError(f"snapshot {snapshot_id} payload differs from durable receipt")
    if expected["schema"] in (2, 3):
        from context_time_travel.execution_receipt import (
            validate_operator_schedule, validate_schedule, schedule_digest,
        )
        validator = validate_schedule if expected["schema"] == 2 else validate_operator_schedule
        digest = validator(manifest["execution_schedule"], history,
                           profile["execution"]["prefill_chunk"])
        if digest != manifest.get("schedule_sha256"):
            raise ValueError(f"snapshot {snapshot_id} execution schedule digest differs")
    payload_hash = manifest["payload_sha256"]
    payload_path = store.root / "checkpoints" / (payload_hash + ".state")
    if _sha256(payload_path) != payload_hash:
        raise ValueError(f"snapshot {snapshot_id} state payload fails its content hash")
    state_bytes = manifest.get("state_bytes")
    if type(state_bytes) is not int or state_bytes < 0:
        raise ValueError(f"snapshot {snapshot_id} has an invalid state offset")
    with payload_path.open("rb") as stream:
        stream.seek(state_bytes)
        frontier = stream.read(FRONTIER_BYTES)
        if stream.read(1):
            raise ValueError(f"snapshot {snapshot_id} payload has trailing bytes")
    _frontier_bytes(frontier, "stored snapshot " + snapshot_id)
    return target, frontier, history


def _validate_generation_state(campaign: dict, store, root_id: str,
                               root_tokens: list[int], entries: list[tuple[str, str, list[int]]],
                               *, prefix_tokens: list[int] | None = None,
                               initial_stream_key: str | None = None,
                               initial_draw: int = -1) -> dict:
    observed = []
    expected_history = list(root_tokens) + list(prefix_tokens or [])
    stream_key = initial_stream_key
    previous_draw = initial_draw
    for label, snapshot_id, tokens in entries:
        expected_history.extend(tokens)
        target = store.get_snapshot(snapshot_id)
        lineage_root, actual_history = store.lineage(snapshot_id)
        result_controller = campaign[label]["controller"]
        controller = target.get("controller")
        if lineage_root != root_id or actual_history != expected_history:
            raise ValueError(f"{label} canonical forced-token history differs")
        if controller != result_controller:
            raise ValueError(f"{label} campaign controller differs from canonical snapshot")
        if controller.get("accepted_in_scope") != len(expected_history) - len(root_tokens):
            raise ValueError(f"{label} controller accepted count differs from generated token span")
        key = controller.get("stream_key")
        draw = controller.get("next_draw")
        if not isinstance(key, str) or type(draw) is not int or draw < previous_draw:
            raise ValueError(f"{label} stream identity/cursor is invalid or regressed")
        if stream_key is None:
            stream_key = key
        elif key != stream_key:
            raise ValueError(f"{label} continuation changed the canonical RNG stream")
        previous_draw = draw
        observed.append({"label": label, "snapshot": snapshot_id,
                         "canonical_history_tokens": len(actual_history),
                         "generated_tokens_in_scope": controller["accepted_in_scope"],
                         "stream_key": key, "next_draw": draw,
                         "controller_sha256": hashlib.sha256(json.dumps(
                             controller, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()})
    return {"scope": "canonical Store/controller cross-check only; not independent sampler/RNG audit",
            "states": observed}


def _validate_reseed_transition(campaign: dict, store, root_id: str,
                                source_snapshot: str, source_history: list[int],
                                source_controller: dict) -> dict:
    reseed = campaign.get("reseed")
    if not isinstance(reseed, dict) or not isinstance(reseed.get("ack"), dict):
        raise ValueError("canonical reseed acknowledgement is missing")
    target_id = reseed["ack"].get("tip")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("canonical reseed target is missing")
    target = store.get_snapshot(target_id)
    lineage_root, history = store.lineage(target_id)
    controller = target.get("controller")
    if (lineage_root != root_id or history != source_history
            or target.get("parent") != source_snapshot):
        raise ValueError("reseed is not a zero-token direct child of the prior snapshot")
    if controller != reseed.get("controller"):
        raise ValueError("reseed campaign controller differs from canonical snapshot")
    expected = dict(source_controller)
    new_key = controller.get("stream_key") if isinstance(controller, dict) else None
    if (not isinstance(new_key, str) or not new_key
            or new_key == source_controller.get("stream_key")
            or type(controller.get("next_draw")) is not int
            or controller.get("next_draw") != 0):
        raise ValueError("reseed did not bind a new stream at cursor zero")
    expected["stream_key"] = new_key
    expected["next_draw"] = 0
    if controller != expected:
        raise ValueError("reseed changed controller state beyond RNG stream and cursor")
    return {"snapshot": target_id, "parent": source_snapshot,
            "history_tokens": len(history), "stream_key": new_key, "next_draw": 0}


def _append_input_ids(read_history: list[int], test_history: list[int],
                      test_tokens: list[int], input_span: object) -> list[int]:
    if not isinstance(input_span, dict):
        raise ValueError("appended TEST input span is missing")
    start = input_span.get("start")
    count = input_span.get("count")
    expected_count = len(test_history) - len(read_history) - len(test_tokens)
    if (type(start) is not int or start != len(read_history)
            or type(count) is not int or count != expected_count or count < 0):
        raise ValueError("appended TEST input span does not bind only new input tokens")
    appended = test_history[start:start + count]
    if len(appended) != count:
        raise ValueError("appended TEST token IDs are truncated")
    return appended


def _reconcile_raw_process(record: dict, run_dir: Path, case_name: str) -> None:
    receipt_path = run_dir / "process.json"
    if not receipt_path.is_file():
        return
    process = json.loads(receipt_path.read_bytes())
    if process.get("gpu_launched_by_audit_runner") is True:
        record["runtime_executed"] = True
        record["gpu_launched"] = True
    record.setdefault("raw_processes", []).append({
        "case": case_name, "pid": process.get("pid"),
        "status": process.get("status"), "exit_code": process.get("exit_code"),
        "pid_absent": process.get("pid_absent"),
        "gpu_launched_by_audit_runner": process.get("gpu_launched_by_audit_runner"),
    })


def _validate_build(binary: Path, build_receipt_path: Path, driver_source: Path,
                    input_verification_path: Path) -> dict:
    build = json.loads(build_receipt_path.read_bytes())
    input_report = json.loads(input_verification_path.read_bytes())
    if (build.get("status") != "BUILT_NOT_EXECUTED" or build.get("runtime_executed") is not False
            or build.get("gpu_launched") is not False):
        raise ValueError("raw oracle build receipt is not a compile-only record")
    driver_hash = _sha256(driver_source)
    binary_hash = _sha256(binary)
    if (build.get("audit_driver_source_sha256") != driver_hash
            or build.get("binary_sha256") != binary_hash
            or Path(build.get("binary", "")).resolve() != binary.resolve()):
        raise ValueError("raw oracle build receipt does not bind the supplied source and ELF")
    if (input_report.get("checked_files") != 53 or input_report.get("mismatches") != []
            or build.get("verified_input_count") != 53
            or build.get("verified_input_mismatches") != []
            or build.get("input_verification_sha256") != _sha256(input_verification_path)):
        raise ValueError("raw oracle build input-verification proof is incomplete")
    rows = input_report.get("rows")
    if not isinstance(rows, list) or len(rows) != 53:
        raise ValueError("raw oracle input report must contain 53 pinned rows")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["expected_sha256"]:
            raise ValueError(f"raw oracle static build input changed: {path}")
    if not os.access(binary, os.X_OK):
        raise ValueError("raw oracle binary is not executable")
    return {"status": "PINNED_BUILD_VALID", "binary": str(binary),
            "binary_sha256": binary_hash, "driver_source_sha256": driver_hash,
            "build_receipt_sha256": _sha256(build_receipt_path),
            "input_verification_sha256": _sha256(input_verification_path),
            "verified_input_count": 53}


def _operation_id(campaign: dict, label: str) -> str:
    rows = [row for row in campaign.get("commands", []) if row.get("label") == label]
    if len(rows) != 1 or not isinstance(rows[0].get("argv_redacted"), list):
        raise ValueError(f"campaign must contain exactly one {label} command row")
    argv = rows[0]["argv_redacted"]
    if argv.count("--operation-id") != 1:
        raise ValueError(f"campaign {label} command does not bind one operation ID")
    index = argv.index("--operation-id") + 1
    if index >= len(argv) or not isinstance(argv[index], str) or not argv[index]:
        raise ValueError(f"campaign {label} operation ID is malformed")
    return argv[index]


def run_audit(*, runtime_root: Path, profile_path: Path, campaign_path: Path,
              store_path: Path, oracle_binary: Path, build_receipt: Path,
              driver_source: Path, input_verification: Path, output: Path,
              timeout_seconds: int = 1200, preflight_only: bool = False) -> dict:
    runtime_root = runtime_root.resolve(strict=True)
    profile_path = profile_path.resolve(strict=True)
    campaign_path = campaign_path.resolve(strict=True)
    store_path = store_path.resolve(strict=True)
    oracle_binary = oracle_binary.resolve(strict=True)
    profile = json.loads(profile_path.read_bytes())
    campaign = json.loads(campaign_path.read_bytes())
    sys.path[:0] = [str(runtime_root), str(runtime_root / "tools"),
                    str(runtime_root / "experimental/branching_context/src"),
                    str(runtime_root / "experimental/branching_context/scripts")]
    for key in list(os.environ):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            os.environ.pop(key, None)
    profile_environment = profile.get("execution", {}).get("environment", {})
    if not isinstance(profile_environment, dict) or any(
            type(key) is not str or type(value) is not str
            for key, value in profile_environment.items()):
        raise ValueError("profile environment must be a string mapping")
    os.environ.update(profile_environment)
    if campaign.get("status") != "CLI_SEED_STOP_FLOW_PASS_NUMERICAL_AUDIT_PENDING":
        raise ValueError("campaign is not the completed, pending-numerics workflow")
    if campaign.get("failures") not in ([], None):
        raise ValueError("campaign contains failures")
    if Path(campaign["profile"]).resolve() != profile_path:
        raise ValueError("campaign profile path differs from supplied profile")
    identity = campaign.get("profile_identity", {})
    if (identity.get("execution_profile_id") != profile.get("execution_profile_id")
            or identity.get("model_contract_id") != profile.get("model_contract_id")):
        raise ValueError("campaign and profile model/runtime identities differ")
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    _validate_execution(profile)
    model_verification = _validate_model(profile)
    build_proof = _validate_build(oracle_binary, build_receipt, driver_source,
                                  input_verification)
    profile_command = profile["execution"].get("oracle_command")
    if not isinstance(profile_command, list) or not profile_command:
        raise ValueError("profile lacks its ordinary raw-oracle command")
    protected_binaries = {str(Path(value[0]).resolve()) for value in (
        profile["execution"].get("sut_command", []), profile_command)
        if isinstance(value, list) and value}
    if str(oracle_binary) in protected_binaries:
        raise ValueError("fresh raw oracle must differ from serving and profile oracle binaries")
    for service in campaign.get("services", []):
        for pid in (service.get("pid"), service.get("worker_shutdown", {}).get("native_pid")):
            if type(pid) is int and Path("/proc", str(pid)).exists():
                raise ValueError(f"campaign process is still alive: {pid}")

    from context_time_travel.store import Store
    store = Store(store_path)
    root_id = campaign["root"]["snapshot"]
    lineage_root, root_tokens = store.lineage(root_id)
    root_tokens = _exact_ids(list(root_tokens), "canonical root history")
    if lineage_root != root_id or len(root_tokens) != campaign["root"].get("history_tokens"):
        raise ValueError("prepared root history differs from the completed campaign")
    model_contract_id = campaign["profile_identity"]["model_contract_id"]
    execution_profile_id = campaign["profile_identity"]["execution_profile_id"]
    if (profile.get("model_contract_id") != model_contract_id
            or profile.get("execution_profile_id") != execution_profile_id):
        raise ValueError("profile bindings differ from campaign identities")

    sampled_partial = _exact_ids(campaign["sampled_partial"]["tokens"], "sampled partial tokens")
    sampled_continue = _exact_ids(campaign["sampled_continue"]["tokens"], "sampled continuation tokens")
    restart_continue = _exact_ids(campaign["restart_continue"]["tokens"], "restarted continuation tokens")
    read_tokens = _exact_ids(campaign["read_stop"]["tokens"], "READ stop tokens")
    test_tokens = _exact_ids(campaign["append_test"]["tokens"], "TEST tokens")
    sampled_targets = [
        ("sampled_partial", campaign["sampled_partial"]["snapshot"], sampled_partial),
        ("sampled_continue", campaign["sampled_continue"]["receipt"]["snapshot"], sampled_continue),
        ("restart_continue", campaign["restart_continue"]["receipt"]["snapshot"], restart_continue),
    ]
    partial_snapshot = sampled_targets[0][1]
    partial_history = root_tokens + sampled_partial
    partial_controller = store.get_snapshot(partial_snapshot).get("controller")
    if not isinstance(partial_controller, dict) or partial_controller != campaign["sampled_partial"]["controller"]:
        raise ValueError("sampled partial controller differs from canonical snapshot")
    reseed_proof = _validate_reseed_transition(
        campaign, store, root_id, partial_snapshot, partial_history, partial_controller)
    sampled_controller_proof = {
        "reseed": reseed_proof,
        "continuations": _validate_generation_state(
            campaign, store, root_id, root_tokens,
            sampled_targets[1:], prefix_tokens=sampled_partial,
            initial_stream_key=reseed_proof["stream_key"], initial_draw=0),
    }
    read_snapshot = campaign["read_stop"]["snapshot"]
    test_snapshot = campaign["append_test"]["receipt"]["snapshot"]
    read_target = store.get_snapshot(read_snapshot)
    read_root, read_history = store.lineage(read_snapshot)
    test_root, test_history = store.lineage(test_snapshot)
    if (read_root != root_id or read_history != root_tokens + read_tokens
            or test_root != root_id or test_history[-len(test_tokens):] != test_tokens):
        raise ValueError("READ/TEST canonical forced-token history differs from campaign")
    input_span = campaign["append_test"].get("input_span")
    appended_ids = _append_input_ids(read_history, test_history, test_tokens, input_span)
    from context_time_travel.real import BranchingModel
    if input_span.get("token_ids_sha256") != BranchingModel._input_identity(appended_ids):
        raise ValueError("appended TEST token span digest differs from canonical token IDs")
    if test_history != root_tokens + read_tokens + appended_ids + test_tokens:
        raise ValueError("TEST canonical history differs from root, READ, append, and output tokens")
    if (read_target.get("controller") != campaign["read_stop"]["controller"]
            or store.get_snapshot(test_snapshot).get("controller") != campaign["append_test"]["controller"]):
        raise ValueError("READ/TEST campaign controllers differ from canonical snapshots")

    checkpoint_specs = [
        ("root", root_id, campaign["root"]["prepare_operation"]),
        ("sampled_partial", sampled_targets[0][1], campaign["sampled_partial"]["operation"]),
        ("sampled_continue", sampled_targets[1][1], _operation_id(campaign, "sampled-continue")),
        ("restart_continue", sampled_targets[2][1], _operation_id(campaign, "restart-continue")),
        ("read_stop", read_snapshot, campaign["read_stop"]["operation"]),
        ("append_test", test_snapshot, campaign["append_test"]["operation"]),
    ]
    saved_frontiers = {}
    snapshot_receipts = {}
    for label, snapshot_id, operation_id in checkpoint_specs:
        target, frontier, history = _receipt_snapshot(
            store, operation_id, snapshot_id, profile, model_contract_id, root_id)
        saved_frontiers[label] = frontier
        snapshot_receipts[label] = {"snapshot": snapshot_id, "history_tokens": len(history),
                                    "payload_sha256": store.checkpoint_receipt(operation_id)[
                                        "checkpoint_receipt"]["payload_sha256"],
                                    "controller_sha256": hashlib.sha256(json.dumps(
                                        target["controller"], sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()}

    output = output.resolve(strict=False)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = {
        "schema": "cli-seed-stop-interleaved-independent-numerics-v1",
        "status": "READY_NOT_RUN" if preflight_only else "RUNNING",
        "runtime_executed": False, "gpu_launched": False,
        "campaign_path": str(campaign_path), "campaign_sha256": _sha256(campaign_path),
        "profile_path": str(profile_path), "profile_sha256": _sha256(profile_path),
        "runtime_root": str(runtime_root), "execution_profile_id": execution_profile_id,
        "model_contract_id": model_contract_id, "model_verification": model_verification,
        "store_path": str(store_path), "oracle_build": build_proof,
        "oracle_checkpoint_imports": 0, "oracle_adapter_imports": 0,
        "repetitions_per_case": 2, "sampled_controller_crosscheck": sampled_controller_proof,
        "saved_checkpoint_frontiers": snapshot_receipts, "comparisons": [],
        "cases": [], "raw_processes": [], "failures": [],
        "scope": "Fresh raw Prefill/Verifier exact full-vocabulary logit frontier audit at canonical durable snapshots. Forced token histories are compared exactly. Grammar legality, sampled token-choice reimplementation, and controller/RNG arithmetic are not independently audited.",
    }
    cases = [
        ("sampled-1-16-10", root_tokens, sampled_partial + sampled_continue + restart_continue,
         (len(sampled_partial), len(sampled_continue), len(restart_continue)), None, None,
         [(0, "root"), (1, "sampled_partial"), (2, "sampled_continue"), (3, "restart_continue")]),
        ("read-stop-append-test", root_tokens, read_tokens + test_tokens,
         (len(read_tokens), len(test_tokens)), appended_ids, 1,
         [(0, "root"), (1, "read_stop"), (2, "append_test")]),
    ]
    result_path = output / "result.json"
    _write_json(result_path, record)
    if preflight_only:
        record.update(phase="preflight-complete", preflight_only=True,
                      planned_cases=[{"name": name, "prompt_tokens": len(prompt),
                                      "forced_outputs": len(forced),
                                      "segments": list(segments),
                                      "appended_tokens": len(appended or [])}
                                     for name, prompt, forced, segments, appended, _boundary, _boundaries
                                     in cases],
                      comparisons_count=0)
        _write_json(result_path, record)
        return record
    try:
        for case_name, prompt_ids, forced_ids, segments, append_ids, append_boundary, boundaries in cases:
            repeated = []
            for repetition in (1, 2):
                record["phase"] = f"{case_name}-fresh-{repetition}"
                _write_json(result_path, record)
                raw_dir = output / f"{case_name}-fresh-{repetition}"
                try:
                    raw = run_raw_case(
                    profile=profile, binary=oracle_binary,
                    run_dir=raw_dir,
                    case_name=f"{case_name}-fresh-{repetition}", prompt_ids=prompt_ids,
                    forced_ids=forced_ids, segments=segments, vocab_size=VOCAB_SIZE,
                    appended_ids=append_ids, append_after_segment=append_boundary,
                    reset_verifier_segments=True, timeout_seconds=timeout_seconds)
                finally:
                    _reconcile_raw_process(record, raw_dir, f"{case_name}-fresh-{repetition}")
                    _write_json(result_path, record)
                repeated.append(raw)
                record["cases"].append({"name": f"{case_name}-fresh-{repetition}",
                                        "pid": raw["pid"], "exit_code": raw["exit_code"],
                                        "pid_absent": raw["pid_absent"], "status": raw["status"],
                                        "work": raw["work"], "schedule": raw["schedule"],
                                        "frontier_sha256": raw["frontier_sha256"]})
                _write_json(result_path, record)
            for field in ("prompt_frontiers", "decode_frontiers", "segment_frontiers"):
                if len(repeated[0][field]) != len(repeated[1][field]):
                    raise ValueError(f"fresh {case_name} repetitions differ in {field} count")
                for index, (left, right) in enumerate(zip(repeated[0][field], repeated[1][field])):
                    record["comparisons"].append(compare_frontiers(left, right,
                        f"{case_name}-repeat-{field}-{index}"))
            if append_ids:
                record["comparisons"].append(compare_frontiers(
                    repeated[0]["append_prefill_frontier"], repeated[1]["append_prefill_frontier"],
                    f"{case_name}-repeat-appended-prefill"))
            record["comparisons"].append(compare_frontiers(
                repeated[0]["prompt_frontiers"][0], saved_frontiers["root"],
                f"{case_name}-prepared-root"))
            for segment_index, snapshot_label in boundaries[1:]:
                record["comparisons"].append(compare_frontiers(
                    repeated[0]["segment_frontiers"][segment_index - 1],
                    saved_frontiers[snapshot_label], f"{case_name}-{snapshot_label}"))
            record["phase"] = case_name + "-audited"
            _write_json(result_path, record)
        record.update(status="PASS_EXACT_FULL_FRONTIER_AUDIT", phase="complete",
                      comparisons_count=len(record["comparisons"]), failures=[])
    except BaseException as error:
        record["status"] = "FAIL"
        record["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        _write_json(result_path, record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--oracle-binary", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--driver-source", type=Path, required=True)
    parser.add_argument("--input-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_audit(runtime_root=args.runtime_root, profile_path=args.profile,
                       campaign_path=args.campaign, store_path=args.store,
                       oracle_binary=args.oracle_binary, build_receipt=args.build_receipt,
                       driver_source=args.driver_source, input_verification=args.input_verification,
                       output=args.output, timeout_seconds=args.timeout_seconds,
                       preflight_only=args.preflight_only)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "comparisons": result["comparisons_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
