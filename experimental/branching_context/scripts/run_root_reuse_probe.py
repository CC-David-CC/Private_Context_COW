"""Bounded duplicate-root reuse probe; not a general pool campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback

import native_request_probe as native_probe_module
import pool_campaign as pool_campaign_module
from native_request_probe import _validate_execution, _validate_model, _record_native_shutdown
from pool_campaign import get_state
from context_time_travel import controller
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig, PoolTransitionTrace
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend
from context_time_travel import coordinator as coordinator_module
from context_time_travel import real as real_module
from context_time_travel import store as store_module
from context_time_travel import pools as pools_module
from context_time_travel import strata_backend as backend_module
from context_time_travel import execution_receipt as execution_receipt_module
from tools.strata_tokenizer import Tokenizer

FRONTIER_WIDTH = 4 * 248320


class VectorMismatch(AssertionError):
    def __init__(self, message, maximum):
        super().__init__(message)
        self.maximum = maximum


def publish(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(profile, outdir, name, ids, request_sizes, generated):
    """Fresh matching native oracle with explicit process PID/return code."""
    outdir.mkdir(parents=True, exist_ok=True)
    tokens_path = outdir / f"{name}.tokens"
    logits_path = outdir / f"{name}.f32"
    tokens_path.write_text(",".join(map(str, ids)), encoding="ascii", newline="")
    command = list(profile["execution"]["oracle_command"])
    if "--max-new" not in command:
        raise ValueError("Pinned oracle command is missing --max-new")
    command[command.index("--max-new") + 1] = str(generated)
    command += ["--tokens-file", str(tokens_path)]
    env = dict(os.environ)
    env.update(profile["execution"].get("environment", {}))
    env["STRATA_DUMP_FIRST_LOGITS"] = str(logits_path)
    env["STRATA_PREFILL_REQUESTS"] = ",".join(map(str, request_sizes))
    started = time.monotonic()
    with (outdir / f"{name}.stdout").open("wb") as stdout, (outdir / f"{name}.stderr").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        pid = child.pid
        process_receipt = {"name": name, "pid": pid, "command": command, "status": "RUNNING"}
        publish(outdir / f"{name}.process.json", process_receipt)
        try:
            code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            _stop_owned_group(pid)
            code = child.wait(timeout=20)
            process_receipt.update(status="TIMEOUT", exit_code=code)
            publish(outdir / f"{name}.process.json", process_receipt)
            raise TimeoutError(f"Independent native reference {name} timed out; pid={pid}, exit={code}")
        process_receipt.update(status="EXITED", exit_code=code)
        publish(outdir / f"{name}.process.json", process_receipt)
    if code != 0:
        raise RuntimeError(f"Independent native reference {name} exited {code}, pid {pid}")
    work = json.loads((outdir / f"{name}.stdout").read_text(encoding="utf-8"))
    schedule = json.loads(Path(str(logits_path) + ".schedule.json").read_text(encoding="utf-8"))
    output_tokens = work["generated"]
    if (schedule.get("requests") != request_sizes
            or work.get("prefill_tokens") != len(ids)
            or work.get("normal_forward_tokens") != len(output_tokens)
            or work.get("consumed") != len(ids) + len(output_tokens)
            or not 0 < len(output_tokens) <= generated):
        raise AssertionError(f"Reference {name} work/schedule differs from the requested operators")
    raw = Path(str(logits_path) + ".decode.f32").read_bytes()
    prompt = Path(str(logits_path) + ".requests.f32").read_bytes()
    if len(raw) != FRONTIER_WIDTH * len(output_tokens) or len(prompt) != FRONTIER_WIDTH * len(request_sizes):
        raise AssertionError(f"Reference {name} frontier dump is incomplete")
    return {"name": name, "pid": pid, "exit_code": code, "command": command, "work": work,
            "schedule": schedule, "tokens": output_tokens,
            "generated_frontiers": [raw[i:i + FRONTIER_WIDTH] for i in range(0, len(raw), FRONTIER_WIDTH)],
            "prompt_frontiers": [prompt[i:i + FRONTIER_WIDTH] for i in range(0, len(prompt), FRONTIER_WIDTH)],
            "seconds": time.monotonic() - started}


def compare_vectors(actual: bytes, expected: bytes, name: str, output: Path):
    if len(actual) != FRONTIER_WIDTH or len(expected) != FRONTIER_WIDTH:
        raise ValueError(f"{name}: incomplete full-vector evidence")
    left = struct.unpack("<248320f", actual)
    right = struct.unpack("<248320f", expected)
    if not all(math.isfinite(x) for x in left + right):
        raise ValueError(f"{name}: nonfinite full-vector value")
    maximum = max(abs(a - b) for a, b in zip(left, right))
    if actual != expected:
        failure_dir = output / "vector-mismatches"
        failure_dir.mkdir(exist_ok=True)
        (failure_dir / f"{name}.actual.f32").write_bytes(actual)
        (failure_dir / f"{name}.expected.f32").write_bytes(expected)
        raise VectorMismatch(f"{name}: vector differs; max_abs_error={maximum:.9g}", maximum)
    return maximum


def validate_root_create_observation(answer, expected_reused, expected_forward, before_stats, after_stats,
                                     require_disk_read=False):
    """Check coordinator proof separately from later checkpoint-activation stats."""
    reused = answer.get("exact_materialization_reused")
    deltas = answer.get("counter_deltas")
    if reused is not expected_reused:
        raise AssertionError("exact materialization reuse evidence differs")
    if not isinstance(deltas, dict) or deltas.get("forward_tokens") != expected_forward:
        raise AssertionError("root forward counter differs")
    stats_delta = {key: value - before_stats[key] for key, value in after_stats.items()
                   if type(value) is int and type(before_stats.get(key)) is int}
    if stats_delta.get("forward_tokens") != expected_forward:
        raise AssertionError("post-capture native forward counter differs from expected root work")
    loaded = stats_delta.get("loaded_file_bytes", 0)
    if require_disk_read and loaded <= 0:
        raise AssertionError("fresh-process root reuse did not read saved checkpoint bytes")
    return {"exact_materialization_reused": reused,
            "coordinator_forward_tokens": deltas["forward_tokens"],
            "coordinator_counter_delta": deltas,
            "post_capture_stats_delta": stats_delta,
            "loaded_file_bytes_after_capture": loaded}


def expected_controller(operation, history, tokens, tokenizer, eos_tokens, vocab_size):
    state = controller.start(dict(temperature=0, eos_tokens=eos_tokens), operation,
                             tuple(history), vocab_size)
    return controller.consume_many(state,
        ((token, tokenizer.token_bytes(token), token in eos_tokens) for token in tokens))


def clear_native_environment():
    for key in list(os.environ):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            del os.environ[key]


def _stop_owned_group(pid):
    """Terminate only a process group created by this probe with a new session."""
    if os.name == "nt":
        raise RuntimeError("Owned process-group cleanup is implemented for the Linux runner")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_request(coordinator, request, output, operation, expected_snapshot=None):
    started = time.monotonic()
    receipt = request.durable.result(timeout=1800)
    durable_seconds = time.monotonic() - started
    answer = request.snapshot.result(timeout=1800)
    total_seconds = time.monotonic() - started
    if expected_snapshot is not None and receipt.get("snapshot") != expected_snapshot:
        raise AssertionError(f"{operation}: receipt snapshot differs from the expected root")
    state = get_state(coordinator, receipt["snapshot"])
    payload = Path(state["payload_file"])
    row = {"operation": operation, "receipt": receipt, "answer": answer, "state": state,
           "payload_sha256": sha256(payload), "durable_wait_seconds": durable_seconds,
           "completion_wait_seconds": total_seconds,
           "capture_seconds": answer.get("capture_seconds"),
           "ram_reservation_bytes": answer.get("ram_reservation_bytes"),
           "counter_deltas": answer.get("counter_deltas")}
    publish(output, row)
    return row


def run_worker(args, profile, prefix, suffix_a, suffix_b):
    stage = args.output / args.worker
    stage.mkdir(parents=True, exist_ok=False)
    record = {"schema": "duplicate-root-reuse-worker-v1", "status": "RUNNING",
              "worker": args.worker, "parent_pid": os.getppid(), "python_pid": os.getpid(),
              "requests": [], "failures": []}
    publish(stage / "phase.json", record)
    process_index = 1 if args.worker == "process1" else 2
    store_path = args.output / "store"
    pool = PoolConfig(gpus=(args.gpu,), ram_bytes=args.ram_gib * 1024**3,
                      disk_bytes=args.disk_gib * 1024**3,
                      min_available_bytes=args.min_available_gib * 1024**3)
    trace = PoolTransitionTrace("root-reuse-process-" + str(process_index), max_events=4096)
    coordinator = None
    workers = {}
    result = {}
    started = time.monotonic()
    try:
        def factory(gpu):
            return BranchingModel(store_path, StrataBackend(profile, gpu_id=gpu), profile)
        coordinator = ContextCoordinator(store_path, pool, factory, transition_trace=trace)
        coordinator.ready()
        workers = {str(gpu): worker["model"].backend.process
                   for gpu, worker in coordinator._workers.items()}
        model = next(worker["model"] for worker in coordinator._workers.values())
        record.update(native_pids={gpu: proc.pid for gpu, proc in workers.items()},
                      gpu_processes=subprocess.check_output(
                          ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                          text=True).strip(), startup_seconds=time.monotonic() - started,
                      pool_before=coordinator.metrics())
        publish(stage / "phase.json", record)

        def create(branch, operation, expected_reuse, expected_forward):
            began = time.monotonic()
            before_stats = dict(model.backend.stats)
            request = coordinator.create(branch, prefix, operation)
            row = _wait_request(coordinator, request, stage / (operation + ".json"), operation)
            after_stats = _wait_worker_idle(coordinator, model)
            answer = row["answer"]
            observed = validate_root_create_observation(answer, expected_reuse, expected_forward,
                before_stats, after_stats, require_disk_read=(process_index == 2))
            deltas = answer["counter_deltas"]
            row.update(request_wall_seconds=time.monotonic() - began,
                       **observed,
                       capture_seconds=answer.get("capture_seconds"),
                       ram_reservation_bytes=answer.get("ram_reservation_bytes"))
            record["requests"].append(row)
            publish(stage / (operation + ".json"), row)
            publish(stage / "phase.json", record)
            return row

        if process_index == 1:
            root_a = create("base-A-p1", "root-A-process-1", False, len(prefix))
            root_b = create("base-B-p1", "root-B-process-1", True, 0)
            if root_a["receipt"]["snapshot"] != root_b["receipt"]["snapshot"]:
                raise AssertionError("Distinct root operations did not reuse the same canonical snapshot")
            if root_a["payload_sha256"] != root_b["payload_sha256"]:
                raise AssertionError("Resident reused root payload differs")
            result.update(root_snapshot=root_a["receipt"]["snapshot"],
                          root_payload_file=root_a["state"]["payload_file"],
                          root_payload_sha256=root_a["payload_sha256"],
                          root_frontier_file=root_a["state"]["frontier_file"],
                          root_frontier_sha256=root_a["state"]["frontier_sha256"],
                          root_a=root_a, root_b=root_b)
            coordinator.fork("branch-A-p1", root_a["receipt"]["snapshot"])
            result["answer_a"] = answer(coordinator, model, "branch-A-p1", suffix_a,
                                         "answer-A-process-1", args.generated, stage, record)
            coordinator.fork("branch-B-p1", root_a["receipt"]["snapshot"])
            result["answer_b"] = answer(coordinator, model, "branch-B-p1", suffix_b,
                                         "answer-B-process-1", args.generated, stage, record)
        else:
            root = create("base-A-p2", "root-A-process-2", True, 0)
            result.update(root_snapshot=root["receipt"]["snapshot"],
                          root_payload_file=root["state"]["payload_file"],
                          root_payload_sha256=root["payload_sha256"],
                          root_frontier_file=root["state"]["frontier_file"],
                          root_frontier_sha256=root["state"]["frontier_sha256"], root_a=root)
            coordinator.fork("branch-A-p2", root["receipt"]["snapshot"])
            result["answer_a"] = answer(coordinator, model, "branch-A-p2", suffix_a,
                                         "answer-A-process-2", args.generated, stage, record)
        record.update(status="PASS", result=result, pool=coordinator.metrics(),
                      pool_transition_trace=trace.snapshot())
        publish(stage / "phase.json", record)
    except BaseException as exc:
        record.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        record["failures"].append(str(exc))
        publish(stage / "phase.json", record)
        raise
    finally:
        _record_native_shutdown(record, workers,
                                coordinator.close if coordinator is not None else lambda: None)
        record["pool_transition_trace"] = trace.snapshot()
        record["elapsed_seconds"] = time.monotonic() - started
        if record.get("native_exit_success") is not True and record.get("status") == "PASS":
            record["status"] = "FAIL"
            record["failures"].append("Native worker shutdown did not exit successfully")
        publish(stage / "phase.json", record)
        if record.get("status") != "PASS" or record.get("native_exit_success") is not True:
            raise RuntimeError("Worker phase failed or native exit status was not zero")
    return 0


def answer(coordinator, model, branch, suffix, operation, generated, stage, phase_record):
    started = time.monotonic()
    request = coordinator.submit(branch, suffix, generated, operation)
    row = _wait_request(coordinator, request, stage / (operation + ".json"), operation)
    answer_value = row["answer"]
    tokens = answer_value.get("tokens", [])
    deltas = answer_value.get("counter_deltas")
    if not isinstance(deltas, dict) or deltas.get("forward_tokens") != len(suffix) + len(tokens):
        raise AssertionError(f"{operation}: forward delta is not suffix plus actual output")
    row.update(request_wall_seconds=time.monotonic() - started,
               capture_seconds=answer_value.get("capture_seconds"),
               ram_reservation_bytes=answer_value.get("ram_reservation_bytes"),
               counter_deltas=deltas,
               actual_forward_tokens=deltas.get("forward_tokens"), suffix_forward_tokens=len(suffix),
               actual_output_tokens=len(tokens))
    phase_record["requests"].append(row)
    publish(stage / (operation + ".json"), row)
    publish(stage / "phase.json", phase_record)
    return row


def _wait_worker_idle(coordinator, model, timeout=60):
    """Copy Python-side backend counters once the worker is idle; issue no native request."""
    deadline = time.monotonic() + timeout
    with coordinator._condition:
        while any(worker["active"] for worker in coordinator._workers.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Coordinator worker did not become idle for stats snapshot")
            coordinator._condition.wait(remaining)
        return dict(model.backend.stats)


def compare_case(row, root_row, operation, expected_history, fresh, tokenizer, eos_tokens, output):
    state = row["state"]
    if state["history"] != list(expected_history):
        raise AssertionError(f"{operation}: canonical history differs")
    actual = Path(state["frontier_file"]).read_bytes()
    numerical = compare_vectors(actual, fresh["generated_frontiers"][-1], operation + "-final", output)
    root_frontier = Path(root_row["state"]["frontier_file"]).read_bytes()
    root_error = compare_vectors(root_frontier, fresh["prompt_frontiers"][0], operation + "-prefix", output)
    if row["answer"].get("tokens") != fresh["tokens"]:
        raise AssertionError(f"{operation}: output tokens differ from fresh matching oracle")
    expected = expected_controller(operation, expected_history[:-len(fresh["tokens"])],
                                   fresh["tokens"], tokenizer, eos_tokens, len(tokenizer.tokens))
    if state["controller"] != expected:
        raise AssertionError(f"{operation}: complete controller differs")
    return {"operation": operation, "history_tokens": len(expected_history),
            "actual_output_tokens": len(fresh["tokens"]), "tokens_exact": True,
            "prefix_frontier_exact": True, "prefix_max_abs_error": root_error,
            "final_frontier_exact": True, "final_max_abs_error": numerical,
            "controller_exact": True,
            "root_frontier_sha256": hashlib.sha256(root_frontier).hexdigest(),
            "final_frontier_sha256": hashlib.sha256(actual).hexdigest()}


def parent_main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "duplicate-root-reuse-probe-v3", "status": "RUNNING",
              "scope": "duplicate-root reuse and fresh-process disk reuse; no HTTP",
              "sizes": {"prefix": args.prefix, "suffix": args.suffix, "generated_limit": args.generated},
              "processes": [], "references": [], "comparisons": [], "failures": [], "skips": [], "xfails": [],
              "suffix_frontier_coverage": "not captured: no canonical checkpoint is published between suffix prefill and first generated token"}
    publish(args.output / "result.json", record)
    started = time.monotonic()
    stop = threading.Event()
    stage_name = ["initializing"]

    def heartbeat():
        while not stop.wait(30):
            path = args.output / "heartbeat.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"event": "heartbeat", "time": time.time(),
                    "stage": stage_name[0], "elapsed_seconds": time.monotonic() - started}) + "\n")
                stream.flush(); os.fsync(stream.fileno())
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        if profile.get("execution", {}).get("GPU") != [args.gpu]:
            raise ValueError("Pinned profile is bound to a different GPU selection")
        oracle = profile["execution"].get("oracle_command", [])
        if "--gpu" not in oracle or oracle[oracle.index("--gpu") + 1] != str(args.gpu):
            raise ValueError("Pinned oracle command is not bound to the selected GPU")
        shutil.copyfile(args.profile, args.output / "profile.json")
        clear_native_environment(); os.environ.update(profile["execution"].get("environment", {}))
        _validate_execution(profile); model_identity = _validate_model(profile)
        prefix = tuple(100 + i % 17 for i in range(args.prefix))
        suffix_a = tuple(117 + i % 13 for i in range(args.suffix))
        suffix_b = tuple(150 + i % 11 for i in range(args.suffix))
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        pool = PoolConfig(gpus=(args.gpu,), ram_bytes=args.ram_gib * 1024**3,
                          disk_bytes=args.disk_gib * 1024**3,
                          min_available_bytes=args.min_available_gib * 1024**3)
        record.update(execution_profile_id=profile["execution_profile_id"],
            model_contract_id=profile["model_contract_id"], model_weights_sha256=profile["model_weights_sha256"],
            native_build_sha256=profile["native_build"]["sha256"], model_identity=model_identity,
            gpu=list(pool.gpus), pool_budgets={"ram_bytes":pool.ram_bytes,"disk_bytes":pool.disk_bytes,
                "min_available_bytes":pool.min_available_bytes}, profile_source=str(args.profile),
            profile_sha256=sha256(args.output/"profile.json"), probe_sha256=sha256(Path(__file__)),
            runtime_source_sha256={name:sha256(Path(module.__file__)) for name,module in (
                ("native_request_probe",native_probe_module),("pool_campaign",pool_campaign_module),
                ("context_time_travel.coordinator",coordinator_module),("context_time_travel.real",real_module),
                ("context_time_travel.store",store_module),("context_time_travel.pools",pools_module),
                ("context_time_travel.strata_backend",backend_module),
                ("context_time_travel.execution_receipt",execution_receipt_module))},
            prefix_token_ids=list(prefix), suffix_a_token_ids=list(suffix_a), suffix_b_token_ids=list(suffix_b),
            python_parent_pid=os.getpid(), python_executable=sys.executable)
        publish(args.output/"result.json",record)
        phases=[]
        for name in ("process1","process2"):
            stage_name[0]=name
            command=[sys.executable,str(Path(__file__).resolve()),"--worker",name,
                "--profile",str(args.profile),"--output",str(args.output),"--gpu",str(args.gpu),
                "--ram-gib",str(args.ram_gib),"--disk-gib",str(args.disk_gib),
                "--min-available-gib",str(args.min_available_gib),"--prefix",str(args.prefix),
                "--suffix",str(args.suffix),"--generated",str(args.generated)]
            started_phase=time.monotonic()
            with (args.output/f"{name}.stdout").open("wb") as stdout, (args.output/f"{name}.stderr").open("wb") as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=os.environ.copy(),start_new_session=True)
                pid=child.pid
                process_receipt={"name":name,"pid":pid,"command":command,"status":"RUNNING"}
                publish(args.output/f"{name}.process.json",process_receipt)
                try: code=child.wait(timeout=2400)
                except subprocess.TimeoutExpired:
                    _stop_owned_group(pid)
                    code=child.wait(timeout=20)
                    process_receipt.update(status="TIMEOUT",exit_code=code)
                    publish(args.output/f"{name}.process.json",process_receipt)
                    raise TimeoutError(f"{name} Python worker exceeded 2400 seconds; pid={pid}, exit={code}")
                process_receipt.update(status="EXITED",exit_code=code)
                publish(args.output/f"{name}.process.json",process_receipt)
            phase_path=args.output/name/"phase.json"
            phase=json.loads(phase_path.read_bytes()) if phase_path.exists() else {"status":"MISSING"}
            phase.update(python_pid=pid, python_exit_code=code,
                         python_wall_seconds=time.monotonic()-started_phase,
                         stdout_sha256=sha256(args.output/f"{name}.stdout"),
                         stderr_sha256=sha256(args.output/f"{name}.stderr"))
            publish(args.output/name/"phase.json",phase); phases.append(phase)
            record["processes"].append({k:v for k,v in phase.items() if k not in ("requests","result")})
            publish(args.output/"result.json",record)
            if code != 0 or phase.get("status") != "PASS" or phase.get("native_exit_success") is not True:
                raise RuntimeError(f"{name} failed: python_exit={code}, native_exit_success={phase.get('native_exit_success')}")
        if phases[0]["result"]["root_snapshot"] != phases[1]["result"]["root_snapshot"]:
            raise AssertionError("Fresh-process disk root did not resolve to the same snapshot")
        if phases[0]["result"]["root_payload_sha256"] != phases[1]["result"]["root_payload_sha256"]:
            raise AssertionError("Fresh-process root payload bytes changed")
        refs={}
        for key,suffix in (("A",suffix_a),("B",suffix_b)):
            stage_name[0]="reference-"+key
            ref=reference(profile,args.output/"references","reference-"+key,
                          prefix+suffix,[len(prefix),len(suffix)],args.generated)
            refs[key]=ref
            saved={k:v for k,v in ref.items() if k not in ("generated_frontiers","prompt_frontiers")}
            saved["prompt_frontier_sha256"]=[hashlib.sha256(v).hexdigest() for v in ref["prompt_frontiers"]]
            saved["generated_frontier_sha256"]=[hashlib.sha256(v).hexdigest() for v in ref["generated_frontiers"]]
            record["references"].append(saved); publish(args.output/"result.json",record)
        stage_name[0]="frontier-controller-comparisons"
        cases=((phases[0]["result"]["answer_a"],phases[0]["result"]["root_a"],suffix_a,refs["A"],
                "answer-A-process-1"),(phases[0]["result"]["answer_b"],phases[0]["result"]["root_a"],suffix_b,refs["B"],
                "answer-B-process-1"),(phases[1]["result"]["answer_a"],phases[1]["result"]["root_a"],suffix_a,refs["A"],
                "answer-A-process-2"))
        for row,root_row,suffix,ref,operation in cases:
            comparison=compare_case(row,root_row,operation,prefix+suffix+tuple(ref["tokens"]),ref,
                                    tokenizer,profile["execution"]["eos_tokens"],args.output)
            record["comparisons"].append(comparison); publish(args.output/"result.json",record)
        saved_root_file = Path(phases[0]["result"]["root_payload_file"])
        final_root_hash = sha256(saved_root_file)
        if final_root_hash != phases[0]["result"]["root_payload_sha256"]:
            raise AssertionError("Parent root payload bytes changed during answer generation")
        record.update(status="PASS", parent_bytes_unchanged=True,
                      root_payload_sha256=phases[0]["result"]["root_payload_sha256"],
                      final_parent_payload_sha256=final_root_hash,
                      exact_comparisons=len(record["comparisons"]),
                      actual_output_counts={key:len(value["tokens"]) for key,value in refs.items()},
                      numerical_max_error=max((item["final_max_abs_error"] for item in record["comparisons"]),default=0.0),
                      executed_checks=["duplicate root has distinct operations and same canonical snapshot",
                        "resident exact-root reuse proves zero SUT forward work",
                        "fresh Python/native process disk exact-root reuse proves zero SUT forward work and positive loaded bytes",
                        "A/B answers match independent exact [prefix,suffix] native operators",
                        "saved root/final frontiers and complete controller match", "root payload bytes remain unchanged"],
                      elapsed_seconds=time.monotonic()-started)
        publish(args.output/"result.json",record)
        return 0
    except BaseException as exc:
        record.update(status="FAIL",error=str(exc),traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started)
        if isinstance(exc, VectorMismatch):
            record["numerical_max_error"] = exc.maximum
            record["numerical_mismatch"] = str(exc)
        record["failures"].append(str(exc)); publish(args.output/"result.json",record)
        raise
    finally:
        stop.set(); thread.join(timeout=5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",choices=("process1","process2"))
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--ram-gib",type=int,default=96)
    parser.add_argument("--disk-gib",type=int,default=128)
    parser.add_argument("--min-available-gib",type=int,default=16)
    parser.add_argument("--prefix",type=int,default=128)
    parser.add_argument("--suffix",type=int,default=4)
    parser.add_argument("--generated",type=int,default=8)
    args=parser.parse_args()
    if min(args.prefix,args.suffix,args.generated)<=0: parser.error("prefix, suffix, and generated must be positive")
    try:
        if args.worker:
            profile=json.loads(args.profile.read_text(encoding="utf-8"))
            clear_native_environment(); os.environ.update(profile["execution"].get("environment",{}))
            prefix=tuple(100+i%17 for i in range(args.prefix))
            suffix_a=tuple(117+i%13 for i in range(args.suffix))
            suffix_b=tuple(150+i%11 for i in range(args.suffix))
            return run_worker(args,profile,prefix,suffix_a,suffix_b)
        return parent_main(args)
    except BaseException as exc:
        print(f"root reuse probe failed: {exc}",file=sys.stderr,flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
