"""Read-only native Typical/DynaTemp selection audit on one fixed frontier.

This compares the native selector with the independent Python policy reference.
It does not certify a generated continuation or throughput.
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


DRAW_SENTINELS = (0, 1 << 52, (1 << 53) - 1)
_COUNTERS = ("sampling_calls", "sampling_v2_calls", "sampling_v3_calls",
             "sampling_v4_calls", "forward_tokens", "prefill_tokens")
_NATIVE_PROTOCOLS = {"v2": "full-vocab-min-p-u53-v2",
                     "v3": "full-vocab-dry-min-p-u53-v3",
                     "v4": "full-vocab-typical-dyna-min-p-u53-v4"}


def _expected_protocol(policy: dict[str, object]) -> str:
    if policy["temperature"] > 0 and (
            policy.get("typical_p", 1.0) < 1.0 or policy.get("dynatemp_range", 0.0) > 0.0):
        return "v4"
    if policy.get("dry_multiplier", 0.0) > 0 and policy.get("dry_penalty_last_n", 0) > 0:
        return "v3"
    return "v2"


def policy_cases(*, include_dry: bool) -> list[dict]:
    """Return the exact bounded policy/mask matrix used by this audit."""
    base = dict(temperature=0.8, min_p=0.06, top_k=0, top_p=1.0,
                repetition_penalty=1.0, dry_multiplier=0.0, dry_base=1.75,
                dry_allowed_length=2, dry_penalty_last_n=256,
                typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0)

    def case(name, updates, mask_kind="full"):
        policy = dict(base)
        policy.update(updates)
        draws = [0] if policy["temperature"] == 0.0 else list(DRAW_SENTINELS)
        return dict(name=name, policy=policy, draws=draws, mask_kind=mask_kind,
                    expected_protocol=_expected_protocol(policy))

    cases = [
        case("greedy", dict(temperature=0.0, min_p=0.0)),
        case("min-p-baseline", dict()),
        case("typical-alone", dict(min_p=0.0, typical_p=0.95)),
        case("dynatemp-alone", dict(temperature=0.55, min_p=0.06,
                                     dynatemp_range=0.35, dynatemp_exponent=1.0)),
        case("typical-dynatemp", dict(typical_p=0.95, dynatemp_range=0.35,
                                      dynatemp_exponent=1.0)),
    ]
    if include_dry:
        cases.append(case("dry-combined", dict(typical_p=0.95, dynatemp_range=0.35,
            dynatemp_exponent=1.0, dry_multiplier=0.8, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256)))
    cases.extend((
        case("threshold-corner", dict(min_p=1.0)),
        case("single-legal", dict(min_p=0.0, typical_p=0.95), mask_kind="single"),
    ))
    return cases


def validate_receipt(receipt: dict) -> bool:
    """Fail closed on incomplete rows, unexpected work, or protocol double-counting."""
    if not isinstance(receipt, dict) or receipt.get("status") != "PASS":
        raise ValueError("selection audit is not PASS")
    include_dry = receipt.get("include_dry")
    if type(include_dry) is not bool:
        raise ValueError("selection audit lacks an exact include_dry flag")
    cases = policy_cases(include_dry=include_dry)
    if receipt.get("cases") != cases:
        raise ValueError("selection audit policy inventory differs from the pinned case matrix")
    for before_key, after_key in (("frontier_before_sha256", "frontier_after_sha256"),
                                  ("parent_before_sha256", "parent_after_sha256")):
        before, after = receipt.get(before_key), receipt.get(after_key)
        if (not isinstance(before, str) or len(before) != 64 or
                any(ch not in "0123456789abcdef" for ch in before) or after != before):
            raise ValueError(f"read-only selection changed or omitted {before_key.removesuffix('_before_sha256')}")
    for name in ("forward_tokens", "prefill_tokens"):
        before, after = receipt.get(name + "_before"), receipt.get(name + "_after")
        if type(before) is not int or type(after) is not int or after != before:
            raise ValueError(f"read-only selection changed {name}")
    if receipt.get("selection_forward_tokens") != 0:
        raise ValueError("selection audit reports forward work")

    expected = [(case, draw) for case in cases for draw in case["draws"]]
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("selection audit has a missing or extra row")
    for index, ((case, draw), row) in enumerate(zip(expected, rows)):
        if not isinstance(row, dict):
            raise ValueError(f"selection row {index} is malformed")
        if (row.get("name") != case["name"] or row.get("mask_kind") != case["mask_kind"] or
                row.get("policy") != case["policy"] or row.get("draw53") != draw or
                row.get("protocol") != case["expected_protocol"]):
            raise ValueError(f"selection row {index} does not match its requested policy and draw")
        token, reference = row.get("token"), row.get("reference_token")
        if type(token) is not int or type(reference) is not int or token != reference:
            raise ValueError(f"selection row {index} differs from the independent Python reference")
        native = row.get("native_response")
        if (not isinstance(native, dict) or type(native.get("token")) is not int or
                native["token"] != token or native.get("protocol") != _NATIVE_PROTOCOLS[case["expected_protocol"]]):
            raise ValueError(f"selection row {index} lacks its native response receipt")
        for key in _COUNTERS:
            if (type(row.get("counters_before", {}).get(key)) is not int or
                    type(row.get("counters_after", {}).get(key)) is not int):
                raise ValueError(f"selection row {index} lacks counter {key}")
        before, after = row["counters_before"], row["counters_after"]
        if after["sampling_calls"] - before["sampling_calls"] != 1:
            raise ValueError(f"selection row {index} did not account for exactly one native selector call")
        for version in ("v2", "v3", "v4"):
            delta = after[f"sampling_{version}_calls"] - before[f"sampling_{version}_calls"]
            if delta != (1 if case["expected_protocol"] == version else 0):
                raise ValueError(f"selection row {index} has wrong or double-counted {version} calls")
        if (after["forward_tokens"] != before["forward_tokens"] or
                after["prefill_tokens"] != before["prefill_tokens"]):
            raise ValueError(f"selection row {index} did model work")
        if case["expected_protocol"] == "v4":
            effective = row.get("effective_temperature")
            kept = row.get("typical_kept_count")
            if (type(effective) not in (int, float) or not math.isfinite(effective) or effective < 0 or
                    type(kept) is not int or not 1 <= kept <= row.get("legal_count", 0)):
                raise ValueError(f"selection row {index} lacks v4 support/temperature receipt")
            if (row.get("effective_temperature") != native.get("effective_temperature") or
                    row.get("typical_kept_count") != native.get("typical_kept_count")):
                raise ValueError(f"selection row {index} disagrees with the native v4 receipt")
    before = receipt.get("selection_counters_before")
    after = receipt.get("selection_counters_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("selection audit lacks its aggregate native counter snapshots")
    route_counts = {version: sum(case["expected_protocol"] == version for case, _ in expected)
                    for version in ("v2", "v3", "v4")}
    for key, delta in (("sampling_calls", len(expected)),
                       ("sampling_v2_calls", route_counts["v2"]),
                       ("sampling_v3_calls", route_counts["v3"]),
                       ("sampling_v4_calls", route_counts["v4"])):
        if type(before.get(key)) is not int or type(after.get(key)) is not int or after[key] - before[key] != delta:
            raise ValueError(f"aggregate native counter {key} differs from the policy matrix")
    for key in ("forward_tokens", "prefill_tokens"):
        if type(before.get(key)) is not int or type(after.get(key)) is not int or after[key] != before[key]:
            raise ValueError(f"aggregate selector calls changed native {key}")
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mask(vocab: int, token_ids=None) -> bytes:
    width = (vocab + 7) // 8
    bits = bytearray(width)
    if token_ids is None:
        for token in range(vocab):
            bits[token >> 3] |= 1 << (token & 7)
    else:
        for token in token_ids:
            bits[token >> 3] |= 1 << (token & 7)
    return bytes(bits)


def _native_sample_record(trace_path: Path, old_size: int, expected_protocol: str) -> dict:
    if not trace_path.is_file():
        raise AssertionError("native request trace is missing")
    with trace_path.open("rb") as stream:
        stream.seek(old_size)
        events = [json.loads(line) for line in stream if line.strip()]
    matches = [event for event in events
               if isinstance(event, dict) and isinstance(event.get("command"), str)
               and event["command"].startswith("sample_mask_")]
    if len(matches) != 1:
        raise AssertionError(f"expected one new native selection event; saw {len(matches)}")
    event = matches[0]
    response = event.get("response")
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise AssertionError("native selection event has no successful response")
    if response.get("protocol") != expected_protocol:
        raise AssertionError("native command used an unexpected selector protocol")
    return dict(command=event["command"], response=response)


def _case_dry_context(policy, history, token_bytes, dry_arguments, dry_enabled):
    if not dry_enabled(policy):
        return {}
    return dry_arguments(policy, history, token_bytes)


def selection_arguments(policy: dict, dry_context: dict) -> dict:
    """Build one nonduplicated argument set for both reference and native calls.

    ``dry_arguments`` repeats the four canonical DRY numeric settings alongside
    its transient history/breaker inputs. Accept that producer shape only when
    those values exactly agree with the selected policy, then include each
    numeric value once in the shared invocation mapping.
    """
    canonical = {key: value for key, value in policy.items()
                 if key != "repetition_penalty"}
    if not isinstance(dry_context, dict):
        raise ValueError("DRY invocation context must be a mapping")
    transient = {"dry_history", "dry_breakers"}
    allowed = transient | {"dry_multiplier", "dry_base", "dry_allowed_length",
                           "dry_penalty_last_n"}
    if set(dry_context) - allowed:
        raise ValueError("DRY invocation context contains unexpected fields")
    for key, value in dry_context.items():
        if key in transient:
            if key not in ("dry_history", "dry_breakers"):
                raise ValueError("unexpected DRY transient field")
            canonical[key] = value
        elif key in canonical:
            if canonical[key] != value:
                raise ValueError(f"DRY context disagrees with canonical {key}")
        else:
            raise ValueError(f"DRY context repeats unknown numeric setting {key}")
    return canonical


def configure_native_trace(profile, trace_path, apply_profile_environment):
    """Apply profile environment first, then restore the probe-owned trace path."""
    apply_profile_environment(profile)
    os.environ["CTT_NATIVE_TRACE"] = str(trace_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--branch", choices=("A", "B"), default="A")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runtime = args.runtime_root.resolve(strict=True)
    sys.path[:0] = [str(runtime), str(runtime / "tools"),
                    str(runtime / "experimental/branching_context/src"),
                    str(runtime / "experimental/branching_context/scripts")]
    from context_time_travel import min_p
    from context_time_travel import dry as dry_module
    from context_time_travel import real as real_module
    from context_time_travel import sampled_controller as sampled_controller_module
    from context_time_travel import strata_backend as strata_backend_module
    from context_time_travel.sampled_controller import dry_arguments
    from context_time_travel.strata_backend import StrataBackend
    from context_time_travel.real import BranchingModel
    import grammar_long_reference
    import run_grammar_first_probe
    from grammar_long_reference import Heartbeat, _apply_profile_environment
    from run_grammar_first_probe import finish_backend
    from tools.strata_tokenizer import Tokenizer
    import tools.strata_tokenizer as tokenizer_module

    os.umask(0o077)
    output = args.run_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    trace_path = output / "native-trace.jsonl"
    profile_bytes = args.profile.read_bytes()
    profile = json.loads(profile_bytes)
    configure_native_trace(profile, trace_path, _apply_profile_environment)
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    root_ids = fixture.get("root_token_ids")
    branch = fixture.get("branches", {}).get(args.branch)
    suffix_ids = branch.get("suffix_token_ids") if isinstance(branch, dict) else None
    if (not isinstance(root_ids, list) or not root_ids or not isinstance(suffix_ids, list) or not suffix_ids or
            any(type(token) is not int or token < 0 for token in (*root_ids, *suffix_ids))):
        raise ValueError("fixture must provide nonempty root_token_ids and branches.<A|B>.suffix_token_ids")
    if profile.get("execution", {}).get("sampling_protocol") != min_p.SAMPLING_PROTOCOL_V4:
        raise ValueError("profile does not pin the v4 native selection capability")

    result = dict(status="RUNNING", scope="Read-only individual policy selection; not continuation or performance qualification",
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(), fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
        execution_profile_id=profile.get("execution_profile_id"),
        model_contract_id=profile.get("model_contract_id"),
        profile_path=str(args.profile.resolve()), fixture_path=str(args.fixture.resolve()),
        runtime_source_sha256={module.__name__: _sha256(Path(module.__file__).resolve())
            for module in (min_p, dry_module, real_module, sampled_controller_module,
                           strata_backend_module, grammar_long_reference,
                           run_grammar_first_probe, tokenizer_module)},
        script_sha256=_sha256(Path(__file__).resolve()),
        runtime_root=str(runtime), branch=args.branch, include_dry=True,
        root_token_count=len(root_ids), suffix_token_count=len(suffix_ids), cases=policy_cases(include_dry=True),
        rows=[], failures=[], python_pid=os.getpid(), heartbeat_path=str(output / "heartbeat.json"),
        native_trace_path=str(trace_path))
    backend = None
    heartbeat = None
    _publish(output / "result.json", result)
    try:
        heartbeat = Heartbeat(output / "heartbeat.json").start()
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        heartbeat.set("native-start")
        started = time.monotonic()
        backend = StrataBackend(profile)
        result.update(native_pid=backend.process.pid, native_startup_seconds=time.monotonic() - started,
                      native_executable=str(backend.process.args[0]),
                      native_executable_sha256=_sha256(Path(backend.process.args[0])))
        model = BranchingModel(output / "store", backend, profile)
        heartbeat.set("consume-full-root-and-suffix", backend.process.pid)
        root = model.create(tuple(root_ids))
        child = model.feed(root, tuple(suffix_ids))
        history = tuple(model.history(child))
        if history != tuple(root_ids) + tuple(suffix_ids):
            raise AssertionError("materialized history differs from fixture root plus selected suffix")

        parent_blob = model.cache[root].blob
        result.update(root_snapshot=str(root), child_snapshot=str(child),
            root_token_ids_sha256=hashlib.sha256(struct.pack("<" + "I" * len(root_ids), *root_ids)).hexdigest(),
            suffix_token_ids_sha256=hashlib.sha256(struct.pack("<" + "I" * len(suffix_ids), *suffix_ids)).hexdigest())
        parent_before = backend.audit(parent_blob)
        with model._materialize(child) as (lease, _):
            before_selection = backend.work_counters()
            values = backend.logits(lease)
            frontier = struct.pack("<" + "f" * len(values), *values)
            if len(values) != backend.vocab_size:
                raise AssertionError("native frontier width differs from the model vocabulary")
            (output / "frontier.f32").write_bytes(frontier)
            result["frontier"] = dict(path="frontier.f32", bytes=len(frontier),
                                      sha256=hashlib.sha256(frontier).hexdigest(),
                                      logits_count=len(values), consumed=lease.consumed)
            base_mask = _mask(backend.vocab_size)
            single_token = next(token for token in range(1, backend.vocab_size)
                                if base_mask[token >> 3] & (1 << (token & 7)))
            full_mask_sha = hashlib.sha256(base_mask).hexdigest()
            single_mask = _mask(backend.vocab_size, (single_token,))
            mask_map = {"full": base_mask, "single": single_mask}
            result["masks"] = dict(full_sha256=full_mask_sha,
                                   single_sha256=hashlib.sha256(single_mask).hexdigest(),
                                   single_token=single_token)

            for case in result["cases"]:
                policy = case["policy"]
                dry = _case_dry_context(policy, history, tokenizer.token_bytes,
                                        dry_arguments, min_p.dry_enabled)
                invocation = selection_arguments(policy, dry)
                mask = mask_map[case["mask_kind"]]
                for draw in case["draws"]:
                    heartbeat.set("select-" + case["name"], backend.process.pid)
                    counters_before = backend.work_counters()
                    trace_offset = trace_path.stat().st_size if trace_path.exists() else 0
                    expected = min_p.choose(values, mask, draw53=draw, **invocation)
                    native_kwargs = {key: value for key, value in invocation.items()
                                     if key not in ("temperature", "min_p")}
                    actual = backend.sample_legal(lease, mask,
                        invocation["temperature"], invocation["min_p"], draw,
                        **native_kwargs)
                    counters_after = backend.work_counters()
                    route = case["expected_protocol"]
                    native_protocol = {"v2": min_p.SAMPLING_PROTOCOL_V2,
                                       "v3": min_p.DRY_PROTOCOL,
                                       "v4": min_p.SAMPLING_PROTOCOL_V4}[route]
                    native_record = _native_sample_record(trace_path, trace_offset, native_protocol)
                    response = native_record["response"]
                    dry_history = dry.get("dry_history", ())
                    dry_breakers = dry.get("dry_breakers", ())
                    row = dict(name=case["name"], mask_kind=case["mask_kind"], policy=policy,
                        draw53=draw, protocol=route, token=actual, reference_token=expected["token"],
                        native_response=response, native_command=native_record["command"],
                        counters_before=counters_before, counters_after=counters_after,
                        legal_count=expected["legal_count"], kept_count=expected["kept_count"],
                        total_weight=expected["total_weight"],
                        dry_penalized_candidates=response.get("dry_penalized_candidates", 0),
                        dry_history_count=len(dry_history),
                        dry_history_sha256=hashlib.sha256(struct.pack(
                            "<" + "I" * len(dry_history), *dry_history)).hexdigest(),
                        dry_breakers=list(dry_breakers),
                        effective_temperature=response.get("effective_temperature"),
                        typical_kept_count=response.get("typical_kept_count"))
                    result["rows"].append(row)
                    _publish(output / "result.json", result)
                    if actual != expected["token"]:
                        raise AssertionError(f"native selector differs from Python reference for {case['name']}")

            after_selection = backend.work_counters()
            frontier_after = struct.pack("<" + "f" * backend.vocab_size, *backend.logits(lease))
            result.update(frontier_before_sha256=hashlib.sha256(frontier).hexdigest(),
                frontier_after_sha256=hashlib.sha256(frontier_after).hexdigest(),
                selection_forward_tokens=after_selection["forward_tokens"] - before_selection["forward_tokens"],
                forward_tokens_before=before_selection["forward_tokens"],
                forward_tokens_after=after_selection["forward_tokens"],
                prefill_tokens_before=before_selection["prefill_tokens"],
                prefill_tokens_after=after_selection["prefill_tokens"],
                selection_counters_before=before_selection,
                selection_counters_after=after_selection)

        parent_after = backend.audit(parent_blob)
        result.update(parent_before_sha256=parent_before["sha256"],
                      parent_after_sha256=parent_after["sha256"],
                      parent_unchanged=parent_before["sha256"] == parent_after["sha256"])
        result["status"] = "PASS"
        validate_receipt(result)
    except BaseException as exc:
        result["status"] = "FAIL"
        result["failures"].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if backend is not None:
            finish_backend(backend, result)
        if trace_path.is_file():
            result["native_trace_sha256"] = _sha256(trace_path)
        if heartbeat is not None:
            heartbeat.close()
        _publish(output / "result.json", result)
    if result["status"] != "PASS":
        raise AssertionError("selection audit cleanup or receipt validation failed")


if __name__ == "__main__":
    main()
