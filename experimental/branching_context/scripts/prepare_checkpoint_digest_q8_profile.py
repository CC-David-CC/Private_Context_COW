#!/usr/bin/env python3
"""Prepare an unqualified Q8 profile for the checkpoint-digest candidate.

This is a CPU/profile admission step only. It refuses model-file stat drift
before invoking the existing profile helper, because that helper would hash a
large model file if its recorded stat identity changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE_PROFILE = Path(
    "/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json"
)
API_ROOT = Path(
    "/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red"
)
PROFILE_HELPER = Path(
    "/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/g0stage/scripts/prepare_native_request_profile.py"
)
BINARY_DIR = Path(
    "/home/dflanag3/branching-g3/native-request-probe-20261014/binaries-digest-v1"
)
IMPORT_WITNESS = Path(
    "/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/source-import-witness.json"
)
ARTIFACT_DIR = Path(
    "/home/dflanag3/spatial-artifacts/checkpoint-digest-q8-attestation-20261015"
)
PROFILE_DIR = ARTIFACT_DIR / "derived-profile-candidate-v1"
EVIDENCE_PATH = ARTIFACT_DIR / "q8-profile-admission-candidate-v1.json"

EXPECTED_BASE_PROFILE_SHA256 = "11eeda86f34fa2b53b824f273a37ad743bbf6c0b88e668f3b9c03073b23a13b6"
EXPECTED_BASE_PROFILE_ID = "f39df09d619ee55032a199c0a089fa964ef1b27f48929bca11f6e70aff9a07c7"
EXPECTED_MODEL_CONTRACT_ID = "fc6d0d643e4b9ae020964f8c4e5006c0da7199b19cce5285c573023dbe82ca46"
EXPECTED_MODEL_WEIGHTS_SHA256 = "846b74d30a22552ad8cc2fabe5b17103049fe68ffdacf56daa457db8ecaa356f"
EXPECTED_IMPORT_WITNESS_SHA256 = "282dbf2af31ea60ba5bfa68d10022b19ae75f387b65ddf9f47063f145270f6a8"
EXPECTED_PROFILE_HELPER_SHA256 = "1550dcac8db5bea22445f022922720bffd1de77e9ce025b0f8c0c79db01e172e"
EXPECTED_BUILD_RECEIPT_SHA256 = "e1b151aa37ddc4a75b9f6e121ccd7d73412e9a545105f1ffbe8cfc6ba053898f"
EXPECTED_SUT_SHA256 = "7bb17b47d2dc8b6252530f377891380613fc2c43352c34ea6f0319f386bf4a2b"
EXPECTED_ORACLE_SHA256 = "a60314b8aff6c3954ec55aabce18c58196231b57f7cf99b340b68b747689100c"
EXPECTED_MODEL_RECEIPT_SHA256 = "99dfc43d8b1cc1117f136bf1e1c22d2d7611f511e8d3642ea3be3502c43c1469"
EXPECTED_MODULE_COUNT = 17
CAPACITY = 98_304
PREFILL_CHUNK = 8_192


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_base_profile(profile: dict, profile_sha: str) -> dict:
    require(profile_sha == EXPECTED_BASE_PROFILE_SHA256, "Q8 base-profile SHA mismatch")
    require(profile.get("execution_profile_id") == EXPECTED_BASE_PROFILE_ID, "unexpected Q8 base execution profile")
    require(profile.get("model_contract_id") == EXPECTED_MODEL_CONTRACT_ID, "unexpected Q8 model contract")
    require(profile.get("model_weights_sha256") == EXPECTED_MODEL_WEIGHTS_SHA256, "unexpected Q8 weight identity")
    execution = profile["execution"]
    require(execution.get("GPU") == [0], "Q8 profile does not select exactly GPU 0")
    require(execution.get("capacity") == CAPACITY, "Q8 profile capacity changed")
    require(execution.get("prefill_chunk") == PREFILL_CHUNK, "Q8 prefill chunk changed")
    require(execution.get("kv_dtype") == "FP16", "Q8 KV type changed")
    require(execution.get("generated_operator") == "normal-verifier-decode", "unexpected generated-token operator")
    require(execution.get("native_request_protocol") == "consumed-greedy-ack-v1", "unexpected native request protocol")
    require(execution.get("resume_contract") == "identical-native-operators-v1", "unexpected resume operator contract")
    require(execution.get("operator") == "batched-input/native-greedy-request/per-write-sha256", "unexpected operator identity")
    require(execution.get("text_template_kwargs") == {"enable_thinking": False}, "Q8 template settings changed")

    model_contract = profile["model_contract"]
    require(model_contract.get("weights_sha256") == EXPECTED_MODEL_WEIGHTS_SHA256, "model contract weight pin changed")
    require(model_contract.get("tokenizer_identity") == EXPECTED_MODEL_WEIGHTS_SHA256,
            "Q8 tokenizer identity differs from the admitted model contract")
    require(model_contract.get("input_kind") == "token_ids", "profile is not token-ID input")

    gpu_cli = {}
    for label in ("sut_command", "oracle_command"):
        command = execution[label]
        require(Path(command[0]).resolve().parent == BINARY_DIR.resolve(), f"{label} binary is outside the pinned bundle")
        require("--pack" in command and command[command.index("--pack") + 1] == "/home/dflanag3/branching-g3/q8-pack",
                f"{label} does not use the Q8 model pack")
        require("--native" in command and "/Q8_0/" in command[command.index("--native") + 1],
                f"{label} does not use the Q8 GGUF")
        for option, expected in (("--max-context", str(CAPACITY)), ("--prefill", str(PREFILL_CHUNK)),
                                 ("--kv", "fp16")):
            require(option in command and command[command.index(option) + 1] == expected,
                    f"{label} config mismatch for {option}")
        if "--gpu" in command:
            require(command[command.index("--gpu") + 1] == "0", f"{label} CLI selects a different GPU")
            gpu_cli[label] = "explicit --gpu 0"
        else:
            require(label == "sut_command" and execution.get("GPU") == [0],
                    f"{label} omits a GPU selector without a single-GPU profile binding")
            gpu_cli[label] = "omitted; execution.GPU=[0] and hardware is pinned to one GPU"
    require(gpu_cli.get("oracle_command") == "explicit --gpu 0", "oracle must explicitly select GPU 0")
    return execution, gpu_cli


def validate_model_receipts(profile: dict) -> dict:
    receipt_path = Path(profile["model_hash_receipt"])
    receipt_sha = sha_file(receipt_path)
    require(receipt_sha == EXPECTED_MODEL_RECEIPT_SHA256, "executed Q8 model hash receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    model_files = profile["model_files"]
    require(len(model_files) == 11, "unexpected Q8 model/pack file count")
    require(sum("Q8_0" in name for name in model_files) == 6, "all six Q8 GGUF shard receipts are required")
    require(any("/q8-pack/" in name for name in model_files), "Q8 pack receipts are missing")
    require(any("expert-profile.bin" in name for name in model_files), "Q8 expert profile receipt is missing")
    reused = 0
    for name, item in model_files.items():
        st = Path(name).stat()
        stat_tuple = (st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        pinned_tuple = (item["bytes"], item["mtime_ns"], item["ctime_ns"])
        require(stat_tuple == pinned_tuple, "model file stat drift; refusing before any weight rehash: " + name)
        prior = receipt.get("model_files", {}).get(name, {})
        require(prior.get("bytes") == item["bytes"] and prior.get("sha256") == item["sha256"],
                "model hash receipt mismatch: " + name)
        require(item.get("hash_reused_from_executed_run") is True, "model receipt is not marked reused: " + name)
        reused += 1
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "model_file_count": len(model_files),
        "model_file_hashes_reused": reused,
        "model_weight_bytes_hashed_by_this_script": 0,
        "model_weight_bytes_hashed_by_profile_helper": 0,
        "basis": "recorded model hash receipt plus exact size/mtime/ctime stat match; no weight-byte hashing",
        "q8_gguf_shard_count": 6,
        "separate_tokenizer_sha256_field": False,
        "tokenizer_identity": profile["model_contract"]["tokenizer_identity"],
        "gguf_reader_path": profile["gguf_reader"],
        "gguf_reader_sha256": profile["native_build"]["sha256"][profile["gguf_reader"]],
    }


def validate_witness(witness: dict, witness_sha: str) -> tuple[list[dict], list[dict]]:
    require(witness_sha == EXPECTED_IMPORT_WITNESS_SHA256, "candidate import witness SHA mismatch")
    require(witness.get("status") == "PASS", "candidate source import witness is not PASS")
    require(Path(witness.get("root", "")).resolve() == API_ROOT.resolve(), "candidate source root differs from witness")
    modules = witness.get("modules", [])
    scripts = witness.get("script_imports", [])
    require(witness.get("module_count") == EXPECTED_MODULE_COUNT and len(modules) == EXPECTED_MODULE_COUNT,
            "candidate runtime module count is not 17")
    require(witness.get("profile_module_pins_match") is True, "module import witness does not match its pinned profile")

    module_rows = []
    names = set()
    for row in modules:
        name = row["name"]
        filename = "__init__.py" if name == "__init__" else name + ".py"
        expected_path = (API_ROOT / "src/context_time_travel" / filename).resolve()
        path = Path(row["path"]).resolve()
        require(path == expected_path, "candidate module is outside the pinned source tree: " + name)
        actual_sha = sha_file(path)
        require(actual_sha == row["sha256"], "candidate module differs from import witness: " + name)
        require(name not in names, "duplicate candidate module in witness: " + name)
        names.add(name)
        module_rows.append({"name": name, "path": str(path), "sha256": actual_sha})
    require(names == {"__init__", "checkpoint_cache", "controller", "coordinator", "digest_attestation",
                      "evaluation", "evaluation_worker", "execution_receipt", "numerics", "oracle", "pools",
                      "presentation_binding", "presentation_request", "real", "slots", "store", "strata_backend"},
            "candidate source module set differs from the admitted 17-module closure")

    script_rows = []
    for row in scripts:
        path = Path(row["path"]).resolve()
        require(path.is_relative_to(API_ROOT.resolve()), "campaign helper import is outside the candidate tree")
        actual_sha = sha_file(path)
        require(actual_sha == row["sha256"], "campaign helper differs from import witness: " + str(path))
        script_rows.append({"path": str(path), "sha256": actual_sha})
    require(witness.get("native_request_probe_api") == str(API_ROOT.resolve()),
            "native request probe import resolved to a stale API root")
    return module_rows, script_rows


def validate_build_and_binaries(profile: dict) -> dict:
    build_path = BINARY_DIR / "build.json"
    build_sha = sha_file(build_path)
    require(build_sha == EXPECTED_BUILD_RECEIPT_SHA256, "Q8 binary build receipt SHA mismatch")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    require(build.get("status") == "BUILT_NOT_EXECUTED", "unexpected build receipt status")
    require(build.get("native_request_protocol") == "consumed-greedy-ack-v1", "build protocol mismatch")
    require(build.get("resume_contract") == "identical-native-operators-v1", "build resume contract mismatch")

    observed = {}
    for name, expected in (("strata-batched-branch", EXPECTED_SUT_SHA256),
                           ("strata-batched-fresh", EXPECTED_ORACLE_SHA256)):
        path = BINARY_DIR / name
        actual = sha_file(path)
        require(actual == expected, "Q8 SUT/oracle binary SHA mismatch: " + name)
        require(build.get("binaries", {}).get(name) == actual, "binary differs from its build receipt: " + name)
        observed[str(path)] = actual
        require(profile["native_build"]["sha256"].get(str(path)) == actual,
                "binary differs from the Q8 base profile pin: " + name)
    return {
        "build_receipt_path": str(build_path),
        "build_receipt_sha256": build_sha,
        "build_receipt_status": build["status"],
        "native_request_protocol": build["native_request_protocol"],
        "resume_contract": build["resume_contract"],
        "binaries": observed,
        "binary_execution_evidence_note": "build.json alone is not execution evidence; prior Q8 browser profile f39df09d execution and receipts are separate historical evidence.",
    }


def validate_actual_import_origins(module_rows: list[dict]) -> list[dict]:
    sys.path.insert(0, str(API_ROOT / "src"))
    actual = []
    for row in module_rows:
        import_name = "context_time_travel" if row["name"] == "__init__" else "context_time_travel." + row["name"]
        module = importlib.import_module(import_name)
        path = Path(module.__file__).resolve()
        require(path == Path(row["path"]).resolve(), "actual module import resolved outside candidate root: " + import_name)
        digest = sha_file(path)
        require(digest == row["sha256"], "actual imported module bytes changed: " + import_name)
        actual.append({"module": import_name, "path": str(path), "sha256": digest})
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-profile", action="store_true", help="derive one fresh, unqualified Q8 profile")
    args = parser.parse_args()
    require(args.prepare_profile, "explicit --prepare-profile is required")
    require(not PROFILE_DIR.exists() and not EVIDENCE_PATH.exists(), "refusing to overwrite prior profile/evidence")
    require(ARTIFACT_DIR.is_dir(), "reviewed artifact directory must already exist")
    require(sha_file(PROFILE_HELPER) == EXPECTED_PROFILE_HELPER_SHA256, "profile-helper source SHA mismatch")

    base, base_sha = load_json(BASE_PROFILE)
    execution, gpu_cli = validate_base_profile(base, base_sha)
    model_receipt = validate_model_receipts(base)
    witness, witness_sha = load_json(IMPORT_WITNESS)
    module_rows, script_rows = validate_witness(witness, witness_sha)
    build = validate_build_and_binaries(base)

    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            del env[key]
    env.update(execution.get("environment", {}))
    command = [
        sys.executable, str(PROFILE_HELPER),
        "--base", str(BASE_PROFILE),
        "--binaries", str(BINARY_DIR),
        "--api", str(API_ROOT),
        "--output", str(PROFILE_DIR),
        "--capacity", str(CAPACITY),
        "--chunk", str(PREFILL_CHUNK),
    ]
    started = time.time()
    result = subprocess.run(command, cwd=PROFILE_HELPER.parent, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0, "existing profile helper failed: " + result.stderr[-2000:])
    try:
        helper_result = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("existing profile helper returned malformed output") from error
    require(helper_result.get("status") == "PREPARED_NOT_EXECUTED", "profile helper did not report an unexecuted profile")
    profile_path = PROFILE_DIR / "profile.json"
    require(profile_path.is_file(), "profile helper returned without writing profile.json")
    candidate, candidate_sha = load_json(profile_path)
    require(candidate["execution"]["capacity"] == CAPACITY, "derived capacity mismatch")
    require(candidate["execution"]["prefill_chunk"] == PREFILL_CHUNK, "derived prefill chunk mismatch")
    require(candidate["execution"] == execution, "derived operator/config differs from the pinned Q8 base")
    require(candidate["model_contract_id"] == EXPECTED_MODEL_CONTRACT_ID, "model contract changed during derivation")
    require(candidate["model_weights_sha256"] == EXPECTED_MODEL_WEIGHTS_SHA256, "model weights identity changed")
    require(candidate["model_files"] == base["model_files"], "model/pack receipts changed during derivation")
    require(candidate["native_build"].get("probe_build_sha256") == EXPECTED_BUILD_RECEIPT_SHA256,
            "derived profile does not bind the reviewed binary build receipt")
    require(helper_result.get("execution_profile_id") == candidate["execution_profile_id"],
            "profile helper stdout ID differs from the written profile")
    for row in module_rows:
        require(candidate["native_build"]["sha256"].get(row["path"]) == row["sha256"],
                "derived profile omitted a candidate module pin: " + row["name"])
    imported = validate_actual_import_origins(module_rows)
    source_manifest = sha_json(module_rows)
    evidence = {
        "schema": 1,
        "status": "PREPARED_NOT_GPU_QUALIFIED",
        "scope": "CPU-only Q8 candidate profile derivation; no native command execution, inference, build, server, or GPU workload",
        "created_unix_seconds": started,
        "base_profile": {
            "path": str(BASE_PROFILE),
            "sha256": base_sha,
            "execution_profile_id": base["execution_profile_id"],
            "model_contract_id": base["model_contract_id"],
            "model_weights_sha256": base["model_weights_sha256"],
        },
        "candidate_profile": {
            "path": str(profile_path),
            "sha256": candidate_sha,
            "execution_profile_id": candidate["execution_profile_id"],
            "model_contract_id": candidate["model_contract_id"],
            "model_weights_sha256": candidate["model_weights_sha256"],
            "profile_builder_status": "PREPARED_NOT_EXECUTED",
        },
        "source": {
            "api_root": str(API_ROOT),
            "import_witness_path": str(IMPORT_WITNESS),
            "import_witness_sha256": witness_sha,
            "module_count": len(module_rows),
            "module_manifest_sha256": source_manifest,
            "modules": module_rows,
            "actual_import_origins": imported,
            "campaign_helpers": script_rows,
            "profile_helper_path": str(PROFILE_HELPER),
            "profile_helper_sha256": EXPECTED_PROFILE_HELPER_SHA256,
        },
        "native_build": build,
        "model_and_tokenizer": model_receipt,
        "operator_config": {
            "GPU": execution["GPU"],
            "capacity": execution["capacity"],
            "prefill_chunk": execution["prefill_chunk"],
            "kv_dtype": execution["kv_dtype"],
            "operator": execution["operator"],
            "generated_operator": execution["generated_operator"],
            "resume_contract": execution["resume_contract"],
            "native_request_protocol": execution["native_request_protocol"],
            "execution_json_sha256": sha_json(execution),
            "gpu_cli_binding": gpu_cli,
            "configuration_identical_to_q8_base": True,
        },
        "profile_builder": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha_file(Path(__file__).resolve()),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            "did_not_execute_sut_oracle": True,
            "did_not_hash_model_weight_bytes": True,
        },
        "qualification_boundary": [
            "This profile has not been used for a candidate GPU/native test.",
            "The pinned Q8 SUT/oracle have prior execution evidence under the older Q8 profile; that does not qualify these candidate module hashes.",
            "The binary build receipt is BUILT_NOT_EXECUTED; binary execution status comes only from separate historical Q8 run evidence.",
            "No timing, performance, or candidate hash-optimization result is claimed.",
        ],
    }
    payload = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(EVIDENCE_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short evidence write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"status": evidence["status"], "profile": str(profile_path),
                      "profile_sha256": candidate_sha,
                      "execution_profile_id": candidate["execution_profile_id"],
                      "evidence": str(EVIDENCE_PATH),
                      "evidence_sha256": hashlib.sha256(payload).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
