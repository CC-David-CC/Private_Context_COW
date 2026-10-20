#!/usr/bin/env python3
"""Compare saved browser checkpoints with post-shutdown fresh native references."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()

    api = args.api.resolve()
    sys.path.insert(0, str(api / "scripts"))
    sys.path.insert(0, str(api / "src"))
    sys.path.insert(0, str(api))
    from native_request_probe import reference
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    from context_time_travel.checkpoint_cache import validate_manifest
    from context_time_travel.store import IntegrityError

    run = args.run.resolve()
    server_exit = run / "server.exit"
    native_exits = run / "native-exits.jsonl"
    if not server_exit.is_file() or server_exit.read_text(encoding="ascii").strip() != "0":
        raise SystemExit("service has not exited successfully; fresh reference is forbidden")
    if not native_exits.is_file():
        raise SystemExit("native exit receipt is missing; fresh reference is forbidden")
    exit_rows = [json.loads(line) for line in native_exits.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if not exit_rows:
        raise SystemExit("native exit receipt is empty; fresh reference is forbidden")
    for row in exit_rows:
        if (type(row.get("pid")) is not int
                or type(row.get("returncode")) is not int
                or row.get("returncode") != 0):
            raise SystemExit("native worker did not exit successfully")
        if Path(f"/proc/{row['pid']}").exists():
            raise SystemExit("native worker PID remains alive; fresh reference is forbidden")
    server_pid_path = run / "server.pid"
    if not server_pid_path.is_file():
        raise SystemExit("recorded HTTP server PID is missing; fresh reference is forbidden")
    try:
        server_pid = int(server_pid_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as error:
        raise SystemExit("recorded HTTP server PID is invalid; fresh reference is forbidden") from error
    if server_pid <= 1 or Path(f"/proc/{server_pid}").exists():
        raise SystemExit("HTTP server PID is invalid or remains alive; fresh reference is forbidden")

    gpu_lock = Path("/home/dflanag3/branching-g3/native-client-speed.gpu0.lock")
    if gpu_lock.is_symlink():
        raise SystemExit("GPU0 lock path is a symlink; fresh reference is forbidden")
    try:
        gpu_lock_fd = os.open(gpu_lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        fcntl.flock(gpu_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as error:
        raise SystemExit("GPU0 lock is held or unavailable; fresh reference is forbidden") from error

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    report = json.loads((run / "browser" / "report.json").read_text(encoding="utf-8"))
    reusable = report["reusable_context_report"]
    expected_execution = profile["execution_profile_id"]
    expected_model = profile["model_contract_id"]
    identity = reusable["server_receipt_identity"]
    if (identity.get("execution_profile_id") != expected_execution
            or identity.get("model_contract_id") != expected_model):
        raise SystemExit("context durable receipt profile/model identity differs")
    for letter in ("A", "B"):
        receipt = reusable["ask_" + letter.lower() + "_request"]["receipt"]
        if (receipt.get("execution_profile_id") != expected_execution
                or receipt.get("model_contract_id") != expected_model):
            raise SystemExit("durable receipt profile/model identity differs for " + letter)

    for model_path, item in profile["model_files"].items():
        info = Path(model_path).stat()
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                item["bytes"], item["mtime_ns"], item["ctime_ns"]):
            raise SystemExit("model file stat changed; stop without rehashing weights")
    for name in list(os.environ):
        if name.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            del os.environ[name]
    os.environ.update(profile["execution"].get("environment", {}))
    _validate_execution(profile)
    model_receipt = _validate_model(profile)
    if model_receipt["model_bytes_hashed_this_process"] != 0:
        raise SystemExit("unexpected model weight hashing during reference preflight")

    root_ids = fixture["root_token_ids"]
    store = run / "store"

    def stored_frontier(snapshot: str, expected_payload: str) -> bytes:
        checkpoint_dir = store / "checkpoints"
        envelope = json.loads((checkpoint_dir / (snapshot + ".json")).read_text(encoding="utf-8"))
        untrusted = envelope.get("manifest")
        if not isinstance(untrusted, dict):
            raise SystemExit("durable checkpoint manifest is missing: " + snapshot)
        try:
            manifest = validate_manifest(envelope, {
                "schema": untrusted.get("schema"),
                "snapshot": snapshot,
                "model_contract_id": expected_model,
                "execution_profile_id": expected_execution,
            })
        except (IntegrityError, KeyError, TypeError, ValueError) as error:
            raise SystemExit("durable checkpoint manifest validation failed: " + snapshot) from error
        if manifest.get("payload_sha256") != expected_payload:
            raise SystemExit("durable checkpoint payload receipt mismatch: " + snapshot)
        path = checkpoint_dir / (expected_payload + ".state")
        expected_size = manifest["state_bytes"] + 4 * manifest["logits_count"]
        if path.stat().st_size != expected_size:
            raise SystemExit("durable checkpoint payload length mismatch: " + snapshot)
        digest = hashlib.sha256()
        with path.open("rb") as payload_stream:
            for block in iter(lambda: payload_stream.read(1024 * 1024), b""):
                digest.update(block)
            payload_stream.seek(manifest["state_bytes"])
            tail = payload_stream.read(4 * manifest["logits_count"])
        if digest.hexdigest() != expected_payload:
            raise SystemExit("durable checkpoint payload hash mismatch: " + snapshot)
        if len(tail) != 4 * manifest["logits_count"]:
            raise SystemExit("saved frontier has unexpected length: " + snapshot)
        if not all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", tail)):
            raise SystemExit("saved frontier contains nonfinite values: " + snapshot)
        return tail

    root_frontier = stored_frontier(reusable["root_snapshot"], reusable["root_payload_sha256"])
    output_dir = run / "fresh-reference"
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = {}
    for letter in ("A", "B"):
        branch = fixture["branches"][letter]
        suffix = branch["suffix_token_ids"]
        fresh = reference(profile, output_dir, "fresh-" + letter,
                          root_ids + suffix, [len(root_ids), len(suffix)], 128)
        ask = reusable["ask_" + letter.lower() + "_request"]
        actual_tokens = ask["answer"]["tokens"]
        if actual_tokens != fresh["tokens"]:
            raise SystemExit("fresh generated token IDs differ for branch " + letter)
        if letter == "A" and root_frontier != fresh["prompt_frontiers"][0]:
            raise SystemExit("saved context-root frontier differs from fresh reference")
        final_receipt = ask["receipt"]
        final_frontier = stored_frontier(final_receipt["snapshot"],
                                         final_receipt["payload_sha256"])
        if final_frontier != fresh["generated_frontiers"][-1]:
            raise SystemExit("saved final frontier differs from fresh reference for " + letter)
        results[letter] = {
            "status": "PASS_EXACT_TOKEN_IDS_AND_FINAL_FRONTIER",
            "generated_tokens": len(actual_tokens),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(actual_tokens, separators=(",", ":")).encode()).hexdigest(),
            "fresh_work": fresh["work"],
            "fresh_seconds": fresh["seconds"],
            "saved_final_frontier_sha256": hashlib.sha256(final_frontier).hexdigest(),
        }

    receipt = {
        "status": "PASS_FRESH_REFERENCE_COMPARISONS",
        "execution_profile_id": expected_execution,
        "model_contract_id": expected_model,
        "model_hash_validation": model_receipt,
        "service_must_be_stopped_before_invocation": True,
        "root_frontier_compared_with_A_prompt_frontier_0": True,
        "branches": results,
        "browser_logits_unavailable": True,
    }
    target = run / "fresh-reference-comparison.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
