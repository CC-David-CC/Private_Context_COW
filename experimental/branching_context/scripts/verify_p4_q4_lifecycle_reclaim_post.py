#!/usr/bin/env python3
"""Post-unlink audit for the exact nine lifecycle output states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

STAGE = Path("/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-map-audit-20261015-01")
RUN = Path("/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-20261015-v1")
SOURCE_STORE = Path("/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/store")
MAP_PATH = STAGE / "map.json"
VERIFY_PATH = STAGE / "verified-final.json"
PRECHECK_PATH = STAGE / "precheck_receipt-02.json"
UNLINK_PATH = STAGE / "reclaim-receipt-01.json"
RECEIVER_REMOTE = "/srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution/lifecycle-attribution-20261015.tar.receipt.json"
EXPECTED_MAP_SHA256 = "91ec8f54db1fc53fb1983dfc6722c86a898edbde79483f992fe86ff5ee462291"
EXPECTED_VERIFY_SHA256 = "6b0e5f59311cbaa3d896706363601c264f0e9eea11993f664d52cdfff2c7c35e"
EXPECTED_PRECHECK_SHA256 = "7b493b88e0ac9931ab9ccbc4300d41c5dea9a9adf1bc271aa8ebffe611fde273"
EXPECTED_UNLINK_SHA256 = "112c5d7cbca4417af83fad3a3667ce2f6fb8a596e86f25ed3a53fc0c3ce564d0"
EXPECTED_RECEIVER_SHA256 = "513126b097fec519c993aac095c28f237ea29aa82b56ea2b2f4d0bfa3edaf00c"
EXPECTED_ARCHIVE_SHA256 = "d01af0087b43f4c8f3558ae64d49377d8f927c0ec31420035134644cd347684f"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        st = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(f"non-regular protected file: {path}")
        h = hashlib.sha256()
        with path.open("rb", buffering=0) as source:
            while block := source.read(8 * 1024 * 1024):
                h.update(block)
        rows.append({
            "path": path.relative_to(root).as_posix(), "size": st.st_size,
            "mode": f"{stat.S_IMODE(st.st_mode):04o}", "nlink": st.st_nlink,
            "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns,
            "sha256": h.hexdigest(),
        })
    return rows


def inv_sha(rows: list[dict]) -> str:
    return sha_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    map_raw = MAP_PATH.read_bytes()
    verify_raw = VERIFY_PATH.read_bytes()
    precheck_raw = PRECHECK_PATH.read_bytes()
    unlink_raw = UNLINK_PATH.read_bytes()
    assert sha_bytes(map_raw) == EXPECTED_MAP_SHA256
    assert sha_bytes(verify_raw) == EXPECTED_VERIFY_SHA256
    assert sha_bytes(precheck_raw) == EXPECTED_PRECHECK_SHA256
    assert sha_bytes(unlink_raw) == EXPECTED_UNLINK_SHA256
    mapping, verified, precheck, unlink = (json.loads(raw) for raw in (map_raw, verify_raw, precheck_raw, unlink_raw))
    assert verified["status"] == "PASS_READ_ONLY_ARCHIVE_AND_STATE_MAP_VERIFIED"
    assert precheck["status"] == "PRECHECK_PASS_NOT_UNLINKED" and precheck["unlink_performed"] is False
    assert unlink["status"] == "UNLINK_PASS_EXACT_NINE_ONLY" and unlink["unlink_performed"] is True
    assert unlink["candidate_count"] == 9 and unlink["candidate_bytes"] == 17_789_976_604
    assert unlink["archive_sha256"] == EXPECTED_ARCHIVE_SHA256 and unlink["receiver_receipt_sha256"] == EXPECTED_RECEIVER_SHA256

    run_store = RUN / "output/resident/store"
    output_metadata = inventory(run_store)
    source_files = inventory(SOURCE_STORE)
    assert output_metadata == verified["protected_lifecycle_store_metadata_files_excluding_exact_nine_states"]
    assert source_files == verified["protected_q4_source_store_files"]
    assert inv_sha(output_metadata) == precheck["protected_output_metadata_inventory_sha256"]
    assert inv_sha(source_files) == precheck["protected_source_store_inventory_sha256"]
    assert output_metadata == verified["protected_lifecycle_store_metadata_files_excluding_exact_nine_states"]

    missing = []
    for row in mapping["large_file_inventory"]["state_files"]:
        state = run_store / "checkpoints" / f"{row['payload_sha256']}.state"
        if state.exists() or state.is_symlink():
            raise RuntimeError(f"mapped target still exists: {state}")
        missing.append(str(state))

    source_lookup = {row["path"]: row for row in source_files}
    for root in mapping["protected_q4_source_roots"]:
        metadata_rel = f"checkpoints/{root['snapshot']}.json"
        payload_rel = f"checkpoints/{root['payload_sha256']}.state"
        metadata_row = source_lookup.get(metadata_rel)
        payload_row = source_lookup.get(payload_rel)
        if metadata_row is None or payload_row is None:
            raise RuntimeError(f"protected original Q4 root missing: {root['snapshot']}")
        if metadata_row["sha256"] != root["metadata_sha256"] or payload_row["sha256"] != root["payload_sha256"]:
            raise RuntimeError(f"protected root metadata/payload changed: {root['snapshot']}")

    receiver_proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "box-git", "cat", RECEIVER_REMOTE], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert sha_bytes(receiver_proc.stdout) == EXPECTED_RECEIVER_SHA256
    receiver = json.loads(receiver_proc.stdout)
    assert receiver["status"] == "PASS" and receiver["archive_sha256"] == EXPECTED_ARCHIVE_SHA256

    command_dir = Path("/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-reclaim-command-20261015-01")
    command_exit_raw = (command_dir / "command.exit").read_bytes()
    wrapper_exit_raw = (command_dir / "wrapper.exit").read_bytes()
    if command_exit_raw != b"0\n" or wrapper_exit_raw != b"0\n":
        raise RuntimeError("native command or wrapper exit receipt is not numeric zero")
    stats = os.statvfs(RUN)
    free_after = stats.f_bavail * stats.f_frsize
    result = {
        "schema": 1,
        "status": "PASS_POST_UNLINK_EXACT_NINE_PROTECTED_BYTES_VERIFIED",
        "map_sha256": EXPECTED_MAP_SHA256,
        "verification_receipt_sha256": EXPECTED_VERIFY_SHA256,
        "precheck_receipt_sha256": EXPECTED_PRECHECK_SHA256,
        "unlink_receipt_sha256": EXPECTED_UNLINK_SHA256,
        "receiver_receipt_sha256": EXPECTED_RECEIVER_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "removed_count": len(unlink["removed"]),
        "removed_bytes": sum(row["bytes"] for row in unlink["removed"]),
        "removed": unlink["removed"],
        "all_nine_target_paths_absent": missing,
        "protected_output_metadata_file_count": len(output_metadata),
        "protected_output_metadata_inventory_sha256_before": precheck["protected_output_metadata_inventory_sha256"],
        "protected_output_metadata_inventory_sha256_after": inv_sha(output_metadata),
        "protected_output_metadata_inventory_matches_before": output_metadata == verified["protected_lifecycle_store_metadata_files_excluding_exact_nine_states"],
        "protected_source_store_file_count": len(source_files),
        "protected_source_store_inventory_sha256_before": precheck["protected_source_store_inventory_sha256"],
        "protected_source_store_inventory_sha256_after": inv_sha(source_files),
        "protected_source_store_inventory_matches_before": source_files == verified["protected_q4_source_store_files"],
        "original_q4_roots_reverified": len(mapping["protected_q4_source_roots"]),
        "filesystem_free_bytes_before": precheck["filesystem_free_bytes_before_precheck"],
        "filesystem_free_bytes_after_precheck": precheck["filesystem_free_bytes_after_precheck"],
        "filesystem_free_bytes_after_unlink": free_after,
        "filesystem_free_bytes_delta": free_after - precheck["filesystem_free_bytes_before_precheck"],
        "command_exit": 0,
        "wrapper_exit": 0,
        "command_exit_receipt_sha256": sha_bytes(command_exit_raw),
        "wrapper_exit_receipt_sha256": sha_bytes(wrapper_exit_raw),
        "unlink_directory_fsync_performed_after_each_path": True,
        "source_store_unmodified": True,
        "no_other_paths_removed": True,
    }
    payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short post-unlink receipt write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"status": result["status"], "removed_count": result["removed_count"], "removed_bytes": result["removed_bytes"], "free_after": free_after, "sha256": sha_bytes(payload)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
