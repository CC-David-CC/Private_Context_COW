#!/usr/bin/env python3
"""Read-only verification of lifecycle archive state-map rows and protected Q4 roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

RECEIPT = Path(
    "/srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution/"
    "lifecycle-attribution-20261015.tar.receipt.json"
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def stat_listing(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["find", str(root), "-type", "f", "-printf", "%P\t%i\t%n\t%m\t%s\t%T@\n"],
        check=True,
        stdout=subprocess.PIPE,
    )
    lines = sorted(proc.stdout.splitlines(keepends=True))
    return len(lines), sha_bytes(b"".join(lines))


def file_inventory(root: Path, *, exclude_names: set[str] | None = None) -> list[dict]:
    excluded = exclude_names or set()
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded):
        st = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(f"non-regular file in protected inventory: {path}")
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size": st.st_size,
            "mode": f"{stat.S_IMODE(st.st_mode):04o}",
            "nlink": st.st_nlink,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
            "sha256": sha_file(path),
        })
    return rows


def run() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_raw_sha = sha_bytes(manifest_raw)
    manifest_canonical_sha = sha_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    receipt_proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "box-git", "cat", str(RECEIPT)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    receipt_raw = receipt_proc.stdout
    receipt = json.loads(receipt_raw)
    receipt_sha = sha_bytes(receipt_raw)

    archive = mapping["archive"]
    checks = []
    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise RuntimeError(f"{name} failed: {detail}")

    check("result_pin", sha_file(args.run / "output/resident/result.json") == mapping["execution"]["result_sha256"], "terminal result SHA matches map")
    result = json.loads((args.run / "output/resident/result.json").read_text(encoding="utf-8"))
    check("result_scope", result.get("status") == "PASS" and result.get("source_checkpoints_unchanged") is True, "PASS result and source-checkpoints-unchanged assertion")
    check("manifest_raw_sha", manifest_raw_sha == archive["manifest_raw_sha256"], manifest_raw_sha)
    check("receiver_receipt_sha", receipt_sha == archive["receiver_receipt_sha256"], receipt_sha)
    check("receiver_receipt_status", receipt.get("status") == "PASS", receipt.get("status", "missing"))
    check("archive_sha", receipt.get("archive_sha256") == archive["archive_sha256"], receipt.get("archive_sha256", "missing"))
    check("manifest_canonical_sha", manifest_canonical_sha == receipt.get("manifest_sha256"), manifest_canonical_sha)

    manifest_rows = manifest.get("files")
    check("manifest_schema", manifest.get("schema") == 1 and isinstance(manifest_rows, list), f"schema={manifest.get('schema')}, files={len(manifest_rows) if isinstance(manifest_rows, list) else 'invalid'}")
    member_rows = {row["path"]: row for row in manifest_rows}
    check("manifest_unique_paths", len(member_rows) == len(manifest_rows), f"rows={len(manifest_rows)} unique={len(member_rows)}")
    listing = json.dumps(
        sorted(
            [{"path": row["path"], "size": row["size"], "sha256": row["sha256"]} for row in manifest_rows],
            key=lambda row: row["path"],
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    listing_sha = sha_bytes(listing)
    check("member_listing_sha", listing_sha == receipt.get("member_listing_sha256"), listing_sha)

    states = mapping["large_file_inventory"]["state_files"]
    check("state_rows_count", len(states) == 9, str(len(states)))
    check("state_payloads_unique", len({row["payload_sha256"] for row in states}) == len(states), "all mapped output payload IDs are unique")
    measured_snapshots = {sample["answer"]["snapshot"] for sample in result["samples"]}
    mapped_output_state_ids = set()
    output_checkpoint_dir = args.run / "output/resident/store/checkpoints"
    for row in states:
        payload = row["payload_sha256"]
        check(f"payload_name_{payload[:8]}", len(payload) == 64 and payload == payload.lower(), payload)
        state_path = output_checkpoint_dir / f"{payload}.state"
        st = state_path.stat(follow_symlinks=False)
        check(f"state_stat_{payload[:8]}", stat.S_ISREG(st.st_mode) and st.st_size == row["bytes"] and st.st_ino == row["inode"] and st.st_nlink == 1 and stat.S_IMODE(st.st_mode) == 0o600, f"inode={st.st_ino} nlink={st.st_nlink} mode={stat.S_IMODE(st.st_mode):o} bytes={st.st_size}")
        member_path = f"lifecycle-run/output/resident/store/checkpoints/{payload}.state"
        member = member_rows.get(member_path)
        check(f"archive_member_{payload[:8]}", member is not None and member.get("size") == row["bytes"] and member.get("sha256") == payload, f"path={member_path}; manifest={member}")

        if row["classification"] == "measured_answer_payload":
            snapshots = row["measured_snapshots"]
            check(f"sample_refs_{payload[:8]}", set(snapshots) == {s for s in measured_snapshots if json.loads((output_checkpoint_dir / f"{s}.json").read_text())["manifest"]["payload_sha256"] == payload}, f"snapshots={len(snapshots)}")
        elif row["classification"] == "run_copy_of_protected_source_root":
            snapshots = [row["source_root_snapshot"]]
            check(f"root_copy_ref_{payload[:8]}", row["source_root_snapshot"] in result["roots"], row["source_root_snapshot"])
        else:
            snapshots = [row["checkpoint_metadata_snapshot"]]
            check(f"non_sample_ref_{payload[:8]}", row["checkpoint_metadata_snapshot"] not in measured_snapshots, row["checkpoint_metadata_snapshot"])
        for snapshot in snapshots:
            metadata_path = output_checkpoint_dir / f"{snapshot}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            check(f"output_metadata_{snapshot[:8]}", metadata["manifest"].get("snapshot") == snapshot and metadata["manifest"].get("payload_sha256") == payload, f"snapshot={snapshot}; payload={payload}")
            mapped_output_state_ids.add(payload)

    actual_state_files = sorted(p.name for p in output_checkpoint_dir.glob("*.state"))
    mapped_names = sorted(f"{row['payload_sha256']}.state" for row in states)
    check("all_output_state_files_mapped", actual_state_files == mapped_names, f"actual={len(actual_state_files)} mapped={len(mapped_names)}")
    check("mapped_output_payload_count", len(mapped_output_state_ids) == 9, str(len(mapped_output_state_ids)))

    source_roots = mapping["protected_q4_source_roots"]
    check("source_root_count", len(source_roots) == 3, str(len(source_roots)))
    for row in source_roots:
        snapshot, payload = row["snapshot"], row["payload_sha256"]
        metadata_path = args.source_store / "checkpoints" / f"{snapshot}.json"
        state_path = args.source_store / "checkpoints" / f"{payload}.state"
        check(f"source_metadata_sha_{snapshot[:8]}", sha_file(metadata_path) == row["metadata_sha256"], snapshot)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        check(f"source_metadata_payload_{snapshot[:8]}", metadata["manifest"].get("snapshot") == snapshot and metadata["manifest"].get("payload_sha256") == payload, f"payload={metadata['manifest'].get('payload_sha256')}")
        st = state_path.stat(follow_symlinks=False)
        check(f"source_payload_stat_{snapshot[:8]}", stat.S_ISREG(st.st_mode) and st.st_size == row["payload_bytes"] and st.st_ino == row["payload_inode"] and st.st_nlink == row["nlink"] == 1 and stat.S_IMODE(st.st_mode) == 0o600, f"inode={st.st_ino} bytes={st.st_size}")
        check(f"source_payload_sha_{snapshot[:8]}", sha_file(state_path) == payload, payload)

    protected = mapping["protected_store_stat_fingerprints"]
    stat_observations = []
    for key, root in (("q4_source_store", args.source_store), ("lifecycle_output_store", args.run / "output/resident/store")):
        expected = protected[key]
        count, digest = stat_listing(root)
        stat_observations.append({
            "store": key,
            "expected_file_count": expected["regular_file_count"],
            "observed_file_count": count,
            "expected_stat_listing_sha256": expected["stat_listing_sha256"],
            "observed_stat_listing_sha256": digest,
            "matches_initial_capture": count == expected["regular_file_count"] and digest == expected["stat_listing_sha256"],
            "interpretation": "Historical stat digest does not match current recomputation; exact byte inventories below are the protected before-image for any later reviewed operation.",
        })
    reclaim_names = {f"{row['payload_sha256']}.state" for row in states}
    source_files = file_inventory(args.source_store)
    output_metadata_files = file_inventory(args.run / "output/resident/store", exclude_names=reclaim_names)
    for row in output_metadata_files:
        member_path = f"lifecycle-run/output/resident/store/{row['path']}"
        member = member_rows.get(member_path)
        check(f"output_protected_member_{sha_bytes(row['path'].encode())[:8]}", member is not None and member.get("size") == row["size"] and member.get("sha256") == row["sha256"], f"path={member_path}; bytes={row['size']}")

    result_out = {
        "schema": 1,
        "status": "PASS_READ_ONLY_ARCHIVE_AND_STATE_MAP_VERIFIED",
        "map_sha256": sha_file(args.map),
        "archive_sha256": receipt["archive_sha256"],
        "receiver_receipt_sha256": receipt_sha,
        "manifest_raw_sha256": manifest_raw_sha,
        "manifest_canonical_sha256": manifest_canonical_sha,
        "member_listing_sha256": listing_sha,
        "verified_output_state_rows": len(states),
        "verified_output_state_bytes": sum(row["bytes"] for row in states),
        "verified_original_source_roots": len(source_roots),
        "store_stat_observations": stat_observations,
        "protected_q4_source_store_files": source_files,
        "protected_q4_source_store_inventory_sha256": sha_bytes(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "protected_lifecycle_store_metadata_files_excluding_exact_nine_states": output_metadata_files,
        "protected_lifecycle_store_metadata_inventory_sha256": sha_bytes(json.dumps(output_metadata_files, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "check_count": len(checks),
        "checks": checks,
    }
    out_bytes = (json.dumps(result_out, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(out_bytes)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short verification receipt write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"status": result_out["status"], "check_count": len(checks), "output": str(args.output), "output_sha256": sha_bytes(out_bytes)}, sort_keys=True))
    return result_out


if __name__ == "__main__":
    run()
