#!/usr/bin/env python3
"""Exact-scope lifecycle-output reclaim guard; default mode is read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath

STAGE = Path("/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-map-audit-20261015-01")
RUN = Path("/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-20261015-v1")
OUTPUT = RUN / "output/resident"
SOURCE_STORE = Path("/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/store")
MAP_PATH = STAGE / "map.json"
VERIFY_PATH = STAGE / "verified-final.json"
MANIFEST_PATH = Path("/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-archive-20261015-03/manifest.json")
RECEIPT_REMOTE = "/srv/data/git/private_context_cow.evidence/20261015-p4-q4-lifecycle-attribution/lifecycle-attribution-20261015.tar.receipt.json"
EXPECTED_MAP_SHA256 = "91ec8f54db1fc53fb1983dfc6722c86a898edbde79483f992fe86ff5ee462291"
EXPECTED_VERIFY_SHA256 = "6b0e5f59311cbaa3d896706363601c264f0e9eea11993f664d52cdfff2c7c35e"
EXPECTED_ARCHIVE_SHA256 = "d01af0087b43f4c8f3558ae64d49377d8f927c0ec31420035134644cd347684f"
EXPECTED_RECEIVER_SHA256 = "513126b097fec519c993aac095c28f237ea29aa82b56ea2b2f4d0bfa3edaf00c"
EXPECTED_CANDIDATES = 9
EXPECTED_BYTES = 17_789_976_604


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, stat.S_IMODE(s.st_mode), s.st_nlink)
    return fields(left) == fields(right)


def digest_regular(path: Path) -> tuple[os.stat_result, str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if not same_identity(before, opened):
            raise RuntimeError(f"identity changed before hashing: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(fd)
        if size != before.st_size or not same_identity(before, after):
            raise RuntimeError(f"file changed while hashing: {path}")
        return before, digest.hexdigest()
    finally:
        os.close(fd)


def file_inventory(root: Path, excluded: set[str] | None = None) -> list[dict]:
    skip = excluded or set()
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in skip):
        st, digest = digest_regular(path)
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size": st.st_size,
            "mode": f"{stat.S_IMODE(st.st_mode):04o}",
            "nlink": st.st_nlink,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
            "sha256": digest,
        })
    return rows


def inventory_sha(rows: list[dict]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha_bytes(encoded)


def free_bytes(path: Path) -> int:
    usage = os.statvfs(path)
    return usage.f_bavail * usage.f_frsize


def inspect() -> tuple[dict, dict, list[tuple[Path, dict, os.stat_result]], list[dict], list[dict]]:
    map_bytes = MAP_PATH.read_bytes()
    map_sha = sha_bytes(map_bytes)
    if map_sha != EXPECTED_MAP_SHA256:
        raise RuntimeError(f"map SHA mismatch: {map_sha}")
    mapping = json.loads(map_bytes)
    if mapping.get("reclaim_status") != "NO_RECLAIM_AUTHORIZED_OR_PERFORMED; map is for review only.":
        raise RuntimeError("map is not the reviewed read-only candidate map")
    if Path(mapping["execution"]["run"]).resolve() != RUN.resolve():
        raise RuntimeError("unexpected lifecycle run root")
    if SOURCE_STORE.resolve() == RUN.resolve() or SOURCE_STORE in RUN.parents or RUN in SOURCE_STORE.parents:
        raise RuntimeError("source Q4 store and candidate run overlap")

    verify_bytes = VERIFY_PATH.read_bytes()
    verify_sha = sha_bytes(verify_bytes)
    if verify_sha != EXPECTED_VERIFY_SHA256:
        raise RuntimeError(f"verification receipt SHA mismatch: {verify_sha}")
    verified = json.loads(verify_bytes)
    if verified.get("status") != "PASS_READ_ONLY_ARCHIVE_AND_STATE_MAP_VERIFIED":
        raise RuntimeError("archive/state map verifier did not pass")
    if verified.get("map_sha256") != map_sha or verified.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("verification receipt is bound to different inputs")
    if verified.get("receiver_receipt_sha256") != EXPECTED_RECEIVER_SHA256 or verified.get("verified_output_state_rows") != EXPECTED_CANDIDATES:
        raise RuntimeError("verification receipt has wrong archive or state scope")

    manifest_raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_raw)
    canonical_manifest = sha_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if sha_bytes(manifest_raw) != verified.get("manifest_raw_sha256") or canonical_manifest != verified.get("manifest_canonical_sha256"):
        raise RuntimeError("archived source manifest differs from executed verification receipt")
    receipt_proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "box-git", "cat", RECEIPT_REMOTE],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    receipt_bytes = receipt_proc.stdout
    if sha_bytes(receipt_bytes) != EXPECTED_RECEIVER_SHA256:
        raise RuntimeError("private receiver receipt bytes changed")
    receipt = json.loads(receipt_bytes)
    if receipt.get("status") != "PASS" or receipt.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("private receiver no longer confirms this archive")

    states = mapping["large_file_inventory"]["state_files"]
    if len(states) != EXPECTED_CANDIDATES or sum(row["bytes"] for row in states) != EXPECTED_BYTES:
        raise RuntimeError("candidate count/bytes differ from reviewed nine-state map")
    payloads = [row.get("payload_sha256", "") for row in states]
    if len(set(payloads)) != EXPECTED_CANDIDATES or any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in payloads):
        raise RuntimeError("candidate payload IDs are malformed or duplicated")
    if RUN.resolve() != RUN or OUTPUT.resolve() != OUTPUT:
        raise RuntimeError("run/output root resolves through an unexpected symlink")
    store_dir = OUTPUT / "store"
    output_root = OUTPUT.resolve()
    checkpoint_dir = output_root / "store/checkpoints"
    if store_dir.resolve() != store_dir or checkpoint_dir.resolve() != checkpoint_dir:
        raise RuntimeError("store/checkpoints resolves through an unexpected symlink")
    if RUN.is_symlink() or OUTPUT.is_symlink() or store_dir.is_symlink() or checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise RuntimeError("exact lifecycle output/checkpoint directory missing or symlinked")
    archive_rows = {row["path"]: row for row in manifest["files"]}
    if len(archive_rows) != len(manifest["files"]) or len(archive_rows) != 338:
        raise RuntimeError(f"unexpected or duplicate archive manifest rows: {len(manifest['files'])} rows, {len(archive_rows)} unique")
    listing = json.dumps(
        sorted(({"path": row["path"], "size": row["size"], "sha256": row["sha256"]} for row in manifest["files"]), key=lambda row: row["path"]),
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if sha_bytes(listing) != verified.get("member_listing_sha256"):
        raise RuntimeError("complete archive member listing differs from executed verifier receipt")

    targets = []
    for row in states:
        payload = row["payload_sha256"]
        rel = PurePosixPath("store/checkpoints") / f"{payload}.state"
        if rel.parent != PurePosixPath("store/checkpoints") or rel.suffix != ".state" or len(rel.parts) != 3:
            raise RuntimeError(f"target outside exact direct-state scope: {rel}")
        path = output_root.joinpath(*rel.parts)
        st, digest = digest_regular(path)
        if digest != payload or st.st_size != row["bytes"] or st.st_nlink != 1 or stat.S_IMODE(st.st_mode) != 0o600:
            raise RuntimeError(f"candidate content/stat mismatch: {path}")
        member_path = f"lifecycle-run/output/resident/{rel.as_posix()}"
        member = archive_rows.get(member_path)
        if member is None or member.get("size") != st.st_size or member.get("sha256") != payload:
            raise RuntimeError(f"candidate is not bound to archive manifest: {member_path}")
        targets.append((path, row, st))

    target_names = {f"{row['payload_sha256']}.state" for row in states}
    current_output_metadata = file_inventory(output_root / "store", excluded=target_names)
    current_source = file_inventory(SOURCE_STORE)
    if current_output_metadata != verified.get("protected_lifecycle_store_metadata_files_excluding_exact_nine_states"):
        raise RuntimeError("protected lifecycle store metadata differs from verified before-image")
    if current_source != verified.get("protected_q4_source_store_files"):
        raise RuntimeError("protected Q4 source store differs from verified before-image")
    if inventory_sha(current_output_metadata) != verified.get("protected_lifecycle_store_metadata_inventory_sha256"):
        raise RuntimeError("protected lifecycle metadata fingerprint mismatch")
    if inventory_sha(current_source) != verified.get("protected_q4_source_store_inventory_sha256"):
        raise RuntimeError("protected Q4 source inventory fingerprint mismatch")
    return mapping, verified, targets, current_output_metadata, current_source


def write_receipt(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short reclaim receipt write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlink", action="store_true", help="remove only the exact nine mapped lifecycle output .state files")
    parser.add_argument("--receipt", type=Path, help="new no-clobber receipt path; required with --unlink")
    args = parser.parse_args()
    if args.unlink and args.receipt is None:
        parser.error("--receipt is required with --unlink")
    if args.receipt is not None and args.receipt.exists():
        raise RuntimeError(f"refusing to overwrite receipt: {args.receipt}")

    free_before = free_bytes(RUN)
    mapping, verified, targets, protected_output, protected_source = inspect()
    base = {
        "schema": 1,
        "status": "PRECHECK_PASS_NOT_UNLINKED" if not args.unlink else "UNLINK_IN_PROGRESS",
        "operation": "precheck_only" if not args.unlink else "explicit_exact_nine_unlink",
        "map_sha256": EXPECTED_MAP_SHA256,
        "verification_receipt_sha256": EXPECTED_VERIFY_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "receiver_receipt_sha256": EXPECTED_RECEIVER_SHA256,
        "candidate_count": len(targets),
        "candidate_bytes": sum(row["bytes"] for _, row, _ in targets),
        "targets": [
            {"path": str(path), "sha256": row["payload_sha256"], "bytes": before.st_size,
             "device": before.st_dev, "inode": before.st_ino, "mode": f"{stat.S_IMODE(before.st_mode):04o}",
             "nlink": before.st_nlink, "mtime_ns": before.st_mtime_ns}
            for path, row, before in targets
        ],
        "protected_output_metadata_inventory_sha256": inventory_sha(protected_output),
        "protected_output_metadata_file_count": len(protected_output),
        "protected_source_store_inventory_sha256": inventory_sha(protected_source),
        "protected_source_store_file_count": len(protected_source),
        "source_roots_outside_reclaim_scope": mapping["protected_q4_source_roots"],
        "unlink_performed": False,
        "filesystem_free_bytes_before_precheck": free_before,
        "filesystem_free_bytes_after_precheck": free_bytes(RUN),
    }
    if not args.unlink:
        if args.receipt is not None:
            base["receipt_sha256"] = write_receipt(args.receipt, base)
        print(json.dumps({k: base[k] for k in ("status", "candidate_count", "candidate_bytes", "unlink_performed", "receipt_sha256") if k in base}, sort_keys=True))
        return 0

    removed = []
    try:
        for path, row, initial in targets:
            current, digest = digest_regular(path)
            if not same_identity(initial, current) or digest != row["payload_sha256"]:
                raise RuntimeError(f"candidate changed immediately before unlink: {path}")
            os.unlink(path)
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            removed.append({"path": str(path), "sha256": digest, "bytes": current.st_size})
        if any(path.exists() for path, _, _ in targets):
            raise RuntimeError("one or more mapped targets remain after unlink")
        if file_inventory(OUTPUT / "store") != protected_output:
            raise RuntimeError("protected lifecycle metadata changed during unlink")
        if file_inventory(SOURCE_STORE) != protected_source:
            raise RuntimeError("protected Q4 source store changed during unlink")
        base.update({"status": "UNLINK_PASS_EXACT_NINE_ONLY", "unlink_performed": True, "removed": removed})
    except BaseException as exc:
        base.update({"status": "UNLINK_FAILED_PARTIAL_OR_UNVERIFIED", "unlink_performed": bool(removed), "removed": removed, "error": f"{type(exc).__name__}: {exc}"})
        if args.receipt is not None:
            base["receipt_sha256"] = write_receipt(args.receipt, base)
        raise
    if args.receipt is not None:
        base["receipt_sha256"] = write_receipt(args.receipt, base)
    print(json.dumps({k: base[k] for k in ("status", "candidate_count", "candidate_bytes", "unlink_performed", "receipt_sha256") if k in base}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
