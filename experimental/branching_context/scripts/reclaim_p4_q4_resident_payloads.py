#!/usr/bin/env python3
"""Exact-scope resident-output reclaim guard; default behavior is read-only precheck."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path, PurePosixPath

OUTPUT = Path("/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01/resident-parent/output")
REVIEW_DIR = Path("/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01/resident-reclaim-readonly-20261015-01")
MAP_PATH = REVIEW_DIR / "resident-reclaim-map.json"
PROOF_PATH = REVIEW_DIR / "archive-match-proof-v2.json"
EXPECTED_MAP_SHA256 = "1d80892dc4deee21897b22dd37ddbfd14843e8af3fcbe0fd92dd241f80e001ea"
EXPECTED_PROOF_SHA256 = "3ab78750a2b599ad66a58a4d163f0a008af1733eb770798208b62b59f6e5ad3c"
EXPECTED_PROTECTED_SHA256 = "905c45ad3a575c792ddc47af59e1ab278a072f03d000b213beb96126ac9f53bc"
EXPECTED_ARCHIVE_SHA256 = "afea6d6117cc9d7df8c3fbc70225d01ceb3dc61a10224192cf5dd3e694b43a60"
EXPECTED_ARCHIVE_BYTES = 17814958080
EXPECTED_CANDIDATES = 9
EXPECTED_CANDIDATE_BYTES = 17789976604


def same_identity(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns, stat.S_IMODE(a.st_mode), a.st_nlink) == (
        b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns, stat.S_IMODE(b.st_mode), b.st_nlink
    )


def digest_open(path: Path) -> tuple[os.stat_result, str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if not same_identity(before, opened):
            raise RuntimeError(f"file identity changed before read: {path}")
        h = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
            total += len(block)
        after = os.fstat(fd)
        if total != before.st_size or not same_identity(before, after):
            raise RuntimeError(f"file changed during read: {path}")
        return before, h.hexdigest()
    finally:
        os.close(fd)


def fingerprint(rows: list[dict]) -> str:
    canonical = json.dumps(sorted(rows, key=lambda row: row["relative_path"]), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def inspect() -> tuple[dict, list[tuple[Path, dict]], list[dict]]:
    map_bytes = MAP_PATH.read_bytes()
    map_sha = hashlib.sha256(map_bytes).hexdigest()
    if map_sha != EXPECTED_MAP_SHA256:
        raise RuntimeError(f"map SHA mismatch: {map_sha}")
    data = json.loads(map_bytes)
    if data.get("status") != "PREPARED_READ_ONLY_NOT_UNLINKED" or Path(data.get("protected_output", {}).get("root", "")).resolve() != OUTPUT.resolve():
        raise RuntimeError("map status or exact output root mismatch")
    if data.get("verified_archive", {}).get("sha256") != EXPECTED_ARCHIVE_SHA256 or data.get("verified_archive", {}).get("bytes") != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("unexpected resident archive binding")
    proof_bytes = PROOF_PATH.read_bytes()
    if hashlib.sha256(proof_bytes).hexdigest() != EXPECTED_PROOF_SHA256:
        raise RuntimeError("executed archive/source proof SHA mismatch")
    proof = json.loads(proof_bytes)
    if proof.get("status") != "PASS_EXACT_ARCHIVE_MANIFEST_SOURCE_MATCH" or proof.get("result", {}).get("archive_manifest_exact_candidate_match") is not True:
        raise RuntimeError("archive/source proof did not pass")
    if proof.get("inputs", {}).get("archive_sha256") != EXPECTED_ARCHIVE_SHA256 or proof.get("result", {}).get("matched_candidate_count") != EXPECTED_CANDIDATES:
        raise RuntimeError("archive/source proof scope mismatch")
    rows = data.get("candidates", [])
    if len(rows) != EXPECTED_CANDIDATES or sum(row.get("bytes", -1) for row in rows) != EXPECTED_CANDIDATE_BYTES:
        raise RuntimeError("map candidate count/bytes mismatch")
    checkpoint_dir = OUTPUT / "store/checkpoints"
    if OUTPUT.is_symlink() or checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise RuntimeError("output/checkpoint directory missing or symlinked")
    targets = []
    for row in rows:
        rel = PurePosixPath(row.get("source_path", "").removeprefix(str(OUTPUT) + "/"))
        if rel.parent != PurePosixPath("store/checkpoints") or rel.suffix != ".state" or len(rel.parts) != 3:
            raise RuntimeError(f"target outside direct checkpoint-state scope: {rel}")
        if row.get("archive_manifest_match") != "verified by executed proof":
            raise RuntimeError(f"candidate lacks executed archive-manifest match: {rel}")
        path = OUTPUT.joinpath(*rel.parts)
        st, actual_sha = digest_open(path)
        expected = (row["bytes"], row["sha256"], int(row["source_mode"], 8), row["source_device"], row["source_inode"], row["source_mtime_ns"], row["source_nlink"])
        actual = (st.st_size, actual_sha, stat.S_IMODE(st.st_mode), st.st_dev, st.st_ino, st.st_mtime_ns, st.st_nlink)
        if actual != expected or st.st_nlink != 1:
            raise RuntimeError(f"candidate changed since inventory: {path}")
        targets.append((path, row))
    expected_rows = data.get("protected_output", {}).get("inventory_rows", [])
    if len(expected_rows) != data["protected_output"].get("protected_file_count") or fingerprint(expected_rows) != EXPECTED_PROTECTED_SHA256:
        raise RuntimeError("protected metadata rows/fingerprint mismatch")
    target_resolved = {path.resolve() for path, _ in targets}
    current_rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in protected output scope: {path}")
        if path.is_file() and path.resolve() not in target_resolved:
            st, digest = digest_open(path)
            current_rows.append({
                "relative_path": path.relative_to(OUTPUT).as_posix(),
                "size": st.st_size,
                "mode": oct(stat.S_IMODE(st.st_mode)),
                "nlink": st.st_nlink,
                "device": st.st_dev,
                "inode": st.st_ino,
                "mtime_ns": st.st_mtime_ns,
                "sha256": digest,
            })
    if fingerprint(current_rows) != EXPECTED_PROTECTED_SHA256 or current_rows != expected_rows:
        raise RuntimeError("protected output tree differs from archived fingerprint")
    return data, targets, current_rows


def write_receipt(path: Path, receipt: dict) -> None:
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short receipt write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlink", action="store_true", help="remove only the nine map-pinned resident output .state files")
    parser.add_argument("--receipt", type=Path, help="new no-clobber receipt path (required with --unlink)")
    args = parser.parse_args()
    if args.unlink and args.receipt is None:
        parser.error("--receipt is required with --unlink")
    if args.receipt is not None and args.receipt.exists():
        raise RuntimeError(f"refusing to overwrite receipt: {args.receipt}")
    data, targets, protected_rows = inspect()
    result = {
        "status": "PRECHECK_PASS_NOT_UNLINKED",
        "operation": "precheck_only" if not args.unlink else "explicit_unlink_requested",
        "map_sha256": EXPECTED_MAP_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "candidate_count": len(targets),
        "candidate_bytes": sum(row["bytes"] for _, row in targets),
        "protected_file_count": len(protected_rows),
        "protected_fingerprint_sha256": EXPECTED_PROTECTED_SHA256,
        "checked_utc_epoch_ns": time.time_ns(),
        "removed": [],
    }
    if args.unlink:
        removed = []
        try:
            for path, row in targets:
                st, digest = digest_open(path)
                expected = (row["bytes"], row["sha256"], row["source_device"], row["source_inode"], row["source_mtime_ns"], row["source_nlink"])
                actual = (st.st_size, digest, st.st_dev, st.st_ino, st.st_mtime_ns, st.st_nlink)
                if actual != expected:
                    raise RuntimeError(f"candidate changed before unlink: {path}")
                path.unlink()
                if path.exists():
                    raise RuntimeError(f"candidate still exists after unlink: {path}")
                removed.append({"path": str(path), "bytes": row["bytes"], "sha256": row["sha256"]})
            after_rows = []
            for path in sorted(OUTPUT.rglob("*")):
                if path.is_symlink():
                    raise RuntimeError(f"symlink after unlink: {path}")
                if path.is_file():
                    st, digest = digest_open(path)
                    after_rows.append({
                        "relative_path": path.relative_to(OUTPUT).as_posix(), "size": st.st_size,
                        "mode": oct(stat.S_IMODE(st.st_mode)), "nlink": st.st_nlink,
                        "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns,
                        "sha256": digest,
                    })
            if after_rows != protected_rows or fingerprint(after_rows) != EXPECTED_PROTECTED_SHA256:
                raise RuntimeError("protected output tree changed after unlink")
            result["status"] = "UNLINK_PASS"
            result["removed"] = removed
        except Exception as exc:
            result["status"] = "UNLINK_FAILED_PARTIAL_OR_GUARD"
            result["removed"] = removed
            result["error"] = f"{type(exc).__name__}: {exc}"
        write_receipt(args.receipt, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("PRECHECK_PASS_NOT_UNLINKED", "UNLINK_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
