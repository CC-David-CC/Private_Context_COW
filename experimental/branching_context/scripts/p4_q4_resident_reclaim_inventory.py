#!/usr/bin/env python3
"""Read-only inventory for the archived P4 resident-arm output checkpoint files."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path("/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01/resident-parent/output")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, stat.S_IMODE(before.st_mode), before.st_nlink) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, stat.S_IMODE(after.st_mode), after.st_nlink
    )


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"missing output root: {ROOT}")
    candidates = sorted((ROOT / "store/checkpoints").glob("*.state"))
    if len(candidates) != 9 or any(p.is_symlink() for p in candidates):
        raise SystemExit(f"unexpected candidate set: {len(candidates)}")
    candidate_rows = []
    for path in candidates:
        before = path.stat(follow_symlinks=False)
        digest = sha256_file(path)
        after = path.stat(follow_symlinks=False)
        stable = stable_identity(before, after)
        row = {
            "path": str(path),
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size": before.st_size,
            "mode": oct(stat.S_IMODE(before.st_mode)),
            "nlink": before.st_nlink,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "sha256": digest,
            "stable_during_hash": stable,
        }
        if not stable or row["mode"] != "0o600" or row["nlink"] != 1 or path.stem != digest:
            raise SystemExit(f"candidate invariant failed: {path}")
        candidate_rows.append(row)

    protected_rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path in candidates:
            continue
        before = path.stat(follow_symlinks=False)
        digest = sha256_file(path)
        after = path.stat(follow_symlinks=False)
        stable = stable_identity(before, after)
        if path.is_symlink() or not stable:
            raise SystemExit(f"protected file changed or symlinked: {path}")
        protected_rows.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size": before.st_size,
            "mode": oct(stat.S_IMODE(before.st_mode)),
            "nlink": before.st_nlink,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "sha256": digest,
        })
    canonical = json.dumps(protected_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    statvfs = os.statvfs(ROOT)
    result = {
        "status": "READ_ONLY_INVENTORY_PASS",
        "root": str(ROOT),
        "candidate_count": len(candidate_rows),
        "candidate_total_bytes": sum(row["size"] for row in candidate_rows),
        "candidate_max_bytes": max(row["size"] for row in candidate_rows),
        "candidates": candidate_rows,
        "protected_count": len(protected_rows),
        "protected_bytes": sum(row["size"] for row in protected_rows),
        "protected_fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
        "protected": protected_rows,
        "free_bytes_at_inventory": statvfs.f_bavail * statvfs.f_frsize,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
