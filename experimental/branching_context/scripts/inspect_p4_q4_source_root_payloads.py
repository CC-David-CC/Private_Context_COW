#!/usr/bin/env python3
"""Read-only hash/stat inventory of the three original Q4 campaign root payloads."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path("/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/store/checkpoints")
EXPECTED = [
    ("cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca", 1830602232, "e3ecceb2ade807fdf33d87e66d9f5f4c4e901ae6a5c14ec49b36ca11ee1e5286", "07a8d31e1d61e71e25f15322ea5ddd8608e9244a242863ad49e24ce389849a69"),
    ("9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7", 1830602232, "c4386e8b4c4ffe8793ab0dfc34e899b52e7185b74114c455268b22b9bf84b316", "e897d52f34c2cd445635e7c7002455ac8d93eeef7b581881d1b0de43b93cc9f2"),
    ("43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7", 1830602232, "bb974fb5a77c22a4c4ee3e9067a404fa937b70fee53cc469c54273aa02fc584a", "5c725a23cafa398c40c24e202cf6b8d23cee900d9f77bccb50b85b2e0605e8e4"),
]


def stable(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns, stat.S_IMODE(a.st_mode), a.st_nlink) == (
        b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns, stat.S_IMODE(b.st_mode), b.st_nlink
    )


def main() -> None:
    rows = []
    for digest, size, snapshot, manifest_sha in EXPECTED:
        path = ROOT / f"{digest}.state"
        before = path.lstat()
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(fd)
            if not stable(before, opened):
                raise SystemExit(f"identity changed before reading {path}")
            h = hashlib.sha256()
            total = 0
            while True:
                block = os.read(fd, 8 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
                total += len(block)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        actual = h.hexdigest()
        if not stable(before, after) or total != size or actual != digest:
            raise SystemExit(f"source payload mismatch: {path}")
        metadata = ROOT / f"{snapshot}.json"
        metadata_stat = metadata.stat()
        metadata_digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
        if metadata_digest != manifest_sha:
            raise SystemExit(f"snapshot metadata digest mismatch: {metadata}")
        rows.append({
            "source_state_path": str(path),
            "source_state_exists": True,
            "bytes": total,
            "sha256": actual,
            "mode": oct(stat.S_IMODE(before.st_mode)),
            "nlink": before.st_nlink,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "stable_during_hash": True,
            "snapshot": snapshot,
            "metadata_path": str(metadata),
            "metadata_exists": True,
            "metadata_bytes": metadata_stat.st_size,
            "metadata_sha256": metadata_digest,
        })
    fs = os.statvfs(ROOT)
    print(json.dumps({
        "status": "READ_ONLY_Q4_SOURCE_ROOT_INVENTORY_PASS",
        "source_directory": str(ROOT),
        "source_directory_exists": ROOT.is_dir(),
        "count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "free_bytes": fs.f_bavail * fs.f_frsize,
        "no_unlink_or_restore": True,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
