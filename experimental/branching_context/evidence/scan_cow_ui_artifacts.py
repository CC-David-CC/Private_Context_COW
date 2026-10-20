#!/usr/bin/env python3
"""Scan retained COW UI run artifacts for the private API key without revealing it."""
import argparse
import json
from pathlib import Path
import sys


def occurrences(path: Path, secret: bytes) -> int:
    count = 0
    overlap = b""
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            data = overlap + block
            count += data.count(secret)
            overlap = data[-(len(secret) - 1):] if len(secret) > 1 else b""
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    key_file = args.key_file.resolve(strict=True)
    secret = key_file.read_bytes().strip()
    if not secret:
        raise SystemExit("empty key file")
    if key_file.is_relative_to(run):
        excluded = key_file
    else:
        excluded = None
    candidates = []
    for fixed in (run / "server.stdout.log", run / "server.stderr.log"):
        if fixed.is_file():
            candidates.append(fixed)
    artifact_suffixes = {".json", ".jsonl", ".ndjson", ".txt", ".log", ".trace"}
    candidates.extend(path for path in run.iterdir()
                      if path.is_file() and path.suffix.lower() in artifact_suffixes)
    for directory in (run / "browser", run / "fresh-reference"):
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    rows = []
    hits = 0
    for path in sorted(set(candidates)):
        if path == excluded or path.is_symlink():
            continue
        found = occurrences(path, secret)
        hits += found
        rows.append({"path": str(path.relative_to(run)), "match_count": found})
    print(json.dumps({"status": "FAIL_KEY_FOUND" if hits else "PASS_NO_KEY_MATCHES",
                      "scanned_file_count": len(rows), "total_matches": hits,
                      "files": rows}, sort_keys=True))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
