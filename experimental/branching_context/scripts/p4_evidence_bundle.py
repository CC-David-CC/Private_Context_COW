"""Small reusable, non-destructive helpers for P4 arm evidence bundles.

These functions deliberately know nothing about a particular run, profile,
archive target, or state count. Callers provide the arm's output/provenance
paths and the receiver-verified member manifest. They do not stream or delete.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Iterable

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink in input path: {component}")


def _member_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"invalid archive member name: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe archive member name: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"archive member name is not canonical: {value!r}")
    return normalized


def _stable_file(path: Path) -> dict:
    _reject_symlink_components(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"archive input is not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
            stat.S_IMODE(item.st_mode), item.st_nlink,
        )
        if identity(before) != identity(opened):
            raise ValueError(f"input changed before hashing: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(fd)
        if size != before.st_size or identity(opened) != identity(after):
            raise ValueError(f"input changed while hashing: {path}")
        return {
            "size": size,
            "sha256": digest.hexdigest(),
            "mode": f"{stat.S_IMODE(before.st_mode):04o}",
            "nlink": before.st_nlink,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(fd)


def collect_archive_inputs(
    trees: Iterable[tuple[str, Path]], files: Iterable[tuple[str, Path]]
) -> list[tuple[str, Path]]:
    """Collect tree and explicit-file inputs, rejecting unsafe or duplicate paths."""
    rows: list[tuple[str, Path]] = []
    names: set[str] = set()

    def add(name: str, path: Path) -> None:
        canonical = _member_name(name)
        if canonical in names:
            raise ValueError(f"duplicate archive member path: {canonical}")
        names.add(canonical)
        _reject_symlink_components(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"input must be an existing regular non-symlink file: {path}")
        rows.append((canonical, path))

    for prefix, root in trees:
        prefix = _member_name(prefix)
        root = Path(root)
        _reject_symlink_components(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"tree must be an existing non-symlink directory: {root}")
        for current, dirs, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirs.sort()
            filenames.sort()
            for directory in dirs:
                child = current_path / directory
                if child.is_symlink():
                    raise ValueError(f"symlink directory rejected: {child}")
            for filename in filenames:
                child = current_path / filename
                if child.is_symlink():
                    raise ValueError(f"symlink file rejected: {child}")
                rel = child.relative_to(root).as_posix()
                add(f"{prefix}/{rel}", child)
    for name, path in files:
        add(name, Path(path))
    return sorted(rows, key=lambda row: row[0])


def build_manifest(rows: Iterable[tuple[str, Path]], bundle_scope: str, metadata: dict | None = None) -> tuple[list[dict], bytes]:
    """Hash a closed input list and return receiver-compatible manifest bytes."""
    records = []
    names: set[str] = set()
    for name, path in rows:
        name = _member_name(name)
        if name in names:
            raise ValueError(f"duplicate archive member path: {name}")
        names.add(name)
        record = _stable_file(Path(path))
        records.append({"path": name, "size": record["size"], "sha256": record["sha256"]})
    records.sort(key=lambda row: row["path"])
    manifest = {
        "schema": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_scope": bundle_scope,
        "metadata": metadata or {},
        "files": records,
    }
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return records, payload


class _HashingWriter:
    def __init__(self, target):
        self.target = target
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        written = self.target.write(data)
        if written != len(data):
            raise OSError("short archive stream write")
        self.digest.update(data)
        self.size += written
        return written

    def flush(self) -> None:
        self.target.flush()


class _HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


def stream_archive(rows: Iterable[tuple[str, Path]], records: list[dict], manifest_bytes: bytes, target) -> dict:
    """Write a tar stream to a caller-owned binary sink without a local spool.

    The caller remains responsible for receiver identity, no-clobber target
    checks, and receiver receipt verification. Inputs are rehashed as streamed
    and rejected if their bytes or file identity changed after manifest build.
    """
    rows = list(rows)
    if [name for name, _ in rows] != [row.get("path") for row in records]:
        raise ValueError("archive input order/names differ from manifest records")
    try:
        manifest = json.loads(manifest_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest bytes are not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("files") != records:
        raise ValueError("manifest file rows differ from stream records")
    index = {row["path"]: row for row in records}
    if len(index) != len(records):
        raise ValueError("duplicate manifest records")
    writer = _HashingWriter(target)
    with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for name, path in rows:
            name = _member_name(name)
            path = Path(path)
            _reject_symlink_components(path)
            before = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(before.st_mode):
                raise ValueError(f"archive input changed to nonregular file: {path}")
            expected = index[name]
            if before.st_size != expected["size"]:
                raise ValueError(f"archive input size changed after manifest: {path}")
            with path.open("rb", buffering=0) as source:
                opened = os.fstat(source.fileno())
                identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
                if identity(before) != identity(opened):
                    raise ValueError(f"archive input changed before streaming: {path}")
                reader = _HashingReader(source)
                info = tarfile.TarInfo(name)
                info.size = expected["size"]
                info.mode = 0o600
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info, reader)
                after = os.fstat(source.fileno())
            if reader.size != expected["size"] or reader.digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"archive input bytes changed after manifest: {path}")
            if identity(before) != identity(after):
                raise ValueError(f"archive input identity changed while streaming: {path}")
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mode = 0o600
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest_bytes))
    return {"bytes": writer.size, "sha256": writer.digest.hexdigest()}


def _manifest_index(manifest_files: Iterable[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in manifest_files:
        if not isinstance(row, dict):
            raise ValueError("archive manifest row must be an object")
        name = _member_name(row.get("path"))
        size, digest = row.get("size"), row.get("sha256")
        if name in index:
            raise ValueError(f"duplicate archive manifest member: {name}")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid size or SHA-256 in manifest row: {name}")
        index[name] = {"path": name, "size": size, "sha256": digest}
    return index


def make_state_map(output_root: Path, archive_prefix: str, manifest_files: Iterable[dict]) -> dict:
    """Hash current direct output checkpoint states and bind each to a manifest row.

    The discovered state count is data-driven. This function does not imply
    archive receipt authenticity; callers must provide the manifest from the
    already receiver-verified archive and retain that receipt separately.
    """
    output_root = Path(output_root)
    _reject_symlink_components(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError(f"output root must be an existing non-symlink directory: {output_root}")
    archive_prefix = _member_name(archive_prefix)
    index = _manifest_index(manifest_files)
    checkpoint_dir = output_root / "store" / "checkpoints"
    _reject_symlink_components(checkpoint_dir)
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory missing or symlinked: {checkpoint_dir}")
    rows = []
    for path in sorted(checkpoint_dir.iterdir(), key=lambda item: item.name):
        if path.suffix != ".state":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint state must be a regular non-symlink file: {path}")
        file_row = _stable_file(path)
        payload = path.name[:-len(".state")]
        if not _SHA256.fullmatch(payload) or file_row["sha256"] != payload:
            raise ValueError(f"checkpoint name does not match payload SHA-256: {path}")
        member = f"{archive_prefix}/store/checkpoints/{path.name}"
        archived = index.get(member)
        if archived != {"path": member, "size": file_row["size"], "sha256": payload}:
            raise ValueError(f"checkpoint does not exactly match archive manifest member: {member}")
        rows.append({"relative_path": path.relative_to(output_root).as_posix(), **file_row})
    rows.sort(key=lambda row: row["relative_path"])
    if not rows:
        raise ValueError("no checkpoint state files found in output root")
    return {
        "schema": "p4-arm-state-map-v1",
        "status": "READ_ONLY_ARCHIVE_BOUND_MAP",
        "output_root": str(output_root.resolve()),
        "archive_prefix": archive_prefix,
        "candidate_count": len(rows),
        "candidate_bytes": sum(row["size"] for row in rows),
        "candidates": rows,
    }


def verify_state_map(
    output_root: Path, archive_prefix: str, mapping: dict, manifest_files: Iterable[dict]
) -> dict:
    """Recompute and compare a map; refuse changed files or archive rows."""
    actual = make_state_map(output_root, archive_prefix, manifest_files)
    if mapping != actual:
        raise ValueError("state map differs from current files and verified archive manifest")
    return actual
