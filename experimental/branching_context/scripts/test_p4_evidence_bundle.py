"""CPU-only tests for parameterized P4 evidence path and state-map helpers."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_evidence_bundle import (  # noqa: E402
    collect_archive_inputs,
    build_manifest,
    make_state_map,
    stream_archive,
    verify_state_map,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceBundleTests(unittest.TestCase):
    def test_collects_parameterized_roots_and_files_without_path_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "arm" / "output"
            receipts = root / "arm" / "receipts"
            source = root / "source"
            for directory in (output, receipts, source):
                directory.mkdir(parents=True)
            (output / "result.json").write_text("{}\n", encoding="utf-8")
            (receipts / "command.exit").write_text("0\n", encoding="ascii")
            extra = root / "profile.json"
            extra.write_text('{"profile":"test"}\n', encoding="utf-8")
            rows = collect_archive_inputs(
                [("arm/output", output), ("arm/receipt", receipts)],
                [("provenance/profile.json", extra)],
            )
            self.assertEqual(
                [row[0] for row in rows],
                ["arm/output/result.json", "arm/receipt/command.exit", "provenance/profile.json"],
            )

    def test_builds_and_streams_receiver_compatible_tar_without_local_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = root / "run"
            tree.mkdir()
            (tree / "result.json").write_bytes(b'{"status":"PASS"}\n')
            (tree / "receipt.txt").write_bytes(b"0\n")
            rows = collect_archive_inputs([("arm/output", tree)], [])
            records, manifest_bytes = build_manifest(rows, "CPU fixture evidence only")
            sink = io.BytesIO()
            receipt = stream_archive(rows, records, manifest_bytes, sink)
            self.assertEqual(receipt["bytes"], len(sink.getvalue()))
            self.assertEqual(receipt["sha256"], hashlib.sha256(sink.getvalue()).hexdigest())
            with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
                names = archive.getnames()
                self.assertEqual(names, ["arm/output/receipt.txt", "arm/output/result.json", "MANIFEST.json"])
                parsed = json.loads(archive.extractfile("MANIFEST.json").read())
                self.assertEqual(parsed["files"], records)
                self.assertEqual(archive.extractfile("arm/output/receipt.txt").read(), b"0\n")
            bad_manifest = json.dumps({"files": []}).encode("utf-8")
            with self.assertRaises(ValueError):
                stream_archive(rows, records, bad_manifest, io.BytesIO())

    def test_stream_refuses_source_changed_after_manifest_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "receipt"
            source.write_bytes(b"before")
            rows = [("run/receipt", source)]
            records, manifest_bytes = build_manifest(rows, "fixture")
            source.write_bytes(b"after!")
            with self.assertRaises(ValueError):
                stream_archive(rows, records, manifest_bytes, io.BytesIO())

    def test_rejects_duplicate_unsafe_and_symlinked_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            source.mkdir()
            file = source / "x"
            file.write_text("x", encoding="ascii")
            with self.assertRaises(ValueError):
                collect_archive_inputs([("../escape", source)], [])
            with self.assertRaises(ValueError):
                collect_archive_inputs([("same", source)], [("same/x", file)])
            if hasattr(os, "symlink"):
                linked = source / "linked"
                try:
                    linked.symlink_to(file)
                except (OSError, NotImplementedError):
                    self.skipTest("symlink creation unavailable")
                with self.assertRaises(ValueError):
                    collect_archive_inputs([("source", source)], [])

    def test_rejects_symlinked_ancestor_of_archive_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            (real / "item").write_text("x", encoding="ascii")
            alias = root / "alias"
            try:
                alias.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ValueError):
                collect_archive_inputs([("tree", alias)], [])

    def test_state_map_derives_count_and_binds_exact_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "output" / "store" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            blobs = [b"first", b"second", b"third"]
            manifest_files = []
            for payload in blobs:
                sha = digest(payload)
                (checkpoint_dir / f"{sha}.state").write_bytes(payload)
                manifest_files.append({
                    "path": f"arm/output/store/checkpoints/{sha}.state",
                    "size": len(payload),
                    "sha256": sha,
                })
            mapping = make_state_map(root / "output", "arm/output", manifest_files)
            self.assertEqual(mapping["candidate_count"], 3)
            self.assertEqual(mapping["candidate_bytes"], sum(map(len, blobs)))
            verified = verify_state_map(root / "output", "arm/output", mapping, manifest_files)
            self.assertEqual(verified, mapping)

    def test_state_map_rejects_unarchived_changed_and_non_sha_named_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "output" / "store" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            payload = b"state"
            sha = digest(payload)
            path = checkpoint_dir / f"{sha}.state"
            path.write_bytes(payload)
            correct = [{"path": f"arm/output/store/checkpoints/{sha}.state", "size": len(payload), "sha256": sha}]
            with self.assertRaises(ValueError):
                make_state_map(root / "output", "arm/output", [])
            wrong = [dict(correct[0], sha256="0" * 64)]
            with self.assertRaises(ValueError):
                make_state_map(root / "output", "arm/output", wrong)
            mapping = make_state_map(root / "output", "arm/output", correct)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_state_map(root / "output", "arm/output", mapping, correct)

    def test_state_map_rejects_duplicate_manifest_rows_and_symlinked_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            checkpoint_dir = output / "store" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            payload = b"state"
            sha = digest(payload)
            (checkpoint_dir / f"{sha}.state").write_bytes(payload)
            row = {"path": f"arm/output/store/checkpoints/{sha}.state", "size": len(payload), "sha256": sha}
            with self.assertRaises(ValueError):
                make_state_map(output, "arm/output", [row, row])
            mapping = make_state_map(output, "arm/output", [row])
            self.assertEqual(mapping["candidate_count"], 1)
            (output / "store").rename(output / "real-store")
            (output / "store").symlink_to(output / "real-store", target_is_directory=True)
            with self.assertRaises(ValueError):
                verify_state_map(output, "arm/output", mapping, [row])

    def test_state_map_rejects_empty_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            (output / "store" / "checkpoints").mkdir(parents=True)
            with self.assertRaises(ValueError):
                make_state_map(output, "arm/output", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
