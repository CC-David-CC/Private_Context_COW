"""Public branch-head catalog contract for the canonical logical Store."""

import json
from pathlib import Path
import tempfile
import threading
import unittest

from context_time_travel import BranchHead, IntegrityError, InvalidOperation, Store


class StoreBranchCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="branch-catalog-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)

    def root(self, token):
        return self.store.create_root("catalog-test", [token], {"cursor": token})

    def test_empty_catalog(self):
        self.assertEqual(self.store.branches(), ())

    def test_forks_and_distinct_roots_are_sorted_exact_records(self):
        root_a = self.root(1)
        root_b = self.root(2)
        expected = [
            self.store.fork("alpha", root_a),
            self.store.fork("beta", root_a),
            self.store.fork("other", root_b),
        ]
        self.assertEqual(self.store.branches(), tuple(sorted(expected, key=lambda h: h.name)))
        self.assertEqual(self.store.branches(root=root_a), tuple(expected[:2]))
        self.assertEqual(self.store.branches(root=root_b), (expected[2],))

    def test_rewind_catalog_reports_exact_committed_revision(self):
        root = self.root(3)
        branch = self.store.fork("rewind", root)
        newer = self.store.append(root, [4], {"cursor": 4})
        moved = self.store.move("rewind", 0, newer, "move1")
        rewound = self.store.move("rewind", 1, root, "move2")
        self.assertEqual((moved.revision, moved.tip), (1, newer))
        self.assertEqual(self.store.branches(), (BranchHead("rewind", 2, root, branch.root),))
        self.assertEqual(self.store.head("rewind"), self.store.branches()[0])

    def test_catalog_does_not_change_branch_or_history_files(self):
        root = self.root(5)
        self.store.fork("left", root)
        self.store.fork("right", root)
        paths = sorted(p for p in Path(self.tmp.name).rglob("*") if p.is_file())
        before = {p.relative_to(self.tmp.name): p.read_bytes() for p in paths}
        self.store.branches()
        after = {p.relative_to(self.tmp.name): p.read_bytes()
                 for p in Path(self.tmp.name).rglob("*") if p.is_file()}
        self.assertEqual(after, before)

    def test_catalog_scans_branch_records_not_unrelated_receipts(self):
        root = self.root(11)
        branch = self.store.fork("busy", root)
        for revision in range(5):
            branch = self.store.move("busy", revision, root, f"receipt{revision}")
        scanned_before = self.store._work["reference_entries_scanned"]
        self.assertEqual(self.store.branches(), (branch,))
        self.assertEqual(self.store._work["reference_entries_scanned"] - scanned_before, 1)

    def test_shared_ancestry_is_hash_validated_once_per_catalog_query(self):
        root = self.root(12)
        shared = root
        for token in range(13, 16):
            shared = self.store.append(shared, [token], {"cursor": token})
        for index in range(32):
            self.store.fork(f"branch{index:02}", shared)

        # Force cold ancestry metadata so every measured read is a real
        # canonical object validation, while retaining the ordinary parse
        # cache (which still hashes complete object bytes on each validation).
        self.store._ancestry_metadata.clear()
        before_reads = self.store._work["snapshot_reads"]
        before_visits = self.store._work["ancestry_nodes_visited"]
        heads = self.store.branches()
        self.assertEqual(len(heads), 32)
        self.assertEqual(self.store._work["snapshot_reads"] - before_reads, 4)
        self.assertEqual(self.store._work["ancestry_nodes_visited"] - before_visits, 4)

    def test_catalog_snapshot_serializes_with_concurrent_move(self):
        root = self.root(6)
        newer = self.store.append(root, [7], {"cursor": 7})
        old = self.store.fork("race", root)
        entered = threading.Event()
        done = threading.Event()
        failures = []

        def move():
            entered.set()
            try:
                self.store.move("race", 0, newer, "race-move")
            except Exception as exc:  # surfaced below in the parent thread
                failures.append(exc)
            finally:
                done.set()

        with self.store._locked():
            worker = threading.Thread(target=move)
            worker.start()
            self.assertTrue(entered.wait(2))
            snapshot = self.store.branches()
            self.assertEqual(snapshot, (old,))
            self.assertFalse(done.is_set())
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(self.store.branches(), (BranchHead("race", 1, newer, root),))

    def test_corrupt_branch_record_fails_closed(self):
        root = self.root(8)
        self.store.fork("visible", root)
        path = self.store.references / "visible.json"
        record = json.loads(path.read_text("ascii"))
        record["events"].append({"type": "extra"})
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="ascii")
        with self.assertRaises(IntegrityError):
            self.store.branches()

    def test_unauthorized_root_and_branch_are_not_disclosed(self):
        private_root = self.root(9)
        self.store.fork("private", private_root)
        public_root = self.root(10)
        self.store.fork("public", public_root)
        self.store.grant(public_root, "reader", {"read"})
        reader = self.store.for_principal("reader")
        self.assertEqual(reader.branches(), (BranchHead("public", 0, public_root, public_root),))
        self.assertEqual(reader.branches(root=private_root), ())
        self.assertEqual(reader.branches(root=public_root), (BranchHead("public", 0, public_root, public_root),))


    def test_branch_and_operation_ids_accept_128_ascii_bytes(self):
        root = self.root(17)
        branch_name = "b" + "x" * 127
        operation_id = "o" + "x" * 127
        head = self.store.fork(branch_name, root)
        self.assertEqual(self.store.head(branch_name), head)
        moved = self.store.move(branch_name, 0, root, operation_id)
        self.assertEqual(moved.name, branch_name)
        self.assertEqual(moved.revision, 1)
        self.assertEqual(self.store.head(branch_name), moved)

        with self.assertRaises(InvalidOperation):
            self.store.fork("b" + "x" * 128, root)
        with self.assertRaises(InvalidOperation):
            self.store.move(branch_name, 1, root, "o" + "x" * 128)


if __name__ == "__main__":
    unittest.main()
