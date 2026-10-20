"""Owner-only presentation namespace records, separate from COW history."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
import re

from context_time_travel import IntegrityError, InvalidOperation, NotFound, Store


class StoreNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="store-namespaces-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)

    def test_distinct_namespace_ids_do_not_follow_identical_roots(self):
        first_root = self.store.create_root("same-model", [1, 2], {"mode": "idle"})
        second_root = self.store.create_root("same-model", [1, 2], {"mode": "idle"})
        self.assertEqual(first_root, second_root)
        first = self.store.create_namespace("Same title", "client-op-A")
        second = self.store.create_namespace("Same title", "client-op-B")
        self.assertRegex(first.namespace_id, re.compile(r"^[0-9a-f]{32}$"))
        self.assertNotEqual(first.namespace_id, second.namespace_id)
        self.assertEqual(self.store.namespaces(), tuple(sorted((first, second),
                                                                    key=lambda item: item.namespace_id)))

    def test_empty_namespace_survives_restart_without_history_or_head(self):
        record = self.store.create_namespace("Empty", "new-chat-1")
        restarted = Store(self.tmp.name)
        self.assertEqual(restarted.get_namespace(record.namespace_id), record)
        self.assertEqual(restarted.namespaces(), (record,))
        self.assertEqual(list(restarted.objects.glob("*.json")), [])
        self.assertEqual(restarted.branches(), ())
        self.assertEqual(self.store.presentation_namespaces.stat().st_mode & 0o777, 0o700)
        record_path = self.store.presentation_namespaces / (record.namespace_id + ".json")
        self.assertEqual(record_path.stat().st_mode & 0o777, 0o600)

    def test_same_operation_retry_returns_exact_record_and_changed_title_conflicts(self):
        created = self.store.create_namespace("Retry me", "stable-operation")
        self.assertEqual(self.store.create_namespace("Retry me", "stable-operation"), created)
        with self.assertRaises(InvalidOperation):
            self.store.create_namespace("Changed title", "stable-operation")
        self.assertEqual(self.store.namespaces(), (created,))

    def test_title_validation(self):
        for title in ("", " \t ", "x" * 121, None, 4):
            with self.subTest(title=repr(title)), self.assertRaises(InvalidOperation):
                self.store.create_namespace(title, "invalid-title-op")

    def test_corrupt_record_and_deleted_record_fail_closed(self):
        created = self.store.create_namespace("Check me", "integrity-op")
        path = self.store.presentation_namespaces / (created.namespace_id + ".json")
        original = path.read_bytes()
        path.write_bytes(original.replace(b"Check me", b"Check us"))
        with self.assertRaises(IntegrityError):
            self.store.get_namespace(created.namespace_id)
        path.unlink()
        with self.assertRaises(NotFound):
            self.store.get_namespace(created.namespace_id)

    def test_listing_rejects_malformed_record_filename(self):
        self.store.create_namespace("Good", "good-op")
        (self.store.presentation_namespaces / "not-a-namespace.json").write_text("{}\n")
        with self.assertRaises(IntegrityError):
            self.store.namespaces()

    def test_concurrent_same_operation_creates_one_record(self):
        with ThreadPoolExecutor(max_workers=12) as pool:
            records = list(pool.map(lambda _: self.store.create_namespace("Concurrent", "same-op"),
                                    range(24)))
        self.assertTrue(all(record == records[0] for record in records))
        self.assertEqual(self.store.namespaces(), (records[0],))

    def test_create_fault_before_rename_leaves_no_namespace_then_retries(self):
        def fault(point):
            if point == "namespace_before_rename":
                raise RuntimeError(point)

        failing = Store(self.tmp.name, fault_hook=fault)
        with self.assertRaisesRegex(RuntimeError, "namespace_before_rename"):
            failing.create_namespace("Before rename", "before-op")
        self.assertEqual(self.store.namespaces(), ())
        recovered = self.store.create_namespace("Before rename", "before-op")
        self.assertEqual(self.store.namespaces(), (recovered,))

    def test_create_fault_after_rename_recovers_exact_receipt_after_restart(self):
        def fault(point):
            if point == "namespace_after_rename_before_dir_fsync":
                raise RuntimeError(point)

        failing = Store(self.tmp.name, fault_hook=fault)
        with self.assertRaisesRegex(RuntimeError, "namespace_after_rename_before_dir_fsync"):
            failing.create_namespace("After rename", "after-op")
        restarted = Store(self.tmp.name)
        recovered = restarted.create_namespace("After rename", "after-op")
        self.assertEqual(restarted.get_namespace(recovered.namespace_id), recovered)
        self.assertEqual(restarted.namespaces(), (recovered,))

    def test_read_and_listing_do_not_change_metadata_files(self):
        created = self.store.create_namespace("Read only", "readonly-op")
        before = {path.relative_to(self.tmp.name): path.read_bytes()
                  for path in Path(self.tmp.name).rglob("*") if path.is_file()}
        self.store.get_namespace(created.namespace_id)
        self.store.namespaces()
        after = {path.relative_to(self.tmp.name): path.read_bytes()
                 for path in Path(self.tmp.name).rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_restricted_principal_cannot_create_read_or_namespaces(self):
        created = self.store.create_namespace("Owner only", "owner-op")
        restricted = self.store.for_principal("reader")
        with self.assertRaises(PermissionError):
            restricted.get_namespace(created.namespace_id)
        with self.assertRaises(PermissionError):
            restricted.namespaces()
        with self.assertRaises(PermissionError):
            restricted.create_namespace("No", "restricted-op")


if __name__ == "__main__":
    unittest.main()
