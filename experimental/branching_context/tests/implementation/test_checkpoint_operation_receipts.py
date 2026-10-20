"""Numerical checkpoint provenance is atomically bound to an existing COW move."""

import copy
import hashlib
import tempfile
import unittest

from context_time_travel import IntegrityError, InvalidOperation, NotFound, Store
from context_time_travel.store import _decode, _json_bytes


def descriptor(store, target, *, model="model-A", profile="profile-1", payload=None):
    return {
        "schema": 1,
        "snapshot": target,
        "payload_sha256": payload or hashlib.sha256(b"native checkpoint bytes").hexdigest(),
        "model_contract_id": model,
        "execution_profile_id": profile,
    }


class CheckpointOperationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="checkpoint-operation-receipt-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.root = self.store.create_root("model-A", [1, 2], {"state": "idle"})
        self.head = self.store.fork("branch", self.root)
        self.target = self.store.append(self.root, [3], {"state": "stopped"})

    def test_optional_move_keeps_legacy_receipt_shape_byte_identical(self):
        moved = self.store.move("branch", 0, self.target, "legacy-move")
        record = _decode((self.store.references / "branch.json").read_bytes())
        legacy_request = {"type": "move", "name": "branch", "expected_revision": 0,
                          "target": self.target}
        self.assertEqual(record["receipts"]["legacy-move"], {
            "digest": hashlib.sha256(_json_bytes(legacy_request)).hexdigest(),
            "result": {"name": "branch", "revision": 1, "tip": self.target, "root": self.root},
        })
        self.assertEqual(set(record["events"][-1]), {
            "type", "branch", "operation_id", "from", "to", "from_revision", "to_revision"
        })
        self.assertEqual(moved.tip, self.target)
        with self.assertRaises(NotFound):
            self.store.checkpoint_receipt("legacy-move")

    def test_receipt_survives_reopen_rewind_retry_and_numerical_file_deletion(self):
        value = descriptor(self.store, self.target)
        original = self.store.move("branch", 0, self.target, "checkpoint-move",
                                   checkpoint_receipt=value)
        receipt = self.store.checkpoint_receipt("checkpoint-move")
        self.assertEqual(receipt["checkpoint_receipt"], value)
        self.assertEqual(receipt["digest"], hashlib.sha256(_json_bytes(value)).hexdigest())
        receipt["checkpoint_receipt"]["execution_profile_id"] = "mutated-caller-copy"
        self.assertEqual(self.store.checkpoint_receipt("checkpoint-move")["checkpoint_receipt"], value)

        rewound = self.store.move("branch", 1, self.root, "rewind-after-checkpoint")
        self.assertEqual(rewound.tip, self.root)
        reopened = Store(self.tmp.name)
        self.assertEqual(reopened.move("branch", 0, self.target, "checkpoint-move",
                                       checkpoint_receipt=value), original)
        # Numerical materializations are outside Store; deleting them cannot alter the receipt.
        numerical_dir = reopened.root / "numerical-checkpoints"
        numerical_dir.mkdir()
        numerical = numerical_dir / (value["payload_sha256"] + ".bin")
        numerical.write_bytes(b"not a canonical Store object")
        numerical.unlink()
        self.assertEqual(reopened.checkpoint_receipt("checkpoint-move")["checkpoint_receipt"], value)

    def test_namespace_move_and_retry_keep_checkpoint_descriptor(self):
        namespace = self.store.create_namespace("chat", "namespace-op").namespace_id
        root = self.store.create_root("model-A", [7], {"state": "idle"})
        self.store.fork("chat-branch", root, namespace_id=namespace, operation_id="fork-chat")
        target = self.store.append(root, [8], {"state": "stopped"})
        value = descriptor(self.store, target)
        first = self.store.move("chat-branch", 0, target, "chat-checkpoint-op",
                                checkpoint_receipt=value)
        self.store.move("chat-branch", 1, root, "chat-rewind")
        reopened = Store(self.tmp.name)
        self.assertEqual(reopened.move("chat-branch", 0, target, "chat-checkpoint-op",
                                       checkpoint_receipt=value), first)
        self.assertEqual(reopened.head("chat-branch").tip, root)
        self.assertEqual(reopened.checkpoint_receipt("chat-checkpoint-op")["checkpoint_receipt"], value)

    def test_reused_operation_id_with_altered_descriptor_is_rejected(self):
        value = descriptor(self.store, self.target)
        self.store.move("branch", 0, self.target, "bound-op", checkpoint_receipt=value)
        changed = dict(value, execution_profile_id="different-profile")
        with self.assertRaises(InvalidOperation):
            self.store.move("branch", 0, self.target, "bound-op", checkpoint_receipt=changed)

    def test_wrong_target_model_and_malformed_descriptor_reject_before_cas(self):
        other = self.store.append(self.root, [4], {"state": "stopped"})
        with self.assertRaises(InvalidOperation):
            self.store.move("branch", 0, self.target, "wrong-target",
                            checkpoint_receipt=descriptor(self.store, other))
        with self.assertRaises(InvalidOperation):
            self.store.move("branch", 0, self.target, "wrong-model",
                            checkpoint_receipt=descriptor(self.store, self.target, model="other-model"))
        for bad in (
            dict(descriptor(self.store, self.target), schema=True),
            dict(descriptor(self.store, self.target), payload_sha256="z" * 64),
            dict(descriptor(self.store, self.target), execution_profile_id=""),
            dict(descriptor(self.store, self.target), execution_profile_id="p" * 257),
            dict(descriptor(self.store, self.target), unexpected="field"),
        ):
            with self.subTest(bad=bad), self.assertRaises(InvalidOperation):
                self.store.move("branch", 0, self.target, "invalid-descriptor", checkpoint_receipt=bad)
        self.assertEqual(self.store.head("branch").revision, 0)

    def test_descriptor_and_digest_tampering_fail_closed(self):
        value = descriptor(self.store, self.target)
        self.store.move("branch", 0, self.target, "tamper-op", checkpoint_receipt=value)
        path = self.store.references / "branch.json"
        original = _decode(path.read_bytes())
        changed = copy.deepcopy(original)
        changed["receipts"]["tamper-op"]["checkpoint_receipt"]["execution_profile_id"] = "tampered"
        path.write_bytes(_json_bytes(changed))
        with self.assertRaises(IntegrityError):
            Store(self.tmp.name).checkpoint_receipt("tamper-op")
        path.write_bytes(_json_bytes(original))
        changed = copy.deepcopy(original)
        changed["receipts"]["tamper-op"]["digest"] = "0" * 64
        path.write_bytes(_json_bytes(changed))
        with self.assertRaises(IntegrityError):
            Store(self.tmp.name).checkpoint_receipt("tamper-op")
        path.write_bytes(_json_bytes(original))
        changed = copy.deepcopy(original)
        changed["events"][-1]["checkpoint_receipt_digest"] = "0" * 64
        path.write_bytes(_json_bytes(changed))
        with self.assertRaises(IntegrityError):
            Store(self.tmp.name).checkpoint_receipt("tamper-op")

    def test_plain_branch_result_tampering_rejected_by_lookup_and_retry(self):
        value = descriptor(self.store, self.target)
        original_head = self.store.move("branch", 0, self.target, "result-tamper-op",
                                        checkpoint_receipt=value)
        path = self.store.references / "branch.json"
        original = _decode(path.read_bytes())
        for field, forged in (("tip", self.root), ("revision", original_head.revision + 1)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed["receipts"]["result-tamper-op"]["result"][field] = forged
                path.write_bytes(_json_bytes(changed))
                with self.assertRaises(IntegrityError):
                    Store(self.tmp.name).checkpoint_receipt("result-tamper-op")
                with self.assertRaises(IntegrityError):
                    Store(self.tmp.name).move("branch", 0, self.target, "result-tamper-op",
                                              checkpoint_receipt=value)
                path.write_bytes(_json_bytes(original))

    def test_post_rename_fsync_failure_is_recovered_by_exact_retry(self):
        fired = {"value": False}
        def fail(point):
            if point == "after_refs_rename_before_dir_fsync" and not fired["value"]:
                fired["value"] = True
                raise OSError("injected directory sync failure")

        value = descriptor(self.store, self.target)
        failing = Store(self.tmp.name, fault_hook=fail)
        # The operation may be committed despite the reported sync error.
        with self.assertRaises(OSError):
            failing.move("branch", 0, self.target, "after-rename", checkpoint_receipt=value)
        reopened = Store(self.tmp.name)
        receipt = reopened.checkpoint_receipt("after-rename")
        self.assertEqual(receipt["checkpoint_receipt"], value)
        self.assertEqual(reopened.move("branch", 0, self.target, "after-rename",
                                       checkpoint_receipt=value).tip, self.target)

    def test_checkpoint_receipt_is_owner_only_and_missing_operation_is_not_found(self):
        with self.assertRaises(NotFound):
            self.store.checkpoint_receipt("absent-op")
        with self.assertRaises(PermissionError):
            self.store.for_principal("restricted").checkpoint_receipt("absent-op")


if __name__ == "__main__":
    unittest.main()
