"""Durable checkpoint-receipt contract tests; synthetic payloads, no GPU claims."""
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import test_prepared_checkpoint as fixtures
from context_time_travel.store import IntegrityError, NotFound, Store, _hash, _json_bytes


def _new_case(test):
    case = fixtures.PreparedCheckpointTests()
    case.setUp()
    test.addCleanup(case.tmp.cleanup)
    return case


class CheckpointReceiptContractTests(unittest.TestCase):
    def assert_checkpoint_receipt(self, store, operation, prepared):
        receipt = store.checkpoint_receipt(operation)
        expected = {
            'schema': 1,
            'snapshot': prepared.snapshot,
            'payload_sha256': prepared.checkpoint.checksum,
            'model_contract_id': prepared.checkpoint.model_contract_id,
            'execution_profile_id': prepared.checkpoint.execution_profile_id,
        }
        self.assertEqual(receipt['checkpoint_receipt'], expected)
        self.assertEqual(receipt['digest'], _hash(_json_bytes(expected)))
        self.assertEqual(receipt['result']['tip'], prepared.snapshot)
        return receipt

    def _publish(self, case, prepared, operation='save'):
        case.model.store.fork('main', case.sid)
        return case.model.save_prepared_branch('main', 0, prepared, operation)

    def test_exact_checkpoint_metadata_is_atomic_and_survives_restart(self):
        case = _new_case(self)
        prepared = case.prepared()
        result = self._publish(case, prepared)
        receipt = self.assert_checkpoint_receipt(case.model.store, 'save', prepared)

        branch = case.model.store._read_branch('main')
        event = next(row for row in branch['events'] if row.get('operation_id') == 'save')
        raw_receipt = branch['receipts']['save']
        self.assertEqual(event['checkpoint_receipt_digest'], receipt['digest'])
        self.assertEqual(raw_receipt['checkpoint_receipt'], receipt['checkpoint_receipt'])
        self.assertEqual(raw_receipt['result']['tip'], result['snapshot'])

        restarted = Store(case.model.store.root)
        restarted_receipt = self.assert_checkpoint_receipt(restarted, 'save', prepared)
        self.assertEqual(restarted_receipt, receipt)
        payload = case.model.checkpoint_directory / (prepared.checkpoint.checksum + '.state')
        self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), prepared.checkpoint.checksum)
        self.assertEqual(restarted.head('main').tip, prepared.snapshot)

    def test_missing_or_altered_atomic_receipt_metadata_is_detected_after_restart(self):
        mutations = (
            ('missing_descriptor', lambda receipt, event: receipt.pop('checkpoint_receipt')),
            ('altered_profile', lambda receipt, event: receipt['checkpoint_receipt'].__setitem__('execution_profile_id', 'other')),
            ('altered_event_digest', lambda receipt, event: event.__setitem__('checkpoint_receipt_digest', '0' * 64)),
        )
        for label, mutate in mutations:
            with self.subTest(mutation=label):
                case = _new_case(self)
                prepared = case.prepared()
                self._publish(case, prepared)
                path = case.model.store.references / 'main.json'
                value = json.loads(path.read_bytes())
                receipt = value['receipts']['save']
                event = next(row for row in value['events'] if row.get('operation_id') == 'save')
                mutate(receipt, event)
                path.write_bytes(_json_bytes(value))
                restarted = Store(case.model.store.root)
                with self.assertRaises(IntegrityError):
                    restarted.head('main')

    def test_contract_assertion_detects_test_only_metadata_stripping_mutant(self):
        """COW_STRIP_CHECKPOINT_RECEIPT=1 is used for an intentional RED run."""
        case = _new_case(self)
        prepared = case.prepared()
        original = case.model.store.move

        if os.environ.get('COW_STRIP_CHECKPOINT_RECEIPT') == '1':
            def strip_metadata(*args, **kwargs):
                kwargs.pop('checkpoint_receipt', None)
                return original(*args, **kwargs)
            with patch.object(case.model.store, 'move', side_effect=strip_metadata):
                self._publish(case, prepared)
                # The assertion intentionally fails under the stripped-metadata mutant.
                self.assert_checkpoint_receipt(case.model.store, 'save', prepared)
        else:
            self._publish(case, prepared)
            self.assert_checkpoint_receipt(case.model.store, 'save', prepared)

    def test_legacy_committed_operation_keeps_its_original_digest(self):
        case = _new_case(self)
        prepared = case.prepared()
        case.model.publish_prepared_checkpoint(prepared)
        case.model.store.fork('main', case.sid)
        legacy_head = case.model.store.move('main', 0, prepared.snapshot, 'legacy-save')
        before = case.model.store._read_branch('main')['receipts']['legacy-save']

        result = case.model.save_prepared_branch('main', 0, prepared, 'legacy-save')
        after = case.model.store._read_branch('main')['receipts']['legacy-save']
        self.assertEqual(result['branch']['revision'], legacy_head.revision)
        self.assertEqual(after['digest'], before['digest'])
        self.assertNotIn('checkpoint_receipt', after)
        with self.assertRaises(NotFound):
            case.model.store.checkpoint_receipt('legacy-save')

    def test_logical_store_move_remains_valid_without_checkpoint_metadata(self):
        case = _new_case(self)
        case.model.store.fork('main', case.sid)
        target = case.model.store.append(case.sid, (), {'status': 'idle'})
        moved = case.model.store.move('main', 0, target, 'logical-move')
        receipt = case.model.store._read_branch('main')['receipts']['logical-move']
        self.assertEqual(moved.tip, target)
        self.assertNotIn('checkpoint_receipt', receipt)
        with self.assertRaises(NotFound):
            case.model.store.checkpoint_receipt('logical-move')


if __name__ == '__main__':
    unittest.main(verbosity=2)
