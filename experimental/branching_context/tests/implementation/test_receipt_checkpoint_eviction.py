"""CPU recovery controls: committed receipts survive derived-byte eviction."""
from dataclasses import asdict
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.real import PreparedCheckpoint
from context_time_travel.store import IntegrityError, InvalidOperation, NotFound
import test_root_computation_reuse as fixtures


class CommittedReceiptEvictionTests(unittest.TestCase):
    setUp = fixtures.RootComputationReuseTests.setUp
    create = fixtures.RootComputationReuseTests.create
    wait_for_pool_release = fixtures.RootComputationReuseTests.wait_for_pool_release

    def restart_without_checkpoints(self):
        old = self.coordinator
        old.close()
        for path in (old.store.root/'checkpoints').iterdir():
            self.assertTrue(path.is_file())
            path.unlink()
        self.coordinator = ContextCoordinator(old.store.root, old.config, old._factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)
        model = self.coordinator._workers[0]['model']

        def no_numerical_read(*args, **kwargs):
            self.fail('receipt recovery attempted to read an evicted numerical checkpoint')

        model.checkpoint_manifest = no_numerical_read
        model.validate_checkpoint = no_numerical_read
        return self.coordinator, model

    def test_new_prepared_publication_binds_checkpoint_metadata_atomically(self):
        receipt = self.create('base', (1, 2), 'create')
        stored = self.coordinator.store.checkpoint_receipt('create')
        metadata = stored['checkpoint_receipt']
        self.assertEqual(metadata, dict(schema=1, snapshot=receipt['snapshot'],
            payload_sha256=receipt['payload_sha256'],
            model_contract_id=receipt['model_contract_id'],
            execution_profile_id=receipt['execution_profile_id']))
        self.assertEqual(stored['result'], receipt['branch'])

    def test_retry_after_process_reopen_and_all_numerical_eviction_returns_original_receipt(self):
        original = self.create('base', (1, 2), 'create')
        coordinator, model = self.restart_without_checkpoints()
        request = coordinator.create('base', (1, 2), 'create')
        self.assertEqual(request.durable.result(5), original)
        self.assertTrue(request.snapshot.result(5)['recovered'])
        self.assertEqual(model.backend.stats['forward_tokens'], 0)
        self.assertEqual(list(model.checkpoint_directory.iterdir()), [])

    def test_old_ack_recovery_does_not_rewind_a_newer_branch(self):
        original = self.create('base', (1, 2), 'create')
        old = self.coordinator
        later = old.store.append(original['snapshot'], (3,), {'status': 'idle'})
        advanced = old.store.move('base', original['branch']['revision'], later, 'later-move')
        coordinator, model = self.restart_without_checkpoints()
        receipt = coordinator.create('base', (1, 2), 'create').durable.result(5)
        self.assertEqual(receipt, original)
        self.assertEqual(coordinator.store.head('base'), advanced)
        self.assertEqual(model.backend.stats['forward_tokens'], 0)

    def test_receipt_from_another_execution_profile_is_rejected_without_forward_work(self):
        self.create('base', (1, 2), 'create')
        self.profile = dict(self.profile, execution_profile_id='different-profile')
        coordinator, model = self.restart_without_checkpoints()
        with self.assertRaisesRegex(IntegrityError, '[Pp]rofile|execution'):
            coordinator.create('base', (1, 2), 'create')
        self.assertEqual(model.backend.stats['forward_tokens'], 0)

    def test_strict_restore_still_requires_saved_numerical_bytes(self):
        original = self.create('base', (1, 2), 'create')
        coordinator, model = self.restart_without_checkpoints()
        # This control deliberately exercises the real numerical loader's
        # missing-file rejection, unlike receipt lookup which must not read it.
        model.checkpoint_manifest = type(model).checkpoint_manifest.__get__(model)
        with self.assertRaisesRegex((IntegrityError, InvalidOperation), '[Cc]heckpoint|manifest'):
            model.restore_checkpoint(original['snapshot'])
        self.assertEqual(model.backend.stats['forward_tokens'], 0)

    def test_retry_of_legacy_prepared_operation_keeps_its_original_move_digest(self):
        original = self.create('base', (1, 2), 'create')
        self.wait_for_pool_release()
        coordinator = self.coordinator
        model = coordinator._workers[0]['model']
        snapshot = original['snapshot']
        coordinator.store.fork('legacy', snapshot)
        legacy = coordinator.store.move('legacy', 0, snapshot, 'legacy-operation')
        cold = model._load_checkpoint(snapshot, model.history(snapshot))
        prepared = PreparedCheckpoint(snapshot, cold,
            model._prepared_envelope(snapshot, cold, model.history(snapshot)))
        before = model.backend.stats['forward_tokens']
        retry = model.save_prepared_branch('legacy', 0, prepared, 'legacy-operation')
        self.assertEqual(retry['branch'], asdict(legacy))
        self.assertEqual(model.backend.stats['forward_tokens'], before)
        with self.assertRaises(NotFound):
            coordinator.store.checkpoint_receipt('legacy-operation')


if __name__ == '__main__':
    unittest.main()
