"""CPU-only coordinator lifecycle tests; fake models do not qualify GPU recovery."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import Checkpoint, PreparedCheckpoint
from context_time_travel.store import IntegrityError, InvalidOperation, Store


class _Process:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class _Backend:
    def __init__(self, model, events):
        self.model = model
        self.events = events
        self._protocol_failed = False
        self._closed = False
        self.fail_close_without_stop = False
        self.exit_without_closed = False
        self.process = _Process(9000 + model.gpu * 100 + model.generation)

    def work_counters(self):
        return dict(self.model.stats)

    def release_checkpoints(self):
        return 0

    def close(self):
        self.close_calls = getattr(self, 'close_calls', 0) + 1
        self.events.append(('old-or-current-backend-close', self.model.gpu,
                            self.model.generation))
        if self.fail_close_without_stop:
            raise RuntimeError('native child remains live')
        if self.exit_without_closed:
            self.process.returncode = 0
            raise RuntimeError('process exited but backend close did not complete')
        self._closed = True
        self.process.returncode = 0
        if getattr(self.model, 'raise_after_stopped_close', False):
            raise RuntimeError('close reported an error after process exit')


class _Model:
    def __init__(self, directory, gpu, generation, events, all_gpus):
        self.store = Store(directory)
        self.cache = {}
        self.gpu = self.gpu_id = gpu
        self.generation = generation
        self.events = events
        self.profile = dict(model_contract_id='contract', execution_profile_id='profile',
                            execution={'GPU': list(all_gpus), 'eos_tokens': []})
        self.checkpoint_directory = self.store.root / 'checkpoints'
        self.checkpoint_directory.mkdir(exist_ok=True)
        self.stats = {'consumed_tokens': 0, 'forward_tokens': 0,
                      'loaded_file_bytes': 0, 'capture_count': 0}
        self.backend = _Backend(self, events)
        self.run_operations = []
        self.fail_next = False
        self.raise_after_stopped_close = False
        self.save_original = None

    def create(self, ids):
        self.stats['forward_tokens'] += len(ids)
        self.stats['consumed_tokens'] += len(ids)
        snapshot = self.store.create_root('contract', ids, {'status': 'idle'})
        self._image(snapshot)
        return snapshot

    def _image(self, snapshot):
        blob = b'portable-state:' + snapshot.encode()
        history = self.history(snapshot)
        self.cache[snapshot] = Checkpoint('contract', 'profile', len(history), blob,
            (1.0,), Checkpoint.digest(blob, (1.0,)), snapshot)

    def history(self, snapshot):
        return tuple(self.store.lineage(snapshot)[1])

    def prepare_checkpoint(self, snapshot):
        checkpoint = self.cache[snapshot]
        envelope = dict(manifest=dict(snapshot=snapshot, state_bytes=len(checkpoint.blob),
            logits_count=1, payload_sha256=checkpoint.checksum))
        return PreparedCheckpoint(snapshot, checkpoint, json.dumps(envelope).encode())

    def _load_checkpoint(self, snapshot, history):
        envelope = json.loads((self.checkpoint_directory / (snapshot + '.json')).read_bytes())
        digest = envelope['manifest']['payload_sha256']
        raw = (self.checkpoint_directory / (digest + '.state')).read_bytes()
        blob = raw[:-4]
        checkpoint = Checkpoint('contract', 'profile', len(history), blob,
            (1.0,), Checkpoint.digest(blob, (1.0,)), snapshot)
        return checkpoint

    def checkpoint_manifest(self, snapshot):
        return json.loads((self.checkpoint_directory / (snapshot + '.json')).read_bytes())['manifest']

    def validate_checkpoint(self, snapshot):
        self._load_checkpoint(snapshot, self.history(snapshot))
        return self.checkpoint_manifest(snapshot)

    def save_prepared_branch(self, name, revision, prepared, operation):
        if self.save_original is not None:
            return self.save_original(name, revision, prepared, operation)
        checkpoint = prepared.checkpoint
        (self.checkpoint_directory / (checkpoint.checksum + '.state')).write_bytes(
            checkpoint.blob + b'\x00\x00\x80?')
        (self.checkpoint_directory / (prepared.snapshot + '.json')).write_bytes(prepared.envelope)
        head = self.store.move(name, revision, prepared.snapshot, operation)
        return dict(snapshot=prepared.snapshot, branch=asdict(head),
            payload_sha256=checkpoint.checksum, model_contract_id='contract',
            execution_profile_id='profile')

    def run_greedy_request(self, snapshot, ids, count, *, stream, on_token, **_):
        self.run_operations.append(stream)
        if self.backend._protocol_failed:
            raise IntegrityError('Native request protocol failed; restart the backend')
        if self.fail_next:
            self.fail_next = False
            self.backend._protocol_failed = True
            self.stats['forward_tokens'] += len(ids) + 1
            self.stats['consumed_tokens'] += 1
            raise IntegrityError('native allocation failed during decode')
        tokens = list(range(count))
        self.stats['forward_tokens'] += len(ids) + count
        self.stats['consumed_tokens'] += len(ids) + count
        for token in tokens:
            on_token(token)
        target = self.store.append(snapshot, tuple(ids) + tuple(tokens), {'status': 'idle'})
        self._image(target)
        return dict(snapshot=target, tokens=tokens, request_seconds=0.001,
                    generation_seconds=0.001)


class PoisonedWorkerReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.events = []
        self.models = []
        self.factory_calls = []
        self.fail_replacement = False
        self.replacement_gate = None
        self.fail_unproven_close = False
        self.exit_without_closed = False
        self.raise_stopped_close = False
        self.mismatch_replacement = None
        self.malformed_replacement_gpu = False
        self.gpus = (0,)

        def factory(gpu):
            generation = sum(1 for model in self.models if model.gpu == gpu)
            self.factory_calls.append((gpu, generation))
            self.events.append(('factory', gpu, generation))
            if generation and self.replacement_gate is not None:
                entered, release = self.replacement_gate
                entered.set()
                release.wait(3)
            model = _Model(self.temp.name, gpu, generation, self.events, self.gpus)
            if generation and self.fail_replacement:
                raise RuntimeError('replacement factory failed')
            if gpu == 0 and generation == 0 and self.exit_without_closed:
                model.backend.exit_without_closed = True
            if generation and self.mismatch_replacement == 'model':
                model.profile['model_contract_id'] = 'wrong-contract'
            if generation and self.mismatch_replacement == 'profile':
                model.profile['execution_profile_id'] = 'wrong-profile'
            if generation and self.malformed_replacement_gpu:
                model.profile['execution']['GPU'] = object()
            if gpu == 0 and generation == 0 and self.raise_stopped_close:
                model.raise_after_stopped_close = True
            if gpu == 0 and generation == 0 and self.fail_unproven_close:
                original_close = model.backend.close
                calls = [0]

                def close_once_unproven():
                    calls[0] += 1
                    if calls[0] == 1:
                        self.events.append(('unproven-close', gpu, generation))
                        raise RuntimeError('native child exit was not observed')
                    original_close()

                model.backend.close = close_once_unproven
            self.models.append(model)
            return model

        self.factory = factory
        self.config = PoolConfig(gpus=(0,), ram_bytes=8192,
            disk_bytes=65536, min_available_bytes=0)
        self.coordinator = ContextCoordinator(self.temp.name, self.config, factory,
                                               admission_timeout=2)
        self.coordinator.ready(timeout=5)

    def tearDown(self):
        self.coordinator.close()
        self.temp.cleanup()

    def _root_and_branches(self, *branches):
        root_receipt = self.coordinator.create('base', (1, 2), 'create-root').durable.result(5)
        root = root_receipt['snapshot']
        for branch in branches:
            self.coordinator.fork(branch, root)
        return root

    def _wait(self, predicate, timeout=1.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return bool(predicate())

    def test_poisoned_job_fails_once_then_replacement_preserves_publication_retry(self):
        root = self._root_and_branches('fault', 'publication')
        first = self.models[0]
        original_save = first.save_prepared_branch

        def fail_one_publication(name, revision, prepared, operation):
            if operation == 'publication-fails':
                raise OSError('temporary disk failure')
            return original_save(name, revision, prepared, operation)

        first.save_prepared_branch = fail_one_publication
        publication = self.coordinator.submit('publication', (3,), 1, 'publication-fails')
        publication_answer = publication.snapshot.result(5)
        with self.assertRaisesRegex(OSError, 'temporary disk failure'):
            publication.durable.result(5)
        publication_snapshot = publication_answer['snapshot']
        publication_key = self.coordinator._key(first, publication_snapshot)
        retained_before = self.coordinator.pool.get(publication_key)
        self.assertIn('publication-fails', self.coordinator._failure_payloads)

        first.save_prepared_branch = original_save
        old_head = self.coordinator.store.head('fault')
        first.fail_next = True
        failed = self.coordinator.submit('fault', (4,), 4, 'native-failure')
        with self.assertRaisesRegex(IntegrityError, 'native allocation failed during decode'):
            failed.durable.result(5)
        self.assertEqual(self.coordinator.store.head('fault'), old_head)
        self.assertNotIn('native-failure', self.coordinator.store._read_refs()['receipts'])
        self.assertIs(self.coordinator.submit('fault', (4,), 4, 'native-failure'), failed)
        self.assertEqual(first.run_operations.count('native-failure'), 1)

        restarted = self._wait(lambda: len(self.models) == 2)
        self.assertTrue(restarted, 'poisoned worker was not replaced')
        replacement = self.models[1]
        self.assertTrue(first.backend._closed)
        self.assertEqual(first.backend.process.poll(), 0)
        self.assertLess(self.events.index(('old-or-current-backend-close', 0, 0)),
                        self.events.index(('factory', 0, 1)))
        self.assertEqual(first.cache, {}, 'old process-local cache aliases must be discarded')
        self.assertEqual(self.coordinator.pool.get(publication_key), retained_before)
        self.assertIn('publication-fails', self.coordinator._failure_payloads,
                      'replacement must preserve portable failed-publication bytes')

        self.coordinator.retry_publication('publication-fails').result(5)
        self.assertNotIn('publication-fails', self.coordinator._failure_payloads)
        self.assertEqual(first.run_operations.count('publication-fails'), 1)
        following = self.coordinator.submit('fault', (5,), 2, 'after-replacement')
        receipt = following.durable.result(5)
        self.assertEqual(receipt['branch']['tip'], following.snapshot.result(5)['snapshot'])
        self.assertEqual(replacement.run_operations, ['after-replacement'])
        self.assertEqual(self.coordinator.store.head('fault').root, root)

    def test_failed_replacement_marks_worker_unavailable_and_new_request_fails_promptly(self):
        self._root_and_branches('fault')
        self.fail_replacement = True
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda: len(self.factory_calls) >= 2),
                        'replacement factory was never attempted')

        next_request = self.coordinator.submit('fault', (4,), 1, 'after-failed-replacement')
        with self.assertRaisesRegex(InvalidOperation, 'worker unavailable'):
            next_request.durable.result(1)
        self.assertEqual(self.coordinator.metrics()['queued'], 0)
        self.assertTrue(next_request.durable.done(), 'request may not hang behind dead worker')

    def test_request_queued_during_replacement_fails_if_factory_fails(self):
        self._root_and_branches('fault')
        entered, release = threading.Event(), threading.Event()
        self.replacement_gate = (entered, release)
        self.fail_replacement = True
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(entered.wait(2))
        queued = self.coordinator.submit('fault', (4,), 1, 'queued-during-replacement')
        self.assertFalse(queued.durable.done())
        release.set()
        with self.assertRaisesRegex(InvalidOperation, 'worker unavailable'):
            queued.durable.result(2)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'))
        self.assertEqual(self.coordinator.metrics()['queued'], 0)
        self.assertEqual(self.models[0].run_operations.count('native-failure'), 1)

    def test_close_during_replacement_does_not_deadlock_or_start_unowned_child(self):
        self._root_and_branches('fault')
        entered, release = threading.Event(), threading.Event()
        self.replacement_gate = (entered, release)
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(entered.wait(1.5), 'worker never entered replacement startup')
        close_error = []
        close_thread = threading.Thread(target=lambda: self._close_capture(close_error))
        close_thread.start()
        try:
            release.set()
            close_thread.join(3)
            self.assertFalse(close_thread.is_alive(), 'close waited indefinitely for replacement')
            self.assertEqual(close_error, [])
            self.assertEqual(len(self.models), 2)
            self.assertTrue(self.models[1].backend._closed)
        finally:
            release.set()
            close_thread.join(3)

    def test_close_failure_without_proven_process_exit_does_not_spawn_replacement(self):
        self.fail_unproven_close = True
        self._root_and_branches('fault')
        old = self.models[0]
        old.backend.fail_close_without_stop = True
        old.fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'),
            repr(self.coordinator._workers[0]))
        self.assertEqual(len(self.factory_calls), 1)
        self.assertFalse(old.backend._closed)
        self.assertIsNone(old.backend.process.poll())

    def test_process_exit_without_backend_closed_marker_is_not_sufficient(self):
        self._root_and_branches('fault')
        old = self.models[0]
        old.backend.exit_without_closed = True
        old.fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'),
            repr(self.coordinator._workers[0]))
        self.assertEqual(len(self.factory_calls), 1)
        self.assertFalse(old.backend._closed)
        self.assertEqual(old.backend.process.poll(), 0)

    def test_close_exception_after_proven_exit_still_allows_replacement(self):
        self._root_and_branches('fault')
        self.models[0].raise_after_stopped_close = True
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda: len(self.models) == 2),
                        'a close exception must not block replacement after exit is proven')
        self.assertEqual(self.models[0].backend.process.poll(), 0)

    def test_mismatched_replacement_model_or_profile_is_unavailable(self):
        for mismatch in ('model', 'profile'):
            with self.subTest(mismatch=mismatch):
                self.coordinator.close()
                self.temp.cleanup()
                self.temp = tempfile.TemporaryDirectory()
                self.events.clear()
                self.models.clear()
                self.factory_calls.clear()
                self.fail_replacement = False
                self.mismatch_replacement = mismatch
                self.coordinator = ContextCoordinator(self.temp.name, self.config, self.factory,
                                                       admission_timeout=2)
                self.coordinator.ready(timeout=5)
                self._root_and_branches('fault')
                self.models[0].fail_next = True
                failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
                with self.assertRaises(IntegrityError):
                    failed.durable.result(5)
                self.assertTrue(self._wait(lambda:
                    self.coordinator._workers[0].get('health') == 'unavailable'))
                with self.assertRaisesRegex(InvalidOperation, 'worker unavailable'):
                    self.coordinator.submit('fault', (4,), 1,
                        'after-mismatch-'+mismatch).durable.result(1)

    def test_malformed_replacement_gpu_is_closed_before_worker_is_unavailable(self):
        self.malformed_replacement_gpu = True
        self._root_and_branches('fault')
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'))
        replacement = self.models[1]
        self.assertTrue(replacement.backend._closed)
        self.assertEqual(replacement.backend.process.poll(), 0)
        self.assertEqual(replacement.backend.close_calls, 1)
        self.assertIs(self.coordinator._workers[0]['model'], self.models[0])

    def test_transition_trace_failure_closes_unpublished_replacement(self):
        self._root_and_branches('fault')
        original_record = self.coordinator.pool.record_transition

        def fail_replaced_transition(event, *args, **kwargs):
            if event == 'native_worker_replaced':
                raise OSError('transition trace unavailable')
            return original_record(event, *args, **kwargs)

        self.coordinator.pool.record_transition = fail_replaced_transition
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'))
        replacement = self.models[1]
        self.assertTrue(replacement.backend._closed)
        self.assertEqual(replacement.backend.process.poll(), 0)
        self.assertEqual(replacement.backend.close_calls, 1)
        self.assertIs(self.coordinator._workers[0]['model'], self.models[0])

    def test_unproven_replacement_exit_remains_owned_until_final_shutdown(self):
        self.malformed_replacement_gpu = True
        factory = self.coordinator._factory

        def factory_with_first_close_failure(gpu):
            model = factory(gpu)
            original_close = model.backend.close
            attempts = []

            def close():
                attempts.append(True)
                if len(attempts) == 1:
                    raise RuntimeError('replacement child remains live')
                return original_close()

            model.backend.close = close
            return model

        self.coordinator._factory = factory_with_first_close_failure
        self._root_and_branches('fault')
        self.models[0].fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'))
        replacement = self.models[1]
        self.assertIsNone(replacement.backend.process.poll())
        self.assertIs(self.coordinator._workers[0]['model'], self.models[0])
        self.coordinator.close()
        self.assertTrue(replacement.backend._closed)
        self.assertEqual(replacement.backend.process.poll(), 0)
        self.assertEqual(len(self.factory_calls), 2)

    def _complete_receipt_then_poison_worker(self, *, replacement_fails=False,
                                              replacement_gate=None):
        self._root_and_branches('fault')
        original = self.coordinator.submit('fault', (5,), 1, 'already-durable')
        receipt = original.durable.result(5)
        model = self.models[0]
        before_operations = list(model.run_operations)
        self.fail_replacement = replacement_fails
        self.replacement_gate = replacement_gate
        model.fail_next = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        return model, receipt, before_operations

    def test_durable_receipt_recovers_while_replacement_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        model, receipt, before_operations = self._complete_receipt_then_poison_worker(
            replacement_gate=(entered, release))
        try:
            self.assertTrue(entered.wait(2))
            with self.coordinator._condition:
                del self.coordinator._requests['already-durable']
            recovered = self.coordinator.submit('fault', (5,), 1, 'already-durable')
            self.assertEqual(recovered.durable.result(2), receipt)
            self.assertEqual(model.run_operations, before_operations + ['native-failure'])
            worker_status = next(iter(self.coordinator.metrics()['workers'].values()))
            self.assertEqual(worker_status['health'], 'recovering')
        finally:
            release.set()

    def test_durable_receipt_recovers_after_replacement_failure(self):
        model, receipt, before_operations = self._complete_receipt_then_poison_worker(
            replacement_fails=True)
        self.assertTrue(self._wait(lambda:
            self.coordinator._workers[0].get('health') == 'unavailable'))
        with self.coordinator._condition:
            del self.coordinator._requests['already-durable']
        recovered = self.coordinator.submit('fault', (5,), 1, 'already-durable')
        self.assertEqual(recovered.durable.result(2), receipt)
        self.assertEqual(model.run_operations, before_operations + ['native-failure'])
        self.assertIs(self.coordinator._workers[0]['model'], model,
                      'retired model metadata remains available for receipt recovery')

    def test_healthy_gpu_continues_when_poisoned_peer_replacement_fails(self):
        self.coordinator.close()
        self.models.clear()
        self.factory_calls.clear()
        self.events.clear()
        self.gpus = (0, 1)
        self.config = PoolConfig(gpus=self.gpus, ram_bytes=8192,
            disk_bytes=65536, min_available_bytes=0)
        self.coordinator = ContextCoordinator(self.temp.name, self.config, self.factory,
                                               admission_timeout=2)
        self.coordinator.ready(timeout=5)
        root = self.coordinator.create('base', (1, 2), 'create-root').durable.result(5)['snapshot']
        self.coordinator.fork('fault', root)
        self.coordinator.fork('healthy', root)
        model0 = next(model for model in self.models if model.gpu == 0)
        model0.fail_next = True
        self.fail_replacement = True
        failed = self.coordinator.submit('fault', (3,), 2, 'native-failure')
        with self.assertRaises(IntegrityError):
            failed.durable.result(5)
        self.assertTrue(self._wait(lambda: len(self.factory_calls) >= 3))

        healthy = self.coordinator.submit('healthy', (4,), 1, 'healthy-peer-request')
        receipt = healthy.durable.result(5)
        self.assertEqual(receipt['branch']['tip'], healthy.snapshot.result(5)['snapshot'])
        model1 = next(model for model in self.models if model.gpu == 1)
        self.assertEqual(model1.run_operations, ['healthy-peer-request'])
        self.assertEqual(model0.run_operations.count('native-failure'), 1)

    def _close_capture(self, errors):
        try:
            self.coordinator.close()
        except BaseException as error:
            errors.append(error)


if __name__ == '__main__':
    unittest.main()
