"""Supplemental cancellation-boundary tests; no GPU/native model qualification."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
import hashlib
import json
import threading
import unittest

import test_durable_restore as fixtures
from context_time_travel.execution_receipt import tokens_digest, schedule_digest
from context_time_travel.store import _json_bytes


def event(operator, position, tokens):
    return dict(operator=operator, position=position, count=len(tokens), tokens_sha256=tokens_digest(tokens),
                chunks=[[position, len(tokens)]] if operator == 'batched-prefill'
                else [[position+i, 1] for i in range(len(tokens))])


class ResidentRequestCancellationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DurableRestoreControlTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model, self.backend, self.sid = self.fixture.model, self.fixture.backend, self.fixture.sid
        self.fixture.persist()
        self.model.profile['execution'] = dict(resume_contract='identical-native-operators-v1',
            prefill_chunk=8, capacity=32, eos_tokens=[1], generated_operator='normal-verifier-decode',
            native_request_protocol='consumed-greedy-ack-v1')
        self.model._allocation_fault = lambda *args: None
        self.model._lease_fault = lambda *args: None
        path = self.model.checkpoint_directory/(self.sid+'.json')
        envelope = json.loads(path.read_bytes())
        schedule = [event('batched-prefill', 0, [0, 1])]
        envelope['manifest'].update(schema=3, execution_schedule=schedule,
                                    schedule_sha256=schedule_digest(schedule))
        envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
        path.write_bytes(_json_bytes(envelope))
        self.backend.piece = lambda token: bytes([65+token])
        self.backend.is_eog = lambda token: token == 1
        self.backend.captures = 0
        self.backend.restores = 0
        self.backend.native_cancels = 0
        original_restore = self.backend.restore

        def restore(*args, **kwargs):
            self.backend.restores += 1
            return original_restore(*args, **kwargs)

        self.backend.restore = restore

        def consume(lease, tokens):
            self.backend.last_execution_receipt = event('batched-prefill', self.backend.count, tokens)
            self.backend.count += len(tokens)
            self.backend.stats['consumed_tokens'] += len(tokens)

        self.backend.consume = consume

        @contextmanager
        def generate(lease, limit):
            start = self.backend.count

            def outputs():
                for token in [0, 1][:limit]:
                    self.backend.count += 1
                    self.backend.stats['consumed_tokens'] += 1
                    yield token
                self.backend.last_execution_receipt = event('normal-verifier-decode', start, [0, 1][:limit])

            try:
                yield outputs()
            except CancelledError:
                # Mirrors the native request context manager's cancel/drain at
                # the currently consumed token boundary.
                self.backend.native_cancels += 1
                raise

        self.backend.generate_request = generate
        self.backend.export = self._export
        self.backend.park = lambda lease: None

    def _export(self, lease):
        self.backend.captures += 1
        return b'complete-native-state'

    def test_cancel_before_start_does_no_restore_or_canonical_write(self):
        objects = set(self.model.store.objects.iterdir())
        source = self.model.snapshot_bytes(self.sid)
        with self.assertRaises(CancelledError):
            self.model.run_greedy_request(self.sid, [0], 2, stream='cancel-before',
                                          cancelled=lambda: True)
        self.assertEqual(self.backend.restores, 0)
        self.assertEqual(self.backend.count, 0)
        self.assertEqual(self.backend.captures, 0)
        self.assertEqual(set(self.model.store.objects.iterdir()), objects)
        self.assertEqual(self.model.snapshot_bytes(self.sid), source)

    def test_cancel_after_consumed_progress_cancels_request_without_snapshot_then_recovers(self):
        objects = set(self.model.store.objects.iterdir())
        source = self.model.snapshot_bytes(self.sid)
        cancel, checks, progress = threading.Event(), [], []

        def cancelled():
            checks.append(True)
            return cancel.is_set()

        def on_token(token):
            progress.append((token, self.backend.count))
            cancel.set()

        with self.assertRaises(CancelledError):
            self.model.run_greedy_request(self.sid, [0], 2, stream='cancel-after-one',
                                          on_token=on_token, cancelled=cancelled)
        self.assertEqual(progress, [(0, 4)])
        self.assertGreaterEqual(len(checks), 2)
        self.assertEqual(self.backend.native_cancels, 1)
        self.assertEqual(self.backend.captures, 0)
        self.assertEqual(set(self.model.store.objects.iterdir()), objects)
        self.assertEqual(self.model.snapshot_bytes(self.sid), source)
        cached = self.model.cache[self.sid]
        self.assertEqual(cached.consumed, 2)
        self.assertEqual(cached.frontier, (1.0, 2.0))
        self.assertEqual(cached.checksum, type(cached).digest(cached.blob, cached.frontier))

        result = self.model.run_greedy_request(self.sid, [0], 2, stream='valid-after-cancel')
        self.assertEqual(tuple(result['tokens']), (0, 1))
        self.assertEqual(self.model.history(result['snapshot']), (0, 1, 0, 0, 1))
        self.assertEqual(self.backend.captures, 1)
        self.assertEqual(self.model.snapshot_bytes(self.sid), source)


if __name__ == '__main__':
    unittest.main()
