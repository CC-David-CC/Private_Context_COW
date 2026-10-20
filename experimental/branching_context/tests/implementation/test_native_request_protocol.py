"""CPU-only pipe/ownership contract tests; these do not qualify GPU execution."""
from concurrent.futures import ThreadPoolExecutor
import io
import threading
from types import SimpleNamespace
import unittest

from context_time_travel.strata_backend import StrataBackend, _Lease
from context_time_travel.store import IntegrityError, InvalidOperation
from context_time_travel.execution_receipt import tokens_digest


class NativeRequestProtocolTests(unittest.TestCase):
    def backend(self, responses):
        backend = object.__new__(StrataBackend)
        backend.profile = {'execution': {'native_request_protocol': 'consumed-greedy-ack-v1',
            'generated_operator': 'normal-verifier-decode', 'capacity': 64, 'eos_tokens': [7]}}
        backend.vocab_size = 8
        backend._closed = False
        backend._protocol_failed = False
        backend.process = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        backend.stats = {'lease_releases': 0}
        backend._lock = SimpleNamespace(release=lambda: None)
        frames = iter(responses)
        backend._read = lambda: next(frames)
        backend._refresh_stats = lambda: {'forward_tokens': 4}
        lease = _Lease(backend, bound=True, consumed=4, state='ACTIVE',
                       _owner_thread=threading.get_ident())
        backend._active = lease
        return backend, lease

    def final(self, count, work=None):
        return {'status': 'ok', 'generated': count, 'consumed': 4+count,
            'forward_tokens': 4+count if work is None else work,
            'operator': 'normal-verifier-decode', 'position': 4, 'count': count,
            'chunks': [[4+i,1] for i in range(count)]}

    def test_completed_receipt_binds_actual_tokens_and_absolute_native_boundaries(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5},
            {'status': 'token', 'token': 3, 'consumed': 6}, self.final(2)])
        self.assertEqual(list(backend.generate_request(lease, 2)), [2,3])
        self.assertEqual(backend.last_execution_receipt, dict(operator='normal-verifier-decode',
            position=4,count=2,chunks=[[4,1],[5,1]],tokens_sha256=tokens_digest([2,3])))

    def test_false_or_missing_native_boundaries_poison_connection(self):
        for field, value in [('position',5),('count',2),('chunks',[[4,2]]),
                             ('chunks',None),('chunks',[[5,1]]),('position',True)]:
            with self.subTest(field=field, value=value):
                response = dict(self.final(1), **{field:value})
                backend, lease = self.backend([
                    {'status': 'token', 'token': 2, 'consumed': 5},response])
                with self.assertRaises(IntegrityError):
                    list(backend.generate_request(lease,1))
                self.assertTrue(backend._protocol_failed)

    def test_one_request_consumes_all_outputs_without_capture_commands(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5},
            {'status': 'token', 'token': 3, 'consumed': 6}, self.final(2)])
        self.assertEqual(list(backend.generate_request(lease, 2)), [2, 3])
        self.assertEqual(lease.consumed, 6)
        self.assertEqual(backend.process.stdin.getvalue().splitlines(),
                         ['generate 2 1 7', 'advance', 'advance'])

    def test_early_close_cancels_before_releasing_consumed_boundary(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5}, self.final(1)])
        request = backend.generate_request(lease, 8)
        self.assertEqual(next(request), 2)
        request.close()
        self.assertEqual(lease.consumed, 5)
        self.assertEqual(backend.process.stdin.getvalue().splitlines(), ['generate 8 1 7', 'cancel'])

    def test_foreign_next_and_close_fail_without_sending_acknowledgment(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5}, self.final(1)])
        request = backend.generate_request(lease, 1)
        with ThreadPoolExecutor(1) as pool:
            for action in (lambda: next(request), request.close):
                with self.assertRaises(InvalidOperation):
                    pool.submit(action).result()
        self.assertEqual(next(request), 2)
        request.close()
        self.assertEqual(backend.process.stdin.getvalue().splitlines(), ['generate 1 1 7', 'cancel'])

    def test_incorrect_forward_work_is_rejected_and_connection_is_poisoned(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5}, self.final(1, work=8)])
        with self.assertRaises(IntegrityError):
            list(backend.generate_request(lease, 1))
        self.assertTrue(backend._protocol_failed)

    def test_lease_cannot_be_released_while_request_is_waiting_for_ack(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5}, self.final(1)])
        request = backend.generate_request(lease, 8)
        self.assertEqual(next(request), 2)
        with self.assertRaises(InvalidOperation):
            lease.__exit__(None, None, None)
        self.assertIs(backend._active, lease)
        self.assertEqual(lease.state, 'ACTIVE')
        request.close()
        lease.__exit__(None, None, None)
        self.assertEqual(backend.stats['lease_releases'], 1)

    def test_precreated_second_iterator_cannot_steal_active_request(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5}, self.final(1)])
        first = backend.generate_request(lease, 8)
        second = backend.generate_request(lease, 8)
        self.assertEqual(next(first), 2)
        with self.assertRaises(InvalidOperation):
            next(second)
        self.assertFalse(backend._protocol_failed)
        first.close()
        self.assertEqual(backend.process.stdin.getvalue().splitlines(), ['generate 8 1 7', 'cancel'])


if __name__ == '__main__':
    unittest.main()
