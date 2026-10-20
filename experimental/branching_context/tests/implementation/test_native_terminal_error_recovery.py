"""CPU-only contract tests for terminal native request errors.

These tests exercise the Python protocol adapter with scripted pipe frames. They
do not claim that a CUDA allocation failure has been reproduced.
"""
import io
import threading
from types import SimpleNamespace
import unittest

from context_time_travel.execution_receipt import tokens_digest
from context_time_travel.store import IntegrityError
from context_time_travel.strata_backend import StrataBackend, _Lease


class NativeTerminalErrorRecoveryTests(unittest.TestCase):
    def backend(self, responses):
        backend = object.__new__(StrataBackend)
        backend.profile = {'execution': {
            'native_request_protocol': 'consumed-greedy-ack-v1',
            'generated_operator': 'normal-verifier-decode',
            'capacity': 64,
            'eos_tokens': [7],
        }}
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

    def test_native_failure_before_first_token_preserves_reason_and_poisons_pipe(self):
        reason = 'CUDA allocation failed: out of memory'
        backend, lease = self.backend([{'status': 'error', 'message': reason}])
        backend.last_execution_receipt = {'sentinel': 'prior receipt'}
        request = backend.generate_request(lease, 8)

        with self.assertRaises(IntegrityError) as caught:
            next(request)

        self.assertEqual(str(caught.exception), reason)
        self.assertTrue(backend._protocol_failed)
        self.assertIsNone(backend._request_active)
        self.assertTrue(request.closed)
        self.assertEqual(lease.consumed, 4)
        self.assertEqual(backend.last_execution_receipt, {'sentinel': 'prior receipt'})
        with self.assertRaisesRegex(IntegrityError, 'protocol failed'):
            backend._request('stats')
        self.assertEqual(backend.process.stdin.getvalue().splitlines(), ['generate 8 1 7'])

    def test_native_failure_after_consumed_token_keeps_boundary_and_never_captures(self):
        reason = 'CUDA allocation failed during decode'
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5},
            {'status': 'error', 'message': reason},
        ])
        backend.last_execution_receipt = {'tokens_sha256': tokens_digest([1])}
        request = backend.generate_request(lease, 8)
        self.assertEqual(next(request), 2)

        with self.assertRaises(IntegrityError) as caught:
            next(request)

        self.assertEqual(str(caught.exception), reason)
        self.assertTrue(backend._protocol_failed)
        self.assertIsNone(backend._request_active)
        self.assertTrue(request.closed)
        self.assertEqual(lease.consumed, 5)
        self.assertEqual(backend.last_execution_receipt, {'tokens_sha256': tokens_digest([1])})
        # Model the production stats helper's first native command in export();
        # a poisoned pipe must reject it before the fake exporter can proceed.
        backend._refresh_stats = lambda: backend._request('stats')
        with self.assertRaisesRegex(IntegrityError, 'protocol failed'):
            backend.export(lease)
        self.assertEqual(backend.process.stdin.getvalue().splitlines(),
                         ['generate 8 1 7', 'advance'])
        self.assertNotIn('capture', backend.process.stdin.getvalue())

    def test_malformed_terminal_error_frames_fail_closed(self):
        for frame in (
            {'status': 'error'},
            {'status': 'error', 'message': ''},
            {'status': 'error', 'message': 27},
            {'status': 'error', 'message': 'failure', 'consumed': 5},
        ):
            with self.subTest(frame=frame):
                backend, lease = self.backend([frame])
                with self.assertRaises(IntegrityError):
                    next(backend.generate_request(lease, 1))
                self.assertTrue(backend._protocol_failed)
                self.assertIsNone(backend._request_active)
                self.assertEqual(lease.consumed, 4)
                self.assertEqual(backend.process.stdin.getvalue().splitlines(),
                                 ['generate 1 1 7'])

    def test_native_error_after_a_token_does_not_create_a_completion_receipt(self):
        backend, lease = self.backend([
            {'status': 'token', 'token': 2, 'consumed': 5},
            {'status': 'error', 'message': 'request stopped'},
        ])
        with self.assertRaises(IntegrityError):
            list(backend.generate_request(lease, 2))
        self.assertFalse(hasattr(backend, 'last_native_request'))
        self.assertNotEqual(lease.consumed, 6)

    def test_invalid_success_completion_still_poisons_pipe(self):
        backend, lease = self.backend([{
            'status': 'ok', 'generated': 0, 'consumed': 4,
            'forward_tokens': 4, 'operator': 'unexpected-operator',
            'position': 4, 'count': 0, 'chunks': [],
        }])
        with self.assertRaises(IntegrityError):
            next(backend.generate_request(lease, 1))
        self.assertTrue(backend._protocol_failed)
        self.assertIsNone(backend._request_active)
        self.assertFalse(hasattr(backend, 'last_native_request'))


if __name__ == '__main__':
    unittest.main()
