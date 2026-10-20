"""HTTP publication error regressions using the deterministic COW FlowModel."""
from concurrent.futures import CancelledError
import errno
import http.client
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.test_cow_presentation_flow import CowPresentationFlowTests, FlowModel


class CowHTTPPublicationErrorTests(unittest.TestCase):
    """The HTTP boundary must report durable writer errors, not hide them."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = {}
        self.trace = []
        gate = threading.Event()
        gate.set()
        self.gate = gate
        config = PoolConfig(gpus=(0,), ram_bytes=1024 * 1024,
                            disk_bytes=8 * 1024 * 1024, min_available_bytes=0)

        def factory(gpu):
            model = FlowModel(self.temp.name, gpu, (0,), self.trace, gate)
            self.models[gpu] = model
            return model

        self.coordinator = ContextCoordinator(self.temp.name, config, factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)
        self.adapter = CowAdapter(self.coordinator)
        self.server = make_server(self.adapter, api_key='secret')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._close_server)
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def _close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def _call(self, path, body):
        request = urllib.request.Request(
            self.base + path,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer secret'},
            data=json.dumps(body).encode('utf-8'))
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()
        except (http.client.HTTPException, OSError):
            # The buggy handler closes the response when a writer OSError is
            # mistaken for a disconnected client. Preserve that as a red test.
            return None, b''

    def _get(self, path):
        request = urllib.request.Request(
            self.base + path,
            headers={'Authorization': 'Bearer secret'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()

    def _exercise_writer_failure(self, error_number, stream):
        branch = f'branch-{error_number}-{int(stream)}'
        operation = f'writer-error-{error_number}-{int(stream)}'
        self.models[0].plans[operation] = (257,)
        self.adapter.create(branch, [11, 12], f'root-{operation}').durable.result(5)
        head_before = self.coordinator.store.head(branch)
        stats_before = self.models[0].work_counters()

        save = self.models[0].save_prepared_branch
        fault_count = [0]
        fault_path = [None]

        def save_with_one_real_writer_fault(name, revision, prepared, op, **kwargs):
            payload = self.models[0].checkpoint_directory / (prepared.checkpoint.checksum + '.state')
            write_bytes = Path.write_bytes

            def fail_payload_write(path, data):
                if Path(path) == payload and fault_count[0] == 0:
                    fault_count[0] += 1
                    fault_path[0] = payload
                    raise OSError(error_number, os.strerror(error_number), str(payload))
                return write_bytes(path, data)

            with mock.patch.object(Path, 'write_bytes', fail_payload_write):
                return save(name, revision, prepared, op, **kwargs)

        self.models[0].save_prepared_branch = save_with_one_real_writer_fault
        response_status, response_body = self._call('/cow/generate', {
            'branch': branch,
            'rendered_token_ids': [11, 12, 13],
            'revision': head_before.revision,
            'operation_id': operation,
            'max_tokens': 1,
            'settings': {'temperature': 0},
            'stream': stream,
        })

        self.assertEqual(fault_count[0], 1, 'the failure must occur at the checkpoint payload writer')
        self.assertEqual(self.coordinator.store.head(branch), head_before,
                         'failed durable publication must not move the branch')
        self.assertIsNotNone(fault_path[0])
        self.assertFalse(fault_path[0].exists(), 'the failed payload write must not publish a partial file')
        status_code, status_body = self._get('/cow/operation?operation=' + operation)
        self.assertEqual(status_code, 200)
        status = json.loads(status_body)
        self.assertEqual(status['state'], 'failed')
        self.assertFalse(status['durable'])
        self.assertTrue(status['publication_retryable'])

        if stream:
            self.assertEqual(response_status, 200)
            events = [json.loads(line) for line in response_body.splitlines()]
            errors = [event for event in events if event['event'] in ('error', 'failure', 'cancelled')]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]['event'], 'error')
            self.assertFalse(errors[0]['cancelled'])
            self.assertEqual(errors[0]['errno'], error_number)
            self.assertNotIn('complete', [event['event'] for event in events])
            reported_error = errors[0]['error']
        else:
            self.assertEqual(response_status, 409)
            error = json.loads(response_body)
            self.assertNotIn('event', error, 'a failed write cannot be acknowledged as complete')
            self.assertEqual(error['errno'], error_number)
            reported_error = error['error']
        self.assertIn(str(error_number), reported_error)
        self.assertIn(os.strerror(error_number), reported_error)

        retry_status, retry_body = self._call('/cow/publication/retry', {'operation_id': operation})
        self.assertEqual(retry_status, 200)
        retry = json.loads(retry_body)
        self.assertEqual(retry['event'], 'complete')
        self.assertEqual(retry['receipt']['branch']['name'], branch)
        self.assertEqual(fault_count[0], 1)
        self.assertEqual(self.models[0].work_counters(), stats_before | {
            'forward_tokens': stats_before['forward_tokens'] + 2,
            'consumed_tokens': stats_before['consumed_tokens'] + 2,
        }, 'publication retry must reuse the prepared request and do no extra model work')

    def test_efbig_stream_reports_writer_cause_without_ack_and_retry_succeeds(self):
        self._exercise_writer_failure(errno.EFBIG, stream=True)

    def test_efbig_nonstream_reports_writer_cause_without_ack_and_retry_succeeds(self):
        self._exercise_writer_failure(errno.EFBIG, stream=False)

    def test_enospc_stream_reports_writer_cause_without_ack_and_retry_succeeds(self):
        self._exercise_writer_failure(errno.ENOSPC, stream=True)

    def test_enospc_nonstream_reports_writer_cause_without_ack_and_retry_succeeds(self):
        self._exercise_writer_failure(errno.ENOSPC, stream=False)

    def test_nonstream_disconnect_cancels_waiting_model_and_handler_exits(self):
        branch = 'disconnect-branch'
        self.adapter.create(branch, [21, 22], 'disconnect-root').durable.result(5)
        model = self.models[0]
        entered = threading.Event()
        observed_cancel = threading.Event()
        original = model.run_greedy_request

        def wait_for_disconnect(snapshot, ids, count, **kwargs):
            entered.set()
            cancelled = kwargs.get('cancelled')
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if callable(cancelled) and cancelled():
                    observed_cancel.set()
                    raise CancelledError()
                time.sleep(.01)
            return original(snapshot, ids, count, **kwargs)

        model.run_greedy_request = wait_for_disconnect
        head = self.coordinator.store.head(branch)
        operation = 'disconnect-nonstream'
        body = json.dumps({
            'branch': branch, 'rendered_token_ids': [21, 22, 23],
            'revision': head.revision, 'operation_id': operation,
            'max_tokens': 1, 'settings': {'temperature': 0}, 'stream': False,
        }).encode('utf-8')
        client = socket.create_connection(self.server.server_address, timeout=5)
        client.sendall(
            b'POST /cow/generate HTTP/1.0\r\n'
            b'Authorization: Bearer secret\r\n'
            b'Content-Type: application/json\r\n'
            b'Content-Length: ' + str(len(body)).encode('ascii') + b'\r\n\r\n' + body)
        self.assertTrue(entered.wait(5), 'request did not enter the worker')
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        self.assertTrue(observed_cancel.wait(5), 'socket disconnect did not cancel the pending request')

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.coordinator.operation_status(operation)
            if status['state'] == 'cancelled':
                break
            time.sleep(.01)
        self.assertEqual(status['state'], 'cancelled')
        self.assertEqual(self.coordinator.store.head(branch), head)
        self.assertEqual(self._get('/cow/head?branch=' + branch)[0], 200,
                         'the HTTP service should remain usable after the disconnected request exits')


if __name__ == '__main__':
    unittest.main()
