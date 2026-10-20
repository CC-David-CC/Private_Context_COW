"""Authenticated status endpoint tests; no GPU inference qualification."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig


class StatusCoordinator:
    def __init__(self):
        self.metrics_calls = 0
        self.forward_calls = 0
        self.canonical_mutations = 0
        self.failure = None
        self.expected = {
            'active_per_gpu': {'0': 0},
            'max_active_per_gpu': {'0': 1},
            'queued': 0,
            'ram': {'resident_bytes': 123, 'byte_limit': 456},
            'disk': {'file_bytes': 78, 'byte_limit': 90},
        }

    def metrics(self):
        self.metrics_calls += 1
        if self.failure is not None:
            raise self.failure
        return dict(self.expected)


class MetricsOnlyModel:
    """Synthetic worker metadata; no backend inference methods are called."""
    backend = None
    profile = {
        'model_contract_id': 'status-test-model',
        'execution_profile_id': 'status-test-execution',
        'execution': {'GPU': [0]},
    }

    def __init__(self):
        self.backend = self

    def close(self):
        pass


class CowStatusRealCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.coordinator = ContextCoordinator(
            self.temp.name,
            PoolConfig(gpus=(0,), ram_bytes=4096, disk_bytes=8192,
                       min_available_bytes=0),
            lambda gpu: MetricsOnlyModel(),
        )
        self.coordinator.ready(timeout=5)
        self.adapter = CowAdapter(self.coordinator)
        self.server = make_server(self.adapter, api_key='secret')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.coordinator.close()
        self.temp.cleanup()

    def test_http_snapshot_matches_real_metrics_without_canonical_or_forward_work(self):
        metrics_before = self.coordinator.metrics()
        branches_before = self.coordinator.store.branches()
        request = urllib.request.Request(
            self.base + '/cow/status',
            headers={'Authorization': 'Bearer secret'},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read().decode()),
                             json.loads(json.dumps(metrics_before)))
        self.assertEqual(self.coordinator.metrics(), metrics_before)
        self.assertEqual(self.coordinator.store.branches(), branches_before)
        self.assertEqual(branches_before, ())
        self.assertEqual(metrics_before['queued'], 0)
        self.assertEqual(metrics_before['active_per_gpu'], {0: 0})


class CowStatusHTTPTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = StatusCoordinator()
        self.adapter = CowAdapter(self.coordinator)
        self.server = make_server(self.adapter, api_key='secret')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def call(self, path, *, key=None, origin=None):
        headers = {}
        if key is not None:
            headers['Authorization'] = 'Bearer ' + key
        if origin is not None:
            headers['Origin'] = origin
        request = urllib.request.Request(self.base + path, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read().decode()

    def test_auth_and_origin_are_checked_before_metrics_read(self):
        self.assertEqual(self.call('/cow/status')[0], 401)
        self.assertEqual(self.call('/cow/status', key='secret',
                                   origin='http://attacker.example')[0], 403)
        self.assertEqual(self.coordinator.metrics_calls, 0)

    def test_status_returns_actual_metrics_without_forward_or_store_mutation(self):
        code, body = self.call('/cow/status', key='secret',
                               origin=self.base)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body), self.coordinator.expected)
        self.assertEqual(self.coordinator.metrics_calls, 1)
        self.assertEqual(self.coordinator.forward_calls, 0)
        self.assertEqual(self.coordinator.canonical_mutations, 0)

    def test_status_rejects_all_query_parameters_before_metrics_read(self):
        self.assertEqual(self.call('/cow/status?anything=1', key='secret')[0], 400)
        self.assertEqual(self.coordinator.metrics_calls, 0)

    def test_metrics_failure_is_not_reported_as_healthy_status(self):
        self.coordinator.failure = RuntimeError('metrics unavailable')
        code, body = self.call('/cow/status', key='secret')
        self.assertEqual(code, 409)
        self.assertEqual(json.loads(body), {'error': 'metrics unavailable'})
        self.assertEqual(self.coordinator.metrics_calls, 1)


if __name__ == '__main__':
    unittest.main()
