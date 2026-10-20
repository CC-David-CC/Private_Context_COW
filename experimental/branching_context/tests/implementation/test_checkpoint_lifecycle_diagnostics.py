from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import time
import unittest
from unittest import mock

from experimental.branching_context.scripts.checkpoint_lifecycle_diagnostics import (
    LifecycleProbe, main, storage_preflight,
)


class LifecycleProbeTests(unittest.TestCase):
    def test_external_storage_gate_reports_shortfall_without_claiming_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            full = storage_preflight(directory, 100,
                disk_usage=lambda path: SimpleNamespace(total=1000, used=950, free=50))
            self.assertEqual(full['available_bytes'], 50)
            self.assertEqual(full['required_free_bytes'], 100)
            self.assertFalse(full['passes'])
            passing = storage_preflight(directory, 50,
                disk_usage=lambda path: SimpleNamespace(total=1000, used=950, free=50))
            self.assertTrue(passing['passes'])
            with self.assertRaises(ValueError):
                storage_preflight(directory, True,
                    disk_usage=lambda path: SimpleNamespace(total=1000, used=0, free=1000))

    def test_runner_phases_partition_worker_lifecycle_without_changing_totals(self):
        from experimental.branching_context.scripts.checkpoint_lifecycle_diagnostics import install_runner_phase_observer

        class RunHeartbeat:
            def update(self, phase, *, root_index=None, repeat=None):
                self.state = (phase, root_index, repeat)
                return self.state

        runner = SimpleNamespace(RunHeartbeat=RunHeartbeat)

        class Target:
            def operation(self):
                return 'ok'

        with LifecycleProbe() as probe:
            install_runner_phase_observer(probe, runner)
            probe.patch(Target, 'operation', 'checkpoint_capture')
            heartbeat = RunHeartbeat()
            self.assertEqual(heartbeat.update('warmup', root_index=1, repeat=-1),
                             ('warmup', 1, -1))
            self.assertEqual(Target().operation(), 'ok')
            self.assertEqual(heartbeat.update('measured_request', root_index=2, repeat=1),
                             ('measured_request', 2, 1))
            worker = threading.Thread(target=Target().operation, name='p4-worker')
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(heartbeat.update('finalizing'), ('finalizing', None, None))
            self.assertEqual(Target().operation(), 'ok')
            result = probe.snapshot()

        self.assertEqual(result['stages']['checkpoint_capture']['calls'], 3)
        self.assertEqual(len(result['phase_stages']), 3)
        warmup = result['phase_stages']['{"phase":"warmup","repeat":-1,"root_index":1}']
        measured = result['phase_stages']['{"phase":"measured_request","repeat":1,"root_index":2}']
        self.assertEqual(warmup['stages']['checkpoint_capture']['calls'], 1)
        self.assertEqual(measured['stages']['checkpoint_capture']['calls'], 1)
        self.assertEqual(result['events'][0]['phase']['phase'], 'warmup')
        self.assertEqual(result['events'][1]['phase']['phase'], 'measured_request')
        self.assertEqual(result['events'][1]['thread_name'], 'p4-worker')
        self.assertEqual(result['events'][2]['phase'], {
            'phase': 'finalizing', 'root_index': 2, 'repeat': 1})

    def test_cli_refuses_diagnostics_inside_source_or_over_existing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.mkdir()
            inside = source/'diagnostics.json'
            with self.assertRaises(SystemExit):
                main(['--source-root', str(source), '--diagnostics', str(inside)])
            self.assertFalse(inside.exists())

            existing = Path(directory)/'existing.json'
            existing.write_text('preserve', encoding='utf-8')
            with self.assertRaises(SystemExit):
                main(['--source-root', str(source), '--diagnostics', str(existing)])
            self.assertEqual(existing.read_text(encoding='utf-8'), 'preserve')

    def test_insufficient_external_disk_preflight_stops_before_runner_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'source'
            (root/'experimental/branching_context/src').mkdir(parents=True)
            (root/'experimental/branching_context/scripts').mkdir(parents=True)
            storage = Path(directory)/'storage'
            storage.mkdir()
            receipt = Path(directory)/'diagnostics.json'
            with mock.patch(
                    'experimental.branching_context.scripts.checkpoint_lifecycle_diagnostics.importlib.import_module',
                    side_effect=AssertionError('must not import P4 or model modules')):
                result = main(['--source-root', str(root), '--diagnostics', str(receipt),
                    '--storage-check-path', str(storage), '--required-free-bytes', str(10**18),
                    '--source-store', 'ignored'])
            self.assertEqual(result, 2)
            document = json.loads(receipt.read_text(encoding='utf-8'))
            self.assertEqual(document['status'], 'PREFLIGHT_REJECTED')
            self.assertFalse(document['storage_preflight']['passes'])
            self.assertEqual(document['measurements']['events'], [])

    def test_installation_observes_named_lifecycle_edges_and_nested_fsync(self):
        class Checkpoint:
            @staticmethod
            def digest(blob, frontier):
                return f'{len(blob)}:{len(frontier)}'

        class Model:
            def _prepared_checkpoint(self, snapshot, saved, blob):
                return Checkpoint.digest(blob, (1.0,))

            def _publish_checkpoint_parts(self, destination, parts):
                with open(destination, 'wb') as output:
                    output.write(b''.join(parts))
                    output.flush()
                    import os
                    os.fsync(output.fileno())

            def _publish_checkpoint_file(self, destination, data):
                return len(data)

            def prepare_checkpoint(self, snapshot):
                return snapshot

            def publish_prepared_checkpoint(self, prepared):
                return prepared

            def validate_checkpoint(self, snapshot):
                return snapshot

            def _load_checkpoint(self, snapshot, history):
                return snapshot, history

        class Backend:
            def _request(self, command):
                return command

            def serialize(self, lease, state):
                return b'abc'

        def pool_digest(checkpoint):
            return str(len(checkpoint.blob))

        class SharedCheckpointPool:
            def admit(self, key, checkpoint, *, durable=False):
                return self._admit(None, key, checkpoint, durable)

            def _admit(self, reservation, key, checkpoint, durable):
                return checkpoint

        real_module = SimpleNamespace(BranchingModel=Model, Checkpoint=Checkpoint)
        backend_module = SimpleNamespace(StrataBackend=Backend)
        pools_module = SimpleNamespace(checkpoint_digest=pool_digest, SharedCheckpointPool=SharedCheckpointPool)

        with LifecycleProbe() as probe:
            from experimental.branching_context.scripts.checkpoint_lifecycle_diagnostics import install_lifecycle_observers
            install_lifecycle_observers(probe, real_module, backend_module, pools_module)
            backend = Backend()
            self.assertEqual(backend._request('write /tmp/export.state'), 'write /tmp/export.state')
            self.assertEqual(backend._request('load /tmp/input.state'), 'load /tmp/input.state')
            self.assertEqual(backend.serialize(None, None), b'abc')
            with tempfile.TemporaryDirectory() as directory:
                export_path = Path(directory)/'export.state'
                export_path.write_bytes(b'state')
                self.assertEqual(export_path.read_bytes(), b'state')
                state_path = Path(directory)/'payload.state'
                self.assertEqual(Model()._publish_checkpoint_parts(state_path, (b'payload',)), None)
                self.assertEqual(state_path.read_bytes(), b'payload')
            self.assertEqual(Model()._prepared_checkpoint('s', None, b'bytes'), '5:1')
            self.assertEqual(Checkpoint.digest(b'blob', (1.0,)), '4:1')
            checkpoint = SimpleNamespace(blob=b'ram', frontier=(0.0,))
            self.assertEqual(pools_module.checkpoint_digest(checkpoint), '3')
            self.assertIs(SharedCheckpointPool().admit('key', checkpoint), checkpoint)
            result = probe.snapshot()

        stages = result['stages']
        for name in ('native_request_write', 'native_request_load', 'backend_serialize',
                     'export_state_readback', 'checkpoint_digest', 'pool_checkpoint_digest',
                     'shared_pool_admit', 'shared_pool_internal_admit', 'prepared_checkpoint',
                     'payload_publish', 'payload_file_fsync'):
            self.assertIn(name, stages)
        self.assertEqual(stages['backend_serialize']['bytes'], 3)
        self.assertEqual(stages['export_state_readback']['bytes'], 5)
        self.assertEqual(stages['payload_publish']['bytes'], 7)
        prepared = next(row for row in result['events'] if row['stage'] == 'prepared_checkpoint')
        digest = next(row for row in result['events'] if row['stage'] == 'checkpoint_digest'
                      and row['parent_call_id'] == prepared['call_id'])
        self.assertEqual(digest['parent_call_id'], prepared['call_id'])

    def test_return_and_exception_identity_are_preserved_and_recorded(self):
        failure = LookupError('original')

        class Target:
            def call(self, value):
                if value == 'raise':
                    raise failure
                return ('result', value)

        with LifecycleProbe() as probe:
            probe.patch(Target, 'call', 'target.call')
            target = Target()
            self.assertEqual(target.call('ok'), ('result', 'ok'))
            with self.assertRaises(LookupError) as caught:
                target.call('raise')
            self.assertIs(caught.exception, failure)
            result = probe.snapshot()

        summary = result['stages']['target.call']
        self.assertEqual(summary['calls'], 2)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual([row['outcome'] for row in result['events']], ['return', 'exception'])
        self.assertEqual(Target.call(Target(), 'after'), ('result', 'after'))

    def test_nested_timing_is_inclusive_and_exclusive_without_double_counting(self):
        class Target:
            def inner(self):
                time.sleep(.005)
                return 7

            def outer(self):
                time.sleep(.003)
                return self.inner()

        with LifecycleProbe() as probe:
            probe.patch(Target, 'inner', 'inner')
            probe.patch(Target, 'outer', 'outer')
            self.assertEqual(Target().outer(), 7)
            events = probe.snapshot()['events']

        outer = next(row for row in events if row['stage'] == 'outer')
        inner = next(row for row in events if row['stage'] == 'inner')
        self.assertGreaterEqual(outer['inclusive_ns'], inner['inclusive_ns'])
        self.assertGreater(outer['exclusive_ns'], 0)
        self.assertEqual(inner['parent_call_id'], outer['call_id'])

    def test_thread_timings_are_separated(self):
        barrier = threading.Barrier(3)

        class Target:
            def call(self):
                barrier.wait(timeout=2)
                time.sleep(.002)

        with LifecycleProbe() as probe:
            probe.patch(Target, 'call', 'threaded')
            threads = [threading.Thread(target=Target().call, name=f'worker-{i}') for i in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)
            result = probe.snapshot()

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        thread_rows = result['stages']['threaded']['threads']
        self.assertEqual(len(thread_rows), 2)
        self.assertEqual({row['thread_name'] for row in thread_rows}, {'worker-0', 'worker-1'})

    def test_event_retention_is_bounded_but_aggregate_counts_continue(self):
        class Target:
            def call(self, size):
                return b'x' * size

            def other(self):
                return 'other'

        with LifecycleProbe(max_events=2, max_stages=1) as probe:
            probe.patch(Target, 'call', 'bounded', byte_count=lambda args, kwargs, result, error: len(result))
            probe.patch(Target, 'other', 'other')
            target = Target()
            for size in (1, 2, 3, 4):
                target.call(size)
            self.assertEqual(target.other(), 'other')
            result = probe.snapshot()

        self.assertEqual(len(result['events']), 2)
        self.assertEqual(result['dropped_events'], 3)
        stage = result['stages']['bounded']
        self.assertEqual(stage['calls'], 4)
        self.assertEqual(stage['bytes'], 10)
        self.assertEqual(result['max_stages'], 1)
        self.assertEqual(result['stages']['<stage-overflow>']['calls'], 1)


if __name__ == '__main__':
    unittest.main()
