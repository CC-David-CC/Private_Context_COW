"""Protect deduplicated bytes before deciding publication's disk reservation."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import DiskCheckpointPool, PoolConfig
from test_pool_coordinator import Model


class PublicationCapacityPinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = {}
        gate = threading.Event()
        gate.set()
        def factory(gpu):
            model = Model(self.temp.name, gpu, (0,), [], gate)
            self.models[gpu] = model
            return model
        self.coordinator = ContextCoordinator(self.temp.name,
            PoolConfig(gpus=(0,),ram_bytes=1024**2,disk_bytes=1024**2,min_available_bytes=0),
            factory,admission_timeout=.2)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)

    def test_existing_payload_cannot_disappear_between_probe_and_admission(self):
        c = self.coordinator
        model = self.models[0]
        original_prepare = model.prepare_checkpoint
        original_exists = Path.exists
        observed = {}
        collector = DiskCheckpointPool(c.store, c.config.disk_bytes)

        def prepare(sid):
            prepared = original_prepare(sid)
            path = model.checkpoint_directory/(prepared.checkpoint.checksum+'.state')
            path.write_bytes(prepared.checkpoint.blob+b'\0\0\x80?')
            observed.update(path=path,manifest_bytes=len(prepared.envelope))
            return prepared

        def exists(path):
            present = original_exists(path)
            if path == observed.get('path') and 'probe' not in observed:
                observed['probe'] = present
                # Deterministically run a separate pool's collector AFTER the
                # existence result, before the publisher sees that result.
                collector.collect()
                observed['survived_collection'] = path.is_file()
                observed['reserved_bytes_at_probe'] = c.disk.metrics()['reserved_bytes']
            return present

        model.prepare_checkpoint = prepare
        with mock.patch.object(Path, 'exists', exists):
            receipt = c.create('main',(1,2),'create').durable.result(5)
        self.assertTrue(observed['probe'])
        self.assertTrue(observed['survived_collection'],
                        'existing payload was unprotected when its presence reduced required bytes')
        self.assertEqual(observed['reserved_bytes_at_probe'],observed['manifest_bytes'])
        self.assertEqual(c.store.head('main').tip,receipt['snapshot'])
        self.assertEqual(c.disk.metrics()['reserved_bytes'],0)

    def test_missing_payload_has_full_reservation_before_writing(self):
        c = self.coordinator
        model = self.models[0]
        original = model.save_prepared_branch
        observed = []
        def save(name,revision,prepared,operation):
            payload_bytes = len(prepared.checkpoint.blob)+4*len(prepared.checkpoint.frontier)
            expected = len(prepared.envelope)+payload_bytes
            observed.append(c.disk.metrics()['reserved_bytes'])
            self.assertEqual(observed[-1],expected)
            return original(name,revision,prepared,operation)
        model.save_prepared_branch = save
        c.create('main',(1,2),'create').durable.result(5)
        self.assertEqual(len(observed),1)
        self.assertEqual(c.disk.metrics()['reserved_bytes'],0)


if __name__ == '__main__':
    unittest.main()
