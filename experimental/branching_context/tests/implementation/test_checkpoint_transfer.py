"""CPU byte transport and authority checks; these are not native restore evidence."""
from concurrent.futures import CancelledError
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from context_time_travel.checkpoint_transfer import CheckpointTransfers
from context_time_travel.pools import DiskCheckpointPool, PoolFull, SharedCheckpointPool
from context_time_travel.real import BranchingModel, Checkpoint, PreparedCheckpoint
from context_time_travel.store import Store, IntegrityError, NotFound, _json_bytes


class CheckpointTransferTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name)
        self.source=self.model('source');self.target=self.model('target')
        self.snapshot=self.source.store.create_root('a'*64,(1,2,3),{'status':'idle'})
        self.assertEqual(self.target.store.create_root('a'*64,(1,2,3),{'status':'idle'}),self.snapshot)
        for model in (self.source,self.target):model.store.fork('main',self.snapshot)
        history=self.source.history(self.snapshot)
        cold=Checkpoint.create('a'*64,'b'*64,len(history),b'native-state'*16,(1.,2.,3.,4.),
                               self.source._input_identity(history))
        self.raw=self.source._prepared_envelope(self.snapshot,cold,history)
        self.envelope=json.loads(self.raw)
        self.payload=cold.blob+struct.pack('<4f',*cold.frontier)
        self.source.publish_prepared_checkpoint(PreparedCheckpoint(self.snapshot,cold,self.raw))
        self.outgoing=self.transport(self.source)
        self.incoming=self.transport(self.target)

    def model(self,name):
        model=object.__new__(BranchingModel)
        model.store=Store(self.base/name)
        model.profile=dict(model_contract_id='a'*64,execution_profile_id='b'*64,execution={})
        model.backend=SimpleNamespace(vocab_size=4)
        model.cache={}
        model.checkpoint_directory=model.store.root/'checkpoints';model.checkpoint_directory.mkdir()
        model._checkpoint_fault=lambda *args:None
        return model

    def transport(self,model,*,disk_bytes=16384,ram_bytes=4096,max_concurrent=2):
        ram=SharedCheckpointPool(ram_bytes);self.addCleanup(ram.close)
        disk=DiskCheckpointPool(model.store,disk_bytes)
        return CheckpointTransfers(ram,disk,chunk_bytes=64,max_bytes=8192,
                                   max_concurrent=max_concurrent,timeout=2)

    def canonical(self,model):
        return {str(p.relative_to(model.store.root)):p.read_bytes()
                for directory in ('objects','references')
                for p in (model.store.root/directory).glob('*.json')}

    def receive(self,raw=None,**kw):
        return self.incoming.receive(self.target,self.snapshot,self.envelope,
                                     io.BytesIO(self.payload if raw is None else raw),**kw)

    def assert_released(self,transfer):
        self.assertEqual(transfer.metrics()['active'],0)
        self.assertEqual(transfer.pool.metrics()['reserved_bytes'],0)
        self.assertEqual(transfer.disk.metrics()['reserved_bytes'],0)
        self.assertEqual(list(transfer.disk.directory.glob('.stage-*')),[])

    def test_streamed_copy_and_duplicate_preserve_canonical_state(self):
        before=self.canonical(self.target)
        with self.outgoing.export(self.source,self.snapshot,'a'*64,'b'*64) as offered:
            result=self.incoming.receive(self.target,self.snapshot,offered.envelope,offered.stream)
        self.assertEqual(result['transferred_bytes'],len(self.payload))
        self.assertEqual(self.target.validate_checkpoint(self.snapshot),self.envelope['manifest'])
        self.assertEqual(self.canonical(self.target),before)
        self.assertEqual(self.receive()['transferred_bytes'],0)
        self.assert_released(self.outgoing);self.assert_released(self.incoming)

    def test_corrupt_short_and_extra_streams_do_not_publish(self):
        for payload in (b'X'+self.payload[1:],self.payload[:-1],self.payload+b'extra'):
            with self.subTest(length=len(payload)):
                with self.assertRaises(IntegrityError):self.receive(payload)
                self.assertFalse((self.target.checkpoint_directory/(self.snapshot+'.json')).exists())
                self.assertEqual(list(self.target.checkpoint_directory.glob('*.state')),[])
                self.assert_released(self.incoming)

    def test_nonfinite_frontier_rejected_even_with_valid_hash(self):
        payload=self.payload[:-4]+struct.pack('<f',float('nan'))
        self.envelope['manifest']['payload_sha256']=hashlib.sha256(payload).hexdigest()
        self.envelope['checksum']=hashlib.sha256(_json_bytes(self.envelope['manifest'])).hexdigest()
        with self.assertRaises(IntegrityError):self.receive(payload)
        self.assert_released(self.incoming)

    def test_wrong_execution_rejected_before_reading_payload(self):
        self.target.profile['execution_profile_id']='c'*64
        class Unreadable:
            def read(self,size):raise AssertionError('incompatible data was read')
        with self.assertRaises(IntegrityError):
            self.incoming.receive(self.target,self.snapshot,self.envelope,Unreadable())
        self.assert_released(self.incoming)

    def test_unknown_history_cannot_be_imported_by_a_cache_peer(self):
        empty=self.model('empty');incoming=self.transport(empty)
        with self.assertRaises(NotFound):
            incoming.receive(empty,self.snapshot,self.envelope,io.BytesIO(self.payload))
        self.assertEqual(empty.store.branches(),())
        self.assertEqual(list(empty.checkpoint_directory.iterdir()),[])
        self.assert_released(incoming)

    def test_disk_and_ram_admission_precede_stream_reads(self):
        class Unreadable:
            def read(self,size):raise AssertionError('unreserved data was read')
        for kwargs in (dict(disk_bytes=64),dict(ram_bytes=64)):
            incoming=self.transport(self.target,**kwargs)
            with self.assertRaises(PoolFull):
                incoming.receive(self.target,self.snapshot,self.envelope,Unreadable())
            self.assert_released(incoming)

    def test_export_checks_identity_and_source_checksum(self):
        with self.assertRaises(IntegrityError):
            with self.outgoing.export(self.source,self.snapshot,'a'*64,'c'*64):self.fail('bad identity')
        path=self.source.checkpoint_directory/(self.envelope['manifest']['payload_sha256']+'.state')
        path.write_bytes(b'X'+self.payload[1:])
        with self.assertRaises(IntegrityError):
            with self.outgoing.export(self.source,self.snapshot,'a'*64,'b'*64):self.fail('corrupt source')
        self.assert_released(self.outgoing)

    def test_export_pins_source_against_collection_and_bounds_concurrency(self):
        outgoing=self.transport(self.source,max_concurrent=1)
        # A cache-only source has no branch pin; the transfer owns its read lease.
        snapshot=self.source.store.append(self.snapshot,(4,),{'status':'idle'})
        history=self.source.history(snapshot)
        cold=Checkpoint.create('a'*64,'b'*64,len(history),self.payload[:-16],(1.,2.,3.,4.),
                               self.source._input_identity(history))
        self.source.publish_prepared_checkpoint(PreparedCheckpoint(snapshot,cold,
            self.source._prepared_envelope(snapshot,cold,history)))
        self.assertNotIn(snapshot,self.source.store.checkpoint_roots())
        with outgoing.export(self.source,snapshot,'a'*64,'b'*64) as offered:
            outgoing.disk._collect_for(outgoing.disk.byte_limit)
            self.assertEqual(offered.stream.read(),self.payload)
            with self.assertRaises(PoolFull):
                with outgoing.export(self.source,snapshot,'a'*64,'b'*64):self.fail('slot overrun')
        self.assert_released(outgoing)

    def test_cancelled_mid_stream_removes_staging_and_releases_reservations(self):
        outer=self
        class Reader(io.BytesIO):
            def read(self,size):
                self.asserted=True
                outer.assertLessEqual(size,64)
                return super().read(size)
        reader=Reader(self.payload);reader.asserted=False
        with self.assertRaises(CancelledError):
            self.incoming.receive(self.target,self.snapshot,self.envelope,reader,
                                  cancelled=lambda:reader.asserted)
        self.assertEqual(list(self.target.checkpoint_directory.glob('*.state')),[])
        self.assert_released(self.incoming)

    def test_existing_corruption_is_rejected_not_silently_overwritten(self):
        self.receive()
        path=self.target.checkpoint_directory/(self.envelope['manifest']['payload_sha256']+'.state')
        path.write_bytes(b'corrupt')
        with self.assertRaises(IntegrityError):self.receive()
        self.assertEqual(path.read_bytes(),b'corrupt')
        self.assert_released(self.incoming)

    def test_size_cap_rejects_untrusted_extent_before_stream_reads(self):
        self.envelope['manifest']['state_bytes']=16384
        self.envelope['checksum']=hashlib.sha256(_json_bytes(self.envelope['manifest'])).hexdigest()
        with self.assertRaises(PoolFull):self.receive()
        self.assert_released(self.incoming)


if __name__=='__main__':unittest.main()
