"""P5 evidence checks: clean RAM eviction followed by same-process disk reload."""
import unittest
from dataclasses import replace
import tempfile
import threading

from pool_campaign import pressure_tree_order, validate_spill_reload_trace
from context_time_travel.real import Checkpoint
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig, PoolTransitionTrace, SharedCheckpointPool
from test_pool_coordinator import Model


def trace(events, *, phase='prepare', pid=71, phase_id='phase-a', overflowed=False):
    return dict(phase=phase, process_id=pid, phase_id=phase_id,
                overflowed=overflowed, event_count=len(events), events=events)


def event(sequence, kind, key, *, phase='prepare', pid=71, phase_id='phase-a', success=None, load_id=None):
    row=dict(sequence=sequence, event=kind, key=key, phase=phase,
             process_id=pid, phase_id=phase_id)
    if kind=='ram_clean_eviction': row['freed_bytes']=8
    if kind=='disk_preload_started': row['load_id']=sequence
    if success is not None:
        row['success']=success
        if kind in ('disk_preload_succeeded','disk_preload_failed'):
            row['load_id']=sequence-1 if load_id is None else load_id
        if kind=='disk_preload_succeeded': row['bytes']=8
    return row


class SpillReloadTraceTests(unittest.TestCase):
    def test_default_order_is_unchanged_and_pressure_order_interleaves_fixture_ids(self):
        self.assertEqual(pressure_tree_order(6, 'default'), list(range(6)))
        self.assertEqual(pressure_tree_order(6, 'interleaved-roots'), [0, 2, 4, 1, 3, 5])

    def test_same_identity_eviction_then_successful_disk_read_is_required(self):
        key=['contract-a', 'profile-a', 'snapshot-abc']
        document=trace([event(0,'ram_clean_eviction',key),
                        event(1,'disk_preload_started',key),
                        event(2,'disk_preload_succeeded',key,success=True)])
        self.assertEqual(validate_spill_reload_trace(document), [dict(key=key,
            eviction_sequence=0,load_start_sequence=1,load_sequence=2,load_id=1,
            freed_bytes=8,loaded_bytes=8)])

    def test_aggregate_without_identity_events_is_not_proof(self):
        with self.assertRaisesRegex(ValueError,'[Nn]o same-process'):
            validate_spill_reload_trace(trace([]))

    def test_wrong_snapshot_key_does_not_match(self):
        with self.assertRaisesRegex(ValueError,'[Nn]o same-process'):
            validate_spill_reload_trace(trace([
                event(0,'ram_clean_eviction',['contract-a','profile-a','snapshot-a']),
                event(1,'disk_preload_started',['contract-a','profile-a','snapshot-b']),
                event(2,'disk_preload_succeeded',['contract-a','profile-a','snapshot-b'],success=True)]))

    def test_reversed_order_does_not_match(self):
        key=['contract-a','profile-a','snapshot-a']
        with self.assertRaisesRegex(ValueError,'[Nn]o same-process'):
            validate_spill_reload_trace(trace([event(0,'disk_preload_started',key),
                event(1,'ram_clean_eviction',key),event(2,'disk_preload_succeeded',key,
                    success=True,load_id=0)]))

    def test_failed_or_cross_process_load_does_not_match(self):
        key=['contract-a','profile-a','snapshot-a']
        failed=trace([event(0,'ram_clean_eviction',key),event(1,'disk_preload_started',key),
                      event(2,'disk_preload_failed',key,success=False)])
        with self.assertRaisesRegex(ValueError,'[Nn]o same-process'):
            validate_spill_reload_trace(failed)
        cross=trace([event(0,'ram_clean_eviction',key),event(1,'disk_preload_started',key),
                     event(2,'disk_preload_succeeded',key,pid=72,success=True)])
        with self.assertRaisesRegex(ValueError,'process identity'):
            validate_spill_reload_trace(cross)
        wrong_phase=trace([event(0,'ram_clean_eviction',key),event(1,'disk_preload_started',key),
                           event(2,'disk_preload_succeeded',key,phase='resume',success=True)])
        with self.assertRaisesRegex(ValueError,'phase identity'):
            validate_spill_reload_trace(wrong_phase)

    def test_truncated_trace_cannot_qualify(self):
        with self.assertRaisesRegex(ValueError,'overflowed'):
            validate_spill_reload_trace(trace([],overflowed=True))

    def test_successful_load_requires_positive_bytes(self):
        key=['contract-a','profile-a','snapshot-a']
        broken=trace([event(0,'ram_clean_eviction',key),event(1,'disk_preload_started',key),
                      dict(event(2,'disk_preload_succeeded',key,success=True),bytes=0)])
        with self.assertRaisesRegex(ValueError,'positive loaded-byte'):
            validate_spill_reload_trace(broken)

    def test_failed_preload_does_not_consume_eviction_before_successful_retry(self):
        key=['contract-a','profile-a','snapshot-a']
        document=trace([event(0,'ram_clean_eviction',key),
            event(1,'disk_preload_started',key),event(2,'disk_preload_failed',key,success=False),
            event(3,'disk_preload_started',key),event(4,'disk_preload_succeeded',key,success=True)])
        self.assertEqual(validate_spill_reload_trace(document),[dict(key=key,
            eviction_sequence=0,load_start_sequence=3,load_sequence=4,load_id=3,
            freed_bytes=8,loaded_bytes=8)])

    def test_success_with_wrong_load_id_is_rejected(self):
        key=['contract-a','profile-a','snapshot-a']
        document=trace([event(0,'ram_clean_eviction',key),event(1,'disk_preload_started',key),
                        event(2,'disk_preload_succeeded',key,success=True,load_id=77)])
        with self.assertRaisesRegex(ValueError,'exact started load'):
            validate_spill_reload_trace(document)

    def test_sequence_gaps_cannot_qualify(self):
        key=['contract-a','profile-a','snapshot-a']
        with self.assertRaisesRegex(ValueError,'sequence'):
            validate_spill_reload_trace(trace([event(0,'ram_clean_eviction',key),
                                               event(2,'disk_preload_succeeded',key,success=True)]))

    def test_pool_emits_success_only_after_loader_and_admission(self):
        key=('contract-a','profile-a','snapshot-a')
        diagnostics=PoolTransitionTrace('prepare',process_id=71,phase_id='phase-a')
        pool=SharedCheckpointPool(8,transition_trace=diagnostics)
        try:
            image=Checkpoint('contract-a','profile-a',1,b'abcd',(1.,),
                Checkpoint.digest(b'abcd',(1.,)),'snapshot-a')
            pool.admit(key,image,durable=True)
            with pool.reserve(8): pass
            reads=[]
            loaded=pool.preload(key,8,lambda:(reads.append('read') or image),trace_disk_read=True)
            self.assertIs(loaded.result(5),image)
            self.assertEqual(reads,['read'])
            events=diagnostics.snapshot()['events']
            self.assertEqual([row['event'] for row in events],
                ['ram_clean_eviction','disk_preload_started','disk_preload_succeeded'])
            self.assertEqual(events[0]['freed_bytes'],8)
            self.assertEqual(events[-1]['key'],list(key))
            self.assertTrue(events[-1]['success'])
        finally:
            pool.close()

    def test_ram_cache_hit_emits_no_disk_read_success(self):
        key=('contract-a','profile-a','snapshot-a')
        diagnostics=PoolTransitionTrace('prepare',process_id=71,phase_id='phase-a')
        pool=SharedCheckpointPool(8,transition_trace=diagnostics)
        try:
            image=Checkpoint('contract-a','profile-a',1,b'abcd',(1.,),
                Checkpoint.digest(b'abcd',(1.,)),'snapshot-a')
            pool.admit(key,image,durable=True)
            self.assertIs(pool.preload(key,8,lambda:self.fail('cache hit called disk loader'),
                trace_disk_read=True).result(1),image)
            self.assertEqual(diagnostics.snapshot()['events'],[])
        finally:
            pool.close()

    def test_shared_payload_alias_eviction_does_not_claim_bytes_freed_for_first_key(self):
        diagnostics=PoolTransitionTrace('prepare',process_id=71,phase_id='phase-a')
        pool=SharedCheckpointPool(8,transition_trace=diagnostics)
        try:
            image=Checkpoint('contract-a','profile-a',1,b'abcd',(1.,),
                Checkpoint.digest(b'abcd',(1.,)),'snapshot-a')
            pool.admit(('contract-a','profile-a','snapshot-a'),image,durable=True)
            alias=replace(image,input_identity='alias')
            pool.admit(('contract-a','profile-a','snapshot-b'),alias,durable=True)
            with pool.reserve(8): pass
            events=diagnostics.snapshot()['events']
            self.assertEqual([row['freed_bytes'] for row in events],[0,8])
            doc=diagnostics.snapshot()
            doc['events'].append(event(2,'disk_preload_started',
                ['contract-a','profile-a','snapshot-a']))
            doc['events'].append(event(3,'disk_preload_succeeded',
                ['contract-a','profile-a','snapshot-a'],success=True))
            doc['event_count']=4
            with self.assertRaisesRegex(ValueError,'[Nn]o same-process'):
                validate_spill_reload_trace(doc)
        finally:
            pool.close()

    def test_failed_disk_loader_never_emits_success(self):
        key=('contract-a','profile-a','snapshot-a')
        diagnostics=PoolTransitionTrace('prepare',process_id=71,phase_id='phase-a')
        pool=SharedCheckpointPool(8,transition_trace=diagnostics)
        try:
            with self.assertRaisesRegex(OSError,'read failed'):
                pool.preload(key,8,lambda:(_ for _ in ()).throw(OSError('read failed')),
                    trace_disk_read=True).result(5)
            self.assertEqual([row['event'] for row in diagnostics.snapshot()['events']],
                ['disk_preload_started','disk_preload_failed'])
        finally:
            pool.close()

    def test_completion_callback_eviction_orders_after_completed_disk_trace(self):
        key=('contract-a','profile-a','snapshot-a')
        diagnostics=PoolTransitionTrace('prepare',process_id=71,phase_id='phase-a')
        pool=SharedCheckpointPool(8,transition_trace=diagnostics)
        image=Checkpoint('contract-a','profile-a',1,b'abcd',(1.,),
            Checkpoint.digest(b'abcd',(1.,)),'snapshot-a')
        loader_entered,allow_loader,callback_done=(threading.Event() for _ in range(3))
        try:
            def loader():
                loader_entered.set()
                if not allow_loader.wait(5): raise TimeoutError('loader release timed out')
                return image
            future=pool.preload(key,8,loader,trace_disk_read=True)
            self.assertTrue(loader_entered.wait(5))
            def evict_as_soon_as_load_is_visible(_):
                with pool.reserve(8): pass
                callback_done.set()
            future.add_done_callback(evict_as_soon_as_load_is_visible)
            allow_loader.set()
            self.assertTrue(callback_done.wait(5))
            future.result(5)
            pool._reader.submit(lambda:None).result(5)
            events=diagnostics.snapshot()['events']
            names=[row['event'] for row in events]
            self.assertLess(names.index('disk_preload_succeeded'),names.index('ram_clean_eviction'))
            with self.assertRaisesRegex(ValueError,'no same-process'):
                validate_spill_reload_trace(diagnostics.snapshot())
        finally:
            pool.close()

    def test_coordinator_marks_only_actual_checkpoint_loader_as_disk_read(self):
        with tempfile.TemporaryDirectory() as directory:
            gate=threading.Event();gate.set()
            models={}
            def factory(gpu):
                models[gpu]=Model(directory,gpu,(0,),[],gate)
                return models[gpu]
            diagnostics=PoolTransitionTrace('prepare',process_id=71,phase_id='phase-a')
            coordinator=ContextCoordinator(directory,
                PoolConfig(gpus=(0,),ram_bytes=4096,disk_bytes=16384,min_available_bytes=0),
                factory,transition_trace=diagnostics)
            try:
                coordinator.ready(5)
                root=coordinator.create('main',(1,2),'create').durable.result(5)['snapshot']
                coordinator.pool.remove(coordinator._key(models[0],root))
                coordinator.submit('main',(3,),1,'request').durable.result(5)
                events=diagnostics.snapshot()['events']
                loads=[row for row in events if row['event']=='disk_preload_succeeded']
                self.assertTrue(loads)
                self.assertEqual(loads[0]['key'],['m','e',root])
                self.assertGreater(loads[0]['bytes'],0)
            finally:
                coordinator.close()


if __name__=='__main__': unittest.main()
