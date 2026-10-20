"""Queued append recovery when a canonical source has no derived image.

These are CPU scheduler controls. They do not claim native/model qualification.
"""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import weakref

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import Checkpoint, PreparedCheckpoint
from context_time_travel.store import Conflict, IntegrityError, InvalidOperation
from context_time_travel.strata_backend import StrataImage
from context_time_travel.execution_receipt import tokens_digest
from test_root_computation_reuse import CPUBackend


def _execution_event(operator,position,tokens):
    return dict(operator=operator,position=position,count=len(tokens),
        tokens_sha256=tokens_digest(tokens),
        chunks=[[position,len(tokens)]] if operator=='batched-prefill' else
            [[position+index,1] for index in range(len(tokens))])


class RebuildableRootFixtureBackend(CPUBackend):
    """Real BranchingModel fixture using the existing root-reuse CPU backend."""
    def consume(self,lease,ids):
        position=len(lease['ids']);ids=tuple(ids)
        super().consume(lease,ids)
        self.last_execution_receipt=_execution_event('batched-prefill',position,ids)

    @contextmanager
    def generate_request(self,lease,limit):
        position=len(lease['ids'])
        tokens=tuple([4]*limit)
        def outputs():
            for token in tokens:
                lease['ids'].append(token)
                self.stats['forward_tokens']+=1
                self.stats['consumed_tokens']+=1
                yield token
            self.last_execution_receipt=_execution_event('normal-verifier-decode',position,tokens)
        yield outputs()


class RebuildModel:
    def __init__(self, directory, gpu, gpu_selection):
        from context_time_travel.store import Store
        self.store = Store(directory)
        self.cache = {}
        self.gpu = self.gpu_id = gpu
        self.profile = dict(model_contract_id='rebuild-model', execution_profile_id='rebuild-profile',
                            execution={'GPU': list(gpu_selection)})
        self.checkpoint_directory = self.store.root/'checkpoints'
        self.checkpoint_directory.mkdir(exist_ok=True)
        self.backend = self
        self.release_calls = 0
        self.stats = dict(forward_tokens=0, consumed_tokens=0, loaded_file_bytes=0)
        self.rebuild_calls = []
        self.rebuild_thread_ids = []
        self.rebuild_gate = None
        self.fail_rebuild = False
        self.cancel_when_rebuilt = False
        self.cancel_after_rebuild = False
        self.bad_rebuild_evidence = None
        self.fail_after_new_tip = False
        self.fail_capture = False
        self.request_calls = []
        self.gate = threading.Event()
        self.gate.set()

    def work_counters(self):
        return dict(self.stats)

    def checkpoint_compatible(self, manifest, directory):
        return (manifest.get('model_contract_id') == self.profile['model_contract_id']
                and manifest.get('execution_profile_id') == self.profile['execution_profile_id'])

    def close(self):
        pass

    def release_checkpoints(self):
        self.release_calls += 1
        return 0

    def history(self, snapshot):
        return tuple(self.store.lineage(snapshot)[1])

    def image(self, snapshot, *, native=False):
        history = self.history(snapshot)
        if native:
            blob = StrataImage(b'native-'+snapshot.encode(), b'process-identity', 1,
                               len(history), 64, 4, ())
            checksum = Checkpoint.digest(blob, (1.0,))
        else:
            blob = b'checkpoint:'+snapshot.encode()
            checksum = Checkpoint.digest(blob, (1.0,))
        checkpoint = Checkpoint('rebuild-model','rebuild-profile',len(history),blob,(1.0,),
                                checksum,snapshot)
        self.cache[snapshot] = checkpoint
        return checkpoint

    def create(self, ids):
        ids = tuple(ids)
        self.stats['forward_tokens'] += len(ids)
        self.stats['consumed_tokens'] += len(ids)
        snapshot = self.store.create_root('rebuild-model',ids,{'status':'idle'})
        self.image(snapshot)
        return snapshot

    def rebuild_materialization(self, snapshot):
        self.rebuild_calls.append(snapshot)
        self.rebuild_thread_ids.append(threading.get_ident())
        if self.rebuild_gate is not None:
            self.rebuild_gate[0].set()
            if not self.rebuild_gate[1].wait(5):
                raise TimeoutError('test rebuild gate timed out')
        if self.fail_rebuild:
            raise IntegrityError('test reconstruction failed')
        if (snapshot in self.cache
                or __import__('os').path.lexists(self.checkpoint_directory/(snapshot+'.json'))):
            raise InvalidOperation('An existing materialization must be validated and reused, not bypassed')
        history = self.history(snapshot)
        before = self.stats['forward_tokens']
        self.stats['forward_tokens'] += len(history)
        self.stats['consumed_tokens'] += len(history)
        self.image(snapshot, native=True)
        if self.cancel_when_rebuilt:
            self.cancel_after_rebuild = True
        result=dict(snapshot=snapshot, source_kind='recomputed-history', durable=False,
                    prior_computation_preserved=False, consumed=len(history),
                    forward_tokens=self.stats['forward_tokens']-before,
                    model_contract_id='rebuild-model', execution_profile_id='rebuild-profile',
                    schedule_sha256='a'*64, seconds=.01)
        if self.bad_rebuild_evidence:
            result.update(self.bad_rebuild_evidence)
        return result

    def run_greedy_request(self, snapshot, ids, max_steps, *, stream, on_token, cancelled=None):
        self.request_calls.append((snapshot,tuple(ids),max_steps,threading.get_ident()))
        if cancelled is not None and cancelled():
            raise CancelledError()
        self.gate.wait(5)
        self.stats['forward_tokens'] += len(ids)+max_steps
        self.stats['consumed_tokens'] += len(ids)+max_steps
        tokens = tuple(range(max_steps))
        for token in tokens:
            on_token(token)
        target = self.store.append(snapshot,tuple(ids)+tokens,{'status':'idle'})
        self.image(target)
        if self.fail_after_new_tip:
            raise RuntimeError('test suffix failure after creating native tip')
        return dict(snapshot=target,tokens=list(tokens),request_seconds=.01,generation_seconds=.01)

    def prepare_checkpoint(self, snapshot):
        if self.fail_capture:
            raise RuntimeError('test capture failure after creating native tip')
        checkpoint = self.cache[snapshot]
        envelope = dict(manifest=dict(snapshot=snapshot,state_bytes=len(checkpoint.blob),
            logits_count=1,payload_sha256=checkpoint.checksum,
            model_contract_id='rebuild-model',execution_profile_id='rebuild-profile'))
        return PreparedCheckpoint(snapshot,checkpoint,json.dumps(envelope).encode())

    def checkpoint_manifest(self, snapshot, history=None):
        envelope = json.loads((self.checkpoint_directory/(snapshot+'.json')).read_bytes())
        if envelope.get('manifest',{}).get('snapshot') != snapshot:
            raise IntegrityError('Manifest is bound to a different snapshot')
        return envelope['manifest']

    def _load_checkpoint(self, snapshot, history):
        manifest = self.checkpoint_manifest(snapshot,history)
        raw = (self.checkpoint_directory/(manifest['payload_sha256']+'.state')).read_bytes()
        if (len(raw) != manifest['state_bytes']+4*manifest['logits_count']
                or hashlib.sha256(raw).hexdigest() != manifest['payload_sha256']):
            raise IntegrityError('Checkpoint payload corruption')
        blob = raw[:-4]
        return Checkpoint('rebuild-model','rebuild-profile',len(history),blob,(1.0,),
                          manifest['payload_sha256'],snapshot)

    def save_prepared_branch(self, name, revision, prepared, operation):
        checkpoint = prepared.checkpoint
        payload = checkpoint.blob+b'\0\0\x80?'
        (self.checkpoint_directory/(checkpoint.checksum+'.state')).write_bytes(payload)
        (self.checkpoint_directory/(prepared.snapshot+'.json')).write_bytes(prepared.envelope)
        head = self.store.move(name,revision,prepared.snapshot,operation)
        return dict(snapshot=prepared.snapshot,branch=asdict(head),
                    payload_sha256=checkpoint.checksum,model_contract_id='rebuild-model',
                    execution_profile_id='rebuild-profile')


class MissingSourceRebuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = {}
        self.config = PoolConfig(gpus=(0,),ram_bytes=1024*1024,disk_bytes=1024*1024,
                                 min_available_bytes=0)
        def factory(gpu):
            model = RebuildModel(self.temp.name,gpu,(0,))
            self.models[gpu] = model
            return model
        self.coordinator = ContextCoordinator(self.temp.name,self.config,factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)

    def create(self, branch='main', prefix=(1,2,3), operation='create'):
        return self.coordinator.create(branch,prefix,operation).durable.result(5)

    def wait_pool_idle(self):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            metrics=self.coordinator.pool.metrics()
            if not metrics['pins'] and not metrics['reserved_bytes']:
                return
            time.sleep(.005)
        self.fail('RAM source pin/reservation did not drain')

    def assert_request_released(self):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            metrics=self.coordinator.pool.metrics()
            if (not metrics['pins'] and not metrics['reserved_bytes']
                    and not self.coordinator._branches):
                break
            time.sleep(.005)
        self.assertFalse(self.coordinator._branches)
        metrics=self.coordinator.pool.metrics()
        self.assertEqual(metrics['pins'],0)
        self.assertEqual(metrics['reserved_bytes'],0)

    def remove_all_materializations(self, snapshot, *, keep_manifest=False):
        self.wait_pool_idle()
        model=self.models[0]
        key=self.coordinator._key(model,snapshot)
        if key in self.coordinator.pool:
            self.coordinator.pool.remove(key)
        model.cache.pop(snapshot,None)
        manifest=model.checkpoint_directory/(snapshot+'.json')
        if not keep_manifest:
            manifest.unlink(missing_ok=True)
        for path in model.checkpoint_directory.glob('*.state'):
            path.unlink()
        return manifest

    def test_absent_source_rebuild_reports_prefix_separately_then_appends(self):
        original=self.create()
        snapshot=original['snapshot']
        model=self.models[0]
        old_head=self.coordinator.store.head('main')
        old_receipt=self.coordinator.store._read_refs()['receipts']['create']
        old_record=self.coordinator.store.get_snapshot(snapshot)
        self.remove_all_materializations(snapshot)
        before=model.stats['forward_tokens']

        request=self.coordinator.submit('main',(4,5),2,'append-after-rebuild',
                                        expected_revision=old_head.revision)
        answer=request.snapshot.result(5)
        durable=request.durable.result(5)

        self.assertEqual(model.rebuild_calls,[snapshot])
        self.assertEqual(answer['source_recomputation']['source_kind'],'recomputed-history')
        self.assertFalse(answer['source_recomputation']['durable'])
        self.assertFalse(answer['source_recomputation']['prior_computation_preserved'])
        self.assertEqual(answer['source_recomputation']['prefix_forward_tokens'],3)
        self.assertEqual(answer['source_recomputation']['prefix_history_tokens'],3)
        self.assertEqual(answer['source_recomputation']['prefix_consumed_tokens'],3)
        self.assertEqual(answer['counter_deltas']['forward_tokens'],4)
        self.assertEqual(model.stats['forward_tokens']-before,7)
        self.assertEqual(durable['snapshot'],answer['snapshot'])
        events=[]
        while not request.events.empty(): events.append(request.events.get())
        names=[row['event'] for row in events]
        self.assertLess(names.index('source_recomputation_started'),names.index('source_recomputed'))
        self.assertIn('source_recomputed',names)
        self.assertEqual(self.coordinator.store.get_snapshot(snapshot),old_record)
        self.assertEqual(self.coordinator.store._read_refs()['receipts']['create'],old_receipt)
        self.assertEqual(self.coordinator.metrics()['max_active_per_gpu'],{0:1})
        self.assert_request_released()

    def test_intact_disk_or_shared_pool_source_is_reused_without_recompute(self):
        original=self.create(prefix=(2,3,4),operation='create-hot')
        model=self.models[0]
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(5,),2,'append-hot',expected_revision=1)
        answer=request.snapshot.result(5)
        request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens']-before,3)
        self.assertNotIn('source_recomputation',answer)

    def test_shared_ram_source_is_reused_without_disk_manifest(self):
        original=self.create(operation='create-ram-only')
        snapshot=original['snapshot'];model=self.models[0]
        self.wait_pool_idle()
        model.cache.pop(snapshot,None)
        for path in model.checkpoint_directory.iterdir(): path.unlink()
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),1,'append-ram-only',expected_revision=1)
        answer=request.snapshot.result(5);request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens']-before,2)
        self.assertNotIn('source_recomputation',answer)

    def test_source_evicted_between_route_and_preload_rebuilds_only_if_now_absent(self):
        original=self.create(operation='create-eviction-race')
        snapshot=original['snapshot'];model=self.models[0]
        self.wait_pool_idle()
        model.cache.pop(snapshot,None)
        for path in model.checkpoint_directory.iterdir(): path.unlink()
        key=self.coordinator._key(model,snapshot)
        self.assertIn(key,self.coordinator.pool)
        original_preload=self.coordinator._preload
        evicted=[]
        def evict_once(target_model,target_snapshot,**kwargs):
            if (target_snapshot==snapshot and not evicted
                    and threading.get_ident()==self.coordinator._threads[0].ident):
                self.coordinator.pool.remove(key)
                evicted.append(snapshot)
            return original_preload(target_model,target_snapshot,**kwargs)
        self.coordinator._preload=evict_once
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),1,'append-eviction-race',expected_revision=1)
        answer=request.snapshot.result(5);request.durable.result(5)
        self.assertEqual(evicted,[snapshot])
        self.assertEqual(model.rebuild_calls,[snapshot])
        self.assertEqual(answer['source_recomputation']['prefix_forward_tokens'],3)
        self.assertEqual(answer['counter_deltas']['forward_tokens'],2)
        self.assertEqual(model.stats['forward_tokens']-before,5)
        self.assert_request_released()

    def test_unaccounted_serialized_cache_alias_is_dropped_then_rebuilt(self):
        original=self.create(operation='create-serialized-alias')
        snapshot=original['snapshot'];model=self.models[0]
        self.remove_all_materializations(snapshot)
        stale=model.image(snapshot)
        self.assertNotIsInstance(stale.blob,StrataImage)
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),1,'append-after-stale-alias',expected_revision=1)
        answer=request.snapshot.result(5);request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[snapshot])
        self.assertEqual(answer['source_recomputation']['prefix_forward_tokens'],3)
        self.assertIsInstance(model.cache[snapshot].blob,StrataImage)
        self.assertEqual(model.stats['forward_tokens']-before,5)

    def test_nonboolean_rebuild_durability_evidence_rejects_and_releases_image(self):
        original=self.create(operation='create-invalid-rebuild-evidence')
        snapshot=original['snapshot'];model=self.models[0]
        self.remove_all_materializations(snapshot)
        model.bad_rebuild_evidence={'durable':0}
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),1,'append-invalid-rebuild-evidence',expected_revision=1)
        with self.assertRaises(IntegrityError): request.durable.result(5)
        self.assertEqual(model.request_calls,[])
        self.assertEqual(model.stats['forward_tokens']-before,3)
        self.assertNotIn(snapshot,model.cache)
        self.assertEqual(model.release_calls,2)
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])
        self.assert_request_released()

    def test_unread_failed_request_does_not_retain_rebuilt_checkpoint_traceback(self):
        original=self.create(operation='create-unread-failure')
        snapshot=original['snapshot'];model=self.models[0]
        self.remove_all_materializations(snapshot)
        # Fail after the reconstructed checkpoint has entered local validation,
        # but before either Future publishes a result. The exception traceback
        # is then the only possible owner after the worker removes the cache row.
        original_rebuild=model.rebuild_materialization
        checkpoint_refs=[]
        def track_rebuild(target):
            receipt=original_rebuild(target)
            checkpoint_refs.append(weakref.ref(model.cache[target]))
            return receipt
        model.rebuild_materialization=track_rebuild
        original_counters=model.work_counters
        counter_reads=0
        def bad_post_rebuild_counters():
            nonlocal counter_reads
            counter_reads+=1
            result=original_counters()
            if counter_reads>=2:
                result['forward_tokens']+=1
            return result
        model.work_counters=bad_post_rebuild_counters
        request=self.coordinator.submit('main',(4,),1,'append-unread-failure',expected_revision=1)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if request.snapshot.done() and request.durable.done():
                break
            time.sleep(.005)
        self.assertTrue(request.snapshot.done())
        self.assertTrue(request.durable.done())
        self.assertEqual(request.state,'failed')
        self.assertEqual(model.rebuild_calls,[snapshot])
        self.assertEqual(model.cache,{})
        gc.collect()
        self.assertEqual(len(checkpoint_refs),1)
        self.assertIsNone(checkpoint_refs[0](),
            'Unread Future exceptions retained a checkpoint through their traceback')
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])
        self.assert_request_released()

    def test_present_manifest_with_missing_payload_fails_without_rebuild(self):
        original=self.create(operation='create-corrupt')
        snapshot=original['snapshot']
        model=self.models[0]
        manifest=self.remove_all_materializations(snapshot,keep_manifest=True)
        data=json.loads(manifest.read_bytes())
        payload=model.checkpoint_directory/(data['manifest']['payload_sha256']+'.state')
        payload.unlink(missing_ok=True)
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),2,'append-missing-payload',expected_revision=1)
        with self.assertRaises((IntegrityError,FileNotFoundError)):
            request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens'],before)
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])

    def test_present_manifest_with_corrupt_payload_fails_without_rebuild(self):
        original=self.create(operation='create-corrupt-payload')
        snapshot=original['snapshot'];model=self.models[0]
        manifest=self.remove_all_materializations(snapshot,keep_manifest=True)
        data=json.loads(manifest.read_bytes())
        payload=model.checkpoint_directory/(data['manifest']['payload_sha256']+'.state')
        payload.write_bytes(b'corrupt-state')
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),2,'append-corrupt-payload',expected_revision=1)
        with self.assertRaises(IntegrityError): request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens'],before)
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])

    def test_malformed_present_manifest_fails_without_rebuild(self):
        original=self.create(operation='create-bad-manifest')
        snapshot=original['snapshot']
        model=self.models[0]
        manifest=self.remove_all_materializations(snapshot,keep_manifest=True)
        manifest.write_bytes(b'{not-json')
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),2,'append-bad-manifest',expected_revision=1)
        with self.assertRaises((IntegrityError,ValueError)):
            request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens'],before)

    def test_dangling_manifest_binding_fails_without_rebuild(self):
        original=self.create(operation='create-dangling-manifest')
        snapshot=original['snapshot'];model=self.models[0]
        manifest=self.remove_all_materializations(snapshot,keep_manifest=True)
        data=json.loads(manifest.read_bytes())
        data['manifest']['snapshot']='f'*64
        manifest.write_text(json.dumps(data))
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),2,'append-dangling-manifest',expected_revision=1)
        with self.assertRaises((IntegrityError,ValueError)):
            request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens'],before)
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])

    def test_failed_rebuild_does_not_execute_suffix_or_move_branch(self):
        original=self.create(operation='create-failed-rebuild')
        snapshot=original['snapshot'];model=self.models[0]
        self.remove_all_materializations(snapshot)
        model.fail_rebuild=True
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),2,'append-failed-rebuild',expected_revision=1)
        with self.assertRaises(IntegrityError): request.durable.result(5)
        self.assertEqual(model.stats['forward_tokens'],before)
        self.assertEqual(model.request_calls,[])
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])
        self.assertEqual(model.rebuild_calls,[snapshot])
        self.assert_request_released()

    def _assert_failed_new_tip_is_reclaimed(self,operation,*,capture):
        original=self.create(operation=operation+'-root')
        snapshot=original['snapshot'];model=self.models[0]
        unrelated_id=model.store.create_root('rebuild-model',(99,),{'status':'idle'})
        model.image(unrelated_id,native=True)
        self.remove_all_materializations(snapshot)
        if capture:
            model.fail_capture=True
        else:
            model.fail_after_new_tip=True
        request=self.coordinator.submit('main',(4,),1,operation,expected_revision=1)
        with self.assertRaisesRegex(RuntimeError,'test (capture|suffix failure)'):
            request.durable.result(5)
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])
        self.assertNotIn(snapshot,model.cache)
        self.assertEqual(set(model.cache),{unrelated_id})
        self.assertEqual(model.release_calls,2)
        self.assert_request_released()

    def test_suffix_failure_after_new_tip_reclaims_only_job_created_images(self):
        self._assert_failed_new_tip_is_reclaimed('append-suffix-fail',capture=False)

    def test_capture_failure_after_new_tip_reclaims_only_job_created_images(self):
        self._assert_failed_new_tip_is_reclaimed('append-capture-fail',capture=True)

    def test_cancellation_after_rebuild_keeps_source_and_old_receipt(self):
        original=self.create(operation='create-cancel')
        snapshot=original['snapshot'];model=self.models[0]
        old_receipt=self.coordinator.store._read_refs()['receipts']['create-cancel']
        self.remove_all_materializations(snapshot)
        model.cancel_when_rebuilt=True
        request=self.coordinator.submit('main',(4,),2,'append-cancelled-after-rebuild',
            expected_revision=1,cancelled=lambda: model.cancel_after_rebuild)
        with self.assertRaises(CancelledError): request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[snapshot])
        self.assertEqual(model.request_calls,[])
        self.assertEqual(asdict(self.coordinator.store.head('main')),original['branch'])
        self.assertEqual(self.coordinator.store._read_refs()['receipts']['create-cancel'],old_receipt)
        self.assertNotIn(snapshot,model.cache)
        self.assertEqual(model.release_calls,2)
        self.assert_request_released()

    def test_cas_conflict_after_gated_rebuild_keeps_external_head_and_receipt(self):
        original=self.create(operation='create-cas')
        snapshot=original['snapshot'];model=self.models[0]
        old_receipt=self.coordinator.store._read_refs()['receipts']['create-cas']
        self.remove_all_materializations(snapshot)
        entered,release=threading.Event(),threading.Event()
        model.rebuild_gate=(entered,release)
        request=self.coordinator.submit('main',(4,),1,'append-cas-conflict',expected_revision=1)
        self.assertTrue(entered.wait(3),'worker did not enter source reconstruction')
        external=self.coordinator.store.append(snapshot,(9,),{'status':'idle'})
        moved=self.coordinator.store.move('main',1,external,'external-during-rebuild')
        release.set()
        with self.assertRaises(Conflict): request.snapshot.result(5)
        with self.assertRaises(Conflict): request.durable.result(5)
        self.assertEqual(asdict(self.coordinator.store.head('main')),asdict(moved))
        self.assertEqual(self.coordinator.store._read_refs()['receipts']['create-cas'],old_receipt)
        self.assertNotIn(snapshot,model.cache)
        self.assertEqual(model.release_calls,2)
        self.assert_request_released()

    def test_hot_native_source_remains_reusable_without_disk_manifest(self):
        original=self.create(operation='create-native-hot')
        snapshot=original['snapshot'];model=self.models[0]
        self.remove_all_materializations(snapshot)
        model.image(snapshot,native=True)
        before=model.stats['forward_tokens']
        request=self.coordinator.submit('main',(4,),1,'append-native-hot',expected_revision=1)
        answer=request.snapshot.result(5);request.durable.result(5)
        self.assertEqual(model.rebuild_calls,[])
        self.assertEqual(model.stats['forward_tokens']-before,2)
        self.assertNotIn('source_recomputation',answer)

    def test_rebuild_happens_only_on_worker_owner_thread(self):
        original=self.create(operation='create-owner-thread')
        snapshot=original['snapshot'];model=self.models[0]
        self.remove_all_materializations(snapshot)
        request=self.coordinator.submit('main',(4,),1,'append-owner-thread',expected_revision=1)
        request.durable.result(5)
        self.assertEqual(model.rebuild_thread_ids[0],self.coordinator._threads[0].ident)
        self.assertEqual(model.request_calls[0][3],self.coordinator._threads[0].ident)


class ActualBranchingModelRebuildTests(unittest.TestCase):
    """Exercise wrapped real checkpoint errors through BranchingModel itself."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile=dict(model_contract_id='cpu-model',execution_profile_id='cpu-exec',
            model_weights_sha256='a'*64,execution=dict(GPU=[0],capacity=64,prefill_chunk=8,
                eos_tokens=[15],resume_contract='identical-native-operators-v1',
                generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1'))
        self.backends={}
        def factory(gpu):
            backend=RebuildableRootFixtureBackend(self.profile)
            backend.piece=lambda token:bytes([65+token])
            backend.is_eog=lambda token:token==15
            self.backends[gpu]=backend
            from context_time_travel.real import BranchingModel
            return BranchingModel(self.temp.name,backend,self.profile)
        self.coordinator=ContextCoordinator(self.temp.name,
            PoolConfig(gpus=(0,),ram_bytes=1024*1024,disk_bytes=1024*1024,
                       min_available_bytes=0),factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)

    def _create(self,operation='actual-create'):
        return self.coordinator.create('main',(1,2,3),operation).durable.result(5)

    def _remove_disk_materialization_keep_pool(self,snapshot,model):
        manifest=model.checkpoint_directory/(snapshot+'.json')
        envelope=json.loads(manifest.read_bytes())
        payload=model.checkpoint_directory/(envelope['manifest']['payload_sha256']+'.state')
        manifest.unlink()
        payload.unlink()

    def _wait_pool_idle(self):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            metrics=self.coordinator.pool.metrics()
            if not metrics['pins'] and not metrics['reserved_bytes']:
                return
            time.sleep(.005)
        self.fail('Actual BranchingModel source pin/reservation did not drain')

    def test_actual_branching_model_rebuilds_after_ram_eviction_races_preload(self):
        original=self._create('actual-eviction-create')
        snapshot=original['snapshot'];model=self.coordinator._workers[0]['model']
        backend=self.backends[0]
        self._wait_pool_idle()
        key=self.coordinator._key(model,snapshot)
        self.assertIn(key,self.coordinator.pool)
        model.cache.pop(snapshot,None)
        self._remove_disk_materialization_keep_pool(snapshot,model)
        original_preload=self.coordinator._preload
        removed=[]
        def evict_before_preload(target_model,target_snapshot,**kwargs):
            if (target_snapshot==snapshot and not removed
                    and threading.get_ident()==self.coordinator._threads[0].ident):
                self.coordinator.pool.remove(key)
                removed.append(snapshot)
            return original_preload(target_model,target_snapshot,**kwargs)
        self.coordinator._preload=evict_before_preload
        before=backend.work_counters()['forward_tokens']
        request=self.coordinator.submit('main',(6,),1,'actual-eviction-append',expected_revision=1)
        answer=request.snapshot.result(5)
        durable=request.durable.result(5)
        self.assertEqual(removed,[snapshot])
        self.assertEqual(answer['source_recomputation']['prefix_forward_tokens'],3)
        self.assertIs(answer['source_recomputation']['durable'],False)
        self.assertIs(answer['source_recomputation']['prior_computation_preserved'],False)
        self.assertEqual(answer['counter_deltas']['forward_tokens'],2)
        self.assertEqual(backend.work_counters()['forward_tokens']-before,5)
        self.assertEqual(durable['snapshot'],answer['snapshot'])
        self.assertEqual(model.history(snapshot),(1,2,3))
        self.assertEqual(model.history(answer['snapshot']),(1,2,3,6,4))
        self.assertEqual(self.coordinator.store.head('main').tip,answer['snapshot'])

    def test_actual_strict_restore_still_rejects_missing_manifest(self):
        original=self._create('actual-strict-restore')
        model=self.coordinator._workers[0]['model']
        snapshot=original['snapshot']
        manifest=model.checkpoint_directory/(snapshot+'.json')
        envelope=json.loads(manifest.read_bytes())
        (model.checkpoint_directory/(envelope['manifest']['payload_sha256']+'.state')).unlink()
        manifest.unlink()
        before=self.backends[0].work_counters()['forward_tokens']
        with self.assertRaises(IntegrityError):
            model.restore_checkpoint(snapshot)
        self.assertEqual(self.backends[0].work_counters()['forward_tokens'],before)


if __name__ == '__main__':
    unittest.main()
