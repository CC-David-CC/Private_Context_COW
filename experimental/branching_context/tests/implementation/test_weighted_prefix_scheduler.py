"""Queue policy controls; these do not claim GPU speed or task quality."""
import threading
from concurrent.futures import CancelledError
import json
import time
import tempfile
import unittest

from context_time_travel.prefix_scheduler import order_jobs, select_worker
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from test_root_computation_reuse import CPUBackend
from test_missing_source_rebuild import RebuildableRootFixtureBackend


class Job:
    def __init__(self, key, created):
        self.prefill_key = key
        self.request = type('Request', (), {'created': created})()


class WeightedQueueTests(unittest.TestCase):
    def test_waiter_weight_and_active_slots_change_priority(self):
        a, b, c = Job('a', 9), Job('b', 9), Job('b', 9)
        self.assertEqual(order_jobs([a, b, c], {}, now=10), [b, c, a])
        self.assertEqual(order_jobs([a, b, c], {'b': 2}, now=10), [a, b, c])

    def test_aging_prevents_a_hot_prefix_starving_an_old_request(self):
        old = Job('old', 1)
        hot = [Job('hot', 9) for _ in range(100)]
        self.assertIs(order_jobs(hot + [old], {}, now=10)[0], old)

    def test_worker_choice_uses_measured_speed_and_keeps_warm_affinity(self):
        self.assertEqual(select_worker([3, 4], [], {3: .2, 4: .01}), 4)
        self.assertEqual(select_worker([3, 4], [3], {3: .2, 4: .01}), 3)
        self.assertEqual(select_worker([3, 4], [], {}), 3)


class ConcurrentRootTests(unittest.TestCase):
    def test_failed_leader_releases_waiting_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            profile=dict(model_contract_id='cpu-model',execution_profile_id='cpu-exec',
                model_weights_sha256='a'*64,execution={'GPU':[3,4]})
            entered,release=threading.Event(),threading.Event()
            failed=[]
            class FailingOnce(CPUBackend):
                def consume(self,lease,ids):
                    if not failed:
                        failed.append(True);entered.set()
                        if not release.wait(5):raise TimeoutError('test failure gate')
                        raise ValueError('leader prefill failed')
                    return super().consume(lease,ids)
            c=ContextCoordinator(directory,PoolConfig(gpus=(3,4),ram_bytes=1024**2,
                disk_bytes=1024**2,min_available_bytes=0,scheduling='weighted-prefix'),
                lambda gpu:BranchingModel(directory,FailingOnce(profile),profile)).ready(5)
            try:
                leader=c.create('leader',(2,5,7),'failed-leader')
                self.assertTrue(entered.wait(2))
                follower=c.create('follower',(2,5,7),'waiting-follower')
                release.set()
                with self.assertRaisesRegex(ValueError,'leader prefill failed'):leader.durable.result(5)
                # Observe both original futures; neither operation is retried.
                with self.assertRaises(ValueError):leader.snapshot.result(5)
                follower.durable.result(5)
                self.assertEqual(c.metrics()['scheduling']['inflight_prefills'],0)
            finally:release.set();c.close()

    def test_queued_cancellation_does_not_release_another_requests_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            profile=dict(model_contract_id='cpu-model',execution_profile_id='cpu-exec',
                model_weights_sha256='a'*64,execution={'GPU':[3,4]})
            entered,release,cancelled=threading.Event(),threading.Event(),threading.Event()
            class Gated(CPUBackend):
                def consume(self,lease,ids):
                    entered.set()
                    if not release.wait(5):raise TimeoutError('test cancellation gate')
                    return super().consume(lease,ids)
            c=ContextCoordinator(directory,PoolConfig(gpus=(3,4),ram_bytes=1024**2,
                disk_bytes=1024**2,min_available_bytes=0,scheduling='weighted-prefix'),
                lambda gpu:BranchingModel(directory,Gated(profile),profile)).ready(5)
            try:
                leader=c.create('leader',(2,5,7),'cancel-owner')
                self.assertTrue(entered.wait(2))
                follower=c.create('cancelled',(2,5,7),'cancel-follower',cancelled=cancelled.is_set)
                cancelled.set()
                with self.assertRaises(CancelledError):follower.durable.result(3)
                self.assertTrue(follower.snapshot.cancelled())
                self.assertEqual(c.metrics()['scheduling']['inflight_prefills'],1)
                self.assertFalse(leader.snapshot.done())
                release.set();leader.durable.result(5)
            finally:release.set();c.close()

    def test_duplicate_inflight_prefill_waits_without_blocking_another_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = dict(model_contract_id='cpu-model', execution_profile_id='cpu-exec',
                model_weights_sha256='a' * 64, execution={'GPU': [3, 4]})
            entered, release = threading.Event(), threading.Event()
            backends = {}

            class SlowBackend(CPUBackend):
                def consume(self, lease, ids):
                    if tuple(ids) == (2, 5, 7):
                        entered.set()
                        if not release.wait(5):
                            raise TimeoutError('test prefill gate')
                    super().consume(lease, ids)

            def factory(gpu):
                backend = SlowBackend(profile)
                backends[gpu] = backend
                return BranchingModel(directory, backend, profile)

            c = ContextCoordinator(directory, PoolConfig(gpus=(3, 4),
                ram_bytes=1024**2, disk_bytes=1024**2, min_available_bytes=0,
                scheduling='weighted-prefix'), factory).ready(5)
            try:
                first = c.create('first', (2, 5, 7), 'create-first')
                self.assertTrue(entered.wait(2))
                duplicate = c.create('second', (2, 5, 7), 'create-second')
                independent = c.create('other', (1, 4), 'create-other')
                independent.durable.result(3)
                self.assertFalse(duplicate.snapshot.done())
                self.assertEqual(c.metrics()['scheduling']['waiting_on_prefill'], 1)
                release.set()
                a = first.durable.result(5)
                b = duplicate.durable.result(5)
                self.assertEqual(a['snapshot'], b['snapshot'])
                calls = [ids for backend in backends.values() for ids in backend.forward_calls]
                self.assertEqual(calls.count((2, 5, 7)), 1)
                self.assertEqual(calls.count((1, 4)), 1)
                self.assertEqual(c.metrics()['scheduling']['inflight_prefills'], 0)
            finally:
                release.set()
                c.close()

    def test_rejects_unknown_scheduler(self):
        with self.assertRaises(ValueError):
            PoolConfig(scheduling='guess')


class SharedRebuildTests(unittest.TestCase):
    def test_evicted_prefix_rebuild_is_shared_between_independent_gpu_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            profile=dict(model_contract_id='cpu-model',execution_profile_id='cpu-exec',
                model_weights_sha256='a'*64,execution=dict(GPU=[3,4],capacity=64,prefill_chunk=8,
                    eos_tokens=[15],resume_contract='identical-native-operators-v1',
                    generated_operator='normal-verifier-decode',
                    native_request_protocol='consumed-greedy-ack-v1'))
            entered,release=threading.Event(),threading.Event()
            gating=threading.Event();backends={}
            class Gated(RebuildableRootFixtureBackend):
                def consume(self,lease,ids):
                    if gating.is_set() and tuple(ids)==(1,2,3):
                        entered.set()
                        if not release.wait(5):raise TimeoutError('test rebuild gate')
                    return super().consume(lease,ids)
            def factory(gpu):
                backend=Gated(profile);backend.piece=lambda t:bytes([65+t]);backend.is_eog=lambda t:t==15
                backends[gpu]=backend
                return BranchingModel(directory,backend,profile)
            c=ContextCoordinator(directory,PoolConfig(gpus=(3,4),ram_bytes=1024**2,
                disk_bytes=1024**2,min_available_bytes=0,scheduling='weighted-prefix'),factory).ready(5)
            try:
                root=c.create('main',(1,2,3),'shared-rebuild-root').durable.result(5)['snapshot']
                c.fork('a',root);c.fork('b',root)
                deadline=time.monotonic()+3
                while (c.pool.metrics()['pins'] or c.pool.metrics()['reserved_bytes']
                        or any(w['busy'] for w in c._workers.values())):
                    if time.monotonic()>deadline:self.fail('root publication did not drain')
                    time.sleep(.005)
                model=c._workers[3]['model'];key=c._key(model,root)
                if key in c.pool:c.pool.remove(key)
                for w in c._workers.values():w['model'].cache.pop(root,None)
                manifest=model.checkpoint_directory/(root+'.json')
                payload=json.loads(manifest.read_bytes())['manifest']['payload_sha256']
                manifest.unlink();(model.checkpoint_directory/(payload+'.state')).unlink()
                for backend in backends.values():backend.forward_calls.clear()
                gating.set()
                first=c.submit('a',(6,),1,'rebuild-a',expected_revision=0)
                self.assertTrue(entered.wait(2))
                second=c.submit('b',(7,),1,'rebuild-b',expected_revision=0)
                c.create('other',(8,9),'rebuild-unrelated').durable.result(3)
                self.assertFalse(second.snapshot.done())
                release.set()
                first.durable.result(5);second.durable.result(5)
                calls=[ids for b in backends.values() for ids in b.forward_calls]
                self.assertEqual(calls.count((1,2,3)),1)
                self.assertEqual(c.store.head('main').tip,root)
                self.assertEqual(c.metrics()['scheduling']['inflight_prefills'],0)
                self.assertTrue(manifest.is_file())
            finally:release.set();c.close()


if __name__ == '__main__':
    unittest.main()
