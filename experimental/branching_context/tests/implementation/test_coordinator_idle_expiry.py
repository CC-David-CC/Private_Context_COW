"""Real threads/files, synthetic scheduler model; no GPU or numerical claims."""
from pathlib import Path
import tempfile
import threading
import time
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from test_pool_coordinator import Model


class CoordinatorIdleExpiryTests(unittest.TestCase):
    def start(self, seconds):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        models={}
        gate=threading.Event()
        gate.set()
        def factory(gpu):
            models[gpu]=Model(temporary.name,gpu,(0,),[],gate)
            return models[gpu]
        coordinator=ContextCoordinator(temporary.name,PoolConfig(gpus=(0,),ram_bytes=4096,
            disk_bytes=16384,min_available_bytes=0,checkpoint_idle_seconds=seconds),factory)
        self.addCleanup(coordinator.close)
        coordinator.ready(timeout=5)
        return coordinator,models

    def wait_for(self, predicate):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if predicate(): return
            time.sleep(.005)
        self.fail('Timed out waiting for idle maintenance')

    def test_duration_is_wired_to_both_pools_and_timer_stops_on_close(self):
        coordinator,_=self.start(1800)
        self.assertEqual(coordinator.pool.checkpoint_idle_seconds,1800)
        self.assertEqual(coordinator.disk.checkpoint_idle_seconds,1800)
        thread=coordinator._expiry_thread
        self.assertTrue(thread.is_alive())
        self.assertEqual(coordinator.metrics()['idle_expiry']['seconds'],1800)
        coordinator.close()
        self.assertFalse(thread.is_alive())

    def test_default_has_no_expiry_thread(self):
        coordinator,_=self.start(None)
        self.assertIsNone(coordinator._expiry_thread)
        self.assertEqual(coordinator.collect_expired(),dict(ram=0,disk=0))

    def test_timer_reclaims_idle_derived_copies_without_a_new_request(self):
        coordinator,_=self.start(.05)
        snapshot=coordinator.create('base',(1,2),'create').durable.result(5)['snapshot']
        manifest=coordinator.disk.directory/(snapshot+'.json')
        self.assertTrue(manifest.exists())
        before={str(path.relative_to(coordinator.store.root)):path.read_bytes()
            for folder in ('objects','references') for path in (coordinator.store.root/folder).rglob('*.json')}
        self.wait_for(lambda:not manifest.exists() and coordinator.pool.metrics()['resident_bytes']==0)
        self.assertEqual(coordinator.store.head('base').tip,snapshot)
        after={str(path.relative_to(coordinator.store.root)):path.read_bytes()
            for folder in ('objects','references') for path in (coordinator.store.root/folder).rglob('*.json')}
        self.assertEqual(before,after)
        self.assertGreater(coordinator.metrics()['idle_expiry']['runs'],0)

    def test_timer_reports_collection_error_and_recovers(self):
        coordinator,_=self.start(.04)
        original=coordinator.disk.collect_expired
        def fail(): raise OSError('injected expiry I/O failure')
        coordinator.disk.collect_expired=fail
        self.wait_for(lambda:coordinator.metrics()['idle_expiry']['last_error'] is not None)
        self.assertIn('injected expiry I/O failure',coordinator.metrics()['idle_expiry']['last_error'])
        coordinator.disk.collect_expired=original
        self.wait_for(lambda:coordinator.metrics()['idle_expiry']['last_error'] is None)
        self.assertTrue(coordinator._expiry_thread.is_alive())

    def test_completed_resident_request_refreshes_disk_source_at_end(self):
        coordinator,models=self.start(1800)
        clock=[time.time_ns()+10_000_000_000]
        coordinator.disk._clock_ns=lambda:clock[0]
        snapshot=coordinator.create('base',(1,2),'create').durable.result(5)['snapshot']
        original=models[0].run_greedy_request
        def run(*args,**kwargs):
            result=original(*args,**kwargs)
            clock[0]+=30_000_000_000
            return result
        models[0].run_greedy_request=run
        coordinator.submit('base',(3,),1,'answer').durable.result(5)
        self.assertGreaterEqual((coordinator.disk.directory/(snapshot+'.json')).stat().st_mtime_ns,clock[0])


if __name__=='__main__':
    unittest.main()
