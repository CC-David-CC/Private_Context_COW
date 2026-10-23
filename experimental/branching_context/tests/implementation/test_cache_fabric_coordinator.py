"""COW integration controls on a CPU backend; no GPU performance claims."""
from dataclasses import replace
import unittest
import time
from test_pool_coordinator import CoordinatorTests
from context_time_travel.cache_inventory import InventoryView
from context_time_travel.coordinator import ContextCoordinator


class FabricCoordinatorTests(unittest.TestCase):
    start=CoordinatorTests.start
    def test_capability_observation_does_not_run_model_or_claim_unknown_native_features(self):
        c,models,trace,gate=self.start()
        before=dict(models[0].stats)
        value=c.capabilities()
        self.assertEqual(models[0].stats,before)
        self.assertEqual(c.store.branches(),())
        self.assertFalse(value['features']['native_residency']['supported'])
        self.assertFalse(value['features']['cache_inventory']['enabled'])
        self.assertEqual(value['limits']['ram_cache_bytes'],c.config.ram_bytes)

    def test_inventory_recovers_current_ram_and_disk_without_model_work(self):
        c,models,trace,gate=self.start()
        # Enable on a fresh coordinator; configuration stays immutable.
        directory,factory=c.store.root,c._factory
        config=replace(c.config,cache_inventory=True,scheduling='transfer-aware')
        c.close(); c=ContextCoordinator(directory,config,factory).ready(5); self.addCleanup(c.close)
        root=c.create('base',(1,2),'create-fabric').durable.result(5)['snapshot']
        before=dict(models[0].stats); head=c.store.head('base')
        view=InventoryView(c._inventory.node); view.apply(c.cache_inventory())
        self.assertEqual({r['tier'] for r in view.entries},{'ram','disk'})
        self.assertEqual({r['snapshot'] for r in view.entries},{root})
        self.assertEqual(models[0].stats,before); self.assertEqual(c.store.head('base'),head)
        cursor=view.cursor
        deadline=time.monotonic()+3
        while c.pool.metrics()['pins']:
            if time.monotonic()>deadline:self.fail('publication pin did not drain')
            time.sleep(.005)
        c.pool.remove(('m','e',root))
        view.apply(c.cache_inventory(**cursor))
        self.assertEqual([r['tier'] for r in view.entries],['disk'])
        self.assertEqual(models[0].stats,before)

    def test_transfer_policy_runs_existing_branch_and_publication_contract(self):
        c,models,trace,gate=self.start((0,1))
        directory,factory=c.store.root,c._factory
        config=replace(c.config,scheduling='transfer-aware')
        c.close(); c=ContextCoordinator(directory,config,factory).ready(5); self.addCleanup(c.close)
        root=c.create('base',(1,2),'fabric-root').durable.result(5)['snapshot']
        c.fork('a',root); c.fork('b',root)
        one=c.submit('a',(3,),2,'fabric-a').durable.result(5)
        two=c.submit('b',(4,),2,'fabric-b').durable.result(5)
        self.assertEqual(c.store.head('base').tip,root)
        self.assertEqual(c.store.lineage(one['snapshot'])[1],[1,2,3,0,1])
        self.assertEqual(c.store.lineage(two['snapshot'])[1],[1,2,4,0,1])
        self.assertIsNotNone(c.metrics()['scheduling']['last_placement'])

    def test_inventory_restart_rebuilds_disk_hints_and_drops_old_ram_locations(self):
        c,models,trace,gate=self.start()
        directory,factory=c.store.root,c._factory
        config=replace(c.config,cache_inventory=True)
        c.close();c=ContextCoordinator(directory,config,factory).ready(5);self.addCleanup(c.close)
        root=c.create('base',(1,2),'restart-inventory').durable.result(5)['snapshot']
        view=InventoryView(c._inventory.node);view.apply(c.cache_inventory());old=view.cursor
        c.close();resumed=ContextCoordinator(directory,config,factory).ready(5);self.addCleanup(resumed.close)
        packet=resumed.cache_inventory(**old)
        self.assertEqual(packet['kind'],'snapshot');view.apply(packet)
        self.assertEqual([(r['snapshot'],r['tier']) for r in view.entries],[(root,'disk')])
        self.assertEqual(resumed.store.head('base').tip,root)


if __name__=='__main__':unittest.main()
