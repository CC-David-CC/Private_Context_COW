"""Branch references remain usable after optional numerical eviction.

CPU-only controls; numerical restore and native reconstruction are separate.
"""
import json
import subprocess
import sys
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.store import Conflict, IntegrityError
from test_missing_source_rebuild import RebuildModel
from serve import test_cow_presentation_flow as flow


class BranchActionModel(RebuildModel):
    def validate_checkpoint(self,snapshot):
        self._load_checkpoint(snapshot,self.history(snapshot))
        return self.checkpoint_manifest(snapshot)


class CanonicalBranchActionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = {}
        def factory(gpu):
            model = BranchActionModel(self.temp.name, gpu, (0,))
            self.models[gpu] = model
            return model
        self.factory = factory
        self.config = PoolConfig(gpus=(0,), ram_bytes=1024**2,
                                 disk_bytes=8*1024**2, min_available_bytes=0)
        self.start()

    def start(self):
        self.c = ContextCoordinator(self.temp.name, self.config, self.factory)
        self.addCleanup(self.c.close)
        self.c.ready(5)

    def root_and_answer(self):
        root = self.c.create('main', (1,2,3), 'create').durable.result(5)
        answer = self.c.submit('main', (4,), 1, 'first', expected_revision=1).durable.result(5)
        return root, answer

    def restart_without_numerical_files(self):
        self.c.close()
        directory = self.c.disk.directory
        # Only this test's TemporaryDirectory is owned; canonical files stay.
        for path in directory.iterdir():
            if path.suffix in ('.json','.state'):
                self.assertEqual(path.parent.resolve(), directory.resolve())
                self.assertFalse(path.is_symlink())
                path.unlink()
        self.start()

    def test_fork_missing_checkpoint_is_metadata_only_then_append_reports_rebuild(self):
        root, answer = self.root_and_answer()
        old_receipt = self.c.store._read_refs()['receipts']['first']
        old_history = self.models[0].history(answer['snapshot'])
        self.restart_without_numerical_files()
        before = self.models[0].work_counters()
        copied = self.c.fork('copy', answer['snapshot'])
        self.assertEqual(copied.tip, answer['snapshot'])
        self.assertEqual(self.models[0].work_counters(), before)
        self.assertEqual(self.c.store._read_refs()['receipts']['first'], old_receipt)
        self.assertEqual(self.models[0].history(copied.tip), old_history)
        request = self.c.submit('copy', (6,), 1, 'copy-answer', expected_revision=copied.revision)
        result = request.snapshot.result(5)
        request.durable.result(5)
        self.assertEqual(result['source_recomputation']['prefix_forward_tokens'],len(old_history))
        self.assertIs(result['source_recomputation']['prior_computation_preserved'],False)
        self.assertEqual(result['counter_deltas']['forward_tokens'],2)
        self.assertEqual(self.c.store.head('main').tip,answer['snapshot'])

    def test_rewind_missing_checkpoint_is_metadata_only_and_retry_is_stable(self):
        root, answer = self.root_and_answer()
        self.restart_without_numerical_files()
        before = self.models[0].work_counters()
        old = self.c.store.head('main')
        rewound = self.c.rewind('main', old.revision, root['snapshot'], 'rewind')
        self.assertEqual(rewound.tip, root['snapshot'])
        self.assertEqual(rewound.revision, old.revision+1)
        self.assertEqual(self.c.rewind('main', old.revision, root['snapshot'], 'rewind'),rewound)
        self.assertEqual(self.models[0].work_counters(),before)
        request = self.c.submit('main',(7,),1,'after-rewind',expected_revision=rewound.revision)
        result = request.snapshot.result(5)
        request.durable.result(5)
        self.assertEqual(result['source_recomputation']['prefix_forward_tokens'],3)
        self.assertEqual(self.models[0].history(result['snapshot']),(1,2,3,7,0))
        self.assertEqual(self.models[0].history(answer['snapshot']),(1,2,3,4,0))

    def test_missing_checkpoint_does_not_remove_rewind_cas_guard(self):
        root, answer = self.root_and_answer()
        self.restart_without_numerical_files()
        before = self.c.store.head('main')
        with self.assertRaises(Conflict):
            self.c.rewind('main',before.revision-1,root['snapshot'],'stale')
        self.assertEqual(self.c.store.head('main'),before)
        self.assertEqual(self.models[0].work_counters()['forward_tokens'],0)

    def test_intact_checkpoint_fork_does_not_nest_independent_store_locks(self):
        # Each model and coordinator has its own Store handle. A nested file
        # lock can deadlock even when both handles belong to the same process.
        # Run in a disposable CPU-only subprocess so a regression is bounded.
        code = '''
import sys
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from test_canonical_branch_actions import BranchActionModel
directory = sys.argv[1]
config = PoolConfig(gpus=(0,),ram_bytes=1024**2,disk_bytes=8*1024**2,min_available_bytes=0)
c = ContextCoordinator(directory,config,lambda gpu: BranchActionModel(directory,gpu,(0,)))
try:
    c.ready(5)
    receipt = c.create('main',(1,2,3),'create').durable.result(5)
    copied = c.fork('copy',receipt['snapshot'])
    assert copied.tip == receipt['snapshot']
finally:
    c.close()
'''
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable,'-c',code,directory],
                capture_output=True,timeout=5)
        self.assertEqual(completed.returncode,0,completed.stderr.decode())


class PresentedCanonicalBranchActionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = {}
        gate = threading.Event();gate.set()
        def factory(gpu):
            model = flow.FlowModel(self.temp.name,gpu,(0,),[],gate)
            self.models[gpu] = model
            return model
        self.factory = factory
        self.config = PoolConfig(gpus=(0,),ram_bytes=1024**2,disk_bytes=8*1024**2,
                                 min_available_bytes=0)
        self.start()
        self.ns = self.c.store.create_namespace('Tree','namespace').namespace_id
        text = flow.CowPresentationFlowTests.make_text(self.c)
        text.prepare('chat',[dict(role='user',content='hello')],'root',
                     namespace_id=self.ns).durable.result(5)
        self.root = self.c.store.head('chat')
        text.answer('chat',self.root.revision,'answer-op',1).durable.result(5)
        self.answered = self.c.store.head('chat')

    def start(self):
        self.c = ContextCoordinator(self.temp.name,self.config,self.factory)
        self.addCleanup(self.c.close)
        self.c.ready(5)

    def drop_target(self,snapshot,*,keep_manifest=False):
        # Metadata actions must not activate inference or depend on resident RAM.
        self.c.close()
        model = self.models[0]
        path = model.checkpoint_directory/(snapshot+'.json')
        manifest = json.loads(path.read_bytes())['manifest']
        (model.checkpoint_directory/(manifest['payload_sha256']+'.state')).unlink()
        if not keep_manifest:
            path.unlink()
        self.start()

    def test_namespace_fork_after_eviction_preserves_presentation_and_is_idempotent(self):
        self.drop_target(self.answered.tip)
        before = self.models[0].work_counters()
        copied = self.c.fork('copy',self.answered.tip,source_branch='chat',operation_id='fork')
        self.assertEqual(copied.tip,self.answered.tip)
        self.assertEqual(self.c.store.branch_namespace('copy'),self.ns)
        self.assertEqual(self.c.fork('copy',self.answered.tip,
            source_branch='chat',operation_id='fork'),copied)
        self.assertEqual(self.c.store.head('chat'),self.answered)
        self.assertEqual(self.models[0].work_counters(),before)
        events=self.c.store.presentation_catalog(self.ns)['events']
        receipt=next(event for event in events if event.get('operation_id')=='fork')
        self.assertEqual(receipt['result']['tip'],copied.tip)
        self.assertEqual(receipt['event']['type'],'fork')

    def test_typed_rewind_after_eviction_preserves_presentation_and_retry(self):
        self.drop_target(self.root.tip)
        before = self.models[0].work_counters()
        result = self.c.rewind_presented('chat',self.answered.revision,self.root.tip,'rewind')
        self.assertEqual(result.tip,self.root.tip)
        self.assertEqual(self.c.store.presentation_receipt('rewind')['presentation']['kind'],'rewind')
        self.assertEqual(self.c.rewind_presented('chat',self.answered.revision,
            self.root.tip,'rewind'),result)
        self.assertEqual(self.models[0].work_counters(),before)

    def test_typed_rewind_retry_outlives_later_numerical_eviction(self):
        result = self.c.rewind_presented('chat',self.answered.revision,self.root.tip,'rewind')
        receipt = self.c.store.presentation_receipt('rewind')
        self.drop_target(self.root.tip)
        self.assertEqual(self.c.rewind_presented('chat',self.answered.revision,
            self.root.tip,'rewind'),result)
        self.assertEqual(self.c.store.presentation_receipt('rewind'),receipt)

    def test_present_manifest_missing_payload_is_not_hidden_by_fork(self):
        self.drop_target(self.answered.tip,keep_manifest=True)
        with self.assertRaises((IntegrityError,ValueError)):
            self.c.fork('copy',self.answered.tip,source_branch='chat',operation_id='fork')
        self.assertEqual(self.c.store.head('chat'),self.answered)
        self.assertNotIn('copy',{branch.name for branch in self.c.store.branches()})

    def test_present_manifest_missing_payload_is_not_hidden_by_typed_rewind(self):
        self.drop_target(self.root.tip,keep_manifest=True)
        with self.assertRaises((IntegrityError,ValueError)):
            self.c.rewind_presented('chat',self.answered.revision,self.root.tip,'rewind')
        self.assertEqual(self.c.store.head('chat'),self.answered)


if __name__ == '__main__':
    unittest.main()
