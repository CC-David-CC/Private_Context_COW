"""Namespace-bound fork/rewind coordinator controls; synthetic CPU fixture only."""
import threading
import tempfile
import unittest
import struct

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.store import InvalidOperation, Conflict, IntegrityError, NotFound
import serve.test_cow_presentation_flow as flow_fixture


class BranchActionModel(flow_fixture.FlowModel):
    plans = {**flow_fixture.FlowModel.plans, 'stop-op': (299,), 'fork-resume': (258, 299),
             'other-answer': (260,)}


class CoordinatorBranchActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trace=[]; self.models={}
        self.gate=threading.Event(); self.gate.set()
        self.config=PoolConfig(gpus=(0,),ram_bytes=1024*1024,disk_bytes=8*1024*1024,
                               min_available_bytes=0)
        def factory(gpu):
            model=BranchActionModel(self.temp.name,gpu,(0,),self.trace,self.gate)
            self.models[gpu]=model
            return model
        self.factory=factory
        self.coordinator=ContextCoordinator(self.temp.name,self.config,factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)

    def namespace_chat(self,name='chat',namespace='ns-one'):
        store=self.coordinator.store
        ns=store.create_namespace('Branch action test',namespace).namespace_id
        text=flow_fixture.CowPresentationFlowTests.make_text(self.coordinator)
        text.prepare(name,[{'role':'user','content':'hello'}],'root-'+name,namespace_id=ns).durable.result(5)
        return ns,store.head(name)

    def append(self,branch,revision,operation,outputs):
        # Use the real flow fixture's text adapter so Store presentations are
        # generated from the actual canonical spans and controller state.
        adapter=flow_fixture.CowPresentationFlowTests.make_text(self.coordinator)
        model=self.models[0]
        model.plans[operation]=tuple(outputs)
        state=self.coordinator.store.get_snapshot(self.coordinator.store.head(branch).tip)['controller']
        if state.get('status')=='idle':
            return adapter.answer(branch,revision,operation,len(outputs))
        return adapter.resume(branch,revision,operation,len(outputs))

    def test_namespaced_fork_is_idempotent_and_preserves_namespace(self):
        ns,head=self.namespace_chat()
        answer=self.append('chat',head.revision,'answer-op',(257,))
        answer.durable.result(5)
        source=self.coordinator.store.head('chat')
        before=dict(self.models[0].stats)
        fork=self.coordinator.fork('copy',source.tip,source_branch='chat',operation_id='fork-op')
        self.assertEqual(fork.tip,source.tip)
        self.assertEqual(self.coordinator.store.branch_namespace('copy'),ns)
        retry=self.coordinator.fork('copy',source.tip,source_branch='chat',operation_id='fork-op')
        self.assertEqual(retry,fork)
        self.assertEqual(self.models[0].stats,before)
        with self.assertRaises(InvalidOperation):
            self.coordinator.fork('copy',head.tip,source_branch='chat',operation_id='fork-op')

    def test_fork_rejects_cross_namespace_savedpoint_and_unpublished_snapshot(self):
        ns1,head=self.namespace_chat()
        ns2,other=self.namespace_chat('other','ns-two')
        self.assertEqual(other.tip,head.tip)  # identical computational roots, distinct namespaces
        self.append('other',other.revision,'other-answer',(260,)).durable.result(5)
        foreign=self.coordinator.store.head('other').tip
        self.assertNotEqual(ns1,ns2)
        before=self.coordinator.store.head('chat')
        with self.assertRaises(InvalidOperation):
            self.coordinator.fork('foreign-copy',foreign,source_branch='chat',operation_id='foreign-op')
        self.assertEqual(self.coordinator.store.head('chat'),before)
        with self.assertRaises(NotFound):self.coordinator.store.head('foreign-copy')
        # A second namespace's presentation receipt cannot make an otherwise
        # unreferenced snapshot a savedpoint in this namespace.
        self.assertTrue(self.coordinator.store.presentation_catalog(ns2)['events'])
        model=self.models[0]
        state=self.coordinator.store.get_snapshot(head.tip)['controller']
        unpublished=self.coordinator.store.append(head.tip,(12,),state)
        model.image(unpublished)
        prepared=model.prepare_checkpoint(unpublished)
        cp=prepared.checkpoint
        packed=cp.blob+struct.pack('<'+str(len(cp.frontier))+'f',*cp.frontier)
        (model.checkpoint_directory/(cp.checksum+'.state')).write_bytes(packed)
        (model.checkpoint_directory/(unpublished+'.json')).write_bytes(prepared.envelope)
        model.validate_checkpoint(unpublished)
        with self.assertRaises(InvalidOperation):
            self.coordinator.fork('unpublished-copy',unpublished,source_branch='chat',
                                  operation_id='unpublished-op')
        with self.assertRaises(NotFound):self.coordinator.store.head('unpublished-copy')

    def test_fork_accepts_published_same_namespace_savedpoint(self):
        ns,head=self.namespace_chat()
        first=self.append('chat',head.revision,'answer-op',(257,))
        first.durable.result(5)
        active=self.coordinator.store.head('chat')
        # Fork at the active point and create a sibling continuation that is
        # catalog-published but not an ancestor of the source branch tip.
        self.coordinator.store.fork('sibling',active.tip,namespace_id=ns,operation_id='seed-sibling')
        sibling=self.coordinator.store.head('sibling')
        continuation=self.append('sibling',sibling.revision,'fork-resume',(258,299))
        continuation.durable.result(5)
        target=self.coordinator.store.head('sibling').tip
        self.assertFalse(self.coordinator._is_ancestor(self.coordinator.store,target,active.tip))
        with self.assertRaises(InvalidOperation):
            self.coordinator.rewind_presented('chat',active.revision,target,'not-ancestor-rewind')
        self.assertEqual(self.coordinator.store.head('chat'),active)
        result=self.coordinator.fork('saved-copy',target,source_branch='chat',operation_id='saved-op')
        self.assertEqual(result.tip,target)
        self.assertEqual(self.coordinator.store.branch_namespace('saved-copy'),ns)

    def test_typed_rewind_binds_exact_source_and_replays_after_later_move(self):
        ns,head=self.namespace_chat()
        self.append('chat',head.revision,'answer-op',(257,)).durable.result(5)
        active=self.coordinator.store.head('chat')
        self.append('chat',active.revision,'stop-op',(299,)).durable.result(5)
        stopped=self.coordinator.store.head('chat')
        before=dict(self.models[0].stats)
        result=self.coordinator.rewind_presented('chat',stopped.revision,active.tip,'rewind-op')
        self.assertEqual(result.tip,active.tip)
        receipt=self.coordinator.store.presentation_receipt('rewind-op')
        descriptor=receipt['presentation']
        self.assertEqual(descriptor['kind'],'rewind')
        self.assertEqual(descriptor['namespace_id'],ns)
        self.assertEqual(descriptor['renderer_id'],'b'*64)
        source_count=len(self.coordinator.store.lineage(stopped.tip)[1])
        self.assertEqual(descriptor['source'],dict(snapshot=stopped.tip,revision=stopped.revision,
                                                    token_count=source_count))
        self.assertFalse(descriptor['turn_complete'])
        self.assertEqual(self.models[0].stats,before)
        moved=self.coordinator.store.move('chat',result.revision,stopped.tip,'advance-after-rewind')
        replay=self.coordinator.rewind_presented('chat',stopped.revision,active.tip,'rewind-op')
        self.assertEqual(replay,result)
        self.assertEqual(self.coordinator.store.head('chat'),moved)
        with self.assertRaises(Conflict):
            self.coordinator.rewind_presented('chat',stopped.revision,stopped.tip,'rewind-op')

    def test_corruption_rejects_but_missing_checkpoint_allows_metadata_fork_and_rewind(self):
        ns,head=self.namespace_chat()
        self.append('chat',head.revision,'answer-op',(257,)).durable.result(5)
        target=self.coordinator.store.head('chat')
        self.append('chat',target.revision,'stop-op',(299,)).durable.result(5)
        stopped=self.coordinator.store.head('chat')
        model=self.models[0]
        manifest=model.checkpoint_manifest(target.tip)
        path=model.checkpoint_directory/(manifest['payload_sha256']+'.state')
        payload=path.read_bytes()
        path.write_bytes(payload[:-1]+bytes([payload[-1]^1]))
        with self.assertRaises(IntegrityError):
            self.coordinator.fork('bad-copy',target.tip,source_branch='chat',operation_id='bad-fork')
        self.assertEqual(self.coordinator.store.head('chat'),stopped)
        with self.assertRaises(NotFound):self.coordinator.store.head('bad-copy')
        with self.assertRaises(IntegrityError):
            self.coordinator.rewind_presented('chat',stopped.revision,target.tip,'bad-rewind')
        self.assertEqual(self.coordinator.store.head('chat'),stopped)
        path.write_bytes(payload)
        (model.checkpoint_directory/(target.tip+'.json')).unlink()
        path.unlink()

        before_missing_actions=dict(model.stats)
        forked=self.coordinator.fork('missing-copy',target.tip,source_branch='chat',
                                     operation_id='missing-fork')
        self.assertEqual(forked.tip,target.tip)
        self.assertEqual(self.coordinator.store.branch_namespace('missing-copy'),ns)
        self.assertEqual(self.coordinator.fork('missing-copy',target.tip,
            source_branch='chat',operation_id='missing-fork'),forked)
        self.assertEqual(model.stats,before_missing_actions,
                         'metadata fork and idempotent retry perform no forward work')

        rewound=self.coordinator.rewind_presented('chat',stopped.revision,target.tip,
                                                   'missing-rewind')
        self.assertEqual(rewound.tip,target.tip)
        self.assertEqual(self.coordinator.store.presentation_receipt('missing-rewind')
                         ['presentation']['kind'],'rewind')
        self.assertEqual(self.coordinator.rewind_presented('chat',stopped.revision,
            target.tip,'missing-rewind'),rewound)
        self.assertEqual(self.coordinator.store.head('chat'),rewound)
        self.assertEqual(model.stats,before_missing_actions,
                         'metadata rewind and idempotent retry perform no forward work')

    def test_rewind_rejects_stale_revision_and_pending_branch_work(self):
        ns,head=self.namespace_chat()
        self.append('chat',head.revision,'answer-op',(257,)).durable.result(5)
        active=self.coordinator.store.head('chat')
        self.gate.clear()
        request=self.coordinator.submit('chat',(),2,'stop-op',expected_revision=active.revision)
        before=self.coordinator.store.head('chat')
        try:
            with self.assertRaises(Conflict):
                self.coordinator.rewind_presented('chat',active.revision,head.tip,'pending-rewind')
            self.assertEqual(self.coordinator.store.head('chat'),before)
        finally:
            self.gate.set()
        request.durable.result(5)
        stopped=self.coordinator.store.head('chat')
        with self.assertRaises(Conflict):
            self.coordinator.rewind_presented('chat',active.revision,head.tip,'stale-rewind')
        self.assertEqual(self.coordinator.store.head('chat'),stopped)

    def test_legacy_rewind_serializes_store_move_with_new_admission(self):
        root_request=self.coordinator.create('legacy',(20,21),'legacy-root')
        root_result=root_request.durable.result(5)
        root_snapshot=root_result['snapshot']
        model=self.models[0]
        model.plans['legacy-first']=(257,)
        first=self.coordinator.submit('legacy',(),1,'legacy-first',expected_revision=1)
        first.durable.result(5)
        source=self.coordinator.store.head('legacy')
        original_move=self.coordinator.store.move
        move_entered=threading.Event();allow_move=threading.Event();admission_done=threading.Event()
        outcome={}

        def delayed_move(*args,**kwargs):
            move_entered.set()
            if not allow_move.wait(5):
                raise TimeoutError('test did not release the serialized Store move')
            return original_move(*args,**kwargs)

        self.coordinator.store.move=delayed_move
        def rewind():
            try:
                outcome['rewind']=self.coordinator.rewind('legacy',source.revision,root_snapshot,'legacy-rewind')
            except BaseException as exc:
                outcome['rewind_error']=exc
        thread=threading.Thread(target=rewind)
        thread.start()
        self.assertTrue(move_entered.wait(5),'legacy rewind did not reach Store.move')

        def admit():
            try:
                outcome['request']=self.coordinator.submit('legacy',(),1,'legacy-second',
                    expected_revision=source.revision)
            except BaseException as exc:
                outcome['admission_error']=exc
            finally:
                admission_done.set()
        admission=threading.Thread(target=admit)
        admission.start()
        try:
            self.assertFalse(admission_done.wait(.05),'admission crossed an in-progress rewind')
        finally:
            allow_move.set()
        thread.join(5);admission.join(5)
        self.assertFalse(thread.is_alive());self.assertFalse(admission.is_alive())
        self.assertNotIn('rewind_error',outcome)
        self.assertEqual(outcome['rewind'].tip,root_snapshot)
        self.assertIsInstance(outcome.get('admission_error'),Conflict)
        self.assertNotIn('request',outcome)


if __name__=='__main__':
    unittest.main()
