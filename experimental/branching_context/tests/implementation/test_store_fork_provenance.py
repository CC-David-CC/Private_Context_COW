"""Additive source-branch provenance for namespace-bound forks."""
import hashlib
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from test_pool_coordinator import Model
from context_time_travel import Conflict, IntegrityError, InvalidOperation, NotFound, Store
from context_time_travel.store import _hash, _json_bytes


class StoreForkProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='store-fork-source-')
        self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name)
        self.namespace=self.store.create_namespace('Source provenance','source-chat').namespace_id
        self.other_namespace=self.store.create_namespace('Other source','other-chat').namespace_id
        self.tokens=[7,8,9]
        self.root=self.store.create_root('model',self.tokens,{'status':'idle'})

    def root_presentation(self,namespace,identity,content):
        return dict(schema=1,namespace_id=namespace,kind='root',
            messages=[dict(id=identity*32,role='user',content=content)],
            source=dict(snapshot=None,revision=0,token_count=0),
            input_span=dict(start=0,count=len(self.tokens),
                token_ids_sha256=_hash(_json_bytes(self.tokens))),
            output_span=None,renderer_id=hashlib.sha256(b'source-renderer').hexdigest(),
            turn_complete=False)

    def namespace_source(self,name,namespace,operation,user,content):
        return self.store.fork(name,self.root,namespace_id=namespace,operation_id=operation,
            presentation=self.root_presentation(namespace,user,content))

    def fork_from(self,name,snapshot,operation,source_branch,revision,tip,namespace=None):
        return self.store.fork(name,snapshot,namespace_id=namespace or self.namespace,
            operation_id=operation,source_branch=source_branch,source_revision=revision,
            source_tip=tip)

    def test_distinct_annotated_origins_are_bound_even_for_same_snapshot(self):
        self.namespace_source('left',self.namespace,'left-root','a','left conversation')
        self.namespace_source('right',self.namespace,'right-root','b','right conversation')
        self.assertEqual(self.store.head('left').tip,self.store.head('right').tip)
        self.fork_from('left-copy',self.root,'fork-left','left',0,self.root)
        self.fork_from('right-copy',self.root,'fork-right','right',0,self.root)
        catalog=self.store.presentation_catalog(self.namespace)
        by_operation={event['operation_id']:event for event in catalog['events']}
        left=by_operation['fork-left'];right=by_operation['fork-right']
        self.assertEqual(left['event']['source'],dict(branch='left',revision=0,tip=self.root))
        self.assertEqual(right['event']['source'],dict(branch='right',revision=0,tip=self.root))
        self.assertNotEqual(left['receipt']['digest'],right['receipt']['digest'])

    def test_exact_retry_uses_captured_source_after_it_moves_and_rejects_other_branch(self):
        self.namespace_source('left',self.namespace,'left-root','a','left')
        self.namespace_source('right',self.namespace,'right-root','b','right')
        original=self.fork_from('copy',self.root,'fork-stable','left',0,self.root)
        child=self.store.append(self.root,[10],{'status':'stopped'})
        self.store.move('left',0,child,'move-left')
        retry=self.fork_from('copy',self.root,'fork-stable','left',0,self.root)
        self.assertEqual(retry,original)
        with self.assertRaises(InvalidOperation):
            self.fork_from('copy',self.root,'fork-stable','right',0,self.root)
        self.assertEqual(self.store.head('copy'),original)

    def test_coordinator_binds_original_source_revision_and_retry(self):
        self.namespace_source('left',self.namespace,'left-root','a','left')
        self.namespace_source('right',self.namespace,'right-root','b','right')
        gate=threading.Event();gate.set()
        coordinator=ContextCoordinator(self.temp.name,
            PoolConfig(gpus=(0,),ram_bytes=4096,disk_bytes=16384,min_available_bytes=0),
            lambda gpu: Model(self.temp.name,gpu,(0,),[],gate)).ready(5)
        self.addCleanup(coordinator.close)
        first=coordinator.fork('coordinator-copy',self.root,source_branch='left',
                               operation_id='coordinator-fork')
        event=next(item for item in self.store.presentation_catalog(self.namespace)['events']
                   if item['operation_id']=='coordinator-fork')
        self.assertEqual(event['event']['source'],
                         dict(branch='left',revision=0,tip=self.root))
        child=self.store.append(self.root,[10],{'status':'stopped'})
        self.store.move('left',0,child,'move-coordinator-source')
        retry=coordinator.fork('coordinator-copy',self.root,source_branch='left',
                               operation_id='coordinator-fork')
        self.assertEqual(retry,first)
        with self.assertRaises(InvalidOperation):
            coordinator.fork('coordinator-copy',self.root,source_branch='right',
                             operation_id='coordinator-fork')

    def test_first_fork_rejects_stale_source_and_wrong_namespace(self):
        self.namespace_source('left',self.namespace,'left-root','a','left')
        child=self.store.append(self.root,[10],{'status':'stopped'})
        self.store.move('left',0,child,'move-left')
        with self.assertRaises(Conflict):
            self.fork_from('stale-copy',self.root,'stale-fork','left',0,self.root)
        with self.assertRaises(InvalidOperation):
            self.fork_from('foreign-copy',self.root,'foreign-fork','left',1,child,
                           namespace=self.other_namespace)
        with self.assertRaises(InvalidOperation):
            self.store.fork('partial-copy',self.root,namespace_id=self.namespace,
                            operation_id='partial-source',source_branch='left')
        with self.assertRaises(NotFound):self.store.head('stale-copy')
        with self.assertRaises(NotFound):self.store.head('foreign-copy')

    def test_source_snapshot_remains_retained_after_source_rewinds(self):
        self.namespace_source('left',self.namespace,'left-root','a','left')
        child=self.store.append(self.root,[10],{'status':'stopped'})
        self.store.move('left',0,child,'move-left')
        self.fork_from('copy',self.root,'fork-from-child','left',1,child)
        self.store.move('left',1,self.root,'rewind-left')
        self.assertIn(child,self.store.retained_snapshots())
        event=next(e for e in self.store.presentation_catalog(self.namespace)['events']
                   if e['operation_id']=='fork-from-child')
        self.assertEqual(event['event']['source']['tip'],child)

    def test_source_less_namespace_fork_keeps_legacy_event_and_digest_shape(self):
        self.store.fork('legacy-copy',self.root,namespace_id=self.namespace,
                        operation_id='legacy-fork')
        record=self.store._read_branch('legacy-copy')
        event=record['events'][0]
        receipt=record['receipts']['legacy-fork']
        self.assertNotIn('source',event)
        request=dict(type='fork',name='legacy-copy',snapshot=self.root,
                     namespace_id=self.namespace)
        self.assertEqual(receipt['digest'],_hash(_json_bytes(request)))


if __name__=='__main__':
    unittest.main()
