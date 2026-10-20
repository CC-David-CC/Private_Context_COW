"""Worker-owned root preparation contract; synthetic CPU metadata, not GPU evidence."""
from contextlib import contextmanager
from concurrent.futures import CancelledError
import errno
from types import SimpleNamespace
from pathlib import Path
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolConfig, PoolFull
from context_time_travel.real import BranchingModel
from context_time_travel.store import Conflict, InvalidOperation
from test_vision_api_binding import NoArtifacts
from test_vision_coordinator import ServingMixedCPUBackend
from test_vision_execution import spans


class RootPreparer:
    identity='a'*64
    def __init__(self):
        self.calls=0;self.active=0;self.entered=threading.Event();self.released=threading.Event()
        self.gate=threading.Event();self.gate.set();self.fail=False;self.bad_history=False
        self.threads=[];self.release_checks=[];self.bad_receipt=False
    def validate(self,request):
        if not isinstance(request,dict) or not isinstance(request.get('embedding'),str):
            raise InvalidOperation('Invalid synthetic input request')
        return request
    @contextmanager
    def prepare(self,model,request,*,pool,disk,cancelled=None):
        self.calls+=1;self.active+=1;self.threads.append(threading.get_ident());self.entered.set()
        try:
            if not self.gate.wait(5):raise TimeoutError('Synthetic preparer was not released')
            if self.fail:raise RuntimeError('Synthetic encoder failed')
            history=InputHistory(spans(embedding_sha256=request['embedding']))
            yield SimpleNamespace(history=history if not self.bad_history else history.native_tokens,
                artifacts=NoArtifacts(model.store),receipt=(dict(invalid=object()) if self.bad_receipt
                    else dict(kind='synthetic-cpu-preparation')))
        finally:
            self.active-=1
            self.release_checks.append(any(model.store.references.glob('*.json')))
            self.released.set()


class RootPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.preparer=RootPreparer();self.models=[];self.workers=[];self.manager=None
        self.write_gate=threading.Event();self.write_gate.set();self.write_entered=threading.Event()
        self.start();self.addCleanup(self.close)
        self.namespace=self.manager.store.create_namespace('Prepared images','namespace').namespace_id

    def start(self):
        profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-root',model_weights_sha256='a'*64,
            execution=dict(GPU=[0],resume_contract='identical-mixed-native-operators-v1',generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1',prefill_chunk=128,embedding_width=4,capacity=128,eos_tokens=[]))
        def factory(gpu):
            self.workers.append(threading.get_ident())
            backend=ServingMixedCPUBackend(profile)
            model=BranchingModel(self.tmp.name,backend,profile);self.models.append(model)
            write=model.save_prepared_branch
            def delayed_write(*args,**kwargs):
                self.write_entered.set()
                if not self.write_gate.wait(5):raise TimeoutError('Writer was not released')
                return write(*args,**kwargs)
            model.save_prepared_branch=delayed_write
            return model
        self.manager=ContextCoordinator(self.tmp.name,
            PoolConfig(gpus=(0,),ram_bytes=1024**2,disk_bytes=1024**2,min_available_bytes=0),
            factory,input_preparer=self.preparer)
        self.manager.ready(5)

    def close(self):
        self.write_gate.set();self.preparer.gate.set()
        if self.manager is not None:self.manager.close();self.manager=None

    def root(self,branch='main',operation='prepare',request=None):
        presentation=dict(schema=1,namespace_id=self.namespace,kind='root',
            messages=[dict(id='b'*32,role='user',content='Retained diagram')],renderer_id='c'*64,output_message_id=None)
        return self.manager.prepare_root(branch,request or dict(embedding='d'*64),operation,presentation=presentation)

    def test_preparation_runs_on_worker_and_context_survives_until_publication(self):
        self.write_gate.clear();request=self.root()
        try:
            snapshot=request.snapshot.result(5)
            self.assertTrue(self.write_entered.wait(5));self.assertEqual(self.preparer.active,1)
            self.assertFalse(request.durable.done());self.assertFalse(self.preparer.released.is_set())
            self.assertEqual(self.preparer.threads,[self.workers[-1]])
            self.assertNotEqual(self.preparer.threads[0],threading.get_ident())
            self.assertEqual(snapshot['counter_deltas']['forward_tokens'],9)
        finally:self.write_gate.set()
        durable=request.durable.result(5);self.assertTrue(self.preparer.released.wait(5))
        self.assertEqual(self.models[-1].history(durable['snapshot']),InputHistory(spans()))
        self.assertEqual(self.preparer.release_checks,[True])

    def test_request_is_detached_before_the_worker_reads_it(self):
        self.preparer.gate.clear();body=dict(embedding='d'*64);request=self.root(request=body)
        try:
            self.assertTrue(self.preparer.entered.wait(5));body['embedding']='e'*64
        finally:self.preparer.gate.set()
        result=request.durable.result(5)
        self.assertEqual(self.models[-1].history(result['snapshot']),InputHistory(spans()))

    def test_identical_roots_from_different_work_ids_reuse_the_existing_prefix(self):
        first=self.root('first','first-op');a=first.durable.result(5)
        second=self.root('second','second-op');b=second.durable.result(5)
        self.assertEqual(a['snapshot'],b['snapshot'])
        self.assertEqual(first.snapshot.result(5)['counter_deltas']['forward_tokens'],9)
        self.assertEqual(second.snapshot.result(5)['counter_deltas']['forward_tokens'],0)
        self.assertTrue(second.snapshot.result(5)['exact_materialization_reused'])

    def test_lost_ack_reopen_does_not_prepare_or_evaluate_again(self):
        first=self.root().durable.result(5);self.close();self.start()
        calls=self.preparer.calls;before=self.models[-1].backend.stats['forward_tokens']
        second=self.root().durable.result(5)
        self.assertEqual(first['branch'],second['branch'])
        self.assertEqual(self.preparer.calls,calls)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_changed_request_with_same_operation_is_rejected_before_preparation(self):
        self.root().durable.result(5);calls=self.preparer.calls
        with self.assertRaises(ValueError):self.root(request=dict(embedding='e'*64))
        self.assertEqual(self.preparer.calls,calls)

    def test_preparer_change_during_preparation_is_rejected_before_forward(self):
        self.preparer.gate.clear();request=self.root()
        try:
            self.assertTrue(self.preparer.entered.wait(5));self.preparer.identity='f'*64
        finally:self.preparer.gate.set()
        with self.assertRaisesRegex(InvalidOperation,'identity'):request.durable.result(5)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)
        self.assertFalse((self.manager.store.references/'main.json').exists())

    def test_encoder_failure_or_wrong_history_type_cannot_publish_a_root(self):
        for operation in ('failed-encoder','bad-history'):
            self.preparer.fail=operation=='failed-encoder';self.preparer.bad_history=not self.preparer.fail
            request=self.root(operation=operation)
            with self.subTest(operation=operation),self.assertRaises((RuntimeError,InvalidOperation)):
                request.durable.result(5)
            self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)
            self.assertFalse((self.manager.store.references/'main.json').exists())

    def test_capacity_refusal_does_not_allocate_an_intent_or_start_preparation(self):
        body=dict(embedding='d'*64,buffer='x'*(600*1024))
        with self.assertRaises(PoolFull):self.root(request=body)
        self.assertEqual(self.preparer.calls,0)
        self.assertFalse((Path(self.tmp.name)/'pool-operations/prepare.json').exists())
        self.assertNotIn('prepare',self.manager._requests)

    def test_existing_branch_rejects_new_work_before_preparation(self):
        self.root().durable.result(5);calls=self.preparer.calls
        with self.assertRaises(Conflict):self.root(operation='overwrite')
        self.assertEqual(self.preparer.calls,calls)

    def test_cancellation_after_preparation_releases_capacity_without_forward(self):
        cancelled=threading.Event();self.preparer.gate.clear()
        request=self.manager.prepare_root('cancelled',dict(embedding='d'*64),'cancelled-op',cancelled=cancelled.is_set)
        try:
            self.assertTrue(self.preparer.entered.wait(5));cancelled.set()
        finally:self.preparer.gate.set()
        with self.assertRaises(CancelledError):request.durable.result(5)
        self.assertTrue(self.preparer.released.wait(5))
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)
        self.assertEqual(self.manager.pool.metrics()['reserved_bytes'],0)
        self.assertFalse((self.manager.store.references/'cancelled.json').exists())

    def test_failed_save_retains_typed_inputs_and_retries_publication_only(self):
        model=self.models[-1];write=model.save_prepared_branch
        def full(*args,**kwargs):raise OSError(errno.ENOSPC,'Test-only rejected save')
        model.save_prepared_branch=full;request=self.root()
        with self.assertRaises(OSError):request.durable.result(5)
        self.assertTrue(self.preparer.released.wait(5))
        self.assertEqual(self.manager.pool.metrics()['reserved_bytes'],0)
        self.assertIsInstance(self.manager._failure_payloads['prepare'].ids,InputHistory)
        before=model.backend.stats['forward_tokens'];calls=self.preparer.calls
        model.save_prepared_branch=write
        durable=self.manager.retry_publication('prepare').result(5)
        self.assertEqual(model.history(durable['snapshot']),InputHistory(spans()))
        self.assertEqual(model.backend.stats['forward_tokens'],before)
        self.assertEqual(self.preparer.calls,calls)

    def test_completed_replay_needs_no_request_reservation(self):
        first=self.root().durable.result(5);self.close();self.start()
        calls=self.preparer.calls
        with self.manager.pool.reserve(self.manager.pool.byte_limit):
            second=self.root().durable.result(5)
        self.assertEqual(second['branch'],first['branch'])
        self.assertEqual(self.preparer.calls,calls)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)

    def test_changed_preparer_identity_rejects_completed_operation(self):
        self.root().durable.result(5);self.close();self.preparer.identity='e'*64;self.start()
        calls=self.preparer.calls
        with self.assertRaises(ValueError):self.root()
        self.assertEqual(self.preparer.calls,calls)

    def test_bad_preparation_receipt_is_rejected_before_forward_or_branch_publication(self):
        self.preparer.bad_receipt=True;request=self.root()
        with self.assertRaises(InvalidOperation):request.durable.result(5)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)
        self.assertFalse((self.manager.store.references/'main.json').exists())

    def test_nonfinite_and_oversized_requests_are_rejected_without_intents(self):
        for value in (dict(embedding='d'*64,bad=float('nan')),
                      dict(embedding='d'*64,buffer='x'*(16*1024**2))):
            with self.assertRaises(InvalidOperation):self.root(request=value)
        self.assertEqual(self.preparer.calls,0)
        self.assertFalse((Path(self.tmp.name)/'pool-operations/prepare.json').exists())


if __name__=='__main__':unittest.main()
