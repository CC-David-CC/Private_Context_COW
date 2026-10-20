"""Canonical image deltas with a synthetic CPU backend; not GPU evidence."""
from contextlib import contextmanager
import copy
import tempfile
import unittest

from context_time_travel.input_spans import InputHistory
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation
from test_vision_api_binding import MixedCPUBackend
from test_vision_execution import receipt, spans


class DeltaBackend(MixedCPUBackend):
    def __init__(self,profile):
        super().__init__(profile);self.restores=0;self.input_calls=[];self.fail_consume=False
    def restore(self,*args,**options):
        self.restores+=1;return super().restore(*args,**options)
    def consume_inputs(self,lease,inputs,*,embeddings):
        self.input_calls.append(copy.deepcopy(inputs))
        super().consume_inputs(lease,inputs,embeddings=embeddings)
        if self.fail_consume:raise InvalidOperation('synthetic consumed-input failure')
    @contextmanager
    def generate_request(self,lease,count):
        start=len(lease['inputs'])
        def tokens():
            for _ in range(count):self.consume(lease,[2]);yield 2
        try:yield tokens()
        finally:self.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')


class DeltaArtifacts:
    def __init__(self,store,*,allowed=None):self.store=store;self.allowed=allowed;self.calls=[];self.active=0
    @contextmanager
    def lease(self,span,**options):
        from types import SimpleNamespace
        self.calls.append(span['embedding_sha256'])
        if self.allowed is not None and span['embedding_sha256'] not in self.allowed:
            raise IntegrityError('missing leased delta embedding')
        self.active+=1
        try:yield SimpleNamespace(embedding=object())
        finally:self.active-=1


class VisionInputDeltaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-delta',model_weights_sha256='a'*64,
            execution=dict(resume_contract='identical-mixed-native-operators-v1',prefill_chunk=128,
                embedding_width=4,capacity=128,eos_tokens=[],generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1'))
        self.backend=DeltaBackend(self.profile);self.model=BranchingModel(self.temp.name,self.backend,self.profile)
        self.artifacts=DeltaArtifacts(self.model.store)
        self.root=self.model.create_inputs(spans(),artifacts=self.artifacts)
        self.parent=self.model.cache[self.root];self.prior=self.model.history(self.root)
        self.delta=InputHistory(spans(original_sha256='e'*64,embedding_sha256='f'*64))

    def test_typed_concatenation_retains_both_images_and_is_not_a_token_iterator(self):
        final=self.prior+self.delta
        self.assertEqual(final,InputHistory(self.prior.spans+self.delta.spans))
        self.assertEqual(final.segment(0,len(self.prior)),self.prior)
        self.assertEqual(final.segment(len(self.prior)),self.delta)
        self.assertEqual(len([s for s in final.spans if s['kind']=='image']),2)
        with self.assertRaises(TypeError):tuple(final)

    def test_feed_inputs_appends_one_child_with_only_delta_work(self):
        before=self.backend.stats['forward_tokens']
        child=self.model.feed_inputs(self.root,self.delta.spans,artifacts=self.artifacts)
        self.assertEqual(self.backend.stats['forward_tokens']-before,len(self.delta))
        self.assertEqual(self.model.history(child),InputHistory(self.prior.spans+self.delta.spans))
        self.assertEqual(self.model.store.get_snapshot(child)['parent'],self.root)
        self.assertEqual(self.model.store.get_snapshot(child)['inputs'],self.delta.spans)
        self.assertEqual(self.model.controller(child),{'status':'idle'})
        self.assertEqual(len(self.model.cache[child].execution_schedule),2)
        self.assertEqual(self.model.cache[self.root],self.parent)

    def test_greedy_request_accepts_typed_delta_and_counts_only_actual_text(self):
        before=self.backend.stats['forward_tokens']
        result=self.model.run_greedy_request(self.root,self.delta,3,stream='delta',artifacts=self.artifacts)
        self.assertEqual(result['tokens'],[2,2,2])
        self.assertEqual(self.backend.stats['forward_tokens']-before,len(self.delta)+3)
        self.assertEqual(self.model.history(result['snapshot']),InputHistory(self.prior.spans+self.delta.spans)+(2,2,2))
        controller=self.model.controller(result['snapshot'])
        self.assertEqual(controller['seen'],{'1':2,'2':5,'3':2})
        self.assertEqual(controller['accepted_in_scope'],3)
        self.assertEqual(self.model.cache[self.root],self.parent)

    def test_only_new_embeddings_are_required_and_held_through_capture(self):
        artifacts=DeltaArtifacts(self.model.store,allowed={'f'*64});capture=self.model._save
        def require_lease(*args,**options):
            self.assertEqual(artifacts.active,1);return capture(*args,**options)
        self.model._save=require_lease
        self.model.run_greedy_request(self.root,self.delta,1,stream='delta',artifacts=artifacts)
        self.assertEqual(artifacts.calls,['f'*64]);self.assertEqual(artifacts.active,0)

    def test_same_pad_projection_cannot_substitute_another_delta_image(self):
        other=InputHistory(spans(original_sha256='1'*64,embedding_sha256='2'*64))
        a=self.model.feed_inputs(self.root,self.delta.spans,artifacts=self.artifacts)
        before=self.backend.stats['forward_tokens']
        b=self.model.feed_inputs(self.root,other.spans,artifacts=self.artifacts)
        self.assertEqual(self.backend.stats['forward_tokens']-before,len(other))
        self.assertEqual(self.model.history(a).native_tokens,self.model.history(b).native_tokens)
        self.assertNotEqual(a,b);self.assertNotEqual(self.model.cache[a].input_identity,self.model.cache[b].input_identity)
        self.assertEqual(self.model.history(self.root),self.prior)

    def test_missing_or_foreign_artifacts_reject_before_restore_or_forward(self):
        before=self.backend.stats['forward_tokens'];restores=self.backend.restores
        missing=DeltaArtifacts(self.model.store,allowed=set())
        for artifacts,error in ((None,InvalidOperation),(missing,IntegrityError)):
            with self.subTest(artifacts=artifacts),self.assertRaises(error):
                self.model.run_greedy_request(self.root,self.delta,2,stream='bad',artifacts=artifacts)
        with tempfile.TemporaryDirectory() as folder:
            foreign=DeltaArtifacts(BranchingModel(folder,DeltaBackend(self.profile),self.profile).store)
            with self.assertRaises(InvalidOperation):self.model.feed_inputs(self.root,self.delta.spans,artifacts=foreign)
        self.assertEqual(self.backend.stats['forward_tokens'],before);self.assertEqual(self.backend.restores,restores)

    def test_incomplete_or_wrong_width_delta_rejects_before_computation(self):
        before=self.backend.stats['forward_tokens'];restores=self.backend.restores
        bad_width=spans(width=5,embedding_bytes=20+6*5*4)
        for source in (self.delta.spans[:-1],bad_width):
            with self.subTest(source=source),self.assertRaises(InvalidOperation):
                self.model.run_greedy_request(self.root,InputHistory(source),2,stream='bad',artifacts=self.artifacts)
        self.assertEqual(self.backend.stats['forward_tokens'],before);self.assertEqual(self.backend.restores,restores)

    def test_consumption_failure_releases_new_artifact_and_keeps_parent(self):
        self.backend.fail_consume=True;before=set(self.model.store.objects.glob('*.json'))
        with self.assertRaises(InvalidOperation):
            self.model.run_greedy_request(self.root,self.delta,2,stream='bad',artifacts=self.artifacts)
        self.assertEqual(set(self.model.store.objects.glob('*.json')),before)
        self.assertEqual(self.model.cache[self.root],self.parent);self.assertEqual(self.artifacts.active,0)

    def test_text_delta_matches_existing_feed_semantics(self):
        a=self.model.feed(self.root,[4,5])
        b=self.model.feed_inputs(self.root,[dict(kind='text',tokens=[4,5])],artifacts=None)
        self.assertEqual(a,b);self.assertEqual(self.model.history(a),self.prior+(4,5))

    def test_active_scope_feed_updates_text_but_not_image_rows(self):
        active=self.model.start_generation(self.root,dict(temperature=0),'active-input')
        original=self.model.controller(active)
        child=self.model.feed_inputs(active,self.delta.spans,artifacts=self.artifacts)
        changed=self.model.controller(child)
        self.assertEqual(changed['seen'],{'1':2,'2':2,'3':2})
        self.assertEqual(changed['accepted_in_scope'],len(self.delta.text_tokens))
        self.assertEqual(self.model.controller(active),original)


if __name__=='__main__':unittest.main()
