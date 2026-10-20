"""Image proposal receipts and retry publication; CPU fixtures, not GPU proof."""
import copy
import unittest
from unittest.mock import patch

from context_time_travel.sampling_proposal import SelectionProposal,XTC_PROPOSAL_PROTOCOL
from context_time_travel.store import IntegrityError
from context_time_travel.xtc_pipeline import choose_v5
import serve.test_cow_vision_sampling_flow as fixtures

POLICY=dict(temperature=.9,min_p=.02,top_k=0,top_p=1.,xtc_probability=1.,xtc_threshold=.15)


class CowVisionXtcFlowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CowVisionSamplingFlowTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.c=self.fixture.f;start=self.c.start
        def configure():
            backend=self.c.models[-1].backend
            backend._ready=dict(sampling_v5_protocol=XTC_PROPOSAL_PROTOCOL,
                sampling_v4_protocol='full-vocab-typical-dyna-min-p-u53-v4')
            def propose(lease,mask,temp,p,*,draw_start,draw_candidates,source_digest,**filters):
                choice=choose_v5(backend.logits(lease),mask,temperature=temp,min_p=p,
                    draw53=draw_candidates[0],draw53_second=draw_candidates[1] if len(draw_candidates)>1 else None,**filters)
                return SelectionProposal(choice['token'],draw_start,choice['draw_count'],None,source_digest)
            backend.sample_selection_proposal=propose
        def restart():start();configure()
        self.c.start=restart;configure()

    def answer(self,operation='answer',*,settings=None,steps=4,grammar=None):
        return self.c.text.answer('main',1,operation,steps,settings=POLICY if settings is None else settings,
            seed=42,grammar_state=grammar)

    def assert_cursor_rejected(self,operation,target,cursor):
        original=self.c.manager.store.get_snapshot
        def altered(snapshot):
            value=original(snapshot)
            if snapshot==target:
                value=copy.deepcopy(value);value['controller']['next_draw']=cursor
            return value
        with patch.object(self.c.manager.store,'get_snapshot',side_effect=altered):
            with self.assertRaisesRegex(IntegrityError,'draw cursor'):
                self.c.manager.committed_generation(operation)

    def test_committed_plain_receipt_preserves_two_draws_per_consumed_token(self):
        self.c.prepare();result=self.answer().durable.result(5)
        state=self.c.store.get_snapshot(result['snapshot'])['controller']
        self.assertEqual((state['schema'],state['next_draw']),(4,8))
        record=self.c.manager.committed_generation('answer')
        self.assertEqual(record['receipt']['result']['tip'],result['snapshot'])
        self.assertEqual(record['generation']['schema'],12)

    def test_changed_retry_uses_original_source_without_replaying_its_prefix(self):
        self.c.prepare();first=self.answer().durable.result(5)
        saved=self.c.store.get_snapshot(first['snapshot'])
        before=self.c.models[-1].backend.stats['forward_tokens']
        changed=self.c.text.retry('answer','main',2,'changed',settings=dict(temperature=.8,top_k=2,top_p=.95))
        result=changed.durable.result(5)
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens']-before,4)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],8)
        self.assertEqual(self.c.store.get_snapshot(first['snapshot']),saved)
        self.assertEqual(self.c.manager.committed_generation('changed')['generation']['retry_source_operation'],'answer')

    def test_exact_replay_after_restart_needs_no_model_work_or_new_draws(self):
        self.c.prepare();request=self.answer();result=request.durable.result(5);text=request.text_result()
        saved=self.c.store.get_snapshot(result['snapshot'])
        self.c.close();self.c.without_encoder=True;self.c.start()
        before=self.c.models[-1].backend.stats['forward_tokens']
        replay=self.c.text.retry('answer','main',2,'same')
        self.assertTrue(replay['replayed']);self.assertEqual(replay['answer']['text'],text)
        self.assertEqual(self.c.store.get_snapshot(replay['receipt']['snapshot']),saved)
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens'],before)

    def test_typed_image_suffix_and_changed_retry_preserve_variable_draws(self):
        self.c.text.create('main',[dict(role='user',content=self.c.parts)],'context',namespace_id=self.c.namespace).durable.result(5)
        result=self.c.text.ask('main',self.c.parts,1,'image',4,settings=POLICY,seed=42).durable.result(5)
        record=self.c.manager.committed_generation('image')
        self.assertEqual(record['generation']['schema'],13)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],8)
        before=self.c.models[-1].backend.stats['forward_tokens'];encoders=len(self.c.encoder.calls)
        reroll=self.c.text.retry('image','main',2,'reroll',settings=dict(temperature=.8))
        result=reroll.durable.result(5);retry=self.c.manager.committed_generation('reroll')
        self.assertEqual(retry['generation']['ids'],record['generation']['ids'])
        count=sum(len(s['tokens']) if s['kind']=='text' else s['rows'] for s in record['generation']['ids'])
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens']-before,count+4)
        self.assertEqual(len(self.c.encoder.calls),encoders)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],8)

    def test_single_legal_grammar_choices_still_use_one_draw_and_resume_base(self):
        self.c.prepare();first=self.answer(steps=1,grammar='READ').durable.result(5)
        self.assertEqual(self.c.store.get_snapshot(first['snapshot'])['controller']['next_draw'],1)
        self.c.manager.committed_generation('answer')
        tail=self.c.text.resume('main',2,'tail',8);result=tail.durable.result(5)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],3)
        self.c.manager.committed_generation('tail')

    def test_reseed_resets_variable_cursor_without_changing_old_snapshot(self):
        self.c.prepare();first=self.answer().durable.result(5);saved=self.c.store.get_snapshot(first['snapshot'])
        self.c.text.reseed('main',99,2,'reseed').durable.result(5)
        result=self.c.text.resume('main',3,'tail',4).durable.result(5)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],8)
        self.assertEqual(self.c.store.get_snapshot(first['snapshot']),saved)
        self.c.manager.committed_generation('tail')

    def test_variable_cursor_outside_policy_bounds_is_rejected(self):
        self.c.prepare();result=self.answer().durable.result(5)
        for cursor in (3,9,-1,True):
            with self.subTest(cursor=cursor):self.assert_cursor_rejected('answer',result['snapshot'],cursor)
        self.c.manager.committed_generation('answer')

    def test_disabled_xtc_retains_exact_single_draw_validation(self):
        self.c.prepare();result=self.answer(settings=dict(POLICY,xtc_probability=0.)).durable.result(5)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],4)
        self.c.manager.committed_generation('answer')
        self.assert_cursor_rejected('answer',result['snapshot'],5)


if __name__=='__main__':unittest.main()
