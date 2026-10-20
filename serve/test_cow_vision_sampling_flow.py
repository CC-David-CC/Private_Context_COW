"""Image RNG/Min-P client integration; real store/parser, CPU selector only."""
import copy
import hashlib
import json
import unittest

from context_time_travel.min_p import choose
from context_time_travel.store import IntegrityError,_json_bytes
from serve.cow_text_adapter import UnsupportedTextRequest
import serve.test_cow_vision_grammar_flow as fixtures

POLICY=dict(temperature=.7,min_p=.05,top_k=0,top_p=1.0)


class CowVisionSamplingFlowTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CowVisionGrammarFlowTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        original_start=self.f.start
        def configure():
            model=self.f.models[-1];backend=model.backend
            model.profile['execution']['sampling_protocol']='full-vocab-adaptive-p-u53-v7'
            backend.logits=lambda lease:tuple(.25 if i==66 else 0. if i==65 else -100. for i in range(backend.vocab_size))
            def sample(lease,mask,temp,p,draw,**filters):
                return choose(backend.logits(lease),mask,temperature=temp,min_p=p,draw53=draw,**filters)['token']
            backend.sample_legal=sample
        def start():original_start();configure()
        self.f.start=start;configure()

    def answer(self,*,steps=4,grammar=None,operation='answer'):
        return self.f.text.answer('main',1,operation,steps,settings=POLICY,seed=42,grammar_state=grammar)

    def test_plain_min_p_seed_and_policy_are_bound_to_a_versioned_receipt(self):
        self.f.prepare();request=self.answer();result=request.durable.result(5)
        state=self.f.store.get_snapshot(result['snapshot'])['controller']
        self.assertEqual((state['schema'],state['next_draw']),(3,4))
        self.assertEqual(state['stream_key'],hashlib.sha256(b'cow-seed-v1:42').hexdigest())
        descriptor=self.f.manager.committed_generation('answer')['generation']
        self.assertEqual(descriptor['schema'],12);self.assertEqual(descriptor['controls']['seed'],42)
        self.assertIsNone(descriptor['grammar']);self.assertEqual(descriptor['sampling']['min_p'],.05)
        self.assertNotIn('260',state['seen'])

    def test_plain_forked_cursor_stays_independent_after_sibling_generation(self):
        self.f.prepare();partial=self.answer().durable.result(5)
        self.f.tokens.fork('B',partial['snapshot'],source_branch='main',operation_id='fork')
        head=self.f.store.head('B');saved=self.f.store.get_snapshot(head.tip)
        left=self.f.text.resume('main',2,'left',4);a=left.durable.result(5)
        self.assertEqual(self.f.store.head('B'),head);self.assertEqual(self.f.store.get_snapshot(head.tip),saved)
        right=self.f.text.resume('B',head.revision,'right',4);b=right.durable.result(5)
        self.assertEqual(left.text_result(),right.text_result())
        self.assertEqual(self.f.store.get_snapshot(a['snapshot'])['controller'],self.f.store.get_snapshot(b['snapshot'])['controller'])

    def test_nonzero_rng_resumes_after_reopen_without_prefix_forward_work(self):
        self.f.prepare();self.answer().durable.result(5)
        first=self.f.text.resume('main',2,'tail',4);first.durable.result(5)
        expected=first.text_result();source=self.f.manager.committed_generation('answer')['receipt']['result']['tip']
        self.f.tokens.fork('B',source,source_branch='main',operation_id='fork-B')
        self.f.close();self.f.without_encoder=True;self.f.start();before=self.f.models[-1].backend.stats['forward_tokens']
        second=self.f.text.resume('B',0,'restored-tail',4);receipt=second.durable.result(5)
        self.assertEqual(second.text_result(),expected)
        self.assertEqual(self.f.store.get_snapshot(receipt['snapshot'])['controller']['next_draw'],8)
        self.assertEqual(self.f.models[-1].backend.stats['forward_tokens']-before,4)

    def test_grammar_min_p_keeps_parser_and_cursor_on_restart(self):
        self.f.prepare();first=self.answer(steps=1,grammar='READ');first.durable.result(5)
        self.f.close();self.f.without_encoder=True;self.f.start()
        tail=self.f.text.resume('main',2,'finish-read',8);result=tail.durable.result(5)
        self.assertEqual(tail.text_result(),'B')
        state=self.f.store.get_snapshot(result['snapshot'])['controller']
        self.assertEqual(state['constraint']['agent_state'],'READ');self.assertEqual(state['next_draw'],3)

    def test_reseed_preserves_image_history_and_progress_but_resets_cursor(self):
        self.f.prepare();first=self.answer().durable.result(5);before=self.f.store.get_snapshot(first['snapshot'])['controller']
        work=self.f.models[-1].backend.stats['forward_tokens']
        receipt=self.f.text.reseed('main',99,2,'new-seed').durable.result(5)
        state=self.f.store.get_snapshot(receipt['snapshot'])['controller'];expected=copy.deepcopy(before)
        expected.update(stream_key=hashlib.sha256(b'cow-seed-v1:99').hexdigest(),next_draw=0)
        self.assertEqual(state,expected);self.assertEqual(self.f.models[-1].backend.stats['forward_tokens'],work)
        self.f.text.resume('main',3,'tail-new-seed',4).durable.result(5)

    def test_new_image_question_accepts_basic_sampling_and_seed(self):
        self.f.text.create('main',[dict(role='user',content=self.f.parts)],'context',namespace_id=self.f.namespace).durable.result(5)
        request=self.f.text.ask('main',self.f.parts,1,'image-question',4,settings=POLICY,seed=42)
        receipt=request.durable.result(5);descriptor=self.f.manager.committed_generation('image-question')['generation']
        self.assertEqual(descriptor['schema'],13);self.assertEqual(descriptor['controls']['seed'],42)
        self.assertIsNone(descriptor['grammar']);self.assertTrue(any(span['kind']=='image' for span in descriptor['ids']))
        self.assertEqual(self.f.store.get_snapshot(receipt['snapshot'])['controller']['next_draw'],4)

    def test_exact_retry_replays_sampled_answer_without_new_draws(self):
        self.f.prepare();request=self.answer();first=request.durable.result(5);text=request.text_result()
        self.f.close();self.f.start();before=self.f.models[-1].backend.stats['forward_tokens']
        replay=self.f.text.retry('answer','main',2,'same-answer')
        self.assertEqual(replay['answer']['text'],text);self.assertEqual(replay['receipt']['snapshot'],first['snapshot'])
        self.assertEqual(self.f.models[-1].backend.stats['forward_tokens'],before)

    def test_changed_retry_reuses_source_prefix_and_binds_new_policy(self):
        self.f.prepare();self.answer().durable.result(5);before=self.f.models[-1].backend.stats['forward_tokens']
        request=self.f.text.retry('answer','main',2,'different-answer',settings=dict(temperature=.9,top_k=2,top_p=.95))
        receipt=request.durable.result(5);state=self.f.store.get_snapshot(receipt['snapshot'])['controller']
        self.assertEqual(state['policy']['temperature'],.9);self.assertEqual(state['next_draw'],4)
        self.assertEqual(self.f.models[-1].backend.stats['forward_tokens']-before,4)

    def test_changed_or_unqualified_sampler_receipts_cannot_be_silently_replayed(self):
        self.f.prepare();self.answer().durable.result(5)
        path=self.f.manager._intents/'answer.json';envelope=json.loads(path.read_text())
        envelope['intent']['generation']['sampling']['temperature']=.9
        envelope['checksum']=hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest();path.write_bytes(_json_bytes(envelope))
        with self.assertRaises(IntegrityError):self.f.manager.committed_generation('answer')

    def test_image_dry_and_text_stops_stay_explicitly_rejected(self):
        self.f.prepare();before=self.f.models[-1].backend.stats['forward_tokens']
        for options in (dict(settings=dict(POLICY,dry_multiplier=.8)),dict(settings=POLICY,text_stops=['END'])):
            with self.subTest(options=options),self.assertRaises(UnsupportedTextRequest):
                self.f.text.answer('main',1,'unsupported',4,**options)
        self.assertEqual(self.f.models[-1].backend.stats['forward_tokens'],before)

    def test_changed_retry_of_image_suffix_reuses_retained_artifacts(self):
        self.f.text.create('main',[dict(role='user',content=self.f.parts)],'context',namespace_id=self.f.namespace).durable.result(5)
        self.f.text.ask('main',self.f.parts,1,'image-question',4,settings=POLICY,seed=42).durable.result(5)
        original=self.f.manager.committed_generation('image-question')['generation']
        before=self.f.models[-1].backend.stats['forward_tokens'];encodings=len(self.f.encoder.calls)
        request=self.f.text.retry('image-question','main',2,'image-reroll',settings=dict(temperature=.9,top_k=2,top_p=.95))
        result=request.durable.result(5);record=self.f.manager.committed_generation('image-reroll')['generation']
        self.assertEqual(record['schema'],13);self.assertEqual(record['ids'],original['ids'])
        self.assertEqual(self.f.models[-1].backend.stats['forward_tokens']-before,sum(
            len(span['tokens']) if span['kind']=='text' else span['rows'] for span in record['ids'])+4)
        self.assertEqual(self.f.store.get_snapshot(result['snapshot'])['controller']['policy']['temperature'],.9)
        self.assertEqual(len(self.f.encoder.calls),encodings)

    def test_generation_policy_override_keeps_cursor_and_original_policy(self):
        self.f.prepare();first=self.answer().durable.result(5);saved=self.f.store.get_snapshot(first['snapshot'])
        request=self.f.text.resume('main',2,'warmer-tail',4,settings=dict(temperature=.9,top_k=2,top_p=.95))
        result=request.durable.result(5);state=self.f.store.get_snapshot(result['snapshot'])['controller']
        self.assertEqual(state['next_draw'],8);self.assertEqual(state['policy']['temperature'],.9)
        self.assertEqual(self.f.store.get_snapshot(first['snapshot']),saved)
        self.assertEqual(self.f.manager.committed_generation('warmer-tail')['generation']['schema'],12)

    def test_greedy_seed_binding_does_not_claim_random_draws(self):
        self.f.prepare();result=self.f.text.answer('main',1,'seeded-greedy',1,seed=42).durable.result(5)
        state=self.f.store.get_snapshot(result['snapshot'])['controller']
        self.assertEqual(state['next_draw'],0)
        self.assertEqual(state['stream_key'],hashlib.sha256(b'cow-seed-v1:42').hexdigest())
        self.assertEqual(self.f.manager.committed_generation('seeded-greedy')['generation']['schema'],12)

    def test_grammar_reseed_preserves_accepted_scope(self):
        self.f.prepare();partial=self.answer(steps=1,grammar='READ').durable.result(5)
        before=self.f.store.get_snapshot(partial['snapshot'])['controller']
        result=self.f.text.reseed('main',99,2,'reseed-grammar').durable.result(5)
        state=self.f.store.get_snapshot(result['snapshot'])['controller']
        self.assertEqual(state['constraint'],before['constraint']);self.assertEqual(state['next_draw'],0)
        tail=self.f.text.resume('main',3,'new-tail',8);end=tail.durable.result(5)
        self.assertEqual(tail.text_result(),'B')
        self.assertEqual(self.f.store.get_snapshot(end['snapshot'])['controller']['next_draw'],2)


if __name__=='__main__':unittest.main()
