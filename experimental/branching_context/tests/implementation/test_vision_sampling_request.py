"""Image sampler/RNG ownership with a CPU selector; not GPU qualification."""
import copy
import unittest

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.input_spans import InputHistory
from context_time_travel.min_p import choose
from context_time_travel.seed_controls import seed_stream
from context_time_travel.store import InvalidOperation
import test_vision_grammar_request as fixtures
from test_vision_execution import spans

POLICY=dict(temperature=.7,min_p=.05,top_k=0,top_p=1.0,repetition_penalty=1.0)


class VisionSamplingRequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.VisionGrammarRequestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.model=self.fixture.model;self.backend=self.fixture.backend;self.root=self.fixture.root
        self.model.profile['execution']['sampling_protocol']='full-vocab-adaptive-p-u53-v7'
        self.backend.logits=lambda lease:(0.,.25,-100.,-100.,-100.,-100.,-100.,-100.)
        self.calls=[]
        def sample(lease,mask,temp,p,draw,**filters):
            self.calls.append(dict(draw=draw,temperature=temp,min_p=p))
            return choose(self.backend.logits(lease),mask,temperature=temp,min_p=p,draw53=draw,**filters)['token']
        self.backend.sample_legal=sample
        self.engine=GrammarEngine(self.fixture.tokenizer,{'EDIT':dict(format='gbnf',source=
            'root ::= choice choice choice choice choice choice choice choice\nchoice ::= "A" | "B"')},[6])

    def request(self,snapshot=None,steps=4,*,grammar=False,resume=False,policy=None,stream=None,**options):
        return self.model.run_state_request(snapshot or self.root,(),steps,engine=self.engine if grammar else None,
            state='EDIT' if grammar else None,stream=stream or seed_stream(42),resume=resume,
            sampling=POLICY if policy is None else policy,**options)

    def test_plain_min_p_advances_rng_and_counts_only_actual_text(self):
        parent=self.model.snapshot_bytes(self.root);before=self.backend.stats['forward_tokens']
        result=self.request();state=self.model.controller(result['snapshot'])
        self.assertEqual(state['schema'],3);self.assertEqual(state['next_draw'],4)
        self.assertEqual(state['accepted_in_scope'],4);self.assertEqual(state['status'],'active')
        self.assertNotIn('7',state['seen']);self.assertEqual(sum(state['seen'].values()),7)
        self.assertEqual(self.backend.stats['forward_tokens']-before,4);self.assertEqual(len(self.calls),4)
        self.assertEqual(self.model.snapshot_bytes(self.root),parent)

    def test_forked_rng_cursor_is_independent_and_reproduces_the_same_tail(self):
        first=self.request();partial=first['snapshot'];saved=self.model.controller(partial)
        a=self.model.store.fork('A',partial);b=self.model.store.fork('B',partial)
        left=self.request(a.tip,resume=True);self.model.store.move('A',a.revision,left['snapshot'],'advance-A')
        self.assertEqual(self.model.store.head('B'),b);self.assertEqual(self.model.controller(partial),saved)
        right=self.request(b.tip,resume=True)
        self.assertEqual(left['tokens'],right['tokens']);self.assertEqual(self.model.controller(left['snapshot']),self.model.controller(right['snapshot']))
        self.assertEqual(self.model.controller(right['snapshot'])['next_draw'],8)

    def test_grammar_min_p_restores_text_scope_and_nonzero_draw_progress(self):
        first=self.request(grammar=True);state=self.model.controller(first['snapshot'])
        self.assertEqual((state['constraint']['scope_start'],state['next_draw']),(3,4))
        tail=self.request(first['snapshot'],steps=8,grammar=True,resume=True)
        full=self.request(steps=16,grammar=True)
        self.assertEqual(first['tokens']+tail['tokens'],full['tokens'])
        self.assertTrue(tail['stopped']);self.assertTrue(tail['grammar_complete'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(full['snapshot']))

    def test_policy_override_is_per_generation_and_keeps_the_saved_cursor(self):
        first=self.request();saved=self.model.controller(first['snapshot'])
        changed=dict(POLICY,temperature=.9,min_p=.1,top_k=2,top_p=.95)
        tail=self.request(first['snapshot'],resume=True,policy=changed)
        state=self.model.controller(tail['snapshot'])
        self.assertEqual(state['next_draw'],8);self.assertEqual(state['policy']['temperature'],.9)
        self.assertEqual(self.model.controller(first['snapshot']),saved)
        self.assertEqual(state['stream_key'],saved['stream_key'])

    def test_plain_reseed_changes_only_rng_and_requires_no_forward_work(self):
        first=self.request();saved=self.model.controller(first['snapshot']);before=self.backend.stats['forward_tokens']
        changed=self.model.set_random_stream(first['snapshot'],seed_stream(99));state=self.model.controller(changed)
        expected=copy.deepcopy(saved);expected.update(stream_key=state['stream_key'],next_draw=0)
        self.assertNotEqual(state['stream_key'],saved['stream_key']);self.assertEqual(state,expected)
        self.assertEqual(self.model.history(changed),self.model.history(first['snapshot']))
        self.assertEqual(self.backend.stats['forward_tokens'],before)
        self.assertEqual(self.request(changed,resume=True)['tokens'],self.request(changed,resume=True)['tokens'])

    def test_grammar_reseed_preserves_parser_progress_without_replaying_model(self):
        first=self.request(grammar=True);saved=self.model.controller(first['snapshot']);before=self.backend.stats['forward_tokens']
        changed=self.model.set_random_stream(first['snapshot'],seed_stream(99),grammar_engine=self.engine)
        state=self.model.controller(changed)
        self.assertEqual(state['constraint'],saved['constraint']);self.assertEqual(state['accepted_in_scope'],4)
        self.assertEqual(state['next_draw'],0);self.assertEqual(self.backend.stats['forward_tokens'],before)
        result=self.request(changed,grammar=True,resume=True,steps=8);self.assertTrue(result['stopped'])

    def test_failed_consumption_cannot_publish_rng_or_parser_progress(self):
        before=set(self.model.store.objects.iterdir());parent=self.model.snapshot_bytes(self.root)
        consume=self.backend.consume_generated
        def fail(*args):raise InvalidOperation('synthetic selected-token consumption failure')
        self.backend.consume_generated=fail
        with self.assertRaises(InvalidOperation):self.request(grammar=True)
        self.assertEqual(set(self.model.store.objects.iterdir()),before);self.assertEqual(self.model.snapshot_bytes(self.root),parent)
        self.backend.consume_generated=consume;self.assertEqual(len(self.request()['tokens']),4)

    def test_unqualified_dry_is_rejected_before_work_even_with_native_sampler_capability(self):
        before=self.backend.restores;policy=dict(POLICY,dry_multiplier=.8)
        with self.assertRaises(InvalidOperation):self.request(policy=policy)
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_sampled_image_suffix_keeps_rows_out_of_controller_counts(self):
        delta=InputHistory(spans(original_sha256='e'*64,embedding_sha256='f'*64))
        before=self.backend.stats['forward_tokens']
        result=self.model.run_sampled_request(self.root,delta,4,sampling=POLICY,stream=seed_stream(42),
            artifacts=self.fixture.artifacts)
        self.assertEqual(self.backend.stats['forward_tokens']-before,len(delta)+4)
        self.assertEqual(sum(self.model.controller(result['snapshot'])['seen'].values()),10)
        self.assertNotIn('7',self.model.controller(result['snapshot'])['seen'])
        self.assertEqual(self.fixture.artifacts.active,0)


if __name__=='__main__':unittest.main()
