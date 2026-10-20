"""Mixed-input variable-draw ownership with CPU proposals, not GPU evidence."""
import copy
import unittest

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.sampling_proposal import SelectionProposal,XTC_PROPOSAL_PROTOCOL
from context_time_travel.store import InvalidOperation
from context_time_travel.xtc_pipeline import choose_v5
import test_vision_sampling_request as fixtures

POLICY=dict(fixtures.POLICY,min_p=0.,xtc_probability=1.,xtc_threshold=.15)


class VisionXtcRequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.VisionSamplingRequestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend;self.model=self.fixture.model;self.calls=[]
        self.backend._ready=dict(sampling_v5_protocol=XTC_PROPOSAL_PROTOCOL,
            sampling_v4_protocol='full-vocab-typical-dyna-min-p-u53-v4')
        def proposal(lease,mask,temp,p,*,draw_start,draw_candidates,source_digest,**filters):
            self.assertNotIn('dry_history',filters)
            choice=choose_v5(self.backend.logits(lease),mask,temperature=temp,min_p=p,
                draw53=draw_candidates[0],draw53_second=draw_candidates[1] if len(draw_candidates)>1 else None,**filters)
            self.calls.append(dict(draw_start=draw_start,draw_count=choice['draw_count'],token=choice['token']))
            return SelectionProposal(choice['token'],draw_start,choice['draw_count'],None,source_digest)
        self.backend.sample_selection_proposal=proposal
        self.fixture.engine=GrammarEngine(self.fixture.fixture.tokenizer,{'EDIT':dict(format='gbnf',source=
            'root ::= "A" choice "A" choice "A" choice "A" choice\nchoice ::= "A" | "B"')},[6])

    def request(self,*args,policy=None,**kwargs):
        return self.fixture.request(*args,policy=POLICY if policy is None else policy,**kwargs)

    def test_plain_xtc_counts_actual_draws_and_excludes_image_rows(self):
        result=self.request();state=self.model.controller(result['snapshot'])
        self.assertEqual(state['schema'],4);self.assertIsNone(state['sampler_state'])
        self.assertEqual(state['next_draw'],8);self.assertEqual(sum(state['seen'].values()),7)
        self.assertNotIn('7',state['seen']);self.assertEqual([r['draw_start'] for r in self.calls],[0,2,4,6])

    def test_forked_nonzero_cursor_resume_is_independent(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']))
        a=self.model.store.fork('xtc-A',first['snapshot']);b=self.model.store.fork('xtc-B',first['snapshot'])
        left=self.request(a.tip,resume=True);self.model.store.move(a.name,a.revision,left['snapshot'],'xtc-A')
        self.assertEqual(self.model.store.head(b.name),b);self.assertEqual(self.model.controller(first['snapshot']),saved)
        right=self.request(b.tip,resume=True)
        self.assertEqual(left['tokens'],right['tokens']);self.assertEqual(self.model.controller(left['snapshot']),self.model.controller(right['snapshot']))
        self.assertEqual(self.model.controller(right['snapshot'])['next_draw'],16)

    def test_grammar_single_and_multiple_legal_tokens_use_different_draw_counts(self):
        first=self.request(grammar=True);saved=copy.deepcopy(self.model.controller(first['snapshot']))
        self.assertEqual([r['draw_count'] for r in self.calls],[1,2,1,2]);self.assertEqual(saved['next_draw'],6)
        tail=self.request(first['snapshot'],steps=16,grammar=True,resume=True)
        whole=self.request(steps=16,grammar=True)
        self.assertEqual(first['tokens']+tail['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(whole['snapshot']))
        self.assertEqual(self.model.controller(first['snapshot']),saved);self.assertTrue(tail['grammar_complete'])

    def test_entropy_combination_uses_the_same_branch_owned_proposal_cursor(self):
        policy=dict(POLICY,typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.)
        first=self.request(policy=policy);state=self.model.controller(first['snapshot'])
        self.assertEqual(state['next_draw'],sum(r['draw_count'] for r in self.calls))
        tail=self.request(first['snapshot'],policy=policy,resume=True)
        full=self.request(steps=8,policy=policy)
        self.assertEqual(first['tokens']+tail['tokens'],full['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(full['snapshot']))

    def test_missing_native_proposal_query_rejects_before_restore(self):
        self.backend._ready={};before=self.backend.restores
        with self.assertRaises(InvalidOperation):self.request()
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_disabling_xtc_preserves_the_existing_proposal_controller_and_cursor(self):
        first=self.request();saved=self.model.controller(first['snapshot']);self.calls.clear()
        tail=self.request(first['snapshot'],policy=fixtures.POLICY,resume=True)
        state=self.model.controller(tail['snapshot'])
        self.assertEqual(state['schema'],4);self.assertIsNone(state['sampler_state'])
        self.assertEqual(state['next_draw'],12);self.assertEqual(self.model.controller(first['snapshot']),saved)
        self.assertEqual([r['draw_start'] for r in self.calls],[8,9,10,11])

    def test_dry_combination_remains_closed_before_work(self):
        before=self.backend.restores
        with self.assertRaises(InvalidOperation):self.request(policy=dict(POLICY,dry_multiplier=.8))
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_saved_proposal_cannot_switch_to_unqualified_dry_before_restore(self):
        first=self.request();saved=self.model.controller(first['snapshot'])
        self.model.profile['execution']['vision_dry_protocol']='mixed-text-runs-dry-u53-v1'
        before=self.backend.restores;calls=copy.deepcopy(self.calls)
        with self.assertRaises(InvalidOperation):
            self.request(first['snapshot'],policy=dict(fixtures.POLICY,dry_multiplier=.8),resume=True)
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,calls)
        self.assertEqual(self.model.controller(first['snapshot']),saved)

    def test_failed_consumption_cannot_publish_a_proposal_or_advance_saved_rng(self):
        first=self.request();saved=self.model.controller(first['snapshot']);before=set(self.model.store.objects.iterdir())
        consume=self.backend.consume_generated
        def fail(*args):raise InvalidOperation('synthetic consumed-token failure')
        self.backend.consume_generated=fail
        with self.assertRaisesRegex(InvalidOperation,'synthetic consumed-token failure'):
            self.request(first['snapshot'],resume=True)
        self.assertEqual(set(self.model.store.objects.iterdir()),before);self.assertEqual(self.model.controller(first['snapshot']),saved)
        self.backend.consume_generated=consume
        self.assertEqual(len(self.request(first['snapshot'],resume=True)['tokens']),4)


if __name__=='__main__':unittest.main()
