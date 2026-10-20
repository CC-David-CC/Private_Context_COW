"""Image Adaptive-P controller ownership with CPU proposals, not GPU evidence."""
import copy
import unittest

from context_time_travel import adaptive_p
from context_time_travel.adaptive_p_pipeline import choose_v7
from context_time_travel.xtc_pipeline import choose_v5
from context_time_travel.sampling_proposal import SelectionProposal,ADAPTIVE_P_PROPOSAL_PROTOCOL,XTC_PROPOSAL_PROTOCOL
from context_time_travel.seed_controls import seed_stream
from context_time_travel.store import InvalidOperation
import test_vision_sampling_request as fixtures

POLICY=dict(fixtures.POLICY,adaptive_target=.55,adaptive_decay=.9)


class VisionAdaptiveRequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.VisionSamplingRequestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.model=self.fixture.model;self.backend=self.fixture.backend;self.calls=[]
        self.backend._ready=dict(adaptive_p_protocol=ADAPTIVE_P_PROPOSAL_PROTOCOL,
            adaptive_p_feedback_protocol='adaptive-p-state-v1',adaptive_p_feedback_version=1,
            sampling_v5_protocol=XTC_PROPOSAL_PROTOCOL)
        def proposal(lease,mask,temp,p,*,draw_start,draw_candidates,source_digest,**filters):
            self.assertNotIn('dry_history',filters);saved=filters.pop('sampler_state',None)
            if filters.get('adaptive_target',-1.)>=0.:
                choice=choose_v7(self.backend.logits(lease),mask,saved,temperature=temp,min_p=p,
                    draw53=draw_candidates[0],**filters);feedback=choice['feedback_state']
            else:
                choice=choose_v5(self.backend.logits(lease),mask,temperature=temp,min_p=p,
                    draw53=draw_candidates[0],**filters);feedback=None
            self.calls.append(dict(draw_start=draw_start,source_feedback=copy.deepcopy(saved),feedback=feedback))
            return SelectionProposal(choice['token'],draw_start,choice['draw_count'],feedback,source_digest)
        self.backend.sample_selection_proposal=proposal

    def request(self,*args,policy=None,**kwargs):
        return self.fixture.request(*args,policy=POLICY if policy is None else policy,**kwargs)

    def test_plain_feedback_counts_only_real_consumed_text(self):
        result=self.request();state=self.model.controller(result['snapshot'])
        self.assertEqual((state['schema'],state['next_draw']),(4,4))
        self.assertEqual(state['sampler_state'],self.calls[-1]['feedback'])
        self.assertNotEqual(state['sampler_state'],adaptive_p.initial_state(.55,.9))
        self.assertEqual(sum(state['seen'].values()),7);self.assertNotIn('7',state['seen'])

    def test_forked_nonzero_feedback_resumes_independently(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']))
        a=self.model.store.fork('ap-A',first['snapshot']);b=self.model.store.fork('ap-B',first['snapshot'])
        left=self.request(a.tip,resume=True);self.model.store.move(a.name,a.revision,left['snapshot'],'ap-A')
        self.assertEqual(self.model.store.head(b.name),b);self.assertEqual(self.model.controller(first['snapshot']),saved)
        right=self.request(b.tip,resume=True);whole=self.request(steps=8)
        self.assertEqual(left['tokens'],right['tokens']);self.assertEqual(first['tokens']+right['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(left['snapshot']),self.model.controller(right['snapshot']))
        self.assertEqual(self.model.controller(whole['snapshot']),self.model.controller(right['snapshot']))

    def test_grammar_resume_preserves_parser_cursor_and_feedback(self):
        first=self.request(grammar=True);saved=copy.deepcopy(self.model.controller(first['snapshot']))
        tail=self.request(first['snapshot'],grammar=True,resume=True,steps=16);whole=self.request(grammar=True,steps=16)
        self.assertTrue(tail['grammar_complete']);self.assertEqual(first['tokens']+tail['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(whole['snapshot']))
        self.assertEqual(self.model.controller(first['snapshot']),saved)

    def test_changed_generation_settings_keep_the_incoming_feedback(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']))
        tail=self.request(first['snapshot'],resume=True,steps=1,policy=dict(POLICY,adaptive_target=.7,adaptive_decay=.8))
        self.assertEqual(self.calls[-1]['source_feedback'],saved['sampler_state'])
        state=self.model.controller(tail['snapshot']);self.assertEqual(state['next_draw'],5)
        self.assertEqual((state['policy']['adaptive_target'],state['policy']['adaptive_decay']),(.7,.8))
        self.assertEqual(self.model.controller(first['snapshot']),saved)

    def test_entropy_and_probability_filters_share_branch_owned_feedback(self):
        policy=dict(POLICY,top_k=2,top_p=.95,typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.)
        first=self.request(policy=policy);tail=self.request(first['snapshot'],policy=policy,resume=True)
        whole=self.request(steps=8,policy=policy)
        self.assertEqual(first['tokens']+tail['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(whole['snapshot']))

    def test_reseed_changes_only_rng_and_does_no_model_work(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']));before=self.backend.stats['forward_tokens']
        changed=self.model.set_random_stream(first['snapshot'],seed_stream(99));state=self.model.controller(changed)
        expected=copy.deepcopy(saved);expected.update(stream_key=state['stream_key'],next_draw=0)
        self.assertEqual(state,expected);self.assertNotEqual(state['stream_key'],saved['stream_key'])
        self.assertEqual(self.backend.stats['forward_tokens'],before)
        self.assertEqual(self.model.controller(self.request(changed,resume=True)['snapshot'])['next_draw'],4)

    def test_disabling_and_reenabling_preserve_cursor_without_leaking_old_feedback(self):
        first=self.request();old=self.model.controller(first['snapshot'])
        off=self.request(first['snapshot'],resume=True,policy=dict(POLICY,adaptive_target=-1.))
        self.assertIsNone(self.model.controller(off['snapshot'])['sampler_state'])
        again=self.request(off['snapshot'],resume=True,steps=1,policy=dict(POLICY,adaptive_target=.4))
        self.assertEqual(self.calls[-1]['source_feedback'],adaptive_p.initial_state(.4,.9))
        self.assertEqual(self.model.controller(again['snapshot'])['next_draw'],9)
        self.assertEqual(self.model.controller(first['snapshot']),old)

    def test_missing_feedback_query_rejects_before_restore(self):
        for changes in ({'adaptive_p_protocol':'wrong'},{'adaptive_p_feedback_protocol':'wrong'},
                        {'adaptive_p_feedback_version':True}):
            with self.subTest(changes=changes):
                ready=self.backend._ready;self.backend._ready=dict(ready,**changes);before=self.backend.restores
                with self.assertRaises(InvalidOperation):self.request()
                self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[]);self.backend._ready=ready

    def test_dry_combination_is_closed_before_model_work(self):
        before=self.backend.restores
        with self.assertRaises(InvalidOperation):self.request(policy=dict(POLICY,dry_multiplier=.8))
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_failed_consumption_cannot_publish_or_advance_parent_feedback(self):
        first=self.request();saved=self.model.controller(first['snapshot']);objects=set(self.model.store.objects.iterdir())
        consume=self.backend.consume_generated
        def fail(*args):raise InvalidOperation('synthetic Adaptive-P consume failure')
        self.backend.consume_generated=fail
        with self.assertRaisesRegex(InvalidOperation,'synthetic Adaptive-P consume failure'):
            self.request(first['snapshot'],resume=True)
        self.assertEqual(set(self.model.store.objects.iterdir()),objects);self.assertEqual(self.model.controller(first['snapshot']),saved)
        self.backend.consume_generated=consume;self.assertEqual(len(self.request(first['snapshot'],resume=True)['tokens']),4)


if __name__=='__main__':unittest.main()
