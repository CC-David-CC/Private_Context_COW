"""Image feedback ownership with independent CPU proposals, not GPU evidence."""
import copy
import unittest

from context_time_travel import mirostat2
from context_time_travel.sampling_proposal import SelectionProposal,MIROSTAT2_PROPOSAL_PROTOCOL
from context_time_travel.seed_controls import seed_stream
from context_time_travel.store import InvalidOperation
import test_vision_sampling_request as fixtures

POLICY=dict(fixtures.POLICY,min_p=0.,mirostat=2,mirostat_tau=4.,mirostat_eta=.05)


class VisionMirostatRequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.VisionSamplingRequestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.model=self.fixture.model;self.backend=self.fixture.backend;self.calls=[]
        self.backend._ready=dict(mirostat2_protocol=MIROSTAT2_PROPOSAL_PROTOCOL,
            mirostat2_feedback_protocol='mirostat2-state-v1',mirostat2_feedback_version=1)
        def proposal(lease,mask,temp,p,*,draw_start,draw_candidates,source_digest,sampler_state,**filters):
            self.assertEqual(p,0.);self.assertNotIn('dry_history',filters)
            saved=mirostat2.validate_state(sampler_state)
            choice=mirostat2.choose(self.backend.logits(lease),mask,temperature=temp,draw53=draw_candidates[0],
                mu=saved['mu'],tau=filters['mirostat_tau'],eta=filters['mirostat_eta'])
            feedback=dict(saved,mu=choice['next_mu']);self.calls.append(dict(draw_start=draw_start,feedback=feedback))
            return SelectionProposal(choice['token'],draw_start,1,feedback,source_digest)
        self.backend.sample_selection_proposal=proposal

    def request(self,*args,policy=None,**kwargs):
        return self.fixture.request(*args,policy=POLICY if policy is None else policy,**kwargs)

    def test_plain_feedback_advances_only_with_real_consumed_text(self):
        result=self.request();state=self.model.controller(result['snapshot'])
        self.assertEqual(state['schema'],4);self.assertEqual(state['next_draw'],4)
        self.assertEqual(state['sampler_state'],self.calls[-1]['feedback'])
        self.assertNotEqual(state['sampler_state'],mirostat2.start_state(4.))
        self.assertEqual(sum(state['seen'].values()),7);self.assertNotIn('7',state['seen'])

    def test_nonzero_feedback_fork_and_resume_are_independent(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']))
        a=self.model.store.fork('miro-A',first['snapshot']);b=self.model.store.fork('miro-B',first['snapshot'])
        left=self.request(a.tip,resume=True);self.model.store.move(a.name,a.revision,left['snapshot'],'miro-A')
        self.assertEqual(self.model.store.head(b.name),b);self.assertEqual(self.model.controller(first['snapshot']),saved)
        right=self.request(b.tip,resume=True);whole=self.request(steps=8)
        self.assertEqual(left['tokens'],right['tokens']);self.assertEqual(first['tokens']+right['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(left['snapshot']),self.model.controller(right['snapshot']))
        self.assertEqual(self.model.controller(whole['snapshot']),self.model.controller(right['snapshot']))

    def test_grammar_resume_preserves_parser_rng_and_feedback(self):
        first=self.request(grammar=True);saved=copy.deepcopy(self.model.controller(first['snapshot']))
        tail=self.request(first['snapshot'],grammar=True,resume=True,steps=16);whole=self.request(grammar=True,steps=16)
        self.assertTrue(tail['grammar_complete']);self.assertEqual(first['tokens']+tail['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(whole['snapshot']))
        self.assertEqual(self.model.controller(first['snapshot']),saved)

    def test_generation_settings_change_preserves_progressed_feedback(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']))
        tail=self.request(first['snapshot'],resume=True,steps=1,policy=dict(POLICY,temperature=.9,mirostat_tau=3.5))
        state=self.model.controller(tail['snapshot']);self.assertEqual(state['next_draw'],5)
        self.assertEqual(state['policy']['mirostat_tau'],3.5)
        self.assertEqual(self.model.controller(first['snapshot']),saved)

    def test_reseed_preserves_feedback_without_model_work(self):
        first=self.request();saved=copy.deepcopy(self.model.controller(first['snapshot']));before=self.backend.stats['forward_tokens']
        changed=self.model.set_random_stream(first['snapshot'],seed_stream(99));state=self.model.controller(changed)
        expected=copy.deepcopy(saved);expected.update(stream_key=state['stream_key'],next_draw=0)
        self.assertEqual(state,expected);self.assertNotEqual(state['stream_key'],saved['stream_key'])
        self.assertEqual(self.backend.stats['forward_tokens'],before)
        tail=self.request(changed,resume=True);self.assertEqual(self.model.controller(tail['snapshot'])['next_draw'],4)

    def test_missing_feedback_capability_rejects_before_restore(self):
        for changes in ({'mirostat2_protocol':'wrong'},{'mirostat2_feedback_protocol':'wrong'},
                        {'mirostat2_feedback_version':True}):
            with self.subTest(changes=changes):
                ready=self.backend._ready;self.backend._ready=dict(ready,**changes);before=self.backend.restores
                with self.assertRaises(InvalidOperation):self.request()
                self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[]);self.backend._ready=ready

    def test_dry_remains_closed_before_model_work(self):
        before=self.backend.restores
        with self.assertRaises(InvalidOperation):self.request(policy=dict(POLICY,dry_multiplier=.8))
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_consumption_failure_cannot_publish_or_advance_saved_feedback(self):
        first=self.request();saved=self.model.controller(first['snapshot']);objects=set(self.model.store.objects.iterdir())
        consume=self.backend.consume_generated
        def fail(*args):raise InvalidOperation('synthetic Mirostat consume failure')
        self.backend.consume_generated=fail
        with self.assertRaisesRegex(InvalidOperation,'synthetic Mirostat consume failure'):
            self.request(first['snapshot'],resume=True)
        self.assertEqual(set(self.model.store.objects.iterdir()),objects);self.assertEqual(self.model.controller(first['snapshot']),saved)
        self.backend.consume_generated=consume;self.assertEqual(len(self.request(first['snapshot'],resume=True)['tokens']),4)


if __name__=='__main__':unittest.main()
