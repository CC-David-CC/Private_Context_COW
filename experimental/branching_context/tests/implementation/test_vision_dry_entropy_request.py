"""Mixed DRY/entropy controller isolation; CPU fixtures, not GPU evidence."""
import copy
import unittest

from context_time_travel.min_p import SAMPLING_PROTOCOL_V4,choose
from context_time_travel.store import InvalidOperation
from context_time_travel.vision_dry import text_window
from test_vision_dry_native import reference
import test_vision_sampling_request as fixtures

POLICY=dict(fixtures.POLICY,dry_multiplier=.8,typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.)


class VisionDryEntropyRequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.VisionSamplingRequestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.model=self.fixture.model;self.backend=self.fixture.backend;self.calls=[]
        self.model.profile['execution']['vision_dry_protocol']='mixed-text-runs-dry-u53-v1'
        self.backend._ready=dict(sampling_v4_protocol=SAMPLING_PROTOCOL_V4,
            vision_dry_protocol='mixed-text-runs-dry-u53-v1',vision_dry_sampling_protocol='mixed-text-runs-sampling-u53-v2',
            vision_sampling_protocols=[SAMPLING_PROTOCOL_V4])
        def select(lease,mask,temp,p,draw,**filters):
            window=text_window(lease['inputs'],filters['dry_penalty_last_n'])
            self.assertEqual(tuple(filters.pop('dry_history')),window['tokens']);breakers=filters.pop('dry_breakers')
            adjusted=reference(self.backend.logits(lease),window['tokens'],breakers,window['boundaries'],
                filters['dry_allowed_length'],filters['dry_multiplier'],filters['dry_base'])
            self.calls.append(window);filters['dry_multiplier']=0.
            return choose(adjusted,mask,temperature=temp,min_p=p,draw53=draw,**filters)['token']
        self.backend.sample_legal=select

    def request(self,*args,**kwargs):return self.fixture.request(*args,policy=POLICY,**kwargs)

    def test_plain_actual_text_counts_and_branch_local_rng(self):
        first=self.request();state=self.model.controller(first['snapshot'])
        self.assertEqual(state['next_draw'],4);self.assertEqual(sum(state['seen'].values()),7)
        self.assertNotIn('7',state['seen']);self.assertTrue(all(w['boundaries']==(2,) for w in self.calls))
        saved=copy.deepcopy(state);left=self.request(first['snapshot'],resume=True);right=self.request(first['snapshot'],resume=True)
        self.assertEqual(left['tokens'],right['tokens']);self.assertEqual(self.model.controller(left['snapshot']),self.model.controller(right['snapshot']))
        self.assertEqual(self.model.controller(first['snapshot']),saved)

    def test_grammar_resume_keeps_parser_rng_and_text_window(self):
        first=self.request(grammar=True);tail=self.request(first['snapshot'],grammar=True,resume=True,steps=16)
        whole=self.request(grammar=True,steps=16)
        self.assertTrue(tail['grammar_complete']);self.assertEqual(first['tokens']+tail['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(whole['snapshot']))

    def test_failed_consumption_cannot_mutate_parent_or_publish(self):
        first=self.request();saved=self.model.controller(first['snapshot']);objects=set(self.model.store.objects.iterdir())
        def fail(*args):raise InvalidOperation('synthetic combined DRY consume failure')
        self.backend.consume_generated=fail
        with self.assertRaisesRegex(InvalidOperation,'synthetic combined DRY consume failure'):
            self.request(first['snapshot'],resume=True)
        self.assertEqual(self.model.controller(first['snapshot']),saved);self.assertEqual(set(self.model.store.objects.iterdir()),objects)


if __name__=='__main__':unittest.main()
