"""Image Typical/DynaTemp controller boundaries; CPU fixtures, not GPU evidence."""
import copy
import unittest

from context_time_travel.min_p import SAMPLING_PROTOCOL_V4, choose
from context_time_travel.store import InvalidOperation
import test_vision_sampling_request as fixtures

POLICY=fixtures.POLICY


class VisionEntropyRequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.VisionSamplingRequestTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend;self.model=self.fixture.model
        self.backend._ready=dict(sampling_v4_protocol=SAMPLING_PROTOCOL_V4)
        self.calls=[]
        def select(lease,mask,temp,p,draw,**filters):
            self.calls.append(dict(temperature=temp,min_p=p,draw53=draw,**filters))
            return choose(self.backend.logits(lease),mask,temperature=temp,min_p=p,draw53=draw,**filters)['token']
        self.backend.sample_legal=select

    def test_typical_dynamic_and_combined_keep_actual_text_controller_counts(self):
        for knobs in (dict(typical_p=.95),dict(dynatemp_range=.35,dynatemp_exponent=1.),
                dict(typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.)):
            with self.subTest(knobs=knobs):
                self.calls.clear();result=self.fixture.request(policy=dict(POLICY,**knobs))
                state=self.model.controller(result['snapshot'])
                self.assertEqual(state['next_draw'],4);self.assertEqual(sum(state['seen'].values()),7)
                self.assertNotIn('7',state['seen']);self.assertEqual(len(self.calls),4)
                for call in self.calls:
                    for key,value in knobs.items():self.assertEqual(call[key],value)
                    self.assertNotIn('dry_history',call)

    def test_grammar_progress_and_rng_resume_match_uninterrupted(self):
        policy=dict(POLICY,typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.)
        first=self.fixture.request(grammar=True,policy=policy)
        saved=copy.deepcopy(self.model.controller(first['snapshot']))
        tail=self.fixture.request(first['snapshot'],steps=16,grammar=True,resume=True,policy=policy)
        whole=self.fixture.request(steps=16,grammar=True,policy=policy)
        self.assertEqual(first['tokens']+tail['tokens'],whole['tokens'])
        self.assertEqual(self.model.controller(tail['snapshot']),self.model.controller(whole['snapshot']))
        self.assertEqual(self.model.controller(first['snapshot']),saved)
        self.assertEqual(saved['next_draw'],4);self.assertTrue(tail['grammar_complete'])

    def test_different_policies_on_siblings_preserve_the_saved_progress(self):
        first=self.fixture.request(policy=dict(POLICY,typical_p=.95));saved=self.model.controller(first['snapshot'])
        a=self.model.store.fork('typical-A',first['snapshot']);b=self.model.store.fork('dynamic-B',first['snapshot'])
        left=self.fixture.request(a.tip,resume=True,policy=dict(POLICY,dynatemp_range=.35))
        self.model.store.move(a.name,a.revision,left['snapshot'],'entropy-A')
        self.assertEqual(self.model.store.head(b.name),b);self.assertEqual(self.model.controller(first['snapshot']),saved)
        right=self.fixture.request(b.tip,resume=True,policy=dict(POLICY,typical_p=.8))
        self.assertEqual(self.model.controller(left['snapshot'])['policy']['dynatemp_range'],.35)
        self.assertEqual(self.model.controller(right['snapshot'])['policy']['typical_p'],.8)
        self.assertEqual(self.model.controller(right['snapshot'])['next_draw'],8)

    def test_missing_or_wrong_native_v4_capability_rejects_before_restore(self):
        for ready in ({},dict(sampling_v4_protocol='wrong')):
            with self.subTest(ready=ready):
                self.backend._ready=ready;before=self.backend.restores
                with self.assertRaises(InvalidOperation):self.fixture.request(policy=dict(POLICY,typical_p=.95))
                self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_dry_entropy_combination_remains_closed_before_model_work(self):
        self.model.profile['execution']['vision_dry_protocol']='mixed-text-runs-dry-u53-v1'
        before=self.backend.restores
        with self.assertRaises(InvalidOperation):
            self.fixture.request(policy=dict(POLICY,typical_p=.95,dry_multiplier=.8))
        self.assertEqual(self.backend.restores,before);self.assertEqual(self.calls,[])

    def test_failed_consumption_cannot_publish_entropy_progress(self):
        before=set(self.model.store.objects.iterdir());parent=self.model.snapshot_bytes(self.fixture.root)
        consume=self.backend.consume_generated
        def fail(*args):raise InvalidOperation('synthetic selected-token consumption failure')
        self.backend.consume_generated=fail
        with self.assertRaises(InvalidOperation):
            self.fixture.request(policy=dict(POLICY,typical_p=.95,dynatemp_range=.35))
        self.assertEqual(set(self.model.store.objects.iterdir()),before)
        self.assertEqual(self.model.snapshot_bytes(self.fixture.root),parent)
        self.backend.consume_generated=consume
        self.assertEqual(len(self.fixture.request(policy=dict(POLICY,typical_p=.95))['tokens']),4)


if __name__=='__main__':unittest.main()
