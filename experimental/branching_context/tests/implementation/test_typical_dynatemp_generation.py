"""Generation-boundary Typical/DynaTemp contracts; CPU fixtures."""
import copy
import math
import unittest
from context_time_travel import min_p, sampled_controller
from context_time_travel.store import IntegrityError

BASE=dict(temperature=.7,min_p=.05,top_k=0,top_p=1.,repetition_penalty=1.)
V4='full-vocab-typical-dyna-min-p-u53-v4'


class EntropyPolicyTests(unittest.TestCase):
    def test_requested_presets_are_settable_without_inventing_dry_settings(self):
        typical=dict(BASE,temperature=.8,min_p=0.,typical_p=.95)
        dynamic=dict(BASE,temperature=.55,min_p=.06,dynatemp_range=.35,dynatemp_exponent=1.)
        self.assertEqual(min_p.validate_policy(typical),typical)
        self.assertEqual(min_p.validate_policy(dynamic),dynamic)
        self.assertEqual(min_p.validate_policy(dict(BASE,dynatemp_range=.35)),
                         dict(BASE,dynatemp_range=.35,dynatemp_exponent=1.))
        self.assertEqual(min_p.validate_policy(BASE),BASE)

    def test_invalid_parameters_reject_booleans_nonfinite_and_overflow(self):
        for change in (dict(typical_p=0),dict(typical_p=True),dict(typical_p=1.1),
                       dict(dynatemp_range=-1),dict(dynatemp_range=True),
                       dict(dynatemp_exponent=0),dict(dynatemp_exponent=float('nan')),
                       dict(temperature=1e308,dynatemp_range=1e308)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                min_p.validate_policy(dict(BASE,**change))

    def test_active_entropy_transforms_require_v4_capability(self):
        policy=dict(BASE,typical_p=.95)
        for protocol in (min_p.MIN_P_PROTOCOL,min_p.SAMPLING_PROTOCOL_V2,min_p.DRY_PROTOCOL):
            self.assertFalse(min_p.supports_policy(protocol,policy))
        self.assertTrue(min_p.supports_policy(V4,policy))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V2,dict(policy,temperature=0)))


class EntropyControllerTests(unittest.TestCase):
    def test_settings_survive_restore_and_are_fixed_until_consumption(self):
        policy=dict(BASE,typical_p=.95,dynatemp_range=.3,dynatemp_exponent=2.)
        state=sampled_controller.start(policy,'entropy',(0,),3,[2])
        parent=copy.deepcopy(state)
        session=sampled_controller.PlainSamplingSession.restore(state,(0,),3,lambda t:b'a',lambda t:False)
        self.assertEqual(session.current_controller,state)
        self.assertEqual(state['sampler_order'],
            ['grammar/support','min-p','typical','dynatemp','categorical'])
        calls=[]
        session.choose(b'\x07',lambda *a,**kw:calls.append(kw) or 0)
        self.assertEqual(calls[0]['typical_p'],.95)
        self.assertEqual(calls[0]['dynatemp_exponent'],2.)
        self.assertNotIn('dry_history',calls[0])
        session._consume_selected(0,b'a',False)
        self.assertEqual(session.current_controller['next_draw'],1)
        self.assertEqual(state,parent)

    def test_complete_override_removes_old_optional_fields_without_resetting_rng(self):
        state=sampled_controller.start(dict(BASE,typical_p=.9,dynatemp_range=.2),'entropy',(0,),3,[2])
        state['next_draw']=4
        session=sampled_controller.PlainSamplingSession.restore(state,(0,),3,lambda t:b'a',lambda t:False)
        session.configure_generation(BASE)
        current=session.current_controller
        self.assertEqual(current['next_draw'],4)
        self.assertNotIn('typical_p',current['policy'])
        self.assertNotIn('dynatemp_range',current['policy'])
        self.assertEqual(current['sampler_order'],sampled_controller.MIN_P_SAMPLER_ORDER)

    def test_partial_canonical_parameter_group_is_rejected(self):
        state=sampled_controller.start(dict(BASE,dynatemp_range=.2),'entropy',(0,),3,[2])
        del state['policy']['dynatemp_exponent']
        with self.assertRaises(IntegrityError):sampled_controller.validate_controller(state)


class EntropySelectionTests(unittest.TestCase):
    def test_typical_can_choose_nonmodal_information_content(self):
        scores=[math.log(.55),math.log(.30),math.log(.15)]
        actual=min_p.choose(scores,b'\x07',temperature=.8,min_p=0,draw53=0,typical_p=.1)
        self.assertEqual(actual['token'],1)
        self.assertEqual(actual['selection_protocol'],V4)

    def test_dynamic_uniform_entropy_reaches_upper_temperature(self):
        actual=min_p.choose([0.,0.,0.,0.],b'\x0f',temperature=.55,min_p=0,draw53=0,
            dynatemp_range=.35,dynatemp_exponent=1.)
        self.assertEqual(actual['effective_temperature'],.55+.35)
        self.assertEqual(actual['selection_protocol'],V4)

    def test_zero_base_temperature_keeps_existing_legal_greedy_behavior(self):
        result=min_p.choose([99.,2.,1.],b'\x06',temperature=0,min_p=0,draw53=0,
            typical_p=.1,dynatemp_range=.35,dynatemp_exponent=1.)
        self.assertEqual(result['token'],1)


if __name__=='__main__':unittest.main()
