import copy
import unittest
from run_sampler_serving_probe import validate_completed_request, sampling_arguments


class ServingProbeTests(unittest.TestCase):
    def test_cli_sampler_arguments_are_detached_from_authentication_state(self):
        policy=dict(temperature=1.1,min_p=.02,top_k=20,top_p=.9)
        before=dict(policy)
        self.assertEqual(sampling_arguments(policy),['--temperature','1.1','--min-p','0.02',
                         '--top-k','20','--top-p','0.9'])
        self.assertEqual(policy,before)
        self.assertEqual(sampling_arguments(dict(policy,temperature=.4),policy),['--temperature','0.4'])

    def record(self):
        return dict(tokens=[2],controller={'next_draw':5},samples={'suffix':'a','generated-1':'b'},
            counter_deltas=dict(forward_tokens=5,prefill_tokens=4,decode_timed_tokens=1,
                sampling_calls=1,sampling_logit_bytes=36,sampling_mask_bytes=2))

    def test_complete_matching_measurements_are_accepted(self):
        row=self.record()
        validate_completed_request(row,row,vocab=9,suffix=4)

    def test_missing_frontier_or_controller_cursor_cannot_qualify(self):
        expected=self.record()
        for field,value in [('samples',{'suffix':'a'}),('controller',{'next_draw':0}),('tokens',[3])]:
            changed=copy.deepcopy(expected);changed[field]=value
            with self.assertRaises(AssertionError):validate_completed_request(changed,expected,vocab=9,suffix=4)

    def test_replayed_prefix_or_missing_byte_instrumentation_cannot_qualify(self):
        expected=self.record()
        for key in expected['counter_deltas']:
            changed=copy.deepcopy(expected);changed['counter_deltas'][key]+=1
            with self.assertRaises(AssertionError):validate_completed_request(changed,expected,vocab=9,suffix=4)
        changed=copy.deepcopy(expected);changed['counter_deltas'].pop('sampling_logit_bytes')
        with self.assertRaises(AssertionError):validate_completed_request(changed,expected,vocab=9,suffix=4)
