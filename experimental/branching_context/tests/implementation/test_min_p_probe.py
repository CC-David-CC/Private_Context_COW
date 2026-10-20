import struct
import unittest

from run_min_p_probe import check_delta, compare_record, frontier_bytes, full_mask


class MinPProbeTests(unittest.TestCase):
    def test_full_mask_has_no_padding_bits(self):
        self.assertEqual(full_mask(10), b'\xff\x03')
        self.assertEqual(full_mask(8), b'\xff')
        for bad in (0, -1, True):
            with self.assertRaises(ValueError): full_mask(bad)

    def test_full_frontier_rejects_nonfinite_and_boolean_values(self):
        self.assertEqual(frontier_bytes([1.25, -2.0], 2), struct.pack('<ff', 1.25, -2))
        for values in ([1.0], [True, 2.0], [float('nan'), 0.0], [float('inf'), 0.0]):
            with self.assertRaises(ValueError): frontier_bytes(values, 2)

    def test_work_requires_actual_suffix_and_selected_token_counts(self):
        before=dict(forward_tokens=128,prefill_tokens=128,decode_timed_tokens=0,sampling_calls=0,
                    sampling_logit_bytes=0,sampling_mask_bytes=0)
        after=dict(forward_tokens=140,prefill_tokens=132,decode_timed_tokens=8,sampling_calls=8,
                   sampling_logit_bytes=65*4*8,sampling_mask_bytes=9*8)
        check_delta(before,after,4,8,vocab=65)
        for field in after:
            changed=dict(after);changed[field]+=1
            with self.assertRaises(AssertionError):check_delta(before,changed,4,8,vocab=65)
        after['sampling_calls']=True
        with self.assertRaises(AssertionError):check_delta(before,after,4,8,vocab=65)

    def test_comparison_rejects_token_controller_and_frontier_disagreement(self):
        expected=dict(tokens=[7,8],controller={'next_draw':2},samples={'suffix':'aa','generated-1':'bb','generated-2':'cc'})
        compare_record(expected,expected)
        for field,value in [('tokens',[7,9]),('controller',{'next_draw':0}),
                            ('samples',{'suffix':'aa','generated-1':'bb','generated-2':'cd'})]:
            changed=dict(expected);changed[field]=value
            with self.assertRaises(AssertionError):compare_record(changed,expected)


if __name__=='__main__':unittest.main()
