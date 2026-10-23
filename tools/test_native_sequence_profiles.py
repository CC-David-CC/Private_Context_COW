import unittest
from pcc_native_training import validate_examples,validate_sequence_profile

class SequenceProfiles(unittest.TestCase):
    def example(self,n):
        return dict(id='owned',tokens=[0]*n,targets=[0]*n,loss_mask=[False]*(n-1)+[True],verified=True,provenance_sha256='a'*64)
    def test_default_and_opt_in_bounds(self):
        with self.assertRaises(ValueError):validate_examples([self.example(8193)])
        self.assertEqual(len(validate_examples([self.example(32768)],sequence_profile='tiny-cpu-long-v1')[0]['tokens']),32768)
        with self.assertRaises(ValueError):validate_examples([self.example(32769)],sequence_profile='tiny-cpu-long-v1')
    def test_long_requires_cpu_and_known_profile(self):
        with self.assertRaises(ValueError):validate_sequence_profile('tiny-cpu-long-v1',['CUDA0'])
        with self.assertRaises(ValueError):validate_sequence_profile('unbounded',['CPU'])
        self.assertEqual(validate_sequence_profile('short-v1',['CUDA0']),'short-v1')
    def test_arrays_and_targets_are_checked(self):
        x=self.example(8);x['tokens']='bad'
        with self.assertRaises(ValueError):validate_examples([x])
        x=self.example(8);x['targets'][0]=1
        with self.assertRaises(ValueError):validate_examples([x])
