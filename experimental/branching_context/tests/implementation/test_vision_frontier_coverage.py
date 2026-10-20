"""Full tiny/medium frontier coverage is stricter than the target sampling policy."""
import unittest
from run_vision_qualification import sample_points,validate_generation,QualificationError

class FullFrontierTests(unittest.TestCase):
    def test_every_consumed_frontier_for_small_cases(self):
        for count in (1,4,8,13,16):
            with self.subTest(count=count):
                self.assertEqual(sample_points(count,policy='every-generated'),list(range(1,count+1)))
    def test_target_keeps_declared_samples_including_early_final(self):
        self.assertEqual(sample_points(128,policy='sampled'),[1,4,16,64,128])
        self.assertEqual(sample_points(13,policy='sampled'),[1,4,13])
    def test_sampled_success_cannot_qualify_every_generated(self):
        row=dict(tokens=[7]*8,forward_tokens=8,prefill_tokens=0,frontiers={'1':'one','4':'four','8':'final'})
        validate_generation(row,limit=8,policy='sampled')
        with self.assertRaises(QualificationError):validate_generation(row,limit=8,policy='every-generated')
        row['frontiers']={str(k):'frontier' for k in range(1,9)}
        validate_generation(row,limit=8,policy='every-generated')
    def test_unknown_or_missing_policy_is_not_silently_sampled(self):
        for policy in ('unknown','',None,True):
            with self.subTest(policy=policy),self.assertRaises(QualificationError):sample_points(8,policy=policy)

if __name__=='__main__':unittest.main()
