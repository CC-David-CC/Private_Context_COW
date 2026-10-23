import copy
import unittest
from tools.pcc_learning_cases import cases,check

class LearningCases(unittest.TestCase):
    def test_thousand_distinct_owned_examples_and_disjoint_evaluations(self):
        rows=cases();self.assertEqual(sum(r['split']=='train' for r in rows),1000)
        self.assertEqual(len({r['prompt'] for r in rows}),len(rows))
        self.assertTrue(all(check(r,r['completion'])['passed'] for r in rows))
    def test_positive_denominator_shortcut_fails_signed_divisors(self):
        row=cases(count=1)[0];a,b=row['parameters']
        self.assertFalse(check(row,'('+a+'+'+b+'-1)//'+b)['passed'])
    def test_checker_does_not_use_the_training_answer(self):
        row=cases(count=1)[0];original=row['completion'];row['completion']=' 123\n'
        self.assertFalse(check(row,row['completion'])['passed']);self.assertTrue(check(row,original)['passed'])
    def test_untrusted_side_effects_are_not_executed(self):
        row=cases(count=1)[0]
        for text in ["__import__('os').system('echo unsafe')",'[x for x in range(100)]','2**10000000']:
            self.assertFalse(check(row,text)['passed'])

if __name__=='__main__':unittest.main()
