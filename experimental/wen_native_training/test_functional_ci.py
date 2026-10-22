import unittest
from functional_ci import compare


def row(name,passed,**diagnostics):
    return dict(id=name,guided=False,distance=0,verdict=dict(passed=passed),**diagnostics)


class FunctionalCITests(unittest.TestCase):
    def test_lower_internal_loss_cannot_hide_broken_actions(self):
        result = compare([row('writes-valid-file',True,loss=10.)],
                         [row('writes-valid-file',False,loss=.001)])
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(len(result['regressed']),1)

    def test_changed_scores_and_words_do_not_reject_correct_task(self):
        result = compare([row('writes-valid-file',False,logits=[1.,0.],text='old attempt')],
                         [row('writes-valid-file',True,logits=[-100.,500.],text='new valid action')])
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(len(result['repaired']),1)
        self.assertFalse(result['base_distribution_preservation'])

    def test_missing_duplicate_and_empty_coverage_fail_closed(self):
        for baseline,candidate in [([row('a',True),row('b',True)],[row('a',True)]),
                                   ([row('a',True)],[row('a',True),row('a',True)]),([],[])]:
            with self.assertRaises(ValueError): compare(baseline,candidate)

    def test_same_old_failures_remain_a_failing_ci_run(self):
        result = compare([row('a',True),row('b',False)],[row('a',True),row('b',False)])
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['candidate_passed'],1)
        self.assertEqual(len(result['still_failing']),1)


if __name__=='__main__':unittest.main()
