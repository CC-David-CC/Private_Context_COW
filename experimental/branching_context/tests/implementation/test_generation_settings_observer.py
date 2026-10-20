"""CPU-only evidence guards for the bounded generation-settings harness."""
import sys
from pathlib import Path
import unittest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from run_generation_settings_probe import CampaignObserver, DEFAULT_POLICIES, _check_policy, reference_selector, continuation_samples


class GenerationSettingsObserverTests(unittest.TestCase):
    def test_continuation_keeps_every_reachable_frontier_and_requires_prefix_witness(self):
        measured={'prefix':'earlier','question':'start','generated-1':'next'}
        self.assertEqual(continuation_samples(measured),{'question':'start','generated-1':'next'})
        self.assertEqual(measured['prefix'],'earlier')
        for missing in ('prefix','question'):
            with self.assertRaises(AssertionError):continuation_samples({k:v for k,v in measured.items() if k!=missing})

    def test_reference_callback_accepts_filters_and_checks_the_recorded_policy(self):
        from context_time_travel.sampled_controller import PlainSamplingSession,start
        policy=dict(DEFAULT_POLICIES[0]['sampling'],top_k=2,top_p=.6)
        state=start(policy,'test',(0,),4,[3])
        session=PlainSamplingSession.restore(state,(0,),4,lambda token:b'A',lambda token:False)
        selector=reference_selector([0.,0.,0.,-100.],policy)
        token=session.choose(b'\x07',selector)
        self.assertIn(token,(0,1))
        with self.assertRaises(AssertionError):selector(b'\x07',.7,.05,0,top_k=3,top_p=.6)

    def test_default_rows_are_independent_settings_and_include_greedy(self):
        self.assertEqual(len(DEFAULT_POLICIES), 7)
        for row in DEFAULT_POLICIES:
            _check_policy(row["sampling"])
        by_name = {row["name"]: row["sampling"] for row in DEFAULT_POLICIES}
        self.assertEqual(by_name["temperature-zero"]["temperature"], 0.0)
        self.assertEqual(by_name["temperature-low"]["min_p"], 0.05)
        self.assertEqual(by_name["temperature-high"]["top_k"], 0)
        self.assertEqual(by_name["min-p-off"]["temperature"], 0.7)
        self.assertEqual(by_name["top-k-40"]["top_p"], 1.0)
        self.assertEqual(by_name["top-p-09"]["top_k"], 0)

    def test_policy_rejects_boolean_integer_and_nonfinite_settings(self):
        policy = dict(DEFAULT_POLICIES[0]["sampling"])
        policy["top_k"] = True
        with self.assertRaises(ValueError):
            _check_policy(policy)
        policy = dict(DEFAULT_POLICIES[0]["sampling"])
        policy["temperature"] = float("nan")
        with self.assertRaises(ValueError):
            _check_policy(policy)

    def test_progress_is_monotonic_and_labels_are_safe(self):
        observer = CampaignObserver(1)
        observer.update("baseline", phase="reference", tokens=2, expected_tokens=8)
        observer.update("baseline", phase="candidate", tokens=4, expected_tokens=8)
        with self.assertRaises(ValueError):
            observer.update("baseline", phase="candidate", tokens=3, expected_tokens=8)
        with self.assertRaises(ValueError):
            observer.update("../../x", phase="complete", tokens=8, expected_tokens=8)

    def test_pass_requires_exact_numeric_native_exit_and_exact_match(self):
        observer = CampaignObserver(1)
        valid = {"name": "baseline", "status": "PASS", "native_exit_code": 0,
                 "native_pid_absent": True, "exact_match": True,
                 "reference_tokens": [1, 2], "candidate_tokens": [1, 2],
                 "reference_next_draw": 8, "candidate_next_draw": 8}
        self.assertTrue(observer.qualify([valid]))
        for mutation in (
            {"native_exit_code": True},
            {"native_exit_code": 7},
            {"native_exit_code": None},
            {"native_pid_absent": False},
            {"exact_match": False},
            {"candidate_tokens": [1, 3]},
            {"candidate_next_draw": 12},
            {"reference_next_draw": None},
        ):
            row = dict(valid, **mutation)
            self.assertFalse(observer.qualify([row]), mutation)
        self.assertFalse(observer.qualify([]))


if __name__ == "__main__":
    unittest.main()
