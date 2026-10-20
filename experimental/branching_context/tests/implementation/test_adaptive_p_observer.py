"""Independent reference dispatch and per-generation Adaptive-P CLI routing."""
import copy
import unittest

import run_min_p_probe
from context_time_travel import adaptive_p


class ReferenceSession:
    def __init__(self, *, dry=False):
        self._state = {'schema': 4, 'constraint': None}
        self.dry = dry
        self.proposal = None

    def choose(self, mask, selector):
        filters = dict(top_k=0, top_p=1., adaptive_target=.55, adaptive_decay=.9,
                       sampler_state=adaptive_p.initial_state(.55, .9))
        if self.dry:
            filters.update(dry_multiplier=4., dry_base=2., dry_allowed_length=1,
                dry_penalty_last_n=8, dry_history=(1, 2, 1), dry_breakers=())
        self.proposal = selector(mask, .8, .05, draw_start=9,
            draw_candidates=(7 * (1 << 53) // 10,), source_digest='a' * 64, **filters)
        return self.proposal.token


class AdaptivePReferenceDispatchTests(unittest.TestCase):
    def test_cli_observer_sends_only_changed_adaptive_settings(self):
        from run_sampler_serving_probe import sampling_arguments
        previous = dict(temperature=.8, adaptive_target=.55, adaptive_decay=.9)
        self.assertEqual(sampling_arguments(dict(previous, adaptive_target=.6), previous),
                         ['--adaptive-target', '0.6'])
        self.assertEqual(sampling_arguments(dict(previous, adaptive_decay=.8), previous),
                         ['--adaptive-decay', '0.8'])
        self.assertEqual(sampling_arguments(previous, previous), [])

    def test_reference_returns_feedback_without_mutating_saved_controller(self):
        from context_time_travel.adaptive_p_pipeline import choose_v7
        session = ReferenceSession()
        before = copy.deepcopy(session._state)
        logits = [1., 1., 3.]
        expected = choose_v7(logits, b'\x07', adaptive_p.initial_state(.55, .9),
            temperature=.8, min_p=.05, top_k=0, top_p=1.,
            adaptive_target=.55, adaptive_decay=.9, draw53=7 * (1 << 53) // 10)
        self.assertEqual(run_min_p_probe.reference_choice(session, logits, b'\x07'), expected['token'])
        self.assertEqual(session.proposal.feedback_proposal, expected['feedback_state'])
        self.assertEqual(session.proposal.draw_count, 1)
        self.assertEqual(session.proposal.draw_start, 9)
        self.assertEqual(session.proposal.source_digest, 'a' * 64)
        self.assertEqual(session._state, before)

    def test_reference_applies_dry_before_adaptation_without_altering_logits(self):
        from context_time_travel.adaptive_p_pipeline import choose_v7
        session = ReferenceSession(dry=True)
        logits = [1., 1., 3.]
        expected = choose_v7([1., 1., -1.], b'\x07', adaptive_p.initial_state(.55, .9),
            temperature=.8, min_p=.05, top_k=0, top_p=1.,
            adaptive_target=.55, adaptive_decay=.9, draw53=7 * (1 << 53) // 10)
        self.assertEqual(run_min_p_probe.reference_choice(session, logits, b'\x07'), expected['token'])
        self.assertEqual(session.proposal.feedback_proposal, expected['feedback_state'])
        self.assertEqual(logits, [1., 1., 3.])


if __name__ == '__main__':
    unittest.main()
