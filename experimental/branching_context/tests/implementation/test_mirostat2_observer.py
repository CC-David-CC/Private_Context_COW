"""Independent Mirostat reference dispatch and CLI observer regressions."""
import copy
import unittest

import run_min_p_probe
from context_time_travel import mirostat2


class ReferenceSession:
    def __init__(self, *, dry=False):
        self._state = {'schema': 4, 'constraint': None}
        self.dry = dry
        self.proposal = None

    def choose(self, mask, selector):
        filters = dict(top_k=0, top_p=1., mirostat=2, mirostat_tau=4.,
            mirostat_eta=.05, sampler_state=mirostat2.start_state(4.))
        if self.dry:
            filters.update(dry_multiplier=4., dry_base=2., dry_allowed_length=1,
                dry_penalty_last_n=8, dry_history=(1, 2, 1), dry_breakers=())
        self.proposal = selector(mask, .8, 0., draw_start=9,
            draw_candidates=(7*(1<<53)//10,), source_digest='a'*64, **filters)
        return self.proposal.token


class MirostatReferenceDispatchTests(unittest.TestCase):
    def test_cli_observer_preserves_mode_and_sends_only_changed_feedback_parameters(self):
        from run_sampler_serving_probe import sampling_arguments
        previous = dict(temperature=.8, mirostat=2, mirostat_tau=4., mirostat_eta=.05)
        self.assertEqual(sampling_arguments(dict(previous, mirostat_tau=3.), previous),
                         ['--mirostat-tau', '3.0'])
        self.assertEqual(sampling_arguments(dict(previous, mirostat=0), previous), ['--mirostat', '0'])
        self.assertEqual(sampling_arguments(previous, previous), [])

    def test_reference_returns_uncommitted_one_draw_feedback_proposal(self):
        session = ReferenceSession()
        before = copy.deepcopy(session._state)
        logits = [1., 1., 3.]
        expected = mirostat2.choose(logits, b'\x07', temperature=.8,
            draw53=7*(1<<53)//10, mu=8., tau=4., eta=.05)
        token = run_min_p_probe.reference_choice(session, logits, b'\x07')
        self.assertEqual(token, expected['token'])
        self.assertEqual(session.proposal.draw_count, 1)
        self.assertEqual(session.proposal.draw_start, 9)
        self.assertEqual(session.proposal.source_digest, 'a'*64)
        self.assertEqual(session.proposal.feedback_proposal,
            dict(mirostat2.start_state(4.), mu=expected['next_mu']))
        self.assertEqual(session._state, before)

    def test_reference_applies_dry_before_mirostat_without_altering_logits(self):
        session = ReferenceSession(dry=True)
        logits = [1., 1., 3.]
        expected = mirostat2.choose([1., 1., -1.], b'\x07', temperature=.8,
            draw53=7*(1<<53)//10, mu=8., tau=4., eta=.05)
        self.assertEqual(run_min_p_probe.reference_choice(session, logits, b'\x07'), expected['token'])
        self.assertEqual(session.proposal.feedback_proposal['mu'], expected['next_mu'])
        self.assertEqual(logits, [1., 1., 3.])


if __name__ == '__main__':
    unittest.main()
