"""Independent-reference dispatch tests; CPU only."""
import unittest

import run_min_p_probe
from context_time_travel.sampled_controller import PlainSamplingSession, start
from test_sampling_proposal import POLICY, ProposalTokenizer, make_mask


class XtcReferenceDispatchTests(unittest.TestCase):
    def test_cli_observer_emits_only_changed_xtc_settings(self):
        from run_sampler_serving_probe import sampling_arguments
        previous = dict(POLICY, xtc_probability=.12, xtc_threshold=.15)
        current = dict(previous, xtc_probability=.2)
        self.assertEqual(sampling_arguments(current, previous), ['--xtc-probability', '0.2'])
        self.assertEqual(sampling_arguments(current, current), [])

    def session(self, policy, *, force_schema4=False):
        state = start(policy, 'reference-dispatch', (0,), 4, [3])
        if force_schema4:
            state['schema'] = 4
            state['sampler_state'] = None
        tokenizer = ProposalTokenizer()
        return PlainSamplingSession.restore(state, (0,), 4,
            tokenizer.token_bytes, tokenizer.is_eog)

    def test_reference_commits_two_draws_for_active_multisupport_xtc(self):
        session = self.session(dict(POLICY, xtc_probability=1., xtc_threshold=0.))
        token = run_min_p_probe.reference_choice(session, [0., 0., 0., 0.], make_mask(4, (0, 1, 2)))
        self.assertEqual(token, 2)
        session._consume_selected(token, b'c', False)
        self.assertEqual(session.current_controller['next_draw'], 2)

    def test_reference_commits_one_draw_when_grammar_support_is_singleton(self):
        import test_state_request as fixtures
        fixture = fixtures.StateRequestTests()
        fixture.setUp()
        try:
            state = fixture.engine.start('edit', 'reference-grammar', (0,), 2,
                sampling=dict(POLICY, xtc_probability=1., xtc_threshold=0.))
            session = fixture.engine.restore(state, (0,))
            token = run_min_p_probe.reference_choice(session, [0., 0.], b'\x03')
            self.assertEqual(token, 0)
            session._consume_selected(token, b'A', False)
            self.assertEqual(session.current_controller['next_draw'], 1)
        finally:
            fixture.tearDown()

    def test_reference_preserves_zero_draw_greedy_after_schema4_restore(self):
        session = self.session(dict(POLICY, temperature=0), force_schema4=True)
        token = run_min_p_probe.reference_choice(session, [0., 2., 1., -1.], b'\x0f')
        self.assertEqual(token, 1)
        session._consume_selected(token, b'b', False)
        self.assertEqual(session.current_controller['next_draw'], 0)

    def test_reference_keeps_legacy_single_draw_dispatch(self):
        session = self.session(POLICY)
        token = run_min_p_probe.reference_choice(session, [0., 2., 1., -1.], make_mask(4, (1,)))
        self.assertEqual(token, 1)
        session._consume_selected(token, b'b', False)
        self.assertEqual(session.current_controller['next_draw'], 1)


if __name__ == '__main__':
    unittest.main()
