"""CPU tests for XTC's versioned, branch-owned variable-draw controller."""
import copy
import unittest

from context_time_travel import sampled_controller
from context_time_travel.store import IntegrityError
from test_sampling_proposal import POLICY, ProposalTokenizer, make_mask


XTC = dict(POLICY, xtc_probability=.12, xtc_threshold=.15)


class XtcControllerTests(unittest.TestCase):
    def test_only_active_xtc_creates_proposal_schema(self):
        state = sampled_controller.start(XTC, 'xtc', (0,), 4, [3])
        self.assertEqual(state['schema'], 4)
        self.assertIsNone(state['sampler_state'])
        self.assertEqual(state['sampler_order'][-2:], ['xtc', 'categorical'])
        self.assertEqual(sampled_controller.validate_controller(state), state)
        for policy in (POLICY, dict(XTC, xtc_probability=0),
                       dict(XTC, xtc_threshold=.6), dict(XTC, temperature=0)):
            with self.subTest(policy=policy):
                legacy = sampled_controller.start(policy, 'legacy', (0,), 4, [3])
                self.assertEqual(legacy['schema'], 3)
                self.assertNotIn('sampler_state', legacy)

    def test_active_xtc_cannot_enter_single_draw_schema(self):
        state = sampled_controller.start(XTC, 'xtc', (0,), 4, [3])
        state.pop('sampler_state', None)
        state['schema'] = 3
        with self.assertRaises(IntegrityError):
            sampled_controller.validate_controller(state)

    def test_generation_boundary_upgrade_preserves_progress_and_binds_draw_start(self):
        from context_time_travel.sampling_proposal import SelectionProposal
        tokenizer = ProposalTokenizer()
        state = sampled_controller.start(POLICY, 'upgrade', (0,), 4, [3])
        state['next_draw'] = 7
        original = copy.deepcopy(state)
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        session.configure_generation(XTC)
        upgraded = session.current_controller
        self.assertEqual(upgraded['schema'], 4)
        self.assertEqual(upgraded['next_draw'], 7)
        for field in ('stream_key', 'seen', 'stop_tail', 'accepted_in_scope',
                      'output_pending_hex', 'output_decoder'):
            self.assertEqual(upgraded[field], original[field])
        starts = []
        def select(mask, temperature, min_p, *, draw_start, source_digest,
                   draw_candidates, **filters):
            starts.append(draw_start)
            self.assertEqual(filters['xtc_probability'], .12)
            self.assertEqual(len(draw_candidates), 2)
            return SelectionProposal(1, draw_start, 2, None, source_digest)
        token = session.choose(make_mask(4, (1, 2)), select)
        session._consume_selected(token, b'b', False)
        self.assertEqual(starts, [7])
        self.assertEqual(session.current_controller['next_draw'], 9)
        self.assertEqual(state, original)

    def test_grammar_start_and_upgrade_keep_parser_progress(self):
        import test_state_request as fixtures
        fixture = fixtures.StateRequestTests()
        fixture.setUp()
        try:
            state = fixture.engine.start('edit', 'xtc-grammar', (0,), 2, sampling=XTC)
            self.assertEqual(state['schema'], 4)
            self.assertEqual(fixture.engine.restore(state, (0,)).current_controller, state)
            legacy = fixture.engine.start('edit', 'upgrade-grammar', (0,), 2, sampling=POLICY)
            session = fixture.engine.restore(legacy, (0,))
            session.configure_generation(XTC)
            upgraded = session.current_controller
            self.assertEqual(upgraded['schema'], 4)
            self.assertEqual(upgraded['constraint'], legacy['constraint'])
            self.assertEqual(upgraded['next_draw'], legacy['next_draw'])
            self.assertEqual(upgraded['seen'], legacy['seen'])
        finally:
            fixture.tearDown()

    def test_real_request_upgrades_active_scope_before_native_selection(self):
        import test_state_request as fixtures
        from context_time_travel.sampling_proposal import SelectionProposal, XTC_PROPOSAL_PROTOCOL
        fixture = fixtures.StateRequestTests()
        fixture.setUp()
        try:
            state = sampled_controller.start(POLICY, 'resume-xtc',
                fixture.model.history(fixture.sid), 2, [1])
            state['next_draw'] = 5
            fixture.model.cache[fixture.sid] = fixture.model._load_checkpoint(
                fixture.sid, fixture.model.history(fixture.sid))
            source = fixture.model._controller_transition(fixture.sid, state)
            fixture.model.profile['execution']['sampling_protocol'] = XTC_PROPOSAL_PROTOCOL
            starts = []
            def select(lease, mask, temperature, min_p, *, draw_start,
                       draw_candidates, source_digest, **filters):
                starts.append(draw_start)
                return SelectionProposal(0, draw_start, 2, None, source_digest)
            fixture.backend.sample_selection_proposal = select
            result = fixture.model.run_state_request(source, (), 1, engine=None,
                state=None, stream='ignored', resume=True, sampling=XTC)
            current = fixture.model.controller(result['snapshot'])
            self.assertEqual((current['schema'], current['next_draw']), (4, 7))
            self.assertEqual(starts, [5])
            self.assertEqual(fixture.model.controller(source), state)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
