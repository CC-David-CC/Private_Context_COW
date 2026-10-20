"""CPU contract tests for consume-bound Mirostat v2 state."""
import copy
import json
import unittest

from context_time_travel import mirostat2, sampled_controller
from context_time_travel.sampling_proposal import (
    SelectionProposal,
    controller_digest,
)
from context_time_travel.store import IntegrityError, InvalidOperation
from test_sampling_proposal import POLICY, ProposalTokenizer, make_mask

V6 = "full-vocab-mirostat2-u53-v6"


def policy(**changes):
    result = dict(POLICY, min_p=0.0, mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1)
    result.update(changes)
    return result


class MirostatControllerTests(unittest.TestCase):
    def test_mode_two_starts_schema_four_with_versioned_state(self):
        state = sampled_controller.start(policy(), "miro-start", (0,), 4, [3])
        self.assertEqual(state["schema"], 4)
        self.assertEqual(state["sampler_state"], mirostat2.start_state(5.0))
        self.assertIn("mirostat2", state["sampler_order"])
        self.assertEqual(sampled_controller.validate_controller(state), state)

    def test_feedback_state_is_strict_json_safe_and_finite_negative_mu_is_valid(self):
        state = sampled_controller.start(policy(), "miro-state", (0,), 4, [3])
        state["sampler_state"] = {"protocol": "mirostat2-state-v1", "version": 1, "mu": -2.25}
        encoded = json.loads(json.dumps(state, allow_nan=False))
        self.assertEqual(sampled_controller.validate_controller(encoded), encoded)
        for bad in (
            {"protocol": "wrong", "version": 1, "mu": 2.0},
            {"protocol": "mirostat2-state-v1", "version": True, "mu": 2.0},
            {"protocol": "mirostat2-state-v1", "version": 1, "mu": float("nan")},
            {"protocol": "mirostat2-state-v1", "version": 1, "mu": 2.0, "extra": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises((IntegrityError, InvalidOperation)):
                sampled_controller.validate_controller(dict(encoded, sampler_state=bad))

    def test_selection_feedback_stays_private_until_matching_token_consumption(self):
        state = sampled_controller.start(policy(), "miro-pending", (0,), 4, [3])
        original = copy.deepcopy(state)
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        seen = []

        def selector(mask, temperature, min_p, *, sampler_state, mirostat_tau,
                     mirostat_eta, draw_start, draw_candidates, source_digest, **filters):
            seen.append((copy.deepcopy(sampler_state), mirostat_tau, mirostat_eta,
                         draw_start, draw_candidates))
            next_state = {"protocol": "mirostat2-state-v1", "version": 1, "mu": 4.75}
            return SelectionProposal(1, draw_start, 1, next_state, source_digest)

        token = session.choose(make_mask(4, (1, 2)), selector)
        self.assertEqual(token, 1)
        self.assertEqual(session._state, original)
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller
        session._consume_selected(token, b"b", False)
        current = session.current_controller
        self.assertEqual(current["sampler_state"]["mu"], 4.75)
        self.assertEqual(current["next_draw"], 1)
        self.assertEqual(seen[0][0], mirostat2.start_state(5.0))
        self.assertEqual((seen[0][1], seen[0][2], seen[0][3]), (5.0, 0.1, 0))
        self.assertEqual(state, original)

    def test_bad_feedback_and_wrong_consumed_token_do_not_commit_state(self):
        for feedback in (
            {"protocol": "wrong", "version": 1, "mu": 3.0},
            {"protocol": "mirostat2-state-v1", "version": 1, "mu": float("inf")},
        ):
            state = sampled_controller.start(policy(), "miro-bad", (0,), 4, [3])
            before = copy.deepcopy(state)
            tokenizer = ProposalTokenizer()
            session = sampled_controller.PlainSamplingSession.restore(
                state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)

            def selector(mask, temperature, min_p, *, draw_start, source_digest, **kwargs):
                return SelectionProposal(1, draw_start, 1, feedback, source_digest)

            with self.subTest(feedback=feedback), self.assertRaises((IntegrityError, InvalidOperation)):
                session.choose(make_mask(4, (1,)), selector)
            self.assertEqual(state, before)
            self.assertEqual(session.current_controller, before)

        state = sampled_controller.start(policy(), "miro-wrong-token", (0,), 4, [3])
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        digest = controller_digest(state)
        def selector(mask, temperature, min_p, *, draw_start, source_digest, **kwargs):
            return SelectionProposal(1, draw_start, 1,
                {"protocol": "mirostat2-state-v1", "version": 1, "mu": 4.0}, source_digest)
        token = session.choose(make_mask(4, (1,)), selector)
        with self.assertRaises(IntegrityError):
            session._consume_selected(2, b"c", False)
        self.assertEqual(session._state["sampler_state"], mirostat2.start_state(5.0))
        self.assertEqual(session._state["next_draw"], 0)
        self.assertEqual(digest, controller_digest(state))

    def test_mode_transition_preserves_or_resets_feedback_only_at_generation_boundary(self):
        state = sampled_controller.start(policy(), "miro-config", (0,), 4, [3])
        state["next_draw"] = 9
        state["sampler_state"]["mu"] = -1.5
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        session.configure_generation(policy(mirostat_tau=7.0, mirostat_eta=0.2))
        changed = session.current_controller
        self.assertEqual(changed["sampler_state"]["mu"], -1.5)
        self.assertEqual(changed["next_draw"], 9)
        self.assertEqual(changed["policy"]["mirostat_tau"], 7.0)

        disabled = sampled_controller.PlainSamplingSession.restore(
            changed, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        disabled.configure_generation(policy(mirostat=0))
        disabled_state = disabled.current_controller
        self.assertEqual(disabled_state["schema"], 4)
        self.assertIsNone(disabled_state["sampler_state"])
        enabled = sampled_controller.PlainSamplingSession.restore(
            disabled_state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        enabled.configure_generation(policy(mirostat_tau=3.0))
        self.assertEqual(enabled.current_controller["schema"], 4)
        self.assertEqual(enabled.current_controller["sampler_state"], mirostat2.start_state(3.0))
        self.assertEqual(enabled.current_controller["next_draw"], 9)

    def test_disabling_xtc_keeps_schema_four_and_draw_cursor(self):
        xtc = dict(POLICY, xtc_probability=0.4, xtc_threshold=0.1)
        state = sampled_controller.start(xtc, "miro-xtc-disable", (0,), 4, [3])
        self.assertEqual(state["schema"], 4)
        state["next_draw"] = 6
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        session.configure_generation(POLICY)
        current = session.current_controller
        self.assertEqual(current["schema"], 4)
        self.assertIsNone(current["sampler_state"])
        self.assertEqual(current["next_draw"], 6)

    def test_grammar_schema_four_survives_disabling_xtc(self):
        from test_grammar_first import GrammarFirstTests
        helper = GrammarFirstTests()
        engine = helper.make_engine()
        xtc = dict(POLICY, xtc_probability=0.4, xtc_threshold=0.1)
        state = engine.start("edit", "miro-grammar-xtc", [], engine.vocab_size,
                             sampling=xtc)
        session = engine.restore(state, [])
        session.configure_generation(POLICY)
        current = session.current_controller
        self.assertEqual(current["schema"], 4)
        self.assertIsNone(current["sampler_state"])
        self.assertEqual(current["next_draw"], state["next_draw"])


class MirostatRealBoundaryTests(unittest.TestCase):
    def setUp(self):
        from test_min_p_integration import MinPRequestTests
        self.fixture = MinPRequestTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model
        self.backend = self.fixture.backend
        self.sid = self.fixture.sid
        self.backend.profile['execution']['sampling_protocol'] = V6
        self.calls = []

        def propose(lease, mask, temperature, min_p, **filters):
            self.calls.append((temperature, min_p, dict(filters)))
            token = next(t for t in range(self.backend.vocab_size)
                         if mask[t >> 3] & (1 << (t & 7)))
            draw_count = 0 if temperature == 0 else 1
            feedback = filters.get('sampler_state') if filters.get('mirostat') == 2 else None
            return SelectionProposal(token, filters['draw_start'], draw_count,
                                     feedback, filters['source_digest'])

        self.backend.sample_selection_proposal = propose

    def test_v6_accepts_schema4_after_mirostat_is_disabled(self):
        first = self.model.run_sampled_request(self.sid, (), 1, sampling=policy(), stream='mode2')
        self.assertEqual(self.model.controller(first['snapshot'])['schema'], 4)
        second = self.model.run_sampled_request(first['snapshot'], (), 1,
            sampling=POLICY, stream='ignored', resume=True)
        current = self.model.controller(second['snapshot'])
        self.assertEqual(current['schema'], 4)
        self.assertIsNone(current['sampler_state'])
        self.assertEqual(second['tokens'], [0])

    def test_v6_profile_accepts_stateless_xtc_schema4_request(self):
        xtc = dict(POLICY, xtc_probability=0.4, xtc_threshold=0.1)
        result = self.model.run_sampled_request(self.sid, (), 1, sampling=xtc, stream='xtc')
        current = self.model.controller(result['snapshot'])
        self.assertEqual(current['schema'], 4)
        self.assertIsNone(current['sampler_state'])
        self.assertEqual(result['tokens'], [0])

    def test_active_mirostat_rejects_v5_before_materialization(self):
        first = self.model.run_sampled_request(self.sid, (), 1, sampling=policy(), stream='mode2')
        self.backend.profile['execution']['sampling_protocol'] = 'full-vocab-xtc-min-p-u53-v5'
        restores = self.backend.restores
        with self.assertRaises(InvalidOperation):
            self.model.run_sampled_request(first['snapshot'], (), 1,
                sampling=None, stream='ignored', resume=True)
        self.assertEqual(self.backend.restores, restores)

    def test_missing_schema4_selector_rejects_before_materialization_even_when_policy_supported(self):
        self.backend.profile['execution']['sampling_protocol'] = "full-vocab-xtc-min-p-u53-v5"
        self.backend.sample_selection_proposal = None
        restores = self.backend.restores
        xtc = dict(POLICY, xtc_probability=0.4, xtc_threshold=0.1)
        with self.assertRaises(InvalidOperation):
            self.model.run_sampled_request(self.sid, (), 1, sampling=xtc, stream='missing-selector')
        self.assertEqual(self.backend.restores, restores)


if __name__ == "__main__":
    unittest.main()
