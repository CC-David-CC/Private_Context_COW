"""CPU tests for Adaptive-P feedback state at the consumed-token boundary."""
import copy
import json
import unittest

from context_time_travel import adaptive_p, mirostat2, sampled_controller
from context_time_travel.sampling_proposal import SelectionProposal, controller_digest
from context_time_travel.store import IntegrityError, InvalidOperation
from test_sampling_proposal import POLICY, ProposalTokenizer, make_mask

V6 = "full-vocab-mirostat2-u53-v6"
V7 = "full-vocab-adaptive-p-u53-v7"


def ap_policy(**changes):
    result = dict(POLICY, min_p=0.0, adaptive_target=0.55, adaptive_decay=0.90)
    result.update(changes)
    return result


class AdaptivePControllerTests(unittest.TestCase):
    def test_adaptive_start_uses_schema4_and_reference_initial_state(self):
        state = sampled_controller.start(ap_policy(), "ap-start", (0,), 4, [3])
        self.assertEqual(state["schema"], 4)
        self.assertEqual(state["sampler_state"], adaptive_p.initial_state(0.55, 0.90))
        self.assertIn("adaptive-p", state["sampler_order"])
        self.assertEqual(sampled_controller.validate_controller(state), state)

    def test_feedback_is_private_until_matching_token_is_consumed(self):
        state = sampled_controller.start(ap_policy(), "ap-consume", (0,), 4, [3])
        before = copy.deepcopy(state)
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        seen = []

        def selector(mask, temperature, min_p, *, sampler_state, adaptive_target,
                     adaptive_decay, draw_start, draw_candidates, source_digest, **filters):
            seen.append((copy.deepcopy(sampler_state), adaptive_target, adaptive_decay,
                         draw_start, draw_candidates))
            proposal = adaptive_p.choose(
                [0.0, 1.0, -1.0, 0.0], mask, sampler_state,
                adaptive_target=adaptive_target, adaptive_decay=adaptive_decay,
                draw53=draw_candidates[0])
            next_state = adaptive_p.accept(sampler_state, proposal["proposal"], proposal["token"])
            return SelectionProposal(proposal["token"], draw_start, 1, next_state, source_digest)

        token = session.choose(make_mask(4, (1, 2)), selector)
        self.assertEqual(state, before)
        self.assertEqual(session._state, before)
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller
        session._consume_selected(token, tokenizer.token_bytes(token), False)
        current = session.current_controller
        self.assertEqual(current["next_draw"], 1)
        self.assertEqual(seen[0][0], adaptive_p.initial_state(0.55, 0.90))
        self.assertEqual(seen[0][1:4], (0.55, 0.90, 0))
        self.assertNotEqual(current["sampler_state"], before["sampler_state"])
        self.assertEqual(state, before)

    def test_feedback_and_cursor_do_not_commit_on_wrong_consumed_token(self):
        state = sampled_controller.start(ap_policy(), "ap-wrong-token", (0,), 4, [3])
        before = copy.deepcopy(state)
        tokenizer = ProposalTokenizer()

        def selector(mask, temperature, min_p, *, draw_start, source_digest, **filters):
            next_state = dict(state["sampler_state"], weighted_sum=8.0)
            return SelectionProposal(1, draw_start, 1, next_state, source_digest)

        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        self.assertEqual(session.choose(make_mask(4, (1,)), selector), 1)
        with self.assertRaises(IntegrityError):
            session._consume_selected(2, b"c", False)
        self.assertEqual(session._state, before)
        self.assertEqual(state, before)

    def test_policy_changes_preserve_feedback_disable_and_reenable_cleanly(self):
        state = sampled_controller.start(ap_policy(), "ap-transition", (0,), 4, [3])
        state["next_draw"] = 5
        state["sampler_state"]["weighted_sum"] += 0.25
        original_feedback = copy.deepcopy(state["sampler_state"])
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        session.configure_generation(ap_policy(adaptive_target=0.7, adaptive_decay=0.8))
        active = session.current_controller
        self.assertEqual(active["sampler_state"], original_feedback)
        self.assertEqual(active["next_draw"], 5)

        disabled = sampled_controller.PlainSamplingSession.restore(
            active, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        disabled.configure_generation(ap_policy(adaptive_target=-1.0))
        inactive = disabled.current_controller
        self.assertEqual(inactive["schema"], 4)
        self.assertIsNone(inactive["sampler_state"])
        self.assertEqual(inactive["next_draw"], 5)

        reenabled = sampled_controller.PlainSamplingSession.restore(
            inactive, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        reenabled.configure_generation(ap_policy(adaptive_target=0.4))
        enabled = reenabled.current_controller
        self.assertEqual(enabled["schema"], 4)
        self.assertEqual(enabled["sampler_state"], adaptive_p.initial_state(0.4, 0.90))
        self.assertEqual(enabled["next_draw"], 5)

    def test_cross_mode_transition_reinitializes_the_new_feedback_state(self):
        state = sampled_controller.start(ap_policy(), "ap-cross", (0,), 4, [3])
        state["next_draw"] = 3
        state["sampler_state"]["weighted_sum"] = 2.5
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        miro = dict(POLICY, min_p=0.0, mirostat=2, mirostat_tau=7.0, mirostat_eta=0.2)
        session.configure_generation(miro)
        current = session.current_controller
        self.assertEqual(current["sampler_state"], mirostat2.start_state(7.0))
        self.assertEqual(current["next_draw"], 3)

    def test_active_adaptive_cannot_be_downgraded_to_schema3(self):
        state = sampled_controller.start(ap_policy(), "ap-schema3", (0,), 4, [3])
        invalid = copy.deepcopy(state)
        invalid['schema'] = 3
        invalid.pop('sampler_state')
        with self.assertRaises(IntegrityError):
            sampled_controller.validate_controller(invalid)

    def test_legacy_mirostat_callback_does_not_receive_inactive_adaptive_knobs(self):
        policy = dict(POLICY, min_p=0.0, mirostat=2, mirostat_tau=5.0,
                      mirostat_eta=0.1, adaptive_target=-1.0, adaptive_decay=0.9)
        state = sampled_controller.start(policy, "ap-filter-routing", (0,), 4, [3])
        observed = {}
        digest = controller_digest(state)

        def selector(mask, temperature, min_p, *, draw_start, draw_candidates,
                     source_digest, sampler_state, mirostat, **filters):
            observed.update(filters)
            return SelectionProposal(1, draw_start, 1,
                dict(sampler_state, mu=sampler_state['mu'] - 0.1), source_digest)

        sampled_controller.choose_proposal_owned(
            state, make_mask(4, (1,)), selector, source_digest=digest)
        self.assertNotIn('adaptive_target', observed)
        self.assertNotIn('adaptive_decay', observed)


class AdaptivePGrammarBoundaryTests(unittest.TestCase):
    def test_grammar_start_and_generation_boundary_keep_schema4_feedback(self):
        from test_grammar_first import GrammarFirstTests
        fixture = GrammarFirstTests()
        engine = fixture.make_engine()
        policy = ap_policy(repetition_penalty=1.0)
        state = engine.start('edit', 'ap-grammar', [], engine.vocab_size, sampling=policy)
        self.assertEqual(state['schema'], 4)
        self.assertEqual(state['sampler_state'], adaptive_p.initial_state(0.55, 0.90))
        session = engine.restore(state, [])
        session.configure_generation(ap_policy(repetition_penalty=1.0,
                                                adaptive_target=0.6, adaptive_decay=0.8))
        changed = session.current_controller
        self.assertEqual(changed['schema'], 4)
        self.assertEqual(changed['sampler_state'], state['sampler_state'])


class AdaptivePRealBoundaryTests(unittest.TestCase):
    def setUp(self):
        from test_min_p_integration import MinPRequestTests
        self.fixture = MinPRequestTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model
        self.backend = self.fixture.backend
        self.sid = self.fixture.sid
        self.backend.profile["execution"]["sampling_protocol"] = V7

        def propose(lease, mask, temperature, min_p, **filters):
            token = next(t for t in range(self.backend.vocab_size)
                         if mask[t >> 3] & (1 << (t & 7)))
            state = filters.get("sampler_state")
            if filters.get("adaptive_target", -1.0) >= 0.0:
                state = dict(state, weighted_sum=state["weighted_sum"] + 0.1)
            elif filters.get("mirostat", 0) == 2:
                state = dict(state, mu=state["mu"] - 0.1)
            else:
                state = None
            return SelectionProposal(token, filters["draw_start"],
                0 if temperature == 0 else 1, state, filters["source_digest"])
        self.backend.sample_selection_proposal = propose

    def test_v7_accepts_active_adaptive_p_and_publishes_feedback(self):
        result = self.model.run_sampled_request(
            self.sid, (), 1, sampling=ap_policy(), stream="v7-ap")
        controller = self.model.controller(result["snapshot"])
        self.assertEqual(controller["schema"], 4)
        self.assertEqual(controller["next_draw"], 1)
        self.assertGreater(controller["sampler_state"]["weighted_sum"],
                           adaptive_p.initial_state(0.55, 0.90)["weighted_sum"])

    def test_v6_profile_rejects_adaptive_before_materialization(self):
        self.backend.profile["execution"]["sampling_protocol"] = V6
        restores = self.backend.restores
        with self.assertRaises(InvalidOperation):
            self.model.run_sampled_request(
                self.sid, (), 1, sampling=ap_policy(), stream="v6-no-ap")
        self.assertEqual(self.backend.restores, restores)

    def test_v7_profile_keeps_legacy_mirostat_v6_and_xtc_v5_paths(self):
        miro = dict(POLICY, min_p=0.0, mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1)
        result = self.model.run_sampled_request(
            self.sid, (), 1, sampling=miro, stream="v7-mirostat-compat")
        self.assertEqual(self.model.controller(result["snapshot"])["sampler_state"]["mu"],
                         mirostat2.start_state(5.0)["mu"] - 0.1)
        xtc = dict(POLICY, xtc_probability=0.4, xtc_threshold=0.1)
        result = self.model.run_sampled_request(
            self.sid, (), 1, sampling=xtc, stream="v7-xtc-compat")
        self.assertIsNone(self.model.controller(result["snapshot"])["sampler_state"])

    def test_v7_profile_rejects_missing_adaptive_selector_before_materialization(self):
        self.backend.sample_selection_proposal = None
        restores = self.backend.restores
        with self.assertRaises(InvalidOperation):
            self.model.run_sampled_request(
                self.sid, (), 1, sampling=ap_policy(), stream="v7-no-selector")
        self.assertEqual(self.backend.restores, restores)


if __name__ == "__main__":
    unittest.main()
