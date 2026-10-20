"""CPU-only selection proposal and commit boundary tests.

These tests exercise controller plumbing only. They do not claim native XTC
support or GPU qualification.
"""
import copy
import json
import threading
import unittest
from unittest.mock import patch

from context_time_travel import controller as legacy_controller
from context_time_travel.sampled_controller import (
    PlainSamplingSession,
    start,
    validate_controller,
)
from context_time_travel.store import IntegrityError, InvalidOperation


POLICY = {
    "temperature": 0.7,
    "min_p": 0.05,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
}


def make_mask(vocab_size, allowed):
    mask = bytearray((vocab_size + 7) // 8)
    for token in allowed:
        mask[token >> 3] |= 1 << (token & 7)
    return bytes(mask)


def make_schema4(history=(0,), stream="proposal-stream"):
    state = start(POLICY, stream, history, 4, [3])
    state["schema"] = 4
    state["sampler_state"] = None
    return state


class ProposalTokenizer:
    pieces = (b"a", b"b", b"c", b"")

    def token_bytes(self, token):
        return self.pieces[token]

    def is_eog(self, token):
        return token == 3


def proposal_binding_or_legacy(state, source_digest):
    if source_digest is not None:
        return source_digest
    from context_time_travel.sampling_proposal import controller_digest
    return controller_digest(state)


class SamplingProposalTests(unittest.TestCase):
    def test_schema3_legacy_roundtrip_and_single_draw_behavior_are_unchanged(self):
        state = start(POLICY, "legacy", (0,), 4, [3])
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
        self.assertEqual(validate_controller(json.loads(encoded)), state)
        legacy = legacy_controller.start({"temperature": 0, "eos_tokens": [3]}, "legacy", (0,), 4)
        self.assertEqual(legacy["schema"], 1)
        session = PlainSamplingSession.restore(state, (0,), 4, ProposalTokenizer().token_bytes,
                                              ProposalTokenizer().is_eog)
        token = session.choose(make_mask(4, (1,)), lambda *args: 1)
        result = session._consume_selected(token, b"b", False)
        self.assertIsNone(result)
        self.assertEqual(session.current_controller["next_draw"], 1)

    def test_schema4_validator_is_closed_and_only_accepts_null_initial_state(self):
        from context_time_travel import controller
        state = make_schema4()
        self.assertEqual(validate_controller(state), state)
        self.assertEqual(controller._validated_state(state), state)
        with self.assertRaises(InvalidOperation):
            controller.consume(state, 1, b"b", False)
        for changed in (
            dict(state, unbound=True),
            dict(state, sampler_state={"mu": 2.0}),
            dict(state, sampler_state=False),
        ):
            with self.subTest(changed=changed), self.assertRaises((IntegrityError, InvalidOperation)):
                validate_controller(changed)
        wrong_schema = dict(state, schema=3)
        with self.assertRaises((IntegrityError, InvalidOperation)):
            validate_controller(wrong_schema)

    def test_proposal_draw_count_is_bounded_and_cursor_commits_exactly_once(self):
        from context_time_travel.sampling_proposal import (
            SelectionProposal,
            commit_proposal,
            controller_digest,
            draw_candidates,
            validate_proposal,
        )

        state = make_schema4()
        before = copy.deepcopy(state)
        candidates = draw_candidates(state)
        self.assertEqual(len(candidates), 2)
        proposal = SelectionProposal(1, 0, 2, None, controller_digest(state))
        checked = validate_proposal(state, make_mask(4, (1, 2)), proposal)
        self.assertEqual(checked, proposal)
        committed = commit_proposal(state, proposal, 1)
        self.assertEqual(committed["next_draw"], 2)
        self.assertEqual(committed["sampler_state"], None)
        self.assertEqual(state, before)
        with self.assertRaises(TypeError):
            json.dumps(proposal)

    def test_plain_session_holds_proposal_private_until_matching_consumption(self):
        from context_time_travel.sampling_proposal import controller_digest

        state = make_schema4()
        original = copy.deepcopy(state)
        tokenizer = ProposalTokenizer()
        session = PlainSamplingSession.restore(state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        observed = []

        def selector(mask, temperature, min_p, *, draw_candidates, source_digest=None, **filters):
            observed.append((mask, temperature, min_p, draw_candidates))
            from context_time_travel.sampling_proposal import SelectionProposal
            proposal = SelectionProposal(
                1, original["next_draw"], 2, None,
                proposal_binding_or_legacy(original, source_digest))
            observed.append(proposal)
            return proposal

        token = session.choose(make_mask(4, (1, 2)), selector)
        self.assertEqual(token, 1)
        self.assertEqual(len(observed[0][3]), 2)
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller
        session._consume_selected(token, b"b", False)
        committed = session.current_controller
        self.assertEqual(committed["next_draw"], 2)
        self.assertEqual(committed["accepted_in_scope"], 1)
        self.assertEqual(state, original)
        from context_time_travel.sampling_proposal import commit_proposal
        before_stale_commit = copy.deepcopy(session._state)
        with self.assertRaises(IntegrityError):
            commit_proposal(session._state, observed[-1], 1,
                            expected_source_digest=session._proposal_source_digest())
        self.assertEqual(session._state, before_stale_commit)

    def test_selector_receives_authoritative_draw_start_across_variable_draw_counts(self):
        from context_time_travel.sampling_proposal import SelectionProposal

        state = make_schema4()
        tokenizer = ProposalTokenizer()
        session = PlainSamplingSession.restore(state, (0,), 4, tokenizer.token_bytes,
                                              tokenizer.is_eog)
        starts = []
        counts = iter((2, 1))

        def selector(mask, temperature, min_p, *, draw_candidates, draw_start,
                     source_digest, **filters):
            starts.append((draw_start, len(draw_candidates)))
            count = next(counts)
            return SelectionProposal(1, draw_start, count, None, source_digest)

        for _ in range(2):
            token = session.choose(make_mask(4, (1, 2)), selector)
            session._consume_selected(token, b"b", False)

        self.assertEqual(starts, [(0, 2), (2, 2)])
        self.assertEqual(session.current_controller["next_draw"], 3)

    def test_proposal_cursor_exhaustion_passes_only_remaining_draw(self):
        from context_time_travel.sampling_proposal import DRAW_LIMIT, SelectionProposal

        state = make_schema4()
        state["next_draw"] = DRAW_LIMIT - 2
        tokenizer = ProposalTokenizer()
        session = PlainSamplingSession.restore(state, (0,), 4, tokenizer.token_bytes,
                                              tokenizer.is_eog)
        calls = []

        def selector(mask, temperature, min_p, *, draw_candidates, draw_start,
                     source_digest, **filters):
            calls.append((draw_start, len(draw_candidates)))
            return SelectionProposal(1, draw_start, 1, None, source_digest)

        token = session.choose(make_mask(4, (1,)), selector)
        session._consume_selected(token, b"b", False)
        self.assertEqual(calls, [(DRAW_LIMIT - 2, 1)])
        self.assertEqual(session.current_controller["next_draw"], DRAW_LIMIT - 1)
        with self.assertRaises(IntegrityError):
            session.choose(make_mask(4, (1,)), selector)
        self.assertEqual(calls, [(DRAW_LIMIT - 2, 1)])

    def test_bad_proposals_fail_without_mutating_source_or_publishing(self):
        from context_time_travel.sampling_proposal import SelectionProposal, controller_digest

        for fields in (
            dict(token=9, draw_start=0, draw_count=1, feedback_proposal=None),
            dict(token=1, draw_start=1, draw_count=1, feedback_proposal=None),
            dict(token=1, draw_start=0, draw_count=True, feedback_proposal=None),
            dict(token=1, draw_start=0, draw_count=3, feedback_proposal=None),
            dict(token=1, draw_start=0, draw_count=1, feedback_proposal={"mu": 2}),
        ):
            state = make_schema4()
            original = copy.deepcopy(state)
            session = PlainSamplingSession.restore(state, (0,), 4, ProposalTokenizer().token_bytes,
                                                  ProposalTokenizer().is_eog)
            fields.setdefault("source_digest", controller_digest(state))
            with self.subTest(fields=fields), self.assertRaises((IntegrityError, InvalidOperation)):
                session.choose(make_mask(4, (1, 2)), lambda *args, **kwargs: SelectionProposal(**fields))
            self.assertEqual(state, original)
            self.assertEqual(session.current_controller, original)

    def test_wrong_token_and_foreign_thread_cannot_commit_pending_proposal(self):
        state = make_schema4()
        tokenizer = ProposalTokenizer()
        session = PlainSamplingSession.restore(state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        from context_time_travel.sampling_proposal import SelectionProposal
        token = session.choose(
            make_mask(4, (1,)),
            lambda *args, source_digest=None, **kwargs:
                SelectionProposal(1, state["next_draw"], 1, None,
                                  proposal_binding_or_legacy(state, source_digest)))
        failures = []

        def foreign_consume():
            try:
                session._consume_selected(token, b"b", False)
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=foreign_consume)
        thread.start()
        thread.join()
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], InvalidOperation)
        with self.assertRaises(IntegrityError):
            session._consume_selected(2, b"c", False)
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller

    def test_grammar_session_uses_the_same_pending_proposal_boundary(self):
        import test_state_request as fixtures
        fixture = fixtures.StateRequestTests()
        fixture.setUp()
        try:
            state = fixture.engine.start("edit", "grammar-proposal", (0,), 2,
                                         sampling=POLICY)
            state["schema"] = 4
            state["sampler_state"] = None
            session = fixture.engine.restore(state, (0,))
            from context_time_travel import sampled_controller
            deepcopy_calls = []
            original_deepcopy = sampled_controller.copy.deepcopy

            def counted_deepcopy(value, *args, **kwargs):
                deepcopy_calls.append(1)
                return original_deepcopy(value, *args, **kwargs)

            def selector(mask, temperature, min_p, *, draw_candidates, source_digest=None, **filters):
                self.assertTrue(mask[0] & 1)
                from context_time_travel.sampling_proposal import SelectionProposal
                return SelectionProposal(
                    0, 0, 2, None, proposal_binding_or_legacy(state, source_digest))

            with patch.object(sampled_controller.copy, "deepcopy", counted_deepcopy):
                token = session._choose_proposal(selector)
                self.assertEqual(token, 0)
                with self.assertRaises(InvalidOperation):
                    _ = session.current_controller
                session._consume_selected(token, b"A", False)
            self.assertEqual(deepcopy_calls, [])
            committed = session.current_controller
            self.assertEqual((committed["schema"], committed["next_draw"],
                              committed["accepted_in_scope"]), (4, 2, 1))
        finally:
            fixture.tearDown()

    def test_native_ack_precedes_schema4_controller_commit(self):
        import test_state_request as fixtures
        from context_time_travel import sampled_controller
        from context_time_travel.sampling_proposal import XTC_PROPOSAL_PROTOCOL, SelectionProposal

        fixture = fixtures.StateRequestTests()
        fixture.setUp()
        try:
            state = sampled_controller.start(POLICY, "native-ack", fixture.model.history(fixture.sid),
                                             2, fixture.model.profile["execution"]["eos_tokens"])
            state["schema"] = 4
            state["sampler_state"] = None
            fixture.model.cache[fixture.sid] = fixture.model._load_checkpoint(
                fixture.sid, fixture.model.history(fixture.sid))
            source = fixture.model._controller_transition(fixture.sid, state)
            fixture.model.profile["execution"]["sampling_protocol"] = XTC_PROPOSAL_PROTOCOL
            events = []

            def select(lease, mask, temperature, min_p, *, draw_start, draw_candidates,
                       source_digest=None, **filters):
                self.assertEqual(len(draw_candidates), 2)
                self.assertEqual(draw_start, state["next_draw"])
                return SelectionProposal(
                    0, draw_start, 2, None,
                    proposal_binding_or_legacy(state, source_digest))

            fixture.backend.sample_selection_proposal = select
            original_ack = fixture.backend.consume_generated

            def ack(lease, tokens):
                events.append("native-ack")
                return original_ack(lease, tokens)

            fixture.backend.consume_generated = ack
            original_commit = PlainSamplingSession._consume_selected

            def commit(session, token, piece, is_eog):
                events.append("controller-commit")
                return original_commit(session, token, piece, is_eog)

            with patch.object(PlainSamplingSession, "_consume_selected", commit):
                result = fixture.model.run_state_request(
                    source, (), 1, engine=None, state=None, stream="ignored",
                    resume=True, sampling=None,
                )
            self.assertEqual(events, ["native-ack", "controller-commit"])
            self.assertEqual(fixture.model.controller(result["snapshot"])["next_draw"], 2)
        finally:
            fixture.tearDown()

    def test_native_consume_failure_leaves_source_and_no_child_publication(self):
        import test_state_request as fixtures
        from context_time_travel import sampled_controller
        from context_time_travel.sampling_proposal import XTC_PROPOSAL_PROTOCOL, SelectionProposal

        fixture = fixtures.StateRequestTests()
        fixture.setUp()
        try:
            state = sampled_controller.start(POLICY, "ack-fails", fixture.model.history(fixture.sid),
                                             2, fixture.model.profile["execution"]["eos_tokens"])
            state["schema"] = 4
            state["sampler_state"] = None
            fixture.model.cache[fixture.sid] = fixture.model._load_checkpoint(
                fixture.sid, fixture.model.history(fixture.sid))
            source = fixture.model._controller_transition(fixture.sid, state)
            before_bytes = fixture.model.snapshot_bytes(source)
            before_objects = set(fixture.model.store.objects.iterdir())
            fixture.model.profile["execution"]["sampling_protocol"] = XTC_PROPOSAL_PROTOCOL

            def select(lease, mask, temperature, min_p, *, draw_start, draw_candidates,
                       source_digest=None, **filters):
                return SelectionProposal(
                    0, draw_start, 2, None,
                    proposal_binding_or_legacy(state, source_digest))

            fixture.backend.sample_selection_proposal = select
            fixture.backend.consume_generated = lambda *args: (_ for _ in ()).throw(
                RuntimeError("native selected-token acknowledgment failed"))
            with self.assertRaisesRegex(RuntimeError, "acknowledgment failed"):
                fixture.model.run_state_request(
                    source, (), 1, engine=None, state=None, stream="ignored",
                    resume=True, sampling=None,
                )
            self.assertEqual(fixture.model.snapshot_bytes(source), before_bytes)
            self.assertEqual(fixture.model.controller(source), state)
            self.assertEqual(set(fixture.model.store.objects.iterdir()), before_objects)
            self.assertEqual(fixture.backend.captures, 0)
        finally:
            fixture.tearDown()

    def test_schema4_token_loop_does_not_redigest_or_deepcopy_full_controller(self):
        from context_time_travel import sampled_controller
        from context_time_travel import sampling_proposal
        from context_time_travel.sampling_proposal import SelectionProposal

        vocab_size = 4096
        state = start(POLICY, "large-proposal-controller", (0,), vocab_size, [3])
        state["schema"] = 4
        state["sampler_state"] = None
        tokenizer = ProposalTokenizer()
        digest_calls = []
        deepcopy_calls = []
        original_digest = sampling_proposal.controller_digest
        original_deepcopy = sampled_controller.copy.deepcopy

        def counted_digest(value):
            digest_calls.append(1)
            return original_digest(value)

        def counted_deepcopy(value, *args, **kwargs):
            deepcopy_calls.append(1)
            return original_deepcopy(value, *args, **kwargs)

        with patch.object(sampling_proposal, "controller_digest", counted_digest):
            session = PlainSamplingSession.restore(
                state, (0,), vocab_size, tokenizer.token_bytes, tokenizer.is_eog)
            initial_digest_calls = len(digest_calls)
            self.assertEqual(initial_digest_calls, 1)
            with patch.object(sampled_controller.copy, "deepcopy", counted_deepcopy):
                for _ in range(8):
                    def selector(mask, temperature, min_p, *, draw_candidates,
                                 source_digest=None, **filters):
                        binding = source_digest
                        if binding is None:
                            # Baseline behavior binds each proposal by serializing
                            # the entire controller; the fixed path supplies an
                            # O(1) request-local binding instead.
                            binding = counted_digest(session._state)
                        return SelectionProposal(
                            token=1,
                            draw_start=session._state["next_draw"],
                            draw_count=2,
                            feedback_proposal=None,
                            source_digest=binding,
                        )

                    token = session.choose(make_mask(vocab_size, (1,)), selector)
                    session._consume_selected(token, b"b", False)
            self.assertEqual(len(digest_calls), initial_digest_calls)
            self.assertEqual(deepcopy_calls, [])
            self.assertEqual(session._state["accepted_in_scope"], 8)

    def test_schema4_greedy_proposal_consumes_zero_draws(self):
        from context_time_travel.sampling_proposal import SelectionProposal

        state = make_schema4()
        state["policy"]["temperature"] = 0.0
        from context_time_travel.sampled_controller import sampler_order
        state["sampler_order"] = sampler_order(state["policy"])
        tokenizer = ProposalTokenizer()
        session = PlainSamplingSession.restore(
            state, (0,), 4, tokenizer.token_bytes, tokenizer.is_eog)
        seen_candidates = []

        def selector(mask, temperature, min_p, *, draw_candidates, draw_start,
                     source_digest=None, **filters):
            seen_candidates.append((draw_start, draw_candidates))
            return SelectionProposal(
                1, draw_start, 0, None, proposal_binding_or_legacy(session._state, source_digest))

        token = session.choose(make_mask(4, (1, 2)), selector)
        self.assertEqual(token, 1)
        self.assertEqual(seen_candidates, [(0, ())])
        session._consume_selected(token, b"b", False)
        self.assertEqual(session.current_controller["next_draw"], 0)

    def test_schema4_positive_temperature_rejects_zero_draw_proposal(self):
        from context_time_travel.sampling_proposal import SelectionProposal

        state = make_schema4()
        session = PlainSamplingSession.restore(
            state, (0,), 4, ProposalTokenizer().token_bytes, ProposalTokenizer().is_eog)
        with self.assertRaises(IntegrityError):
            session.choose(
                make_mask(4, (1,)),
                lambda mask, temperature, min_p, *, source_digest=None, **filters:
                    SelectionProposal(
                        1, 0, 0, None,
                        proposal_binding_or_legacy(session._state, source_digest)),
            )


if __name__ == "__main__":
    unittest.main()
