"""CPU receipt and command checks for native Adaptive-P V7 selection."""
import math
import shlex
import struct
import unittest
from pathlib import Path

from context_time_travel import adaptive_p, sampled_controller, strata_backend as sb
from context_time_travel.sampling_proposal import SelectionProposal
from context_time_travel.store import IntegrityError, InvalidOperation
import test_min_p_integration as minp_fixtures
from test_sampling_proposal import ProposalTokenizer, make_mask

V5 = "full-vocab-xtc-min-p-u53-v5"
V6 = "full-vocab-mirostat2-u53-v6"
V7 = "full-vocab-adaptive-p-u53-v7"
SOURCE = "b" * 64
POLICY = dict(temperature=0.7, min_p=0.0, top_k=0, top_p=1.0,
              adaptive_target=0.55, adaptive_decay=0.90)
STATE = adaptive_p.initial_state(0.55, 0.90)


class AdaptivePBackendTests(unittest.TestCase):
    def setUp(self):
        fixture = minp_fixtures.MinPBackendTests(
            "test_native_sample_is_read_only_and_cleans_transient_mask")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.backend = fixture.backend
        self.lease = fixture.lease
        self.backend.profile["execution"]["sampling_protocol"] = V7
        self.backend.profile["execution"]["adaptive_p_protocol"] = V7
        self.backend._ready = {"sampling_protocol": V7, "adaptive_p_protocol": V7,
                               "mirostat2_protocol": V6,
                               "sampling_v5_protocol": V5,
                               "sampling_v4_protocol": "full-vocab-typical-dyna-min-p-u53-v4",
                               "sampling_v3_protocol": "full-vocab-dry-min-p-u53-v3",
                               "sampling_v2_protocol": "full-vocab-min-p-u53-v2",
                               "sampling_v1_protocol": "full-vocab-min-p-u53-v1",
                               "sampling_v6_protocol": V6,
                               "adaptive_p_feedback_protocol": "adaptive-p-state-v1",
                               "adaptive_p_feedback_version": 1,
                               "mirostat2_feedback_protocol": "mirostat2-state-v1",
                               "mirostat2_feedback_version": 1}
        self.commands = []
        self.payloads = []
        self.history_bytes = []
        self.breaker_bytes = []
        self.response = self._response()

        def request(command):
            fields = shlex.split(command)
            self.commands.append(fields)
            if fields[0] != "sample_mask_adaptive_p":
                raise AssertionError(f"Unexpected native command: {fields[0]}")
            self.payloads.append(Path(fields[7]).read_bytes())
            self.history_bytes.append(Path(fields[12]).read_bytes())
            self.breaker_bytes.append(Path(fields[13]).read_bytes())
            return dict(self.response)

        self.backend._request = request

    @staticmethod
    def _response(**changes):
        original_probability = 0.4
        adapted_target = min(1.0, max(0.0, 2.0 * 0.55 - STATE["weighted_sum"] / STATE["total_weight"]))
        response = dict(
            status="ok", protocol=V7, operator="full-vocab-adaptive-p",
            feedback_protocol="adaptive-p-state-v1", feedback_version=1,
            token=3, consumed=4, logits_count=9, mask_bytes=2,
            temperature=0.7, min_p=0.0, top_k=0, top_p=1.0,
            legal_count=2, min_p_kept_count=2, top_k_kept_count=2,
            typical_kept_count=2, kept_count=2, categorical_total_weight=1.25,
            effective_temperature=0.7, original_probability=0.4,
            adapted_target=adapted_target, adaptive_target=0.55, adaptive_decay=0.9,
            state_weighted_sum=STATE["weighted_sum"], state_total_weight=STATE["total_weight"],
            next_weighted_sum=original_probability + 0.9 * STATE["weighted_sum"],
            next_total_weight=1.0 + 0.9 * STATE["total_weight"],
            draw53=123, draw_count=1,
            dry_multiplier=0.0, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256,
            dry_history_count=0, dry_breaker_count=0,
            dry_penalized_candidates=0,
            typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0,
        )
        response.update(changes)
        return response

    def sample(self, mask=b"\x0a\x00", **changes):
        temperature = changes.pop("temperature", POLICY["temperature"])
        min_p = changes.pop("min_p", POLICY["min_p"])
        top_k = changes.pop("top_k", POLICY["top_k"])
        top_p = changes.pop("top_p", POLICY["top_p"])
        args = dict(
            sampler_state=STATE, adaptive_target=0.55, adaptive_decay=0.9,
            draw_start=7, draw_candidates=(123,), source_digest=SOURCE,
            dry_multiplier=0.0, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256,
            dry_history=(), dry_breakers=(),
            typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0,
            xtc_probability=0.0, xtc_threshold=0.1,
        )
        args.update(changes)
        return self.backend.sample_selection_proposal(
            self.lease, mask, temperature, min_p,
            top_k=top_k, top_p=top_p, **args)

    def test_v7_ready_advertises_adaptive_and_legacy_protocols(self):
        profile = {"execution": {"sampling_protocol": V7}}
        ready = {"sampling_protocol": V7, "adaptive_p_protocol": V7,
                 "mirostat2_protocol": V6, "sampling_v5_protocol": V5,
                 "sampling_v4_protocol": "full-vocab-typical-dyna-min-p-u53-v4",
                 "sampling_v3_protocol": "full-vocab-dry-min-p-u53-v3",
                 "sampling_v2_protocol": "full-vocab-min-p-u53-v2",
                 "sampling_v1_protocol": "full-vocab-min-p-u53-v1",
                 "sampling_v6_protocol": V6,
                 "adaptive_p_feedback_protocol": "adaptive-p-state-v1",
                 "adaptive_p_feedback_version": 1,
                 "mirostat2_feedback_protocol": "mirostat2-state-v1",
                 "mirostat2_feedback_version": 1}
        sb._validate_sampling_capability(profile, ready)
        for changed in (
            dict(ready, adaptive_p_protocol=V6),
            dict(ready, mirostat2_protocol=V5),
            dict(ready, sampling_v4_protocol=V5),
            {key: value for key, value in ready.items() if key != "sampling_v6_protocol"},
        ):
            with self.subTest(changed=changed), self.assertRaises(IntegrityError):
                sb._validate_sampling_capability(profile, changed)

    def test_v7_command_binds_mask_draw_state_and_returns_next_envelope(self):
        proposal = self.sample(mask=b"\x0a\x00")
        self.assertIs(type(proposal), SelectionProposal)
        self.assertEqual(proposal.token, 3)
        self.assertEqual((proposal.draw_start, proposal.draw_count), (7, 1))
        self.assertEqual(proposal.feedback_proposal,
                         {"version": "adaptive-p-state-v1",
                          "weighted_sum": self.response["next_weighted_sum"],
                          "total_weight": self.response["next_total_weight"]})
        fields = self.commands[0]
        self.assertEqual(fields[:6], ["sample_mask_adaptive_p", "4", "0.7", "0.0", "0", "1.0"])
        self.assertEqual(fields[6], "123")
        self.assertEqual(self.payloads, [b"\x0a\x00"])
        self.assertEqual(self.history_bytes, [b""])
        self.assertEqual(self.breaker_bytes, [b""])
        self.assertEqual(fields[-4:], [repr(0.55), repr(0.9),
                                       repr(STATE["weighted_sum"]),
                                       repr(STATE["total_weight"])])

    def test_receipt_rejects_wrong_state_probability_counts_and_protocol(self):
        for change in (
            {"feedback_protocol": "wrong"},
            {"feedback_version": True},
            {"state_weighted_sum": 5.0},
            {"next_total_weight": 0.0},
            {"original_probability": float("nan")},
            {"original_probability": 1.1},
            {"adapted_target": 0.1},
            {"min_p_kept_count": 1},
            {"top_k_kept_count": 3},
            {"typical_kept_count": 3},
            {"kept_count": 0},
            {"draw_count": 2},
            {"adaptive_decay": 0.8},
            {"categorical_total_weight": 0.0},
            {"dynatemp_range": 0.2},
        ):
            self.response = self._response(**change)
            with self.subTest(change=change), self.assertRaises((IntegrityError, InvalidOperation)):
                self.sample()

    def test_v7_active_dry_binds_bounded_history_and_breakers(self):
        self.response = self._response(
            dry_multiplier=0.5, dry_history_count=4, dry_breaker_count=2)
        self.sample(dry_multiplier=0.5, dry_history=(1, 2, 3, 4),
                    dry_breakers=(1, 2))
        self.assertEqual(self.history_bytes, [struct.pack('<4I', 1, 2, 3, 4)])
        self.assertEqual(self.breaker_bytes, [struct.pack('<2I', 1, 2)])

    def test_v7_allows_the_documented_pre_adaptive_filter_pipeline(self):
        policy = dict(temperature=0.7, min_p=0.1, top_k=1, top_p=0.8,
                      typical_p=0.9, dynatemp_range=0.1,
                      dry_multiplier=0.5,
                      dynatemp_exponent=1.0, adaptive_target=0.55,
                      adaptive_decay=0.9)
        self.response = self._response(
            legal_count=4, min_p_kept_count=3, top_k_kept_count=1,
            typical_kept_count=1, kept_count=1, top_k=1,
            min_p=0.1, top_p=0.8, typical_p=0.9,
            dynatemp_range=0.1, dynatemp_exponent=1.0,
            effective_temperature=0.75, dry_multiplier=0.5,
            dry_history_count=4, dry_breaker_count=0,
            original_probability=1.0,
            next_weighted_sum=1.0 + 0.9 * STATE["weighted_sum"])

        proposal = self.sample(
            mask=b"\x0f\x00", sampler_state=STATE,
            temperature=policy['temperature'], min_p=policy['min_p'],
            top_k=policy['top_k'], top_p=policy['top_p'],
            adaptive_target=0.55, adaptive_decay=0.9,
            dry_multiplier=policy['dry_multiplier'],
            dry_history=(0, 1, 2, 3), dry_breakers=(),
            typical_p=policy['typical_p'],
            dynatemp_range=policy['dynatemp_range'],
            dynatemp_exponent=policy['dynatemp_exponent'],
            xtc_probability=0.0, xtc_threshold=0.1)

        self.assertEqual(proposal.token, 3)
        self.assertEqual(self.commands[0][0], "sample_mask_adaptive_p")
        self.assertEqual(self.commands[0][3:6], ["0.1", "1", "0.8"])
        self.assertEqual(self.commands[0][-7:-4], ["0.9", "0.1", "1.0"])
        self.assertEqual(self.history_bytes, [struct.pack('<4I', 0, 1, 2, 3)])

    def test_session_two_preview_draws_commits_only_native_acknowledged_first_draw(self):
        policy = dict(temperature=0.7, min_p=0.0, top_k=0, top_p=1.0,
                      repetition_penalty=1.0, adaptive_target=0.55,
                      adaptive_decay=0.9)
        state = sampled_controller.start(policy, "ap-two-preview", (0,), 9, [3])
        before = dict(state)
        before_state = dict(state["sampler_state"])
        tokenizer = ProposalTokenizer()
        session = sampled_controller.PlainSamplingSession.restore(
            state, (0,), 9, tokenizer.token_bytes, tokenizer.is_eog)
        previews = []

        def selector(mask, temperature, min_p, **filters):
            draws = filters["draw_candidates"]
            previews.append(draws)
            self.response = self._response(
                draw53=draws[0], token=2, legal_count=1, min_p_kept_count=1,
                top_k_kept_count=1, typical_kept_count=1, kept_count=1,
                original_probability=1.0,
                next_weighted_sum=1.0 + 0.9 * STATE["weighted_sum"])
            return self.backend.sample_selection_proposal(
                self.lease, mask, temperature, min_p, **filters)

        token = session.choose(make_mask(9, (2,)), selector)
        self.assertEqual(len(previews), 1)
        self.assertEqual(len(previews[0]), 2)
        self.assertEqual(session._state, before)
        self.assertEqual(state["next_draw"], 0)
        self.assertEqual(state["sampler_state"], before_state)
        self.assertEqual(self.commands[0][6], str(previews[0][0]))
        self.assertEqual(self.commands[0][6], str(self.response["draw53"]))
        self.assertEqual(token, 2)

        session._consume_selected(token, tokenizer.token_bytes(token), False)
        committed = session.current_controller
        self.assertEqual(committed["next_draw"], 1)
        self.assertEqual(committed["sampler_state"]["total_weight"],
                         self.response["next_total_weight"])
        self.assertEqual(state["next_draw"], 0)
        self.assertEqual(state["sampler_state"], before_state)

    def test_inputs_and_capability_fail_before_native_io(self):
        for change in (
            {"sampler_state": {"version": "wrong", "weighted_sum": 1.0, "total_weight": 1.0}},
            {"sampler_state": {"version": "adaptive-p-state-v1", "weighted_sum": 1.0, "total_weight": 0.0}},
            {"adaptive_target": 1.01},
            {"adaptive_decay": 1.0},
            {"draw_candidates": ()},
            {"draw_candidates": (1, 2, 3)},
            {"source_digest": "not-a-digest"},
        ):
            count = len(self.commands)
            with self.subTest(change=change), self.assertRaises((IntegrityError, InvalidOperation)):
                self.sample(**change)
            self.assertEqual(len(self.commands), count)
        self.backend.profile["execution"]["sampling_protocol"] = V6
        count = len(self.commands)
        with self.assertRaises(InvalidOperation):
            self.sample()
        self.assertEqual(len(self.commands), count)


if __name__ == "__main__":
    unittest.main()
