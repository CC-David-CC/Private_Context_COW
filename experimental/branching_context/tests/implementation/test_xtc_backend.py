"""CPU protocol tests for the schema-4 native XTC adapter.

These use a protocol fixture and make no GPU inference claim.
"""
import copy
import shlex
import struct
import unittest
from pathlib import Path

from context_time_travel import strata_backend as sb
from context_time_travel.sampling_proposal import SelectionProposal, XTC_PROPOSAL_PROTOCOL
from context_time_travel.store import IntegrityError, InvalidOperation
import test_min_p_integration as minp_fixtures


V5 = XTC_PROPOSAL_PROTOCOL
U53 = 1 << 53
SOURCE = "a" * 64


class XtcBackendTests(unittest.TestCase):
    def setUp(self):
        fixture = minp_fixtures.MinPBackendTests(
            "test_native_sample_is_read_only_and_cleans_transient_mask")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.backend = fixture.backend
        self.lease = fixture.lease
        self.backend.profile["execution"]["sampling_protocol"] = V5
        self.commands = []
        self.mask_bytes = []
        self.history_bytes = []
        self.breaker_bytes = []
        self.response = self._response()

        def request(command):
            fields = shlex.split(command)
            self.commands.append(fields)
            if fields[0] != "sample_mask_v5":
                raise AssertionError(f"Unexpected native command: {fields[0]}")
            self.mask_bytes.append(Path(fields[7]).read_bytes())
            self.history_bytes.append(Path(fields[12]).read_bytes())
            self.breaker_bytes.append(Path(fields[13]).read_bytes())
            return dict(self.response)

        self.backend._request = request

    @staticmethod
    def _response(**changes):
        response = dict(
            status="ok", operator="full-vocab-min-p", protocol=V5,
            token=8, consumed=4, logits_count=9, mask_bytes=2,
            legal_count=1, min_p_kept_count=1, top_k_kept_count=1,
            typical_kept_count=1, effective_temperature=0.7,
            kept_count=1, draw53=123, draw53_second=456,
            draw_count=1, activation_draws=0, activated=False,
            removed_count=0, pre_xtc_count=1,
            top_k=0, top_p=1.0,
            dry_multiplier=0.0, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256,
            dry_history_count=0, dry_breaker_count=0,
            dry_penalized_candidates=0,
            typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0,
            xtc_probability=0.0, xtc_threshold=0.1,
            total_weight=1.0,
        )
        response.update(changes)
        return response

    def sample(self, mask=b"\0\1", **changes):
        args = dict(
            draw_start=5, draw_candidates=(123, 456), source_digest=SOURCE,
            top_k=0, top_p=1.0,
            dry_multiplier=0.0, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256,
            dry_history=(), dry_breakers=(),
            typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0,
            xtc_probability=0.0, xtc_threshold=0.1,
        )
        args.update(changes)
        return self.backend.sample_selection_proposal(
            self.lease, mask, 0.7, 0.05, **args)

    def test_v5_ready_capability_is_pinned_exactly(self):
        profile = {"execution": {"sampling_protocol": V5}}
        sb._validate_sampling_capability(profile, {"sampling_protocol": V5})
        for ready in ({}, {"sampling_protocol": "full-vocab-min-p-u53-v4"}):
            with self.subTest(ready=ready), self.assertRaises(IntegrityError):
                sb._validate_sampling_capability(profile, ready)

    def test_command_binds_cursor_draw_candidates_settings_and_transient_inputs(self):
        self.response = self._response(
            dry_multiplier=0.8, dry_history_count=4, dry_breaker_count=1,
            dry_penalized_candidates=0, typical_p=0.95,
            dynatemp_range=0.2, dynatemp_exponent=1.25)
        result = self.sample(
            dry_multiplier=0.8, dry_history=(0, 1, 3, 0), dry_breakers=(3,),
            typical_p=0.95, dynatemp_range=0.2, dynatemp_exponent=1.25,
            xtc_probability=0.0, xtc_threshold=0.1)
        self.assertIs(type(result), SelectionProposal)
        self.assertEqual((result.token, result.draw_start, result.draw_count), (8, 5, 1))
        fields = self.commands[0]
        self.assertEqual(fields[0], "sample_mask_v5")
        self.assertEqual(fields[1:7], ["4", "0.7", "0.05", "0", "1.0", "123"])
        self.assertEqual(fields[8:12], ["0.8", "1.75", "2", "256"])
        self.assertEqual(self.history_bytes, [struct.pack("<4I", 0, 1, 3, 0)])
        self.assertEqual(self.breaker_bytes, [struct.pack("<I", 3)])
        self.assertEqual(fields[14:], ["0.95", "0.2", "1.25", "0.0", "0.1", "456"])
        self.assertEqual(self.mask_bytes, [b"\0\1"])
        self.assertEqual(list(self.backend._directory.iterdir()), [])
        self.assertEqual(self.lease.consumed, 4)

    def test_activation_uses_second_candidate_and_reports_actual_removed_support(self):
        self.response = self._response(
            legal_count=2, min_p_kept_count=2, top_k_kept_count=2,
            typical_kept_count=2, pre_xtc_count=2, kept_count=1,
            xtc_probability=0.4, xtc_threshold=0.1,
            activated=True, activation_draws=1, removed_count=1,
            draw53_second=456, draw_count=2)
        result = self.sample(
            mask=b"\x02\x01",
            xtc_probability=0.4, xtc_threshold=0.1,
            draw_candidates=(123, 456))
        self.assertEqual((result.draw_start, result.draw_count), (5, 2))
        self.assertEqual(self.commands[0][6], "123")
        self.assertEqual(self.commands[0][-1], "456")
        self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_eligible_xtc_consumes_second_draw_even_when_activation_test_is_false(self):
        first = 1 << 52
        self.response = self._response(
            draw53=first, legal_count=2, min_p_kept_count=2,
            top_k_kept_count=2, typical_kept_count=2,
            pre_xtc_count=2, kept_count=2,
            xtc_probability=0.1, xtc_threshold=0.1,
            activated=False, activation_draws=1, removed_count=0,
            draw53_second=456, draw_count=2, total_weight=1.5)
        proposal = self.sample(mask=b"\x02\x01", draw_candidates=(first, 456),
                              xtc_probability=0.1)
        self.assertEqual(proposal.draw_count, 2)

    def test_inactive_xtc_echoes_second_candidate_or_absence_sentinel(self):
        self.response = self._response(
            xtc_probability=0.0, xtc_threshold=0.1, draw53_second=456,
            activated=False, activation_draws=0, removed_count=0, draw_count=1)
        proposal = self.sample()
        self.assertEqual(proposal.draw_count, 1)
        self.assertEqual(self.commands[0][-1], "456")
        self.response = self._response(draw53_second=U53)
        proposal = self.sample(draw_candidates=(123,))
        self.assertEqual(proposal.draw_count, 1)
        self.assertEqual(self.commands[1][-1], str(U53))

    def test_zero_temperature_delegates_greedy_and_consumes_no_rng_draw(self):
        self.backend.sample_legal = lambda lease, mask, temperature, min_p, draw53, **kw: 8
        proposal = self.backend.sample_selection_proposal(
            self.lease, b"\0\1", 0.0, 0.05,
            draw_start=7, draw_candidates=(), source_digest=SOURCE,
            xtc_probability=0.4, xtc_threshold=0.1)
        self.assertEqual((proposal.token, proposal.draw_start, proposal.draw_count), (8, 7, 0))
        self.assertEqual(self.commands, [])

    def test_bad_receipt_fields_fail_closed_and_remove_every_temp_file(self):
        bad = (
            ("consumed", 5), ("logits_count", 8), ("mask_bytes", 3),
            ("legal_count", True), ("token", True), ("token", 1),
            ("draw53", 124), ("draw53_second", 123), ("kept_count", 0),
            ("pre_xtc_count", 0), ("draw_count", 2),
            ("activation_draws", True), ("activated", 1),
            ("removed_count", 1), ("protocol", "wrong"),
            ("xtc_probability", 0.2), ("xtc_threshold", 0.2),
            ("effective_temperature", float("nan")),
            ("min_p_kept_count", 0), ("min_p_kept_count", True),
            ("min_p_kept_count", None), ("top_k_kept_count", 2),
            ("top_k_kept_count", None), ("typical_kept_count", None),
            ("effective_temperature", None),
            ("total_weight", None), ("total_weight", -1.0),
            ("total_weight", 0.0), ("total_weight", float("inf")),
            ("total_weight", True), ("status", "error"),
        )
        for field, value in bad:
            self.response = self._response(**{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(IntegrityError):
                self.sample()
            self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_v5_top_k_receipt_requires_exact_integer_type(self):
        self.response = self._response(top_k=0.0)
        with self.assertRaises(IntegrityError):
            self.sample(top_k=0)
        self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_v5_stage_counts_obey_requested_top_k_bound(self):
        cases = (
            (0, 2, 1),  # no Top-K cap preserves all Min-P candidates
            (1, 2, 2),  # a positive cap cannot retain more than K
        )
        for top_k, min_p_count, top_k_count in cases:
            self.response = self._response(
                token=1, legal_count=2, min_p_kept_count=min_p_count,
                top_k_kept_count=top_k_count, typical_kept_count=top_k_count,
                pre_xtc_count=top_k_count, kept_count=top_k_count,
                top_k=top_k)
            with self.subTest(top_k=top_k, min_p_count=min_p_count,
                              top_k_count=top_k_count):
                with self.assertRaises(IntegrityError):
                    self.sample(mask=b"\x03\x00", top_k=top_k)
            self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_v5_positive_top_k_accepts_a_valid_truncated_stage_count(self):
        self.response = self._response(
            token=1, legal_count=2, min_p_kept_count=2,
            top_k_kept_count=1, typical_kept_count=1,
            pre_xtc_count=1, kept_count=1, top_k=1)
        proposal = self.sample(mask=b"\x03\x00", top_k=1)
        self.assertEqual(proposal.token, 1)
        self.assertEqual(self.commands[0][3], "0.05")
        self.assertEqual(self.commands[0][4], "1")

    def test_bad_inputs_reject_before_native_io(self):
        good = dict(draw_start=5, draw_candidates=(123, 456), source_digest=SOURCE)
        cases = (
            dict(draw_start=True), dict(draw_start=1 << 128),
            dict(draw_candidates=()), dict(draw_candidates=(True, 4)),
            dict(draw_candidates=(1 << 53,)), dict(source_digest="x"),
            dict(xtc_probability=float("nan")), dict(xtc_probability=1.1),
            dict(xtc_threshold=float("inf")), dict(xtc_threshold=-0.1),
        )
        for changed in cases:
            args = dict(good)
            args.update(changed)
            with self.subTest(changed=changed), self.assertRaises(InvalidOperation):
                self.backend.sample_selection_proposal(self.lease, b"\0\1", 0.7, 0.05, **args)
            self.assertEqual(self.commands, [])
            self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_transient_native_error_still_cleans_mask_history_and_breaker_files(self):
        def fail(_command):
            raise RuntimeError("fixture native error")

        self.backend._request = fail
        with self.assertRaisesRegex(RuntimeError, "fixture native error"):
            self.sample(dry_multiplier=0.8, dry_history=(0, 0, 0, 0), dry_breakers=())
        self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_v5_profile_preserves_legacy_v2_routing_for_plain_policy(self):
        self.response = dict(self.fixture.response,
                             protocol="full-vocab-min-p-u53-v2", top_k=0, top_p=1.0)
        self.backend._request = lambda command: (
            self.commands.append(shlex.split(command)), dict(self.response))[1]
        self.assertEqual(self.backend.sample_legal(self.lease, b"\0\1", 0.7, 0.05, 123), 8)
        self.assertEqual(self.commands[0][0], "sample_mask_v2")
        self.assertEqual(list(self.backend._directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
