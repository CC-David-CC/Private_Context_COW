"""CPU-only receipt and command tests for the native Mirostat v2 adapter."""
import copy
import math
import shlex
import unittest
from pathlib import Path

from context_time_travel import strata_backend as sb
from context_time_travel.mirostat2 import start_state
from context_time_travel.sampling_proposal import SelectionProposal
from context_time_travel.store import IntegrityError, InvalidOperation
import test_min_p_integration as minp_fixtures


V6 = "full-vocab-mirostat2-u53-v6"
SOURCE = "b" * 64
POLICY = dict(
    temperature=0.7, min_p=0.0, top_k=0, top_p=1.0,
    mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1,
)


class MirostatBackendTests(unittest.TestCase):
    def setUp(self):
        fixture = minp_fixtures.MinPBackendTests(
            "test_native_sample_is_read_only_and_cleans_transient_mask")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.backend = fixture.backend
        self.lease = fixture.lease
        self.backend.profile["execution"]["sampling_protocol"] = V6
        self.commands = []
        self.mask_bytes = []
        self.history_bytes = []
        self.breaker_bytes = []
        self.response = self._response()

        def request(command):
            fields = shlex.split(command)
            self.commands.append(fields)
            if fields[0] != "sample_mask_mirostat2":
                raise AssertionError(f"Unexpected native command: {fields[0]}")
            self.mask_bytes.append(Path(fields[4]).read_bytes())
            self.history_bytes.append(Path(fields[12]).read_bytes())
            self.breaker_bytes.append(Path(fields[13]).read_bytes())
            return dict(self.response)

        self.backend._request = request

    @staticmethod
    def _response(**changes):
        response = dict(
            status="ok", operator="full-vocab-mirostat2", protocol=V6,
            feedback_protocol="mirostat2-state-v1", feedback_version=1,
            token=8, consumed=4, logits_count=9, mask_bytes=2,
            legal_count=1, kept_count=1,
            legal_total_weight=1.0, total_weight=1.0,
            draw53=123, draw_count=1,
            temperature=0.7, tau=5.0, eta=0.1, mu=10.0,
            selected_log_probability=0.0, selected_probability=1.0,
            surprise_bits=0.0, next_mu=10.5,
            dry_multiplier=0.0, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256,
            dry_history_count=0, dry_breaker_count=0,
            dry_penalized_candidates=0,
        )
        response.update(changes)
        return response

    def sample(self, mask=b"\0\1", **changes):
        args = dict(
            sampler_state=start_state(5.0), mirostat_tau=5.0, mirostat_eta=0.1,
            draw_start=7, draw_candidates=(123,), source_digest=SOURCE,
            dry_multiplier=0.0, dry_base=1.75,
            dry_allowed_length=2, dry_penalty_last_n=256,
            dry_history=(), dry_breakers=(),
        )
        args.update(changes)
        return self.backend.sample_selection_proposal(
            self.lease, mask, POLICY["temperature"], POLICY["min_p"],
            top_k=POLICY["top_k"], top_p=POLICY["top_p"],
            mirostat=POLICY["mirostat"], **args)

    def test_v6_ready_capability_is_pinned(self):
        profile = {"execution": {"sampling_protocol": V6}}
        sb._validate_sampling_capability(profile, {"sampling_protocol": V6})
        for ready in ({}, {"sampling_protocol": "full-vocab-min-p-u53-v5"}):
            with self.subTest(ready=ready), self.assertRaises(IntegrityError):
                sb._validate_sampling_capability(profile, ready)

    def test_native_command_binds_state_draw_and_dry_inputs(self):
        self.response = self._response(
            token=3, consumed=4, logits_count=9, mask_bytes=2,
            legal_count=2, kept_count=2, legal_total_weight=1.5, total_weight=1.25,
            selected_log_probability=-0.25, selected_probability=math.exp(-0.25),
            surprise_bits=0.25 / math.log(2.0),
            mu=20.0, next_mu=20.0-0.1*(0.25/math.log(2.0)-5.0),
            dry_multiplier=0.75, dry_history_count=4, dry_breaker_count=1,
            dry_penalized_candidates=1)
        proposal = self.sample(mask=b"\x0a\x00", sampler_state=start_state(10.0),
            mirostat_tau=5.0, mirostat_eta=0.1,
            dry_multiplier=0.75, dry_history=(0, 1, 2, 3), dry_breakers=(1,))
        self.assertIs(type(proposal), SelectionProposal)
        self.assertEqual(proposal.token, 3)
        self.assertEqual(proposal.draw_start, 7)
        self.assertEqual(proposal.draw_count, 1)
        self.assertEqual(proposal.feedback_proposal['protocol'], 'mirostat2-state-v1')
        self.assertEqual(proposal.feedback_proposal['version'], 1)
        self.assertEqual(proposal.feedback_proposal['mu'],
                         20.0 - 0.1 * (0.25 / math.log(2.0) - 5.0))
        fields = self.commands[0]
        self.assertEqual(fields[:4], ["sample_mask_mirostat2", "4", "0.7", "123"])
        self.assertEqual(self.mask_bytes, [b"\x0a\x00"])
        self.assertEqual(len(self.history_bytes[0]), 16)
        self.assertEqual(len(self.breaker_bytes[0]), 4)
        self.assertEqual(fields[5:12], ["5.0", "0.1", "20.0", "0.75", "1.75", "2", "256"])

    def test_underflowed_probability_is_valid_only_with_finite_log_probability(self):
        logp = -1000.0
        self.response = self._response(
            token=8, selected_log_probability=logp,
            selected_probability=math.exp(logp),
            surprise_bits=-logp/math.log(2.0),
            next_mu=10.0-0.1*((-logp/math.log(2.0))-5.0))
        proposal = self.sample()
        self.assertEqual(proposal.token, 8)
        self.assertEqual(self.response["selected_probability"], 0.0)

    def test_receipt_rejects_probability_logprob_surprise_and_feedback_mismatch(self):
        cases = (
            {"selected_log_probability": float("nan")},
            {"selected_log_probability": 0.1},
            {"selected_probability": 0.5},
            {"surprise_bits": 2.0},
            {"next_mu": 9.0},
            {"feedback_protocol": "wrong"},
            {"feedback_version": True},
            {"kept_count": 2, "total_weight": 0.0},
            {"temperature": 0.8},
            {"draw_count": 0},
            {"selected_probability": 10 ** 400},
        )
        for change in cases:
            self.response = self._response(**change)
            with self.subTest(change=change), self.assertRaises((IntegrityError, InvalidOperation)):
                self.sample()

    def test_request_inputs_fail_before_native_io(self):
        for change in (
            {"mirostat_tau": 0.0},
            {"mirostat_eta": -0.1},
            {"sampler_state": {"protocol": "bad", "version": 1, "mu": 10.0}},
            {"sampler_state": {"protocol": "mirostat2-state-v1", "version": 1, "mu": float("nan")}},
            {"draw_candidates": ()},
            {"source_digest": "not-a-digest"},
        ):
            count = len(self.commands)
            with self.subTest(change=change), self.assertRaises((IntegrityError, InvalidOperation)):
                self.sample(**change)
            self.assertEqual(len(self.commands), count)


if __name__ == "__main__":
    unittest.main()
