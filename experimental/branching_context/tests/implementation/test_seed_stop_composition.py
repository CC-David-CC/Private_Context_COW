"""Composition contracts for seed controls and text-stop presentation metadata."""

import unittest

from context_time_travel.presentation_request import detach_request
from context_time_travel.store import InvalidOperation


class SeedStopCompositionTests(unittest.TestCase):
    def test_presentation_request_keeps_seed_and_stop_controls_in_generation_descriptor(self):
        request = {
            "schema": 2,
            "namespace_id": "a" * 32,
            "kind": "answer",
            "messages": [],
            "renderer_id": "b" * 64,
            "output_message_id": "c" * 32,
            "seed": 0,
        }

        normalized = detach_request(request, expected_kind="answer")

        self.assertEqual(normalized, request)
        request["controls"] = {"schema": 1, "seed": 0, "text_stops": ["<END>"]}
        with self.assertRaises(InvalidOperation):
            detach_request(request, expected_kind="answer")

    def test_stop_only_control_uses_null_seed_without_changing_legacy_seed_schema(self):
        request = {
            "schema": 1,
            "namespace_id": "a" * 32,
            "kind": "answer",
            "messages": [],
            "renderer_id": "b" * 64,
            "output_message_id": "c" * 32,
        }

        normalized = detach_request(request, expected_kind="answer")

        self.assertEqual(normalized, request)


if __name__ == "__main__":
    unittest.main()
