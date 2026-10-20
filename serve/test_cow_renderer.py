"""CPU-only regression tests for presentation identity pinning."""

from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from strata_tokenizer import Tokenizer  # noqa: E402
from serve import cow_renderer  # noqa: E402


CONTRACT = "fc6d0d643e4b9ae020964f8c4e5006c0da7199b19cce5285c573023dbe82ca46"
TEMPLATE = "{{ messages|length }}:{{ add_generation_prompt }}"
KWARGS = {"enable_thinking": False, "tools": [{"name": "lookup", "mode": "safe"}]}


def tokenizer_fixture(seed: int = 17) -> Tokenizer:
    rng = random.Random(seed)
    variant = rng.randrange(1000)
    tokens = ["a", "b", "ab", "ba", f"x{variant}", f"<|stop{variant}|>"]
    token_types = [1, 1, 1, 1, 1, 3]
    special_ids = {
        "tokenizer.ggml.eos_token_id": 5,
        "tokenizer.ggml.bos_token_id": 0,
    }
    return Tokenizer(tokens, ["a b", "b a"], token_types, "qwen35", special_ids)


class RendererIdentityTests(unittest.TestCase):
    def identity(self, tokenizer=None, template=TEMPLATE, kwargs=None, contract=CONTRACT):
        return cow_renderer.renderer_identity(
            tokenizer or tokenizer_fixture(), template, KWARGS if kwargs is None else kwargs,
            model_contract_id=contract)

    def test_same_metadata_is_stable_across_mapping_order_and_cache_mutation(self):
        first = tokenizer_fixture(17)
        second = tokenizer_fixture(17)
        first.encode("ab", parse_special=False)
        before = self.identity(first)
        first._piece_ids["runtime-cache-only"] = [1, 2, 3]
        first._bytes_cache[9] = b"mutable cache content"
        after_cache_change = self.identity(first)

        kwargs_reordered = {
            "tools": [{"mode": "safe", "name": "lookup"}],
            "enable_thinking": False,
        }
        same_fixture_identity = self.identity(second, kwargs=kwargs_reordered)
        self.assertEqual(before, after_cache_change)
        self.assertEqual(before, same_fixture_identity)
        self.assertRegex(before, r"^[0-9a-f]{64}$")

    def test_seeded_fixture_metadata_and_each_rendering_input_change_identity(self):
        baseline = self.identity()
        self.assertNotEqual(baseline, self.identity(tokenizer=tokenizer_fixture(18)))
        self.assertNotEqual(baseline, self.identity(template=TEMPLATE + " "))
        self.assertNotEqual(baseline, self.identity(kwargs={**KWARGS, "add_generation_prompt": True}))
        self.assertNotEqual(baseline, self.identity(contract="qualified-contract-v2"))

        cases = (
            lambda t: setattr(t, "pre", "different-pre-tokenizer"),
            lambda t: t.tokens.__setitem__(4, "changed-token"),
            lambda t: t.token_types.__setitem__(0, 4),
            lambda t: t.special_ids.__setitem__("tokenizer.ggml.eos_token_id", 4),
            lambda t: t.ranks.__setitem__(("a", "b"), 9),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                changed = tokenizer_fixture()
                mutate(changed)
                self.assertNotEqual(baseline, self.identity(tokenizer=changed))

    def test_implementation_policy_hashes_are_bound_without_paths(self):
        normal = self.identity()
        with mock.patch.object(cow_renderer, "_implementation_hashes", return_value={
                "serve.cow_prompt_encoder": "0" * 64,
                "serve.frontend": "1" * 64,
                "serve.server": "2" * 64,
                "tools.strata_tokenizer": "3" * 64}):
            changed = self.identity()
        self.assertNotEqual(normal, changed)

        receipt = cow_renderer.renderer_identity_receipt(tokenizer_fixture(), TEMPLATE, KWARGS,
                                                         model_contract_id=CONTRACT)
        self.assertEqual(set(receipt["implementation_sha256"]), {
            "serve.cow_prompt_encoder", "serve.frontend", "serve.server", "tools.strata_tokenizer"})
        self.assertNotIn(str(ROOT), repr(receipt))

    def test_receipt_contains_raw_input_fingerprints_and_qualified_contract(self):
        receipt = cow_renderer.renderer_identity_receipt(tokenizer_fixture(), TEMPLATE, KWARGS,
                                                         model_contract_id=CONTRACT)
        self.assertEqual(receipt["model_contract_id"], CONTRACT)
        self.assertRegex(receipt["tokenizer"]["tokens_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(receipt["template_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(set(receipt["dependencies"]), {"regex", "jinja2"})

    def test_missing_or_incomplete_tokenizer_metadata_is_rejected(self):
        incomplete = copy.copy(tokenizer_fixture())
        del incomplete.special_ids
        with self.assertRaisesRegex(ValueError, "special_ids"):
            self.identity(tokenizer=incomplete)

        malformed = tokenizer_fixture()
        malformed.token_types = [1]
        with self.assertRaisesRegex(ValueError, "token_types"):
            self.identity(tokenizer=malformed)

        malformed = tokenizer_fixture()
        malformed.ranks = {}
        with self.assertRaisesRegex(ValueError, "ranks"):
            self.identity(tokenizer=malformed)

    def test_empty_contract_and_non_json_kwargs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_contract_id"):
            self.identity(contract="  ")
        with self.assertRaises((TypeError, ValueError)):
            self.identity(kwargs={1: "non-string key"})


if __name__ == "__main__":
    unittest.main()
