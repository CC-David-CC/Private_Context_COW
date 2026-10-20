"""Exact token-prefix checks required before a native COW request can reuse history."""
import unittest

from serve.cow_prefix import RenderedPrefixMismatch, exact_append_suffix


class BoundaryMergingTokenizer:
    """Tiny example where encoding the concatenation changes the old token."""
    def encode(self, text):
        if text == 'prefix':
            return [41]
        if text == 'prefix + suffix':
            return [77, 12]
        raise ValueError(text)


class ExactRenderedPrefixTests(unittest.TestCase):
    def test_exact_prefix_returns_only_new_rendered_token_ids(self):
        self.assertEqual(exact_append_suffix((41, 8), [41, 8, 3, 5]), (3, 5))

    def test_empty_canonical_history_returns_complete_prompt(self):
        self.assertEqual(exact_append_suffix((), (9, 4)), (9, 4))

    def test_equal_history_has_empty_suffix(self):
        self.assertEqual(exact_append_suffix((2, 4), (2, 4)), ())

    def test_rejects_character_prefix_when_reencoding_changes_bpe_boundary(self):
        tokenizer = BoundaryMergingTokenizer()
        old_text = 'prefix'
        full_text = 'prefix + suffix'
        self.assertTrue(full_text.startswith(old_text))
        self.assertNotEqual(tokenizer.encode(old_text), tokenizer.encode(full_text)[:1])
        with self.assertRaisesRegex(RenderedPrefixMismatch, 'token IDs'):
            exact_append_suffix(tokenizer.encode(old_text), tokenizer.encode(full_text))

    def test_rejects_short_or_divergent_rendered_history(self):
        for rendered in ((1,), (1, 9, 3)):
            with self.subTest(rendered=rendered):
                with self.assertRaises(RenderedPrefixMismatch):
                    exact_append_suffix((1, 2), rendered)

    def test_rejects_non_token_ids_instead_of_coercing_them(self):
        for values in ((True,), (-1,), ('3',), (1.0,)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, 'nonnegative integer token IDs'):
                    exact_append_suffix((), values)


if __name__ == '__main__':
    unittest.main()
