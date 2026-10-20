"""Audit output IDs from canonical spans without inventing image tokens."""
import unittest

from context_time_travel.store import IntegrityError, InvalidOperation
from run_json_schema_probe import output_token_span


class JsonSchemaProbeReaderTests(unittest.TestCase):
    def test_reads_the_requested_text_boundary(self):
        spans = [dict(kind="text", tokens=[1, 2]), dict(kind="text", tokens=[3, 4, 5])]
        self.assertEqual(output_token_span(spans, 2, 2), [3, 4])

    def test_rejects_invalid_or_truncated_receipt_boundaries(self):
        for start, count in ((True, 1), (0, False), (-1, 1), (0, -1), (1, 2)):
            with self.subTest(start=start, count=count), self.assertRaises((ValueError, InvalidOperation)):
                output_token_span([dict(kind="text", tokens=[1, 2])], start, count)

    def test_never_treats_image_padding_as_generated_tokens(self):
        image = dict(kind="image", original_sha256="1" * 64, normalized_sha256="2" * 64,
            encoder_profile_id="3" * 64, embedding_sha256="4" * 64,
            embedding_bytes=28, rows=2, width=1, grid_width=2, grid_height=1, pad_token_id=9)
        spans = [dict(kind="text", tokens=[1]), image, dict(kind="text", tokens=[7, 8])]
        self.assertEqual(output_token_span(spans, 3, 2), [7, 8])
        with self.assertRaises(IntegrityError):
            output_token_span(spans, 1, 2)
        with self.assertRaises(InvalidOperation):
            output_token_span(spans, 2, 1)


if __name__ == "__main__":
    unittest.main()
