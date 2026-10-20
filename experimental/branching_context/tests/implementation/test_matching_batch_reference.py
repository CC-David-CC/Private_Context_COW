"""Source-generation checks; these do not qualify GPU execution."""
import importlib.util
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(scripts))
import build_batched_resume_probe as builder


class MatchingBatchReferenceTests(unittest.TestCase):
    def test_reference_preserves_requests_and_records_completed_chunks(self):
        source = builder.matching_schedule_source(builder.fresh_batched_source(builder.base.FRESH))
        self.assertIn('STRATA_PREFILL_REQUESTS', source)
        self.assertIn('request_offset != initial_tokens.size()', source)
        self.assertIn('batched.run(initial_tokens.data()+request_offset, request_count, request_offset, error)', source)
        self.assertIn('completed_chunks', source)
        self.assertIn('.requests.f32', source)
        self.assertNotIn('Verifier decode;', source)
        self.assertNotIn('session.restore', source)


if __name__ == '__main__': unittest.main()
