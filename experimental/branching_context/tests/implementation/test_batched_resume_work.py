"""Reject hidden decode or replay work in the batches-only milestone."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('resume_work',
    Path(__file__).resolve().parents[2]/'scripts/durable_resume.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BatchedResumeWorkTests(unittest.TestCase):
    def test_decode_and_replay_cannot_qualify_batch_consumption(self):
        before = dict(prefill_tokens=132,consumed_tokens=132,decode_timed_tokens=0)
        good = dict(prefill_tokens=140,consumed_tokens=140,decode_timed_tokens=0)
        runner.assert_batched_generation_work(before,good,8)
        for bad in (dict(prefill_tokens=139,consumed_tokens=140,decode_timed_tokens=1),
                    dict(prefill_tokens=272,consumed_tokens=272,decode_timed_tokens=0)):
            with self.assertRaises(AssertionError):
                runner.assert_batched_generation_work(before,bad,8)


if __name__ == '__main__': unittest.main()
