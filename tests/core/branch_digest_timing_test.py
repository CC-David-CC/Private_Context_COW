"""Actual CUDA digest timings and independent hashlib comparison; no GPU mock."""
import hashlib
import json
import os
import random
import subprocess
import unittest


class DigestTimingTests(unittest.TestCase):
    def test_blocking_stage_times_preserve_full_ordered_hashes_and_work(self):
        rng = random.Random(1938001962)
        pages = [rng.randbytes(65536) for _ in range(35)]
        process = subprocess.run([os.environ['CTT_DIGEST_TIMING_BINARY']],
            input=b''.join(pages),capture_output=True,check=True)
        report = json.loads(process.stdout)
        expected = [hashlib.sha256(page[i:i+4096]).hexdigest()
            for page in reversed(pages) for i in range(0,len(page),4096)]
        self.assertEqual(report['digests'],expected)
        stages = [report[key] for key in ('plan_upload_ns','page_hash_ns','page_digest_download_ns')]
        self.assertTrue(all(value > 0 for value in stages))
        self.assertLessEqual(sum(stages),report['wall_ns'])
        self.assertEqual(report['device_hashed_bytes'],3*len(pages)*65536)
        self.assertEqual(report['kernel_calls'],3)
        self.assertEqual(report['workspace_live_bytes'],0)
        print(json.dumps({key:value for key,value in report.items() if key != 'digests'}))


if __name__ == '__main__': unittest.main()
