"""Verify persistent scratch on CUDA, independently checking all SHA-256 tiles."""
import hashlib
import json
import os
import random
import subprocess
import unittest


class DigestViewTests(unittest.TestCase):
    def test_reused_scratch_fresh_bytes_and_changed_page_order(self):
        rng = random.Random(1938001962)
        pages = [rng.randbytes(65536) for _ in range(35)]
        process = subprocess.run([os.environ['CTT_DIGEST_VIEW_BINARY']],
            input=b''.join(pages),capture_output=True,check=True)
        report = json.loads(process.stdout)
        expected = [hashlib.sha256(page[i:i+4096]).hexdigest()
            for page in reversed(pages) for i in range(0,len(page),4096)]
        self.assertEqual(report['digests'],expected)
        self.assertTrue(report['fresh_mutation_detected'])
        self.assertEqual(report['host_allocations_during_measurement'],0)
        self.assertEqual(report['host_workspace_live_bytes'],0)
        self.assertEqual(report['workspace_live_bytes'],0)
        self.assertEqual(report['device_hashed_bytes'],3*len(pages)*65536)
        self.assertEqual(report['kernel_calls'],3)
        print(json.dumps({key:value for key,value in report.items() if key != 'digests'}))


if __name__ == '__main__': unittest.main()
