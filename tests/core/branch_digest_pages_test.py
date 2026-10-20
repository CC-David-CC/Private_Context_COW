"""Check the all-pool CUDA work list against independent hashlib SHA-256."""
import hashlib
import json
import os
import random
import subprocess
import unittest


class CudaPageDigestTests(unittest.TestCase):
    def test_seeded_first_and_last_byte_mutations_change_only_expected_tiles(self):
        rng = random.Random(20261015)
        original = [rng.randbytes(65536) for _ in range(3)]

        def run(pages):
            process = subprocess.run([os.environ['CTT_DIGEST_PAGES_BINARY'], '65536'],
                input=b''.join(pages), capture_output=True, check=True)
            report = json.loads(process.stdout)
            expected = [hashlib.sha256(page[i:i+4096]).hexdigest()
                for page in reversed(pages) for i in range(0,65536,4096)]
            self.assertEqual(report['digests'], expected)
            self.assertEqual(report['device_hashed_bytes'], len(pages)*65536)
            self.assertEqual(report['kernel_calls'], 1)
            self.assertEqual(report['plan_copy_bytes'], len(pages)*8)
            self.assertEqual(report['workspace_live_bytes'], 0)
            return report['digests']

        baseline = run(original)
        first = [bytearray(page) for page in original]
        first[0][0] ^= 1
        first_digests = run([bytes(page) for page in first])
        self.assertNotEqual(first_digests[32], baseline[32])  # reversed page order: original page 0
        self.assertEqual([i for i,(a,b) in enumerate(zip(baseline,first_digests)) if a != b], [32])

        last = [bytearray(page) for page in original]
        last[-1][-1] ^= 1
        last_digests = run([bytes(page) for page in last])
        self.assertNotEqual(last_digests[15], baseline[15])  # last tile of original page 2
        self.assertEqual([i for i,(a,b) in enumerate(zip(baseline,last_digests)) if a != b], [15])

    def test_reordered_pages_and_uneven_tile_count(self):
        rng = random.Random(1938001962)
        for count, page_bytes in [(1, 4096), (3, 69632), (35, 69632)]:
            with self.subTest(pages=count, page_bytes=page_bytes):
                pages = [rng.randbytes(page_bytes) for _ in range(count)]
                process = subprocess.run([os.environ['CTT_DIGEST_PAGES_BINARY'], str(page_bytes)],
                                         input=b''.join(pages), capture_output=True, check=True)
                report = json.loads(process.stdout)
                expected = [hashlib.sha256(page[i:i+4096]).hexdigest()
                            for page in reversed(pages) for i in range(0, page_bytes, 4096)]
                self.assertEqual(report['digests'], expected)
                self.assertEqual(report['device_hashed_bytes'], count * page_bytes)
                self.assertEqual(report['kernel_calls'], 1)
                self.assertEqual(report['plan_copy_bytes'], count * 8)
                self.assertEqual(report['workspace_live_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
