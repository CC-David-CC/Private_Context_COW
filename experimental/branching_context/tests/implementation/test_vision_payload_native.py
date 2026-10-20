"""VC008 native payload boundary; no GPU inference is simulated here."""
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


def sve():
    return struct.pack('<5i24f', 0x31455653, 6, 3, 2, 4, *[i/32 for i in range(24)])


class VisionPayloadNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='cow-vision-payload-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.exe = cls.directory/'probe'
        compiler = shutil.which('g++')
        if compiler is None:
            raise RuntimeError('The real native compiler is required')
        proc = subprocess.run([compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I', str(ROOT/'include'), str(ROOT/'experimental/branching_context/native/vision_payload_probe.cpp'),
            '-o', str(cls.exe)], capture_output=True, text=True, timeout=60)
        if proc.returncode:
            raise AssertionError(proc.stderr)

    def read(self, raw=None, *, path=None, sha=None, width=4, max_rows=1024):
        if path is None:
            path = self.directory/'payload.sve'
            path.write_bytes(raw)
        if sha is None:
            sha = hashlib.sha256(raw if raw is not None else b'').hexdigest()
        run = subprocess.run([str(self.exe), str(path), sha, str(width), str(max_rows)],
                             capture_output=True, text=True, timeout=5)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout.strip()

    def test_vc008_native_bounded_payload_preserves_dimensions(self):
        self.assertEqual(self.read(sve()), 'OK 6 3 2 4 24')

    def test_vc008_native_rejects_changed_finite_byte_with_original_hash(self):
        original = sve()
        changed = bytearray(original)
        changed[20] ^= 1
        self.assertIn('digest', self.read(changed, sha=hashlib.sha256(original).hexdigest()))

    def test_vc008_native_rejects_size_grid_width_and_nonfinite_values(self):
        for raw in (sve()[:-1], sve()+b'\0', sve()+sve(), b'',
                    struct.pack('<5i', 0x31455653, 6, 2, 2, 4)+sve()[20:],
                    sve()[:20]+struct.pack('<f', float('nan'))+sve()[24:],
                    sve()[:20]+struct.pack('<f', float('inf'))+sve()[24:]):
            with self.subTest(raw=raw[:20]):
                self.assertTrue(self.read(raw).startswith('ERR'))
        self.assertTrue(self.read(sve(), width=5).startswith('ERR'))
        self.assertTrue(self.read(sve(), max_rows=5).startswith('ERR'))

    def test_vc008_native_rejects_unsafe_file_types_without_blocking(self):
        regular = self.directory/'regular'
        regular.write_bytes(sve())
        link = self.directory/'link'
        link.symlink_to(regular)
        fifo = self.directory/'fifo'
        os.mkfifo(fifo)
        for path in (link, fifo, self.directory, self.directory/'missing'):
            with self.subTest(path=path):
                self.assertTrue(self.read(path=path).startswith('ERR'))

    def test_vc008_native_rejects_oversized_sparse_file_before_read(self):
        path = self.directory/'oversized'
        with path.open('wb') as stream:
            stream.truncate(1024**3)
        self.assertTrue(self.read(path=path).startswith('ERR'))

    def test_vc008_native_rejects_invalid_pinned_expectations(self):
        for options in (dict(sha='a'*63), dict(sha='A'*64), dict(width=0),
                        dict(max_rows=0), dict(width=2**31), dict(max_rows=2**31)):
            with self.subTest(options=options):
                self.assertTrue(self.read(sve(), **options).startswith('ERR'))


if __name__ == '__main__':
    unittest.main()
