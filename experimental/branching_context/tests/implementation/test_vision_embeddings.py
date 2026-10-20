"""VC02 bounded encoder payload admission. These are CPU parser tests."""
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest

from context_time_travel import IntegrityError, InvalidOperation


def sve(rows=6, nx=3, ny=2, width=4, values=None):
    header = struct.pack('<5i', 0x31455653, rows, nx, ny, width)
    if values is None:
        values = [(i % 19 - 9) / 32 for i in range(max(0, rows * width))]
    return header + struct.pack('<' + 'f' * len(values), *values)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class VisionEmbeddingTests(unittest.TestCase):
    def parse(self, raw, **overrides):
        from context_time_travel.vision_embeddings import parse_sve
        options = dict(expected_sha256=digest(raw), expected_width=4, max_rows=1024)
        options.update(overrides)
        return parse_sve(raw, **options)

    def test_vc004_valid_payload_keeps_exact_immutable_bytes_and_dimensions(self):
        raw = sve()
        record = self.parse(raw)
        self.assertIs(record.payload, raw)
        self.assertEqual((record.rows, record.grid_width, record.grid_height, record.width), (6, 3, 2, 4))
        self.assertEqual(record.sha256, digest(raw))
        self.assertEqual(record.nbytes, len(raw))
        self.assertTrue(record.data.readonly)
        self.assertEqual(record.data.tobytes(), raw[20:])

    def test_vc004_checksum_rejection_includes_one_finite_changed_byte(self):
        raw = sve()
        changed = bytearray(raw)
        changed[20] ^= 1
        self.assertTrue(math.isfinite(struct.unpack_from('<f', changed, 20)[0]))
        with self.assertRaisesRegex(IntegrityError, 'digest|checksum'):
            self.parse(bytes(changed), expected_sha256=digest(raw))

    def test_vc004_rejects_wrong_model_width_even_with_valid_hash(self):
        with self.assertRaisesRegex(IntegrityError, 'width'):
            self.parse(sve(), expected_width=8)

    def test_vc004_rejects_mutable_buffers_before_borrowing_them(self):
        from context_time_travel.vision_embeddings import parse_sve
        raw = sve()
        for value in (bytearray(raw), memoryview(raw), None):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(InvalidOperation):
                    parse_sve(value, expected_sha256=digest(raw), expected_width=4)

    def test_vc005_rejects_nan_and_both_infinities_with_matching_hashes(self):
        for number in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(number=number):
                with self.assertRaisesRegex(IntegrityError, 'finite'):
                    self.parse(sve(values=[number] + [0.] * 23))

    def test_vc005_rejects_truncation_trailing_bytes_and_concatenated_chunks(self):
        raw = sve()
        for changed in (b'', raw[:19], raw[:-1], raw + b'\0', raw + raw):
            with self.subTest(length=len(changed)):
                with self.assertRaises(IntegrityError):
                    self.parse(changed)

    def test_vc005_rejects_bad_magic_nonrectangular_and_negative_dimensions(self):
        raw = sve()
        headers = [(0x53455631, 6, 3, 2, 4), (0x31455653, 6, 2, 2, 4),
                   (0x31455653, 0, 3, 2, 4), (0x31455653, -1, 3, 2, 4),
                   (0x31455653, 6, 3, 2, -4)]
        for header in headers:
            with self.subTest(header=header):
                with self.assertRaises(IntegrityError):
                    self.parse(struct.pack('<5i', *header) + raw[20:])

    def test_vc005_enforces_row_cap_including_huge_size_arithmetic(self):
        with self.assertRaisesRegex(IntegrityError, 'row|bound|limit'):
            self.parse(sve(), max_rows=5)
        raw = struct.pack('<5i', 0x31455653, 2**31-1, 2**31-1, 1, 2**31-1)
        with self.assertRaises(IntegrityError):
            self.parse(raw)

    def test_vc005_rejects_invalid_pinned_expectations(self):
        for options in (dict(expected_width=True), dict(expected_width=0),
                        dict(max_rows=0), dict(max_rows=True),
                        dict(max_rows=2**31), dict(expected_sha256='x'*64),
                        dict(expected_sha256='A'*64)):
            with self.subTest(options=options):
                with self.assertRaises(InvalidOperation):
                    self.parse(sve(), **options)

    def test_vc006_file_load_round_trip_validates_without_an_encoder(self):
        from context_time_travel.vision_embeddings import read_sve
        raw = sve()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'payload.sve'
            path.write_bytes(raw)
            loaded = read_sve(path, expected_sha256=digest(raw), expected_width=4)
            self.assertEqual(loaded.payload, raw)

    def test_vc006_file_load_rejects_oversize_sparse_file_without_reading_it_all(self):
        from context_time_travel.vision_embeddings import read_sve
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'oversize.sve'
            with path.open('wb') as stream:
                stream.write(sve())
                stream.truncate(1024**3)
            with self.assertRaisesRegex(IntegrityError, 'size|length|limit'):
                read_sve(path, expected_sha256='0'*64, expected_width=4, max_rows=8)

    def test_vc006_file_load_refuses_symlink_and_fifo(self):
        from context_time_travel.vision_embeddings import read_sve
        raw = sve()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'real.sve').write_bytes(raw)
            (folder/'link.sve').symlink_to(folder/'real.sve')
            os.mkfifo(folder/'fifo.sve')
            for path in (folder/'link.sve', folder/'fifo.sve'):
                with self.subTest(path=path.name):
                    with self.assertRaises(IntegrityError):
                        read_sve(path, expected_sha256=digest(raw), expected_width=4)

    def test_vc006_encoding_identity_binds_full_normalized_digest_and_profile(self):
        from context_time_travel.vision_embeddings import encoding_key
        first = encoding_key('a'*64, 'b'*64)
        self.assertEqual(first, encoding_key('a'*64, 'b'*64))
        self.assertNotEqual(first, encoding_key('a'*63+'c', 'b'*64))
        self.assertNotEqual(first, encoding_key('a'*64, 'c'*64))
        self.assertEqual(len(first), 64)
        with self.assertRaises(InvalidOperation):
            encoding_key('a'*32, 'b'*64)


if __name__ == '__main__':
    unittest.main()
