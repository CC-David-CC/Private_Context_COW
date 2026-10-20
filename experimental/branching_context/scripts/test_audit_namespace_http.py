"""Post-run evidence comparator controls, not GPU acceptance tests."""
import struct
import unittest
from audit_namespace_http import compare_vectors


class AuditVectorTests(unittest.TestCase):
    def test_exact_finite_vectors(self):
        raw = struct.pack('<ff', 1.0, -2.0)
        self.assertEqual(compare_vectors(raw, raw, 2), 0.0)

    def test_incomplete_vector_cannot_pass(self):
        raw = struct.pack('<ff', 1.0, -2.0)
        with self.assertRaises(ValueError): compare_vectors(raw[:-1], raw, 2)

    def test_changed_value_cannot_pass(self):
        with self.assertRaises(ValueError):
            compare_vectors(struct.pack('<f', 1.0), struct.pack('<f', 1.5), 1)

    def test_matching_nonfinite_bytes_cannot_pass(self):
        raw = struct.pack('<f', float('nan'))
        with self.assertRaises(ValueError): compare_vectors(raw, raw, 1)


if __name__ == '__main__': unittest.main()
