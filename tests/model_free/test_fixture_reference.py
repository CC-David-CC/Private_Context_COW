import hashlib
import unittest

import fixture_reference as ref


class FixtureReferenceContract(unittest.TestCase):
    def test_seeded_known_answers_and_coordinate_sensitivity(self):
        self.assertEqual(ref.byte_at(0x0123456789ABCDEF, 2, 3, 17, 5, 64), 0x8D)
        self.assertEqual(ref.byte_at(0x0123456789ABCDEF, 2, 3, 17, 5, 65), 0xDB)
        self.assertNotEqual(ref.byte_at(9, 1, 2, 3, 4, 64), ref.byte_at(10, 1, 2, 3, 4, 64))
        base = ref.byte_at(9, 1, 2, 3, 4, 64)
        for coord in ((2,2,3,4,64),(1,3,3,4,64),(1,2,4,4,64),(1,2,3,5,64),(1,2,3,4,65)):
            self.assertNotEqual(base, ref.byte_at(9, *coord))

    def test_logical_page_markers_and_physical_independence(self):
        seed = 0xC0FFEE
        for page in (0, 1, 7, 1024):
            marker = ref.read_bytes(seed, 0, 1, 2, 3, page * ref.LOGICAL_PAGE_BYTES, 16)
            self.assertEqual(marker[:4], b"CFX1")
            self.assertEqual(int.from_bytes(marker[4:12], "little"), page)
            self.assertEqual(marker, ref.read_bytes(seed, 0, 1, 2, 3, page * ref.LOGICAL_PAGE_BYTES, 16))
        self.assertEqual(ref.LOGICAL_PAGE_BYTES, 4096)

    def test_offset_and_byte_order_are_observable(self):
        seed = 0x5EED
        data = ref.read_bytes(seed, 3, 8, 11, 2, 0, 64)
        self.assertEqual(len(data), 64)
        self.assertNotEqual(data, ref.read_bytes(seed, 3, 8, 11, 2, 1, 64))
        self.assertEqual(data.hex(), "434658310000000000000000a55a3cc37c75a439057d2f3fe5b479daaaa9dfbcf7c37c2245bece4afa16b98b66f8dce2fe26547a90256b3a57e624e0c3f3d66a")

    def test_canonical_state_excludes_device_canary(self):
        state = ref.canonical_state(seed=41, layers=2, tokens=3, channels=4, bytes_per_cell=8)
        canary = bytes(ref.canary_byte(41, 0, 0, 0, 0, i) for i in range(len(state)))
        other_device = bytes(ref.canary_byte(41, 1, 0, 0, 0, i) for i in range(len(state)))
        self.assertNotEqual(state, canary)
        self.assertNotEqual(canary, other_device)
        self.assertEqual(state, ref.canonical_state(seed=41, layers=2, tokens=3, channels=4, bytes_per_cell=8))

    def test_canonical_known_hash_and_stream_order(self):
        data = b"".join(ref.iter_canonical(0x5EED, 1, 1, 1, 8192, chunk_size=257))
        self.assertEqual(len(data), 8192)
        self.assertEqual(hashlib.sha256(data).hexdigest(), "2eb34e7ed0da08b0012daa8bdb83c2555bf6e4d43d8c9c9b08a8e1c76287a7f3")

    def test_short_cells_keep_seed_and_coordinates_in_canonical_payload(self):
        for bytes_per_cell in (1, 2, 8):
            with self.subTest(bytes_per_cell=bytes_per_cell):
                state = ref.canonical_state(41, 2, 3, 4, bytes_per_cell)
                other_seed = ref.canonical_state(42, 2, 3, 4, bytes_per_cell)
                self.assertNotEqual(state, other_seed)

                # A true cell-order mutation witness: retain each destination byte offset
                # while swapping source coordinates, so only coordinate/stride handling
                # can distinguish this wrong payload from the canonical traversal.
                coordinates = [(layer, token, channel)
                               for layer in range(2) for token in range(3)
                               for channel in range(4)]
                expected = bytearray()
                global_offset = 0
                for layer, token, channel in coordinates:
                    for _ in range(bytes_per_cell):
                        expected.append(ref.byte_at(41, ref.REGION_CANONICAL, layer,
                                                    token, channel, global_offset))
                        global_offset += 1
                self.assertEqual(state, bytes(expected))

                coordinates[1], coordinates[-2] = coordinates[-2], coordinates[1]
                swapped = bytearray()
                global_offset = 0
                for layer, token, channel in coordinates:
                    for _ in range(bytes_per_cell):
                        swapped.append(ref.byte_at(41, ref.REGION_CANONICAL, layer,
                                                   token, channel, global_offset))
                        global_offset += 1
                self.assertNotEqual(state, bytes(swapped))

    def test_canonical_pages_follow_global_offsets_across_cell_boundaries(self):
        # 1200 cells of 7 bytes cross two 4 KiB boundaries inside cells.
        state = ref.canonical_state(73, 2, 2, 300, 7)
        self.assertEqual(len(state), 8400)
        marker = lambda page: (b"CFX1" + page.to_bytes(8, "little")
                               + b"\xA5\x5A\x3C\xC3")
        self.assertEqual(state[0:16], marker(0))
        self.assertEqual(state[4096:4112], marker(1))
        self.assertEqual(state[8192:8208], marker(2))
        # Seven-byte cell starts must not restart the global page marker.
        for cell_start in range(7, len(state), 7):
            self.assertNotEqual(state[cell_start:cell_start + 16],
                                marker(cell_start // ref.LOGICAL_PAGE_BYTES))
        self.assertNotEqual(state[4095:4111], marker(1))

    def test_float_values_are_finite_bounded_and_dyadic(self):
        values = [ref.float_value(123, 2, 8, 31, i, precision="fp16") for i in range(4096)]
        self.assertTrue(all(-8.0 <= x <= 8.0 for x in values))
        self.assertTrue(all(x == 0 or (x * 16).is_integer() for x in values))
        self.assertTrue(any(x < 0 for x in values))
        self.assertTrue(any(x > 0 for x in values))
        self.assertIn(0.0, values)


if __name__ == "__main__":
    unittest.main()
