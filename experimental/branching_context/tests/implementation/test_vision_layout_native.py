"""VC007: compiled native layout versus an independent CPU position oracle.

These tests do not execute a GPU or certify image model continuation.
"""
import hashlib
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
PAD = 99


def placement(start=2, nx=3, ny=2, **changes):
    value = dict(start=start, rows=nx*ny, grid_width=nx, grid_height=ny,
                 width=4, pad_token=PAD, embedding_sha256='d'*64)
    value.update(changes)
    return value


def oracle(ids, images, capacity):
    """Construct whole text runs and image grids, independently of the C++ loop."""
    positions, references = [], []
    cell, rotary = 0, 0
    for index, image in enumerate(images):
        text_count = image['start'] - cell
        positions.extend((p, p, p) for p in range(rotary, rotary+text_count))
        references.extend([(-1, -1)] * text_count)
        rotary += text_count
        positions.extend((rotary, rotary+y, rotary+x)
                         for y in range(image['grid_height'])
                         for x in range(image['grid_width']))
        references.extend((index, row) for row in range(image['rows']))
        cell = image['start'] + image['rows']
        rotary += max(image['grid_width'], image['grid_height'])
    tail = len(ids) - cell
    positions.extend((p, p, p) for p in range(rotary, rotary+tail))
    references.extend([(-1, -1)] * tail)
    cursor = rotary + tail
    consumed_positions = [p for point in positions for p in point]
    payload = b'COW-MROPE-v1\0' + struct.pack('<QQ', len(ids), cursor)
    payload += struct.pack('<'+'i'*len(consumed_positions), *consumed_positions)
    positions.extend((p, p, p) for p in range(cursor, cursor+capacity-len(ids)))
    return dict(consumed=len(ids), cursor=cursor, image_rows=sum(p['rows'] for p in images),
                digest=hashlib.sha256(payload).hexdigest(),
                positions=positions, rows=references)


class VisionLayoutNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++') or shutil.which('c++')
        if compiler is None:
            raise RuntimeError('The native layout tests require a real C++ compiler')
        cls.temp = tempfile.TemporaryDirectory(prefix='cow-vision-layout-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.exe = Path(cls.temp.name) / 'layout'
        built = subprocess.run([compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I', str(ROOT/'include'), str(ROOT/'experimental/branching_context/native/vision_layout_probe.cpp'),
            '-o', str(cls.exe)], capture_output=True, text=True, timeout=60)
        if built.returncode:
            raise AssertionError('Native layout compilation failed:\n' + built.stderr)

    def case(self, ids, images=(), capacity=32, vocab=128, width=4, max_rows=1024):
        fields = [capacity, vocab, width, max_rows, len(ids), len(images), *ids]
        for item in images:
            fields.extend(item[name] for name in ('start', 'rows', 'grid_width', 'grid_height',
                'width', 'pad_token', 'embedding_sha256'))
        proc = subprocess.run([str(self.exe)], input=' '.join(map(str, fields))+'\n',
                              capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        fields = proc.stdout.strip().split()
        if fields[0] == 'ERR':
            return ' '.join(fields[1:])
        self.assertEqual(fields[0], 'OK', proc.stdout)
        self.assertEqual(len(fields), 5+3*capacity+2*len(ids))
        data = list(map(int, fields[5:]))
        return dict(consumed=int(fields[1]), cursor=int(fields[2]), image_rows=int(fields[3]),
                    digest=fields[4], positions=list(zip(*[iter(data[:3*capacity])]*3)),
                    rows=list(zip(*[iter(data[3*capacity:])]*2)))

    def good(self, ids, images=(), capacity=32):
        actual = self.case(ids, images, capacity)
        self.assertEqual(actual, oracle(ids, images, capacity))
        return actual

    def test_vc007_text_empty_and_literal_pad_ids_keep_identity_positions(self):
        for ids in ([], [1, 2, 3], [PAD]*9):
            with self.subTest(ids=ids):
                plan = self.good(ids)
                self.assertEqual(plan['positions'], [(p, p, p) for p in range(32)])
                self.assertEqual(plan['image_rows'], 0)

    def test_vc007_rectangular_image_has_explicit_expected_coordinates(self):
        plan = self.good([1, 2] + [PAD]*6 + [3, 4], [placement()])
        self.assertEqual(plan['positions'][:12], [
            (0, 0, 0), (1, 1, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4),
            (2, 3, 2), (2, 3, 3), (2, 3, 4), (5, 5, 5), (6, 6, 6),
            (7, 7, 7), (8, 8, 8)])
        self.assertEqual((plan['consumed'], plan['cursor']), (10, 7))

    def test_vc007_two_images_keep_separate_rows_and_following_cursor(self):
        ids = [1, 2] + [PAD]*6 + [3] + [PAD]*8 + [4, 5]
        plan = self.good(ids, [placement(), placement(start=9, nx=2, ny=4)])
        self.assertEqual(plan['cursor'], 12)
        self.assertEqual(plan['rows'][9:17], [(1, row) for row in range(8)])

    def test_vc007_same_row_count_different_grid_changes_positions(self):
        ids = [1, 2]+[PAD]*6+[3]
        a = self.good(ids, [placement()])
        b = self.good(ids, [placement(nx=6, ny=1)])
        self.assertNotEqual(a['digest'], b['digest'])
        self.assertNotEqual(a['cursor'], b['cursor'])

    def test_vc007_prefix_coordinates_survive_suffix_and_batch_row_slices(self):
        ids = [1, 2]+[PAD]*6+[3]
        prefix = self.good(ids, [placement()])
        longer = self.good(ids+[4, 5, 6], [placement()])
        self.assertEqual(prefix['positions'][:len(ids)], longer['positions'][:len(ids)])
        # Physical batches can split images; the absolute row mapping remains correct.
        chunks = [longer['rows'][start:start+4] for start in range(0, len(ids)+3, 4)]
        self.assertEqual([row for chunk in chunks for row in chunk], longer['rows'])
        self.assertEqual(chunks[0][-2:], [(0, 0), (0, 1)])
        self.assertEqual(chunks[1], [(0, 2), (0, 3), (0, 4), (0, 5)])

    def test_vc007_layout_reconstruction_needs_metadata_not_embedding_files(self):
        ids = [1, 2]+[PAD]*6+[3]
        a = self.good(ids, [placement()])
        b = self.good(ids, [placement(embedding_sha256='e'*64)])
        self.assertEqual(a, b)  # This digest binds positions; input identity binds pixels separately.
        text = self.good([7]*len(ids))
        self.assertNotEqual(text['digest'], a['digest'])

    def test_vc007_rejects_bad_grid_width_row_cap_and_digest(self):
        changes = [dict(rows=5), dict(grid_width=0), dict(grid_height=-1), dict(width=5),
                   dict(rows=2**62, grid_width=2**31, grid_height=2**31),
                   dict(embedding_sha256='d'*63), dict(embedding_sha256='D'*64),
                   dict(embedding_sha256='x'*64)]
        for change in changes:
            with self.subTest(change=change):
                self.assertIsInstance(self.case([1, 2]+[PAD]*6+[3], [placement(**change)]), str)
        self.assertIsInstance(self.case([1, 2]+[PAD]*6+[3], [placement()], max_rows=5), str)

    def test_vc007_rejects_unsorted_overlapping_and_out_of_range_images(self):
        ids = [PAD]*20+[1]
        for images in ([placement(start=-1)], [placement(start=20)],
                       [placement(start=0), placement(start=2)],
                       [placement(start=8), placement(start=0)]):
            with self.subTest(images=images):
                self.assertIsInstance(self.case(ids, images), str)

    def test_vc007_rejects_non_pad_cells_without_recognizing_text_as_an_image(self):
        ids = [1, 2]+[PAD]*6+[3]
        ids[5] = 3
        self.assertIsInstance(self.case(ids, [placement()]), str)
        self.good(ids)

    def test_vc007_rejects_public_boundary_inside_or_at_end_of_image(self):
        for count in (3, 5, 6):
            with self.subTest(count=count):
                self.assertIsInstance(self.case([1, 2]+[PAD]*count, [placement()]), str)

    def test_vc007_rejects_capacity_and_token_errors_before_large_allocation(self):
        for capacity in (0, -1, 2, 2**31):
            with self.subTest(capacity=capacity):
                self.assertIsInstance(self.case([1, 2, 3], capacity=capacity), str)
        for ids in ([-1], [128]):
            self.assertIsInstance(self.case(ids), str)
        for options in (dict(vocab=0), dict(width=0), dict(max_rows=0)):
            self.assertIsInstance(self.case([1], **options), str)

    def test_vc007_seeded_mixed_histories_match_independent_oracle(self):
        rng = random.Random(7319)
        for case in range(64):
            ids, images = [], []
            for _ in range(rng.randrange(5)):
                ids.extend(rng.randrange(128) for _ in range(rng.randrange(1, 5)))
                nx, ny = rng.randrange(1, 9), rng.randrange(1, 9)
                images.append(placement(start=len(ids), nx=nx, ny=ny))
                ids.extend([PAD]*(nx*ny))
            ids.extend(rng.randrange(128) for _ in range(rng.randrange(1, 5)))
            with self.subTest(case=case):
                self.good(ids, images, len(ids)+8)


if __name__ == '__main__':
    unittest.main()
