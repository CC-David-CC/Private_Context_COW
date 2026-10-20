"""Campaign fixture integrity and independent multi-image coordinate checks."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from vision_fixture import fixture_parts, configure_prefill, reference_positions, boundary_coverage


class VisionFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = []
        for index, (nx, ny) in enumerate(((3, 2), (2, 2))):
            original = self.root/f'{index}.png'; original.write_bytes(bytes([index, 10, 30]))
            sve = self.root/f'{index}.sve'
            sve.write_bytes(struct.pack('<5i', 0x31455653, nx*ny, nx, ny, 4)+bytes(nx*ny*16))
            self.images.append(dict(original=self.fp(original), embedding=self.fp(sve)))
        self.tokens = [1, 99, 99, 99, 99, 99, 99, 2, 99, 99, 99, 99, 3]
        self.fixture = self.root/'result.json'
        self.fixture.write_text(json.dumps(dict(input_tokens=self.tokens, images=self.images)))
        self.manifest = self.root/'input.manifest'
        self.manifest.write_text('COW_VISION_INPUT_V1 0 13 2\n'+' '.join(map(str,self.tokens))+'\n'+
            '\n'.join(f'{start} {nx*ny} {nx} {ny} 4 99 {self.images[i]["embedding"]["sha256"]} '+
                json.dumps(self.images[i]['embedding']['path']) for i,(start,nx,ny) in enumerate(((1,3,2),(8,2,2))))+'\n')

    def fp(self, path):
        return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_two_images_preserve_order_and_the_complete_input(self):
        parts = fixture_parts(self.fixture, [Path(x['original']['path']) for x in self.images])
        self.assertEqual([x['start'] for x in parts], [1,8])
        self.assertEqual([x['rows'] for x in parts], [6,4])
        self.assertEqual(parts[1]['embedding']['sha256'], self.images[1]['embedding']['sha256'])

    def test_changed_original_embedding_or_token_transport_is_rejected(self):
        for target in (self.root/'0.png',self.root/'1.sve',self.manifest):
            before = target.read_bytes()
            try:
                target.write_bytes(before+b'corrupt')
                with self.subTest(target=target.name), self.assertRaises(ValueError):
                    fixture_parts(self.fixture, [Path(x['original']['path']) for x in self.images])
            finally: target.write_bytes(before)

    def test_wrong_image_order_and_missing_original_are_rejected(self):
        paths = [Path(x['original']['path']) for x in self.images]
        for values in (paths[::-1],paths[:1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fixture_parts(self.fixture, values)

    def test_legacy_one_image_fixture_remains_supported(self):
        payload = dict(input_tokens=[1]+[99]*6+[2],embedding=self.images[0]['embedding'])
        self.fixture.write_text(json.dumps(payload))
        line = self.manifest.read_text().splitlines()[2]
        self.manifest.write_text('COW_VISION_INPUT_V1 0 8 1\n1 99 99 99 99 99 99 2\n'+line+'\n')
        parts = fixture_parts(self.fixture,[self.root/'0.png'])
        self.assertEqual(len(parts),1);self.assertEqual(parts[0]['rows'],6)

    def test_reference_positions_use_each_grid_and_shift_following_text(self):
        images = [dict(start=1,rows=6,grid_width=3,grid_height=2),
                  dict(start=8,rows=4,grid_width=2,grid_height=2)]
        positions = reference_positions(13,images,16)
        self.assertEqual(positions[:8],[(0,0,0),(1,1,1),(1,1,2),(1,1,3),
            (1,2,1),(1,2,2),(1,2,3),(4,4,4)])
        self.assertEqual(positions[8:13],[(5,5,5),(5,5,6),(5,6,5),(5,6,6),(7,7,7)])
        self.assertEqual(positions[-1],(10,10,10))

    def test_prefill_profile_updates_both_actual_commands_without_mutating_input(self):
        profile = dict(execution=dict(capacity=98304,prefill_chunk=8192,
            sut_command=['sut','--prefill','8192'],oracle_command=['oracle','--prefill','8192']))
        before = copy.deepcopy(profile); changed = configure_prefill(profile,512)
        self.assertEqual(profile,before)
        self.assertEqual(changed['execution']['prefill_chunk'],512)
        self.assertEqual(changed['execution']['sut_command'][-1],'512')
        self.assertEqual(changed['execution']['oracle_command'][-1],'512')
        for value in (0,True,98305):
            with self.subTest(value=value),self.assertRaises(ValueError):configure_prefill(profile,value)

    def test_boundary_claims_require_observed_batches_and_physical_geometry(self):
        images = [dict(start=1030,rows=1024)]
        audit = dict(consumed=2097,qsa_page_cells=4,index_block_cells=4,
            regions=[dict(kind='k',pool=0,pool_offset=0,bytes=2097152,bytes_per_cell=1024),
                     dict(kind='k',pool=0,pool_offset=2097152,bytes=2097152,bytes_per_cell=1024)])
        coverage = boundary_coverage(images,[[0,512],[512,512],[1024,512],[1536,512],[2048,49]],audit)
        self.assertEqual(coverage['batch_boundaries_inside_images'],[1536,2048])
        self.assertEqual(coverage['physical_k_boundaries_inside_images'],[2048])
        self.assertEqual(coverage['partial_qsa_tail_cells'],1)

    def test_missing_physical_crossing_cannot_be_reported_as_one(self):
        audit = dict(consumed=13,qsa_page_cells=4,index_block_cells=4,
            regions=[dict(kind='k',pool=0,pool_offset=0,bytes=2097152,bytes_per_cell=1024)])
        result = boundary_coverage([dict(start=1,rows=6)],[[0,13]],audit)
        self.assertEqual(result['physical_k_boundaries_inside_images'],[])
        self.assertEqual(result['batch_boundaries_inside_images'],[])


if __name__ == '__main__':unittest.main()
