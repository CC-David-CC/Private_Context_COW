import copy,hashlib,struct,unittest
from context_time_travel.input_spans import InputHistory
from context_time_travel.qwen35_mixed_binding import native_binding

class Qwen35MixedBindingTests(unittest.TestCase):
 def image(self,nx=2,ny=2):
  return dict(kind='image',original_sha256='12'*32,normalized_sha256='34'*32,encoder_profile_id='56'*32,embedding_sha256='78'*32,embedding_bytes=20+nx*ny*4096*4,rows=nx*ny,width=4096,grid_width=nx,grid_height=ny,pad_token_id=17)
 def history(self,image=None):return InputHistory([dict(kind='text',tokens=[7,8]),image or self.image(),dict(kind='text',tokens=[9,10])])
 def test_native_grid_positions_and_logical_cells_are_distinct(self):
  v=native_binding(self.history());self.assertEqual((v['consumed'],v['rotary_cursor']),(8,6))
  positions=[(0,0,0),(1,1,1),(2,2,2),(2,2,3),(2,3,2),(2,3,3),(4,4,4),(5,5,5)]
  expected=hashlib.sha256(b'COW-QWEN35-POS-v2\0'+b''.join(struct.pack('<iii',*p) for p in positions)).hexdigest()
  self.assertEqual(v['positions_sha256'],expected)
 def test_transposed_grid_cannot_restore_as_same_layout(self):
  a=native_binding(self.history(self.image(2,3)));b=native_binding(self.history(self.image(3,2)))
  self.assertEqual(a['consumed'],b['consumed']);self.assertNotEqual(a['positions_sha256'],b['positions_sha256']);self.assertNotEqual(a['cells_sha256'],b['cells_sha256'])
 def test_changed_token_changes_input_proof_without_changing_positions(self):
  h=self.history();spans=h.spans;spans[-1]['tokens'][-1]=11;a=native_binding(h);b=native_binding(InputHistory(spans))
  self.assertNotEqual(a['cells_sha256'],b['cells_sha256']);self.assertEqual(a['positions_sha256'],b['positions_sha256'])
 def test_native_proof_is_not_canonical_provenance_identity(self):
  h=self.history();spans=h.spans;spans[1]['original_sha256']='ab'*32;changed=InputHistory(spans)
  self.assertNotEqual(h.digest,changed.digest);self.assertEqual(native_binding(h),native_binding(changed))
