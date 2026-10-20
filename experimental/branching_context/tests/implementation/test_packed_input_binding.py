"""Fresh packed bindings versus independent unchanged scalar equations."""
import random,struct,unittest
from unittest.mock import patch
from context_time_travel.input_spans import InputHistory
from context_time_travel.store import InvalidOperation
from context_time_travel.vision_execution import input_binding
from test_vision_execution import spans

class PackedBindingTests(unittest.TestCase):
 def packed(self,h):
  from context_time_travel.packed_input_binding import fresh_input_binding
  return fresh_input_binding(h)
 def test_empty_text_images_and_all_small_prefixes_match_scalar_equations(self):
  for source in [[],[dict(kind='text',tokens=[0,2147483647])],spans(),spans()[:-1],spans()[1:]]:
   h=InputHistory(source)
   self.assertEqual(self.packed(h),input_binding(h))
   for end in range(len(h)+1):
    try:p=h.prefix(end)
    except InvalidOperation:continue
    self.assertEqual(self.packed(p),input_binding(p))
 def test_seeded_mixed_grids_offsets_and_append_match_reference(self):
  rng=random.Random(20261020)
  for _ in range(80):
   nx,ny=rng.randint(1,8),rng.randint(1,8)
   source=spans(grid_width=nx,grid_height=ny,rows=nx*ny,embedding_bytes=20+nx*ny*4*4)
   source[0]['tokens']=[rng.randrange(200000) for _ in range(rng.randrange(32))]
   source[-1]['tokens']=[rng.randrange(200000) for _ in range(rng.randrange(32))]
   h=InputHistory(source)
   for i in range(3):
    self.assertEqual(self.packed(h),input_binding(h));h=h+(rng.randrange(200000),)
 def test_large_text_and_image_boundaries_have_exact_hashes(self):
  for count in [8192,65536,131072]:
   h=InputHistory([dict(kind='text',tokens=[(i*7919+17)%200000 for i in range(count)])]+spans()[1:])
   self.assertEqual(self.packed(h),input_binding(h))
 def test_position_payload_is_exact_little_endian_scalar_bytes(self):
  from context_time_travel.packed_input_binding import _position_bytes
  for source in [[],[dict(kind='text',tokens=[1,2,3])],spans(),spans(grid_width=2,grid_height=3)]:
   cursor=0;points=[]
   for s in source:
    if s['kind']=='text':
     for _ in s['tokens']:points.append((cursor,cursor,cursor));cursor+=1
    else:
     for y in range(s['grid_height']):
      for x in range(s['grid_width']):points.append((cursor,cursor+y,cursor+x))
     cursor+=max(s['grid_width'],s['grid_height'])
   self.assertEqual(_position_bytes(source),(cursor,b''.join(struct.pack('<iii',*p) for p in points)))
 def test_signed_word_bytes_and_out_of_range_positions(self):
  from context_time_travel.packed_input_binding import _i32_bytes,_text_positions
  values=[-2147483648,-1,0,1,2147483647]
  self.assertEqual(_i32_bytes(values),struct.pack('<5i',*values))
  self.assertEqual(_text_positions(2147483647,1),struct.pack('<3i',2147483647,2147483647,2147483647))
  for start,count in [(2147483647,2),(-2147483649,1)]:
   with self.assertRaises(struct.error):_text_positions(start,count)
 def test_mutable_subclass_is_not_given_owned_fast_path(self):
  from context_time_travel import packed_input_binding as module
  class Mutable(InputHistory):
   @property
   def spans(self):return self.dynamic
  h=Mutable(spans());object.__setattr__(h,'dynamic',spans())
  with patch.object(module,'reference_input_binding',wraps=input_binding) as ref:
   self.assertEqual(self.packed(h),input_binding(h));h.dynamic[-1]['tokens']=[9]
   self.assertEqual(self.packed(h),input_binding(h));self.assertEqual(ref.call_count,2)
 def test_altered_encoding_uses_full_reference_and_does_not_reuse_old_views(self):
  h=InputHistory(spans());self.packed(h);other=InputHistory(spans(embedding_sha256='e'*64))
  object.__setattr__(h,'_encoded',other._encoded)
  self.assertEqual(self.packed(h),input_binding(other))
 def test_different_image_hashes_and_grid_layouts_remain_distinct(self):
  a=InputHistory(spans());b=InputHistory(spans(embedding_sha256='e'*64));c=InputHistory(spans(grid_width=2,grid_height=3))
  pa,pb,pc=self.packed(a),self.packed(b),self.packed(c)
  self.assertNotEqual(pa['native_input_sha256'],pb['native_input_sha256'])
  self.assertEqual(pa['position_sha256'],pb['position_sha256'])
  self.assertNotEqual(pa['position_sha256'],pc['position_sha256'])

class PackedProjectionWorkTests(unittest.TestCase):
 def test_full_audit_constructs_identical_payloads_without_scalar_per_cell_pack(self):
  from context_time_travel import input_schedule_projection as module
  h=InputHistory([dict(kind='text',tokens=list(range(8192)))]+spans()[1:])
  original=struct.pack;calls=[]
  def counted(fmt,*args):
   if fmt in ('<i','<iii'):calls.append(fmt)
   return original(fmt,*args)
  with patch.object(module.struct,'pack',side_effect=counted):
   projection=module._InputScheduleProjection(h,[len(h)])
  self.assertEqual(calls,[], 'full audit still packs native/position words once per consumed cell')
  self.assertEqual(bytes(projection._native),struct.pack('<'+'i'*len(h),*h.native_tokens))
  from context_time_travel.packed_input_binding import _position_bytes
  self.assertEqual(bytes(projection._positions),_position_bytes(h.spans)[1])
  self.assertEqual(projection.binding(len(h)),dict(inputs_sha256=h.digest,**input_binding(h)))
