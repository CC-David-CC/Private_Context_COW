"""Independent pixel/format fixtures for the real image CLI campaign; CPU only."""
import importlib.util,io,json,tempfile,unittest
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
class VisionFormatFixtureTests(unittest.TestCase):
 def module(self):
  spec=importlib.util.spec_from_file_location('vision_format_fixture',ROOT/'scripts/vision_format_fixture.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_all_sixteen_cases_have_independent_exact_pixel_witnesses(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as d:
   rows=m.prepare(Path(d))
   self.assertEqual(len(rows),16);self.assertEqual({(x['format'],x['orientation']) for x in rows},{(f,o) for f in ('PNG','JPEG') for o in range(1,9)})
   self.assertEqual(len({x['original_sha256'] for x in rows}),16)
   for row in rows:
    with self.subTest(format=row['format'],orientation=row['orientation']):
     data=Path(row['original']).read_bytes();self.assertLess(len(data),4096)
     with Image.open(io.BytesIO(data)) as image:
      self.assertEqual(image.getexif()[274],row['orientation']);image.load();pixels=list(image.convert('RGB').getdata())
     # Literal independent index selection for horizontal/vertical/transpose controls.
     if row['orientation']==1:expected=pixels;size=[3,2]
     elif row['orientation']==2:expected=[pixels[i] for i in [2,1,0,5,4,3]];size=[3,2]
     elif row['orientation']==6:expected=[pixels[i] for i in [3,0,4,1,5,2]];size=[2,3]
     else:continue
     self.assertEqual(row['expected_size'],size);self.assertEqual(row['expected_pixels'],[list(x) for x in expected])
 def test_oracle_rejects_corruption_and_does_not_use_production_transpose(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as d:
   row=m.prepare(Path(d))[5]
   # The expected raster is fixed before importing a production normalizer.
   with Image.open(row['original']) as image:image.load();payload=io.BytesIO();image.save(payload,format='PNG')
   wrong=Image.new('RGB',(row['expected_size'][0],row['expected_size'][1]),(7,7,7));payload=io.BytesIO();wrong.save(payload,format='PNG')
   with self.assertRaisesRegex(ValueError,'pixel'):m.check_normalized(row,payload.getvalue())
   import inspect
   text=inspect.getsource(m)
   self.assertNotIn('ImageOps',text);self.assertNotIn('exif_transpose',text);self.assertNotIn('serve.cow_vision',text)
 def test_normalizer_output_matches_all_literal_oracles(self):
  m=self.module();from serve.cow_vision import CowImageNormalizer
  normalizer=CowImageNormalizer()
  with tempfile.TemporaryDirectory() as d:
   for row in m.prepare(Path(d)):
    with self.subTest(format=row['format'],orientation=row['orientation']):
     got=normalizer.normalize(Path(row['original']).read_bytes());m.check_normalized(row,got.payload)
     self.assertEqual([got.width,got.height],row['expected_size'])
 def test_matrix_adds_both_ordered_two_image_controls(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as d:
   rows=m.prepare(Path(d));plan=m.case_plan(rows)
   self.assertEqual(len(plan),18);self.assertEqual([x['images'] for x in plan[-2:]],[[rows[0]['name'],rows[13]['name']],[rows[13]['name'],rows[0]['name']]])
   self.assertEqual(len({x['name'] for x in plan}),18)
   with self.assertRaises(ValueError):m.case_plan(rows[:-1])
   with self.assertRaises(ValueError):m.case_plan(rows+[rows[0]])
 def test_fixture_output_is_reproducible_and_refuses_overwrite(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   x=m.prepare(Path(a));y=m.prepare(Path(b));self.assertEqual([r['original_sha256'] for r in x],[r['original_sha256'] for r in y])
   with self.assertRaises(FileExistsError):m.prepare(Path(a))
if __name__=='__main__':unittest.main()
