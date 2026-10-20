"""Asymmetric pixel oracles for all eight EXIF orientations; CPU only."""
import hashlib,io,unittest
from PIL import Image
from serve.cow_vision import CowImageNormalizer
from context_time_travel.store import InvalidOperation
# Independent row-major permutations of a 3x2 raster. No production/Pillow
# transpose helper participates in the expected result.
ORDER={1:(3,2,(0,1,2,3,4,5)),2:(3,2,(2,1,0,5,4,3)),
 3:(3,2,(5,4,3,2,1,0)),4:(3,2,(3,4,5,0,1,2)),
 5:(2,3,(0,3,1,4,2,5)),6:(2,3,(3,0,4,1,5,2)),
 7:(2,3,(5,2,4,1,3,0)),8:(2,3,(2,5,1,4,0,3))}
class CowVisionOrientationTests(unittest.TestCase):
 def check_format(self,format):
  normalizer=CowImageNormalizer()
  for orientation,(width,height,order) in ORDER.items():
   with self.subTest(format=format,orientation=orientation):
    image=Image.new('RGB',(3,2));image.putdata([(11,27,189),(220,9,41),(17,203,83),(251,130,6),(67,31,244),(5,160,191)])
    exif=Image.Exif();exif[274]=orientation;stream=io.BytesIO();image.save(stream,format=format,exif=exif,**({'quality':100,'subsampling':0} if format=='JPEG' else {}));raw=stream.getvalue()
    with Image.open(io.BytesIO(raw)) as decoded:decoded.load();pixels=list(decoded.getdata())
    result=normalizer.normalize(raw);repeat=normalizer.normalize(raw)
    self.assertEqual((result.width,result.height),(width,height))
    self.assertEqual(result.original_sha256,hashlib.sha256(raw).hexdigest())
    self.assertEqual(result.normalized_sha256,hashlib.sha256(result.payload).hexdigest());self.assertEqual(result,repeat)
    with Image.open(io.BytesIO(result.payload)) as output:
     self.assertEqual(output.size,(width,height));self.assertEqual(list(output.convert('RGB').getdata()),[pixels[i] for i in order])
     self.assertIn(output.getexif().get(274),(None,1))
    if orientation==1:self.assertEqual(result.payload,raw)
 def test_all_png_orientations_preserve_exact_asymmetric_pixels(self):self.check_format('PNG')
 def test_all_jpeg_orientations_preserve_exact_decoded_pixels(self):self.check_format('JPEG')
 def test_invalid_orientation_rejects_before_producing_a_normalized_identity(self):
  image=Image.new('RGB',(3,2));exif=Image.Exif();exif[274]=9;stream=io.BytesIO();image.save(stream,format='PNG',exif=exif)
  with self.assertRaisesRegex(InvalidOperation,'orientation'):CowImageNormalizer().normalize(stream.getvalue())
if __name__=='__main__':unittest.main()
