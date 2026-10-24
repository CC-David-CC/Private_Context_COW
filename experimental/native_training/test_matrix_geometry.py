import tempfile,unittest,math
from pathlib import Path
import numpy as np
from .matrix_geometry import spectrum_metrics,extract,measure,sha,read_subset
from .make_llama_fixture import make_fixture

class MatrixGeometryTests(unittest.TestCase):
 def test_known_spectra_scale_invariance_and_zero(self):
  a=spectrum_metrics(np.eye(3));b=spectrum_metrics(np.eye(3)*.001)
  for key in ('stable_rank','entropy_effective_rank','energy_participation_rank'):
   self.assertAlmostEqual(a[key],3);self.assertAlmostEqual(a[key],b[key])
  c=spectrum_metrics(np.diag([3.,0.,0.]));z=spectrum_metrics(np.zeros((3,2)))
  for key in ('stable_rank','entropy_effective_rank','energy_participation_rank'):
   self.assertAlmostEqual(c[key],1);self.assertEqual(z[key],0)
  for x in [np.array([1.]),np.array([[float('nan')]])]:
   with self.assertRaises(ValueError):spectrum_metrics(x)
 def test_actual_gguf_extraction_and_zero_accumulated_change(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);model=make_fixture(r/'base.gguf');subset=r/'subset'
   v=extract(model,sha(model),['blk.0.attn_q.weight'],subset);self.assertEqual(len(v['matrices']),1)
   out=measure(subset,subset,r/'result.json');self.assertEqual(out['matrices'][0]['change_from_base']['frobenius_norm'],0);self.assertFalse(out['promoted'])
   (subset/'000.npy').write_bytes(b'changed')
   with self.assertRaises(ValueError):read_subset(subset)
 def test_wrong_model_hash_never_exports(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);model=make_fixture(r/'base.gguf')
   with self.assertRaises(ValueError):extract(model,'0'*64,['blk.0.attn_q.weight'],r/'subset')
   self.assertFalse((r/'subset').exists())
if __name__=='__main__':unittest.main()
