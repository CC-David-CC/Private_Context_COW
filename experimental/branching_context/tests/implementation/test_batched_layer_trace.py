import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('batch_trace',
    Path(__file__).resolve().parents[2]/'scripts/compare_batched_layer_trace.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class BatchedLayerTraceTests(unittest.TestCase):
    def write(self,path,rows):
        with path.open('wb') as f:
            for layer,half,values in rows:
                f.write(struct.pack('<4q',132,layer,half,len(values)))
                f.write(struct.pack('<'+'f'*len(values),*values))

    def test_first_difference_survives_later_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            self.write(a,[(0,0,[1,2]),(0,1,[3,4]),(1,0,[5,6]),(1,1,[7,8])])
            self.write(b,[(0,0,[1,2]),(0,1,[3,9]),(1,0,[5,6]),(1,1,[7,8])])
            result=probe.compare(a,b,132,2)
            self.assertEqual(result['first_difference']['layer'],0)
            self.assertEqual(result['first_difference']['half'],1)
            self.assertEqual(result['first_difference']['max_absolute_error'],5)

    def test_missing_or_nonfinite_readback_fails(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            self.write(a,[(0,0,[1,2]),(0,1,[3,4])])
            for rows in ([(0,0,[1,2])],[(0,0,[1,2]),(0,1,[3,float('nan')])]):
                self.write(b,rows)
                with self.assertRaises(ValueError): probe.compare(a,b,132,1)


if __name__=='__main__': unittest.main()
