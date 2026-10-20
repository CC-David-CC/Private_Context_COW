import unittest
from intervals import merge_intervals
class IntervalTests(unittest.TestCase):
 def test_empty(self):self.assertEqual(merge_intervals([]),[])
 def test_overlap(self):self.assertEqual(merge_intervals([(1,4),(2,6)]),[(1,6)])
 def test_nested(self):self.assertEqual(merge_intervals([(1,8),(2,3)]),[(1,8)])
 def test_touch(self):self.assertEqual(merge_intervals([(1,3),(3,5)]),[(1,5)])
 def test_gaps(self):self.assertEqual(merge_intervals([(1,2),(3,4)]),[(1,2),(3,4)])
 def test_unsorted(self):self.assertEqual(merge_intervals([(6,8),(1,3),(2,5)]),[(1,5),(6,8)])
 def test_negative(self):self.assertEqual(merge_intervals([(-4,-1),(-2,2)]),[(-4,2)])
 def test_points(self):self.assertEqual(merge_intervals([(2,2),(2,3)]),[(2,3)])
 def test_no_mutation(self):
  data=[[3,5],[1,4]];before=[x[:] for x in data];self.assertEqual(merge_intervals(data),[(1,5)]);self.assertEqual(data,before)
 def test_reversed(self):
  with self.assertRaises(ValueError):merge_intervals([(3,1)])
 def test_shape(self):
  with self.assertRaises(ValueError):merge_intervals([(1,2,3)])
 def test_type(self):
  for data in [[(1,'3')],[(True,2)],[(1.5,3)]]:
   with self.assertRaises(TypeError):merge_intervals(data)
