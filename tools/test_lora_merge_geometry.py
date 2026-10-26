"""Rank-one GGUF geometry recovery must not admit ambiguous/malformed pairs."""
import unittest
import numpy as np
from tools.test_real_lora_merge_weights import adapter_matrices

class AdapterGeometryTests(unittest.TestCase):
    def test_rank_one_collapsed_a_matches_independent_outer_product(self):
        a=np.array([2.,-3.,5.]);b=np.array([[7.],[-2.]])
        left,right=adapter_matrices(a,b,(2,3))
        self.assertEqual(left.shape,(1,3))
        np.testing.assert_array_equal(right@left,np.array([[14.,-21.,35.],[-4.,6.,-10.]]))

    def test_rank_two_geometry_is_preserved(self):
        a=np.array([[1.,2.,3.],[4.,5.,6.]]);b=np.array([[7.,8.],[9.,10.]])
        left,right=adapter_matrices(a,b,(2,3))
        self.assertIs(left,a);self.assertIs(right,b)
        np.testing.assert_array_equal(right@left,np.array([[39.,54.,69.],[49.,68.,87.]]))

    def test_explicit_rank_one_is_preserved(self):
        a=np.ones((1,3));b=np.ones((2,1))
        left,right=adapter_matrices(a,b,(2,3));self.assertIs(left,a);self.assertIs(right,b)

    def test_malformed_or_ambiguous_pairs_refuse(self):
        cases=[(np.ones(3),np.ones((2,2)),(2,3)),
               (np.ones(4),np.ones((2,1)),(2,3)),
               (np.ones(3),np.ones((3,1)),(2,3)),
               (np.ones(3),np.ones(2),(2,3)),
               (np.ones((1,3,1)),np.ones((2,1)),(2,3)),
               (np.ones((2,3)),np.ones((2,1)),(2,3)),
               (np.ones(3),np.ones((2,1)),(3,)),
               (np.ones((1,3)),np.ones((2,1)),(2,4))]
        for a,b,shape in cases:
            with self.subTest(a=a.shape,b=b.shape,weight=shape):
                with self.assertRaises(ValueError):adapter_matrices(a,b,shape)

if __name__=='__main__':unittest.main()
