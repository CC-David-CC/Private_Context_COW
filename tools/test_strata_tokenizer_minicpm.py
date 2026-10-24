"""MiniCPM segmentation differs from Qwen even when both round-trip."""
import unittest
from tools.strata_tokenizer import Tokenizer,BYTE_TO_UNICODE

class MiniCPMTokenizerTests(unittest.TestCase):
 def tokenizer(self,pre):
  base=[BYTE_TO_UNICODE[b] for b in range(256)]
  merges=['1 7','2 6','1 0','10 0','_ j']
  return Tokenizer(base+[m.replace(' ','') for m in merges],merges,pre=pre)
 def test_numbers_merge_within_three_digit_groups_only(self):
  mini=self.tokenizer('minicpm5');qwen=self.tokenizer('qwen35')
  self.assertEqual(mini.encode('17 26 1001'),[256,32,257,32,259,49])
  self.assertEqual(qwen.encode('17 26 1001'),[49,55,32,50,54,32,49,48,48,49])
  self.assertEqual(mini.decode(mini.encode('17 26 1001')),'17 26 1001')
  self.assertEqual(mini._pieces('1234567890'),['123','456','789','0'])
 def test_combining_mark_boundary_does_not_merge_underscore_j(self):
  mini=self.tokenizer('minicpm5');qwen=self.tokenizer('qwen35')
  self.assertEqual(mini._pieces('e\u0301_j'),['e','\u0301_','j'])
  self.assertNotEqual(mini.encode('e\u0301_j'),qwen.encode('e\u0301_j'))
  for t in [mini,qwen]:self.assertEqual(t.decode(t.encode('e\u0301_j')),'e\u0301_j')

 def test_numeric_split_preserves_trailing_whitespace_before_numbers(self):
  mini=self.tokenizer('minicpm5')
  self.assertEqual(mini._pieces('  17\n\n26'),['  ','17','\n\n','26'])
