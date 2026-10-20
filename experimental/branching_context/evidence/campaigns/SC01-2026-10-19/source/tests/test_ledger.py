import unittest
from ledger import summarize
class LedgerTests(unittest.TestCase):
 def test_repeated_names_add(self):
  self.assertEqual(summarize(["oak:2","oak:3","ash:0"]),{"oak":5,"ash":0})
 def test_trim(self):
  self.assertEqual(summarize(["  oak : 2  "]),{"oak":2})
 def test_comments_and_empty(self):
  self.assertEqual(summarize(["","  "," # note","ash:1"]),{"ash":1})
 def test_unicode_name(self):
  self.assertEqual(summarize(["café:4","café:1"]),{"café":5})
 def test_empty_input(self):
  self.assertEqual(summarize([]),{})
 def test_negative_count(self):
  with self.assertRaises(ValueError):summarize(["oak:-1"])
 def test_empty_name(self):
  with self.assertRaises(ValueError):summarize([":1"])
 def test_malformed(self):
  for row in ["oops","a:x","a:1:2"]:
   with self.subTest(row=row),self.assertRaises(ValueError):summarize([row])
