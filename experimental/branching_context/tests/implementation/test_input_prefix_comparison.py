"""Compare immutable canonical prefixes without rebuilding a decoded history."""
import random,unittest
from unittest.mock import patch
from context_time_travel.input_spans import InputHistory
from context_time_travel.store import InvalidOperation
from context_time_travel.execution_receipt import _InputScheduleCursor
from test_vision_execution import spans,receipt

class InputPrefixComparisonTests(unittest.TestCase):
 def test_same_empty_shorter_and_longer_prefixes(self):
  root=InputHistory([]);a=InputHistory([dict(kind='text',tokens=[1,2])]);b=a+(3,)
  for value,prefix,expected in ((root,root,True),(a,root,True),(a,a,True),(b,a,True),(a,b,False)):
   self.assertEqual(value.starts_with(prefix),expected)
 def test_numeric_text_prefix_is_not_a_token_prefix(self):
  for left,right in (([10],[1]),([1,20],[1,2]),([12,3],[1]),([1,2],[1,20])):
   self.assertFalse(InputHistory([dict(kind='text',tokens=left)]).starts_with(InputHistory([dict(kind='text',tokens=right)])))
 def test_text_extension_and_new_span_after_images_match_reference(self):
  source=InputHistory(spans());image_end=InputHistory(spans()[:-1])
  values=[source,source+(4,),source+InputHistory(spans(embedding_sha256='e'*64))]
  for value in values:
   for n in range(len(value)+1):
    try:prefix=value.prefix(n)
    except InvalidOperation:continue
    self.assertTrue(value.starts_with(prefix),(value,n))
  self.assertTrue(source.starts_with(image_end))
 def test_image_provenance_grid_and_placement_cannot_collide(self):
  root=InputHistory(spans())
  variants=[spans(original_sha256='e'*64),spans(embedding_sha256='e'*64),spans(grid_width=2,grid_height=3),
   [dict(kind='text',tokens=[2,1]),spans()[1],dict(kind='text',tokens=[3])]]
  for value in variants:self.assertFalse((InputHistory(value)+(4,)).starts_with(root))
 def test_seeded_prefix_comparisons_match_independent_reconstruction(self):
  rng=random.Random(521014)
  for _ in range(40):
   values=[dict(kind='text',tokens=[rng.randrange(100000) for _ in range(rng.randrange(1,20))])]
   if rng.randrange(2):values+=spans()
   values.append(dict(kind='text',tokens=[rng.randrange(100000) for _ in range(10)]))
   history=InputHistory(values)
   prefixes=[InputHistory([]),InputHistory(spans())]
   for n in range(len(history)+1):
    try:prefixes.append(history.prefix(n))
    except InvalidOperation:pass
   for prefix in prefixes:
    try:expected=history.prefix(len(prefix))==prefix
    except InvalidOperation:expected=False
    self.assertEqual(history.starts_with(prefix),expected)
 def test_receipt_cursor_still_rejects_ancestry_rewrite_without_prefix_reconstruction(self):
  root=InputHistory(spans());cursor=_InputScheduleCursor([receipt(root)],root,128);final=root+(4,)
  with patch.object(InputHistory,'prefix',side_effect=AssertionError('rebuilt old prefix')):
   cursor.append(receipt(final,position=len(root)),final)
  cursor.finish()
  bad=InputHistory(spans(original_sha256='e'*64))+(4,)
  cursor=_InputScheduleCursor([receipt(root)],root,128)
  with self.assertRaises(ValueError):cursor.append(receipt(bad,position=len(root)),bad)
 def test_caller_mutations_cannot_change_comparison_inputs(self):
  spans_=[dict(kind='text',tokens=[1,2])];old=InputHistory(spans_);extended=old+(3,)
  spans_[0]['tokens'][0]=99;detached=old.spans;detached[0]['tokens'][0]=99
  self.assertTrue(extended.starts_with(old));self.assertEqual(old.native_tokens,(1,2))
