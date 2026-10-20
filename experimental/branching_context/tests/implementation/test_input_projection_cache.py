"""Derived hash cache versus the unchanged complete binding equations."""
import unittest,random,threading,copy
from unittest.mock import patch
from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_execution import input_binding
from context_time_travel import input_schedule_projection as projection
from test_vision_execution import spans

class InputProjectionCacheTests(unittest.TestCase):
 def setUp(self):
  self.cache=projection._BindingCache(4);p=patch.object(projection,'_BINDINGS',self.cache);p.start();self.addCleanup(p.stop)
 def binding(self,h):
  from context_time_travel.input_projection import cached_input_binding
  return cached_input_binding(h)
 def test_distinct_equal_objects_share_derived_hashes_but_not_mutable_results(self):
  from context_time_travel import input_projection as module
  a=InputHistory(spans());b=InputHistory(spans())
  with patch.object(module,'reference_input_binding',wraps=input_binding) as original:
   first=self.binding(a);first['native_input_sha256']='wrong'
   self.assertEqual(self.binding(b),input_binding(b));self.assertEqual(original.call_count,1)
 def test_all_seeded_text_image_and_append_bindings_match_reference(self):
  rng=random.Random(20261020)
  for initial in [[],[dict(kind='text',tokens=[0,1])],spans(),spans()[:-1]]:
   h=InputHistory(initial)
   for _ in range(30):
    self.assertEqual(self.binding(h),input_binding(h));h=h+(rng.randrange(200000),)
 def test_provenance_grid_placement_and_suffix_changes_do_not_collide(self):
  histories=[InputHistory(spans()),InputHistory(spans(original_sha256='f'*64)),InputHistory(spans(embedding_sha256='e'*64)),InputHistory(spans(grid_width=2,grid_height=3))]
  for h in histories:
   self.assertEqual(self.binding(h),input_binding(h));self.assertEqual(self.binding(h+(31,)),input_binding(h+(31,)))
 def test_changed_encoded_bytes_and_mutable_subclass_use_uncached_reference(self):
  h=InputHistory(spans());self.binding(h);other=InputHistory(spans(embedding_sha256='f'*64));object.__setattr__(h,'_encoded',other._encoded)
  self.assertEqual(self.binding(h),input_binding(other))
  class Mutable(InputHistory):
   @property
   def spans(self):return self.dynamic
  m=Mutable(spans());object.__setattr__(m,'dynamic',spans());self.binding(m);m.dynamic[-1]['tokens']=[19]
  self.assertEqual(self.binding(m),input_binding(m))
 def test_cache_eviction_changes_work_not_bindings_or_size_bound(self):
  for i in range(20):
   h=InputHistory([dict(kind='text',tokens=[i])]);self.assertEqual(self.binding(h),input_binding(h))
  self.assertLessEqual(len(self.cache),4)
 def test_complete_schedule_validation_and_incremental_binding_share_only_hashes(self):
  from test_vision_execution import receipt
  from context_time_travel.execution_receipt import validate_input_schedule
  from context_time_travel import input_projection as module
  h=InputHistory(spans());validate_input_schedule([receipt(h)],h,128)
  with patch.object(module,'reference_input_binding',side_effect=AssertionError('recomputed already audited binding')):
   self.assertEqual(self.binding(h),input_binding(h))
 def test_cached_expected_does_not_replace_a_native_query_or_accept_wrong_reply(self):
  from context_time_travel.strata_backend import StrataBackend
  import test_vision_backend_binding as fixtures
  fixtures.VisionBackendBindingTests.setUp(self)
  self.backend._check_input_binding(self.history)
  before=len(self.calls);self.info['position_sha256']='bad'
  from context_time_travel.store import IntegrityError
  with self.assertRaises(IntegrityError):self.backend._check_input_binding(self.history)
  self.assertEqual(len(self.calls),before+1)
