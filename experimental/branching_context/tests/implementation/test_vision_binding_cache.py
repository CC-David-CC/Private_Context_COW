"""Host protocol/cache tests, not real GPU numerical evidence."""
import gc,threading,unittest,weakref
from unittest.mock import patch
from context_time_travel.input_spans import InputHistory
from context_time_travel.strata_backend import _Lease
import context_time_travel.strata_backend as module
from context_time_travel.vision_execution import input_binding
from context_time_travel.store import IntegrityError,InvalidOperation
import test_vision_backend_binding as fixtures
from test_vision_execution import spans
class BindingCacheTests(unittest.TestCase):
 def setUp(self):fixtures.VisionBackendBindingTests.setUp(self)
 def test_repeated_check_reuses_expected_digest_but_queries_native_every_time(self):
  with patch.object(module,'input_binding',wraps=input_binding) as binding:
   for _ in range(3):self.backend._check_input_binding(self.history)
  self.assertEqual(binding.call_count,1)
  self.assertEqual(self.calls,['vision_info']*3)
 def test_cached_expected_still_rejects_each_wrong_native_binding(self):
  with patch.object(module,'input_binding',wraps=input_binding) as binding:
   self.backend._check_input_binding(self.history)
   for key in ['native_input_sha256','position_sha256','rotary_cursor','consumed','vision_protocol','enabled']:
    old=self.info[key];self.info[key]='wrong'
    with self.subTest(field=key),self.assertRaises(IntegrityError):
     self.backend._check_input_binding(self.history)
    self.info[key]=old
  self.assertEqual(binding.call_count,1)
  self.assertEqual(len(self.calls),7)
 def test_alternating_histories_never_reuse_the_other_expected_identity(self):
  changed=spans();changed[-1]['tokens']=[4]
  other=InputHistory(changed)
  with patch.object(module,'input_binding',wraps=input_binding) as binding:
   self.backend._check_input_binding(self.history)
   with self.assertRaises(IntegrityError):self.backend._check_input_binding(other)
   self.info.update(input_binding(other));self.backend._check_input_binding(other)
   with self.assertRaises(IntegrityError):self.backend._check_input_binding(self.history)
  self.assertEqual(binding.call_count,3)
 def test_same_object_changed_encoded_bytes_cannot_hit_old_binding(self):
  other=InputHistory(spans(embedding_sha256='e'*64))
  with patch.object(module,'input_binding',wraps=input_binding) as binding:
   self.backend._check_input_binding(self.history)
   object.__setattr__(self.history,'_encoded',other._encoded)
   with self.assertRaises(IntegrityError):self.backend._check_input_binding(self.history)
  self.assertEqual(binding.call_count,2)
 def test_equal_native_tokens_different_grid_cannot_share_expected_binding(self):
  other=InputHistory(spans(grid_width=2,grid_height=3))
  self.assertEqual(other.native_tokens,self.history.native_tokens)
  self.backend._check_input_binding(self.history)
  with self.assertRaises(IntegrityError):self.backend._check_input_binding(other)
 def test_mutable_subclass_is_never_cached(self):
  class Mutable(InputHistory):
   @property
   def spans(self):return self.dynamic
  other=Mutable(spans());object.__setattr__(other,'dynamic',spans())
  self.backend._check_input_binding(other)
  other.dynamic[-1]['tokens']=[4]
  with self.assertRaises(IntegrityError):self.backend._check_input_binding(other)
 def test_cache_is_lease_local_and_release_discards_derived_history(self):
  self.backend._lock=threading.RLock();self.backend._lock.acquire()
  self.backend.stats={'lease_releases':0}
  with patch.object(module,'input_binding',wraps=input_binding) as binding:
   self.backend._check_input_binding(self.history)
   self.lease.__exit__(None,None,None)
   other=_Lease(self.backend,bound=True,consumed=9,state='ACTIVE',
    _owner_thread=threading.get_ident(),inputs=self.history)
   self.backend._active=other
   self.backend._check_input_binding(self.history)
  self.assertEqual(binding.call_count,2)
  transient=InputHistory(spans(embedding_sha256='e'*64));ref=weakref.ref(transient)
  self.info.update(input_binding(transient));self.backend._check_input_binding(transient)
  del transient;self.backend._lock.acquire();other.__exit__(None,None,None)
  gc.collect();self.assertIsNone(ref())
 def test_foreign_thread_cannot_read_write_or_release_warm_lease(self):
  with patch.object(module,'input_binding',wraps=input_binding) as binding:
   self.backend._check_input_binding(self.history);before=list(self.calls);errors=[]
   def foreign():
    for action in [lambda:self.backend.logits(self.lease),
      lambda:self.backend.consume(self.lease,[4]),lambda:self.lease.__exit__(None,None,None)]:
     try:action()
     except BaseException as error:errors.append(error)
   thread=threading.Thread(target=foreign);thread.start();thread.join()
   self.assertEqual(len(errors),3);self.assertTrue(all(isinstance(x,InvalidOperation) for x in errors))
   self.assertEqual(self.calls,before);self.assertEqual(self.lease.state,'ACTIVE')
   self.backend._check_input_binding(self.history)
  self.assertEqual(binding.call_count,1)

 def test_interruption_at_each_cache_update_line_cannot_publish_partial_entry(self):
  import sys
  class Interrupted(Exception):pass
  code=self.backend._expected_input_binding.__func__.__code__
  lines=set()
  def observe(frame,event,arg):
   if frame.f_code is code and event=='line':lines.add(frame.f_lineno)
   return observe
  self.backend._check_input_binding(self.history)
  other=InputHistory(spans(embedding_sha256='e'*64))
  try:
   sys.settrace(observe)
   with self.assertRaises(IntegrityError):self.backend._check_input_binding(other)
  finally:sys.settrace(None)
  for line in sorted(lines):
   with self.subTest(interrupted_line=line):
    fixtures.VisionBackendBindingTests.setUp(self)
    self.backend._check_input_binding(self.history)
    other=InputHistory(spans(embedding_sha256='e'*64))
    def interrupt(frame,event,arg):
     if frame.f_code is code and event=='line' and frame.f_lineno==line:raise Interrupted()
     return interrupt
    try:
     sys.settrace(interrupt)
     try:self.backend._check_input_binding(other)
     except (Interrupted,IntegrityError):pass
    finally:sys.settrace(None)
    with self.assertRaises(IntegrityError):self.backend._check_input_binding(other)
