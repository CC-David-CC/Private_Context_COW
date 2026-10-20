"""Immutable derived views; canonical bytes and equations remain the oracle."""
import gc,hashlib,json,random,unittest,weakref
from unittest.mock import patch
from context_time_travel.input_spans import InputHistory,canonical_inputs
from context_time_travel.store import _json_bytes,InvalidOperation
from test_vision_execution import spans

class InputHistoryViewTests(unittest.TestCase):
 def test_append_matches_independent_canonical_encoding_at_every_step(self):
  rng=random.Random(20261020)
  for initial in [[],[dict(kind='text',tokens=[1,2])],spans(),spans()[:-1]]:
   expected=canonical_inputs(initial);history=InputHistory(initial)
   for _ in range(40):
    delta=[rng.randrange(200000) for _ in range(rng.randrange(1,5))]
    old=history;old_bytes=old._encoded;history=old+tuple(delta)
    expected=canonical_inputs(expected+[dict(kind='text',tokens=delta)])
    raw=_json_bytes(dict(schema=1,spans=expected))
    self.assertEqual(history._encoded,raw);self.assertEqual(history.digest,hashlib.sha256(raw).hexdigest())
    self.assertEqual(history.native_tokens,tuple(t for s in expected for t in (s['tokens'] if s['kind']=='text' else [s['pad_token_id']]*s['rows'])))
    self.assertEqual(old._encoded,old_bytes);self.assertTrue(history.starts_with(old))
 def test_owned_append_and_native_view_do_not_decode_prior_json(self):
  history=InputHistory([dict(kind='text',tokens=list(range(8192)))])
  with patch('context_time_travel.input_spans.json.loads',side_effect=AssertionError('decoded owned prefix')):
   child=history+(17,)
   self.assertEqual(child.native_tokens,tuple(range(8192))+(17,))
   self.assertEqual(child.text_runs,(tuple(range(8192))+(17,),))
 def test_returned_views_are_detached_and_do_not_poison_future_appends(self):
  initial=spans();history=InputHistory(initial);ids=history.native_tokens
  initial[0]['tokens'][0]=999
  view=history.spans;view[0]['tokens'][0]=888;view[1]['embedding_sha256']='f'*64
  self.assertEqual(history.native_tokens,ids);self.assertEqual((history+(19,)).native_tokens,ids+(19,))
  self.assertNotEqual(history.spans,view)
 def test_changed_encoded_identity_misses_all_old_derived_views(self):
  history=InputHistory(spans());history.native_tokens
  other=InputHistory(spans(embedding_sha256='f'*64));body=other.spans;body[-1]['tokens']=[23];other=InputHistory(body)
  object.__setattr__(history,'_encoded',other._encoded)
  self.assertEqual(history.spans,other.spans);self.assertEqual(history.native_tokens,other.native_tokens)
  self.assertEqual((history+(9,))._encoded,(other+(9,))._encoded)
 def test_mutable_subclass_views_are_never_trusted_or_cached(self):
  class Mutable(InputHistory):
   @property
   def spans(self):return self.dynamic
  history=Mutable(spans());object.__setattr__(history,'dynamic',spans())
  first=history.native_tokens;history.dynamic[-1]['tokens']=[29]
  self.assertNotEqual(history.native_tokens,first)
  self.assertEqual((history+(17,)).native_tokens,history.native_tokens+(17,))
 def test_append_still_rejects_invalid_new_ids_and_preserves_uint32(self):
  history=InputHistory([dict(kind='text',tokens=[0,2**32-1])])
  for values in [(True,),(-1,),(2**32,),('2',)]:
   with self.subTest(values=values),self.assertRaises(InvalidOperation):history+values
  self.assertEqual((history+())._encoded,history._encoded)
  self.assertEqual((history+(i for i in [3,4])).native_tokens,(0,2**32-1,3,4))
 def test_append_does_not_retain_a_chain_of_prior_full_histories(self):
  root=InputHistory([dict(kind='text',tokens=list(range(100)))]);current=root;refs=[]
  for i in range(100):refs.append(weakref.ref(current));current=current+(i,)
  gc.collect();self.assertLessEqual(sum(ref() is not None for ref in refs),1)
  self.assertEqual(current.native_tokens,tuple(range(100))+tuple(range(100)))
 def test_image_rows_remain_lazy_when_only_building_history(self):
  image=spans()[1];image.update(rows=1000000,grid_width=1000,grid_height=1000,embedding_bytes=20+1000000*image['width']*4)
  history=InputHistory([image]);self.assertEqual(len(history),1000000)
  self.assertEqual(history.spans,[image]);self.assertLess(len(history._encoded),1024)
