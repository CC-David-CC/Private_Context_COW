"""Derived hash reuse must preserve exact equations and reject untrusted receipts."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import unittest
from unittest.mock import patch
from context_time_travel import input_schedule_projection as projection
from context_time_travel.input_spans import InputHistory
from context_time_travel.execution_receipt import validate_input_schedule
from context_time_travel.vision_execution import input_binding
from test_vision_execution import spans
from test_vision_schedule_projection import events_for

class BindingReuseTests(unittest.TestCase):
    def setUp(self):
        self.patcher=patch.object(projection,'_BINDINGS',projection._BindingCache(65536))
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def test_repeated_validation_hashes_each_immutable_prefix_body_once(self):
        history=InputHistory([dict(kind='text',tokens=list(range(713)))])
        events=events_for(history,[128,512,713])
        original=hashlib.sha256; bodies=[]
        def measured(data=b'',**kwargs):
            if data in (b'COW-VISION-INPUT-v1\0',b'COW-MROPE-v1\0'):
                bodies.append(data)
            return original(data,**kwargs)
        with patch.object(projection.hashlib,'sha256',side_effect=measured):
            validate_input_schedule(events,history,128)
            first=len(bodies)
            validate_input_schedule(copy.deepcopy(events),InputHistory(history.spans),128)
            self.assertEqual(len(bodies),first,'old immutable bodies were hashed again')
        self.assertEqual(first,6)

class BindingCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache=projection._BindingCache(3)
        self.patcher=patch.object(projection,'_BINDINGS',self.cache)
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def binding(self,history,end=None):
        end=len(history) if end is None else end
        return projection._InputScheduleProjection(history,[end]).binding(end)

    def test_independent_equations_match_cold_and_warm_for_mixed_inputs(self):
        for value in (spans(),spans(original_sha256='e'*64),spans(grid_width=2,grid_height=3)):
            history=InputHistory(value)
            expected=dict(inputs_sha256=history.digest,**input_binding(history))
            self.assertEqual(self.binding(history),expected)
            self.assertEqual(self.binding(InputHistory(value)),expected)

    def test_extension_and_sibling_share_only_identical_prefix(self):
        history=InputHistory(spans());expected=self.binding(history)
        with patch.object(projection,'_hash_binding',side_effect=AssertionError('rehashed old prefix')):
            for tail in ((11,),(12,13)):
                self.assertEqual(self.binding(history+tail,len(history)),expected)
        self.assertNotEqual(self.binding(history+(11,)),self.binding(history+(12,)))

    def test_all_receipt_fields_rejected_even_with_warm_cache(self):
        history=InputHistory(spans());events=events_for(history,[2,9])
        validate_input_schedule(events,history,128)
        changes=dict(inputs_sha256='0'*64,native_input_sha256='0'*64,
            position_sha256='0'*64,rotary_cursor=True,tokens_sha256='0'*64,
            position=1,count=True,operator='bad',chunks=[])
        for key,value in changes.items():
            bad=copy.deepcopy(events);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_input_schedule(bad,history,128)

    def test_changed_canonical_content_cannot_inherit_warm_binding(self):
        history=InputHistory(spans());events=events_for(history,[2,9])
        validate_input_schedule(events,history,128)
        variants=[spans(original_sha256='e'*64),spans(embedding_sha256='e'*64),
            spans(grid_width=2,grid_height=3),
            [dict(kind='text',tokens=[2,1]),spans()[1],dict(kind='text',tokens=[3])]]
        for value in variants:
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_input_schedule(events,InputHistory(value),128)

    def test_false_declared_history_digest_rejected_before_cache_lookup(self):
        history=InputHistory(spans());self.binding(history)
        bad=InputHistory(spans(original_sha256='e'*64));history_digest=history.digest
        with patch.object(InputHistory,'digest',new=property(lambda _:history_digest)):
            with self.assertRaises(ValueError):self.binding(bad)

    def test_growing_schedule_hashes_new_boundary_but_reuses_prior_boundaries(self):
        history=InputHistory(spans());events=events_for(history,[2,9])
        validate_input_schedule(events,history,128)
        extended=history+(101,)
        new_events=events_for(extended,[2,9,10])
        with patch.object(projection,'_hash_binding',wraps=projection._hash_binding) as hashing:
            validate_input_schedule(new_events,extended,128)
            self.assertEqual(hashing.call_count,1)
        bad=copy.deepcopy(new_events);bad[0]['position_sha256']='0'*64
        with self.assertRaises(ValueError):validate_input_schedule(bad,extended,128)

    def test_deliberately_wrong_hash_computation_is_detected(self):
        history=InputHistory(spans());events=events_for(history,[2,9])
        with patch.object(projection,'_hash_binding',return_value=('0'*64,'0'*64)):
            with self.assertRaises(ValueError):validate_input_schedule(events,history,128)

    def test_returned_dict_is_detached_from_cached_binding(self):
        history=InputHistory(spans());expected=self.binding(history)
        actual=self.binding(history);actual['native_input_sha256']='0'*64;actual['rotary_cursor']=99
        self.assertEqual(self.binding(history),expected)

    def test_bounded_lru_eviction_recomputes_without_retaining_history(self):
        for n in range(5):
            self.binding(InputHistory([dict(kind='text',tokens=[n])]))
        self.assertEqual(len(self.cache),3)
        with patch.object(projection,'_hash_binding',wraps=projection._hash_binding) as hashing:
            self.binding(InputHistory([dict(kind='text',tokens=[4])]))
            self.assertEqual(hashing.call_count,0)
            self.binding(InputHistory([dict(kind='text',tokens=[0])]))
            self.assertEqual(hashing.call_count,1)
        self.assertEqual(len(self.cache),3)

    def test_cache_hit_cannot_allow_image_interior_or_image_end_boundary(self):
        history=InputHistory(spans());self.binding(history)
        for end in range(3,9):
            with self.subTest(end=end),self.assertRaises(ValueError):self.binding(history,end)

    def test_concurrent_lookup_and_eviction_preserve_independent_results(self):
        def check(n):
            history=InputHistory([dict(kind='text',tokens=[n%7,n%11])])
            return self.binding(history)==dict(inputs_sha256=history.digest,**input_binding(history))
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertTrue(all(pool.map(check,range(200))))
        self.assertLessEqual(len(self.cache),3)

    def test_invalid_cache_capacity_rejected(self):
        for capacity in (0,-1,True,1.5):
            with self.subTest(capacity=capacity),self.assertRaises(ValueError):
                projection._BindingCache(capacity)
