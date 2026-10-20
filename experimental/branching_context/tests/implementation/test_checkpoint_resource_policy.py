"""CPU admission controls, not measured RAM/VRAM usage or GPU qualification."""
from types import SimpleNamespace
import unittest
from context_time_travel.strata_backend import StrataBackend
from context_time_travel.store import IntegrityError


class Extent:
    def __init__(self,size): self.size = size
    def __len__(self): return self.size


class CheckpointResourceTests(unittest.TestCase):
    def backend(self, **policy):
        backend = object.__new__(StrataBackend)
        backend.profile = {'execution':{'checkpoint_cache':policy}}
        return backend

    def test_explicit_reference_extent_budget_admits_and_evicts_without_unbounded_retention(self):
        cache = self.backend(byte_limit=1500,entry_limit=2).new_cache()
        cache['large'] = SimpleNamespace(blob=Extent(1200))
        cache['small'] = SimpleNamespace(blob=Extent(200))
        cache['next'] = SimpleNamespace(blob=Extent(400))
        self.assertNotIn('large',cache)
        self.assertEqual(set(cache),{'small','next'})
        with self.assertRaises(MemoryError): cache['oversized'] = SimpleNamespace(blob=Extent(1501))
        self.assertEqual(set(cache),{'small','next'})

    def test_unknown_incomplete_and_invalid_budgets_fail_before_model_execution(self):
        for policy in ({}, {'byte_limit':10}, {'byte_limit':True,'entry_limit':2},
                       {'byte_limit':0,'entry_limit':2}, {'byte_limit':10,'entry_limit':-1},
                       {'byte_limit':10,'entry_limit':2,'unlimited':True},
                       {'byte_limit':33*1024**3,'entry_limit':2}):
            with self.subTest(policy=policy),self.assertRaises(IntegrityError):
                self.backend(**policy).new_cache()

    def test_absent_policy_preserves_previous_bound(self):
        backend = self.backend()
        del backend.profile['execution']['checkpoint_cache']
        cache = backend.new_cache()
        self.assertEqual((cache.byte_limit,cache.entry_limit),(1024**3,256))


if __name__ == '__main__': unittest.main()
