"""Native mutant construction policy; does not simulate GPU execution."""
import importlib.util
from pathlib import Path
import unittest
class NativeMutationTests(unittest.TestCase):
    def test_native_mutant_isolated(self):
        p=Path(__file__).resolve().parents[2]/'scripts/build_page_mutant.py'
        self.assertTrue(p.exists(),'Missing isolated native page mutant builder')
        spec=importlib.util.spec_from_file_location('page_mutant',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        source='HEADER\n                if (!session.begin(position, 1, error) ||\nTAIL'
        changed=m.mutate_source(source)
        self.assertIn('cuMemSetAccess',changed)
        self.assertIn('cudaMemcpyHostToDevice',changed)
        self.assertIn('page.owners < 2',changed)
        self.assertIn('M05_SHARED_PAGE_WRITE',changed)
        self.assertNotIn('getenv',changed)
        with self.assertRaises(ValueError):m.mutate_source('wrong source')
        with self.assertRaises(ValueError):m.mutate_source(changed)
if __name__=='__main__':unittest.main()
