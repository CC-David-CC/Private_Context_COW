"""Sampler parsing/qualification checks only; no model/GPU is simulated."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

class SoakTests(unittest.TestCase):
    def test_memory_and_long_soak_boundaries(self):
        path=ROOT/'scripts/lifecycle_soak.py'
        self.assertTrue(path.is_file(),'Missing actual lifecycle sampler')
        spec=importlib.util.spec_from_file_location('soak',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        memory=module.parse_memory('Name: strata\nVmRSS: 1234 kB\n','17, 12\n19, 99\n17, 8\n',17)
        self.assertEqual(memory,dict(rss_bytes=1234*1024,gpu_bytes=20*1024*1024))
        for status,gpu in [('Name: strata','17, 12'),('VmRSS: 1 kB','19, 12'),('VmRSS: 1 kB','17, [N/A]')]:
            with self.assertRaises(ValueError):module.parse_memory(status,gpu,17)
        report=dict(status='MEASURED',cycles=64,seconds=86400,warmup_cycles=8,
                    numerical_comparisons=256,max_absolute_error=0,summary=dict(status='MEASURED',frozen_host_heap_bound=100))
        self.assertTrue(module.qualifies_long_soak(report))
        for key,value in [('status','RUNNING'),('cycles',63),('seconds',86399),('warmup_cycles',0),('numerical_comparisons',0),('max_absolute_error',float('nan'))]:
            self.assertFalse(module.qualifies_long_soak(dict(report,**{key:value})),key)
        self.assertFalse(module.qualifies_long_soak(dict(report,summary={})))

if __name__=='__main__':unittest.main()
