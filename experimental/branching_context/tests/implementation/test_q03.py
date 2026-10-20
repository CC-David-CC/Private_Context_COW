"""Lifecycle evidence checks only; these tests execute no GPU."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]

class EvidenceTests(unittest.TestCase):
    def test_q03_invariant(self):
        path=ROOT/'scripts/lifecycle_evidence.py'
        self.assertTrue(path.is_file(),'Missing strict lifecycle evidence collector')
        spec=importlib.util.spec_from_file_location('lifecycle_evidence',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        counters={name:0 for name in module.REQUIRED}
        rows=[]
        for index in range(10):
            before=dict(counters)
            for allocated,released in module.PAIRS:
                counters[allocated]+=2;counters[released]+=2
            counters['host_heap_allocated_bytes']=100
            rows.append(dict(before=before,after=dict(counters),seconds=index+1.,rss_bytes=200,gpu_bytes=300))
        result=module.summarize(rows,warmup_cycles=8)
        self.assertEqual(result['cycles'],10)
        self.assertEqual(result['status'],'MEASURED')
        self.assertEqual(result['latency_seconds']['minimum'],1)
        self.assertEqual(result['latency_seconds']['maximum'],10)
        self.assertEqual(sum(b['count'] for b in result['latency_histogram']),len(rows))
        self.assertEqual(result['peak_rss_bytes'],200)
        self.assertEqual(result['peak_gpu_bytes'],300)
        self.assertIn('rss_histogram',result,'Memory histograms must retain every measured sample')
        self.assertEqual(sum(b['count'] for b in result['rss_histogram']),len(rows))
        self.assertEqual(sum(b['count'] for b in result['gpu_histogram']),len(rows))
        mutations=[('live_bytes',1),('allocations',99),('host_heap_allocated_bytes',101),('integrity_mismatches',1)]
        for name,value in mutations:
            bad=[dict(r,before=dict(r['before']),after=dict(r['after'])) for r in rows]
            bad[-1]['after'][name]=value
            with self.subTest(counter=name),self.assertRaises(ValueError):module.summarize(bad,warmup_cycles=8)
        bad=[dict(r,before=dict(r['before']),after=dict(r['after'])) for r in rows]
        del bad[-1]['after']['handle_live_entries']
        with self.assertRaises(ValueError):module.summarize(bad,warmup_cycles=8)
        for seconds in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):module.summarize([dict(rows[0],seconds=seconds)],warmup_cycles=0)
        with self.assertRaises(ValueError):module.summarize([],warmup_cycles=0)
        for text in ['ERROR: AddressSanitizer: heap-use-after-free','runtime error: misaligned address','WARNING: ThreadSanitizer: data race','ERROR SUMMARY: 3 errors']:
            self.assertEqual(module.sanitizer_status(0,text,instrumented=True),'FAIL')
        self.assertEqual(module.sanitizer_status(1,'',instrumented=True),'FAIL')
        self.assertEqual(module.sanitizer_status(0,'ERROR SUMMARY: 0 errors',instrumented=False),'BLOCKED')
        self.assertEqual(module.sanitizer_status(0,'ERROR SUMMARY: 0 errors',instrumented=True),'PASS')

if __name__=='__main__':unittest.main()
