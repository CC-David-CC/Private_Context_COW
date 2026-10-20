"""Compute Sanitizer receipt parsing only; no GPU execution."""
import importlib.util
from pathlib import Path
import unittest

class RunnerTests(unittest.TestCase):
    def test_missing_instrumentation_and_diagnostics_cannot_pass(self):
        path=Path(__file__).resolve().parents[2]/'scripts/run_compute_sanitizer.py'
        self.assertTrue(path.is_file(),'Missing retained Compute Sanitizer runner')
        spec=importlib.util.spec_from_file_location('runner',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.classify(0,'COMPUTE-SANITIZER'),'BLOCKED')
        self.assertEqual(module.classify(0,'COMPUTE-SANITIZER\nERROR SUMMARY: 0 errors'),'PASS')
        for code,text in [(0,'ERROR SUMMARY: 1 errors'),(0,'ERROR SUMMARY: 0 errors\nError: instrumented API failed'),(86,'ERROR SUMMARY: 0 errors')]:
            self.assertEqual(module.classify(code,text),'FAIL')

if __name__=='__main__':unittest.main()
