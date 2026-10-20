"""Sanitizer suite policy tests; no inference or tool execution."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

class SuiteTests(unittest.TestCase):
    def test_partial_and_unsupported_results_cannot_pass(self):
        path=ROOT/'scripts/sanitizer_suite.py'
        self.assertTrue(path.is_file(),'Missing actual sanitizer suite runner')
        spec=importlib.util.spec_from_file_location('suite',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        complete={k:{'status':'PASS','instrumented':True} for k in ['asan-ubsan','tsan','cuda-memcheck']}
        self.assertEqual(module.suite_status(complete),'PASS')
        for name in complete:
            missing=dict(complete);del missing[name]
            self.assertEqual(module.suite_status(missing),'BLOCKED')
            partial=dict(complete,**{name:dict(status='PASS',instrumented=False)})
            self.assertEqual(module.suite_status(partial),'BLOCKED')
            failed=dict(complete,**{name:dict(status='FAIL',instrumented=True)})
            self.assertEqual(module.suite_status(failed),'FAIL')
        self.assertEqual(module.suite_status(dict(complete,tsan={'status':'BLOCKED','reason':'runtime unsupported'})),'BLOCKED')

if __name__=='__main__':unittest.main()
