"""Supplemental profiler accounting checks; no inference qualification."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from profile_matching_resume import Timings


class WallAttributionTests(unittest.TestCase):
    def test_nested_time_is_not_double_counted(self):
        timings = Timings()
        inner = timings.wrap('inner',lambda: 'value')
        outer = timings.wrap('outer',inner)
        with patch('profile_matching_resume.time.perf_counter_ns',side_effect=[0,10,30,50]):
            self.assertEqual(outer(),'value')
        self.assertEqual(timings.rows['inner']['inclusive_ns'],20)
        self.assertEqual(timings.rows['outer']['inclusive_ns'],50)
        self.assertEqual(timings.rows['outer']['exclusive_ns'],30)
        self.assertEqual(timings.stack,[])

    def test_failure_keeps_accounting_and_original_exception(self):
        timings = Timings()
        def fail(): raise ValueError('fixture')
        wrapped = timings.wrap('failure',fail)
        with patch('profile_matching_resume.time.perf_counter_ns',side_effect=[2,9]):
            with self.assertRaisesRegex(ValueError,'fixture'): wrapped()
        self.assertEqual(timings.rows['failure']['exclusive_ns'],7)
        self.assertEqual(timings.stack,[])


if __name__ == '__main__': unittest.main()
