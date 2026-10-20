"""Evidence serialization checks; no inference or GPU simulation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.numerics import compare_logits

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/durable_resume.py'
spec = importlib.util.spec_from_file_location('durable_resume_report',SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ResumeReportTests(unittest.TestCase):
    def test_nonzero_error_at_zero_tolerance_is_preserved(self):
        report = compare_logits((1.,),(0.,),atol=0,rtol=0)
        record = runner.numerical_record(report)
        self.assertEqual(record['max_absolute_error'],1.)
        self.assertEqual(record['max_normalized_error'],'infinity')
        self.assertFalse(record['pass_elementwise'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            runner.publish(path,record)
            self.assertEqual(json.loads(path.read_text()),record)

    def test_exact_comparison_remains_numeric_zero(self):
        record = runner.numerical_record(compare_logits((1.,),(1.,),atol=0,rtol=0))
        self.assertEqual(record['max_normalized_error'],0.)
        self.assertTrue(record['pass_elementwise'])


if __name__=='__main__': unittest.main(verbosity=2)
