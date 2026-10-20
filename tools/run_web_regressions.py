"""Run named unittest suites and save executed IDs, failures and skips as JSON."""
import argparse
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Evidence(unittest.TextTestResult):
    def startTest(self, test):
        self.ids.append(test.id())
        super().startTest(test)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = []
        self.measurements = {}

    def addSuccess(self, test):
        super().addSuccess(test)
        if hasattr(test, 'measurements'):
            self.measurements[test.id()] = test.measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('suites', nargs='+')
    options = parser.parse_args()
    start = time.perf_counter()
    result = unittest.TextTestRunner(resultclass=Evidence, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(options.suites))
    report = {'status': 'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
              'seconds': time.perf_counter()-start, 'executed_ids': result.ids,
              'executed_methods': result.testsRun,
              'measurements': result.measurements,
              'failures': [(t.id(), e) for t, e in result.failures],
              'errors': [(t.id(), e) for t, e in result.errors],
              'skips': [(t.id(), e) for t, e in result.skipped]}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
