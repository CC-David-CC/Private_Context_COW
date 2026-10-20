"""Executed CPU/filesystem evidence, explicitly separate from GPU qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests/implementation'),str(ROOT/'scripts')]
# Crash witnesses start new Python interpreters; they need the same real package.
os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT/'scripts'),
    os.environ.get('PYTHONPATH','')])
MODULES = ['test_shared_pools','test_disk_pool','test_gpu_workers','test_prepared_checkpoint',
    'test_pool_coordinator','test_durable_restore','test_host_checkpoint_parking',
    'test_resident_request_publication','test_execution_receipt','test_native_operator_receipts',
    'test_operator_bound_checkpoint','test_state_staging','test_checkpoint_resource_policy',
    'test_client_stream_metrics','test_controller_bulk_consumption','test_pool_campaign',
    'test_pool_request_stop','test_resident_request_cancel','test_native_request_protocol',
    'test_pool_source_trace','test_pool_spill_reload_trace','test_codebase_fixture','test_store_branch_catalog',
    'test_store_namespaces','test_presentation_binding','test_prepared_presentation',
    'test_store_presentation','test_pool_presentation','test_store_presentation_catalog',
    'test_pool_branch_actions','test_store_fork_provenance',
    'test_namespace_http_fixture','test_namespace_http_fixture_8k','test_audit_namespace_http']


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.executed=[]
    def startTest(self,test):
        self.executed.append(test.id());super().startTest(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    record = dict(scope='CPU/filesystem control only; no mocked GPU qualification',
        status='PASS' if result.wasSuccessful() and not result.skipped and not result.expectedFailures else 'FAIL',
        tests=result.testsRun,executed_ids=result.executed,failures=[t.id() for t,_ in result.failures],
        errors=[t.id() for t,_ in result.errors],skips=[t.id() for t,_ in result.skipped],
        xfails=[t.id() for t,_ in result.expectedFailures],python=sys.version,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src/context_time_travel').glob('*.py'))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    return 0 if record['status']=='PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
