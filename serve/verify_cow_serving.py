"""Supplemental CPU/HTTP evidence; never a real-backend acceptance gate."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import unittest


class Results(unittest.TextTestResult):
    def startTest(self,test):
        super().startTest(test);self.executed.append(test.id())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    suite=unittest.TestSuite()
    for pattern in ('test_cow_prefix.py','test_cow_adapter.py','test_cow_http.py','test_cow_text_adapter.py','test_cow_prompt_encoder.py','test_cow_context_root_capability.py','test_cow_presentation_adapter.py','test_cow_presentation_flow.py','test_cow_renderer.py','test_cow_namespace_http.py','test_cow_workspace.py','test_cow_workspace_http.py','test_cow_browser_fixture.py','test_cow_workspace_graph.py','test_cow_workspace_graph_http.py','test_cow_status_http.py','test_cow_reusable_context_smoke.py'):
        found=unittest.defaultTestLoader.discover(str(root),pattern=pattern)
        if found.countTestCases()==0: raise RuntimeError('Missing required supplemental suite: '+pattern)
        suite.addTests(found)
    Results.executed=[]
    result=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(suite)
    value=dict(status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
        scope='supplemental CPU/control and real HTTP socket tests with synthetic computation; not GPU/model/ordinary-chat qualification',
        host=socket.gethostname(),python=platform.python_version(),tests_run=result.testsRun,
        executed_ids=result.executed,failures=[(t.id(),s) for t,s in result.failures],
        errors=[(t.id(),s) for t,s in result.errors],skips=[(t.id(),s) for t,s in result.skipped],
        expected_failures=len(result.expectedFailures),unexpected_successes=len(result.unexpectedSuccesses),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in root.glob('*cow*.py')})
    value['dependency_sha256']={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name,module in sys.modules.items()
        if (name.startswith('context_time_travel.') or name in ('serve.server','serve.frontend')) and getattr(module,'__file__',None)
        and str(module.__file__).endswith('.py')}
    template=root/'chat_template.jinja'
    value['template_sha256']=hashlib.sha256(template.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    raise SystemExit(0 if value['status']=='PASS' else 1)


if __name__=='__main__': main()
