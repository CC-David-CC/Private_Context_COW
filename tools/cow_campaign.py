#!/usr/bin/env python3
"""Select model-specific tests before import; report prerequisites honestly.

This runner is complementary to the unchanged G0-G5 acceptance gate runner.
It qualifies only the selected campaign, never the union of excluded campaigns.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BRANCHING = ROOT/'experimental/branching_context'
TESTS = BRANCHING/'tests/implementation'
MANIFEST = BRANCHING/'tests/campaigns.json'
CAMPAIGNS = ('cpu', 'moe-native', 'dense-native', 'legacy-build', 'retained-evidence')


def validate_manifest(directory, modules):
    if type(modules) is not dict: raise ValueError('Invalid campaign manifest modules')
    present = {p.stem for p in Path(directory).glob('test_*.py')}
    unknown, missing = present - modules.keys(), modules.keys() - present
    if unknown: raise ValueError('Unclassified test modules: ' + ', '.join(sorted(unknown)))
    if missing: raise ValueError('Missing test modules: ' + ', '.join(sorted(missing)))
    for name, row in modules.items():
        if (type(row) is not dict or set(row) != {'campaign','requires'}
                or row['campaign'] not in CAMPAIGNS or type(row['requires']) is not list
                or any(type(key) is not str or not re.fullmatch('[A-Z][A-Z0-9_]*',key)
                       for key in row['requires'])
                or len(row['requires']) != len(set(row['requires']))):
            raise ValueError('Invalid campaign manifest row: ' + name)


def classify(row, campaign, environment):
    if row['campaign'] != campaign:
        return dict(status='NOT_APPLICABLE', reason='belongs to ' + row['campaign'])
    missing = [key for key in row.get('requires', []) if not environment.get(key)]
    if missing:
        return dict(status='BLOCKED', reason='missing ' + ', '.join(missing))
    return dict(status='READY')


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite): yield from cases(item)
        else: yield item


@contextmanager
def campaign_environment(paths):
    """Keep subprocess import resolution bound to the selected checkout too."""
    selected=list(map(str,paths))
    previous_path=list(sys.path); previous_env=os.environ.get('PYTHONPATH')
    sys.path[:0]=selected
    os.environ['PYTHONPATH']=os.pathsep.join(selected+([previous_env] if previous_env else []))
    try:yield
    finally:
        sys.path[:]=previous_path
        if previous_env is None:os.environ.pop('PYTHONPATH',None)
        else:os.environ['PYTHONPATH']=previous_env


def run(campaign, *, modules=None, environment=None, stream=None):
    env = os.environ if environment is None else environment
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (type(manifest) is not dict or set(manifest) != {'schema','modules'}
            or type(manifest['schema']) is not int or manifest['schema'] != 1):
        raise ValueError('Unsupported campaign manifest schema')
    registered = manifest['modules']
    validate_manifest(TESTS, registered)
    selected = set(registered) if modules is None else set(modules)
    if selected - registered.keys(): raise ValueError('Requested unregistered test module')
    observations = {name: classify(row, campaign, env) for name, row in sorted(registered.items())
                    if name in selected}
    report = dict(schema=1, campaign=campaign, modules=observations,
                  manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  status='BLOCKED', tests_run=0, failures=[], errors=[], skipped=[])
    git = ['git','-C',str(ROOT)]
    try:
        report['source_commit'] = subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
        report['source_dirty'] = bool(subprocess.check_output(git+['status','--porcelain'],text=True))
    except (OSError,subprocess.CalledProcessError):
        report['reason'] = 'Source revision unavailable; run from the checked-out repository'
        return report
    if os.name != 'posix':
        report['reason'] = 'This campaign requires POSIX filesystem/process semantics'
        return report
    if campaign=='cpu' and modules is None and shutil.which('g++') is None:
        report['reason']='The complete CPU campaign includes compiled host controls and requires g++'
        return report
    # Abort the requested campaign before importing any tests if it is incomplete.
    if any(row['status'] == 'BLOCKED' for row in observations.values()): return report
    ready = [name for name, row in observations.items() if row['status'] == 'READY']
    if not ready:
        report['reason'] = 'No applicable tests selected'
        return report
    if campaign in ('moe-native', 'dense-native'):
        profile = json.loads(Path(env['CTT_PROFILE']).read_text())
        expected = 'strata' if campaign == 'moe-native' else 'qwen35-llama.cpp'
        if profile.get('backend') != expected:
            report['reason'] = 'Selected profile backend differs from campaign'
            return report
        report['profile'] = {key: profile.get(key) for key in
            ('backend', 'model_contract_id', 'execution_profile_id', 'model_weights_sha256')}
    with campaign_environment((ROOT, BRANCHING/'src', BRANCHING/'scripts', TESTS)):
        suite = unittest.TestSuite()
        seen = set()
        for name in ready:
            loaded = unittest.defaultTestLoader.loadTestsFromName(name)
            for test in cases(loaded):
                # Imported fixture TestCase classes must not silently multiply coverage.
                if test.id() not in seen:
                    suite.addTest(test); seen.add(test.id())
        result = unittest.TextTestRunner(stream=stream or sys.stderr, verbosity=1).run(suite)
    report.update(tests_run=result.testsRun,
        failures=[t.id() for t, _ in result.failures], errors=[t.id() for t, _ in result.errors],
        skipped=[dict(test=t.id(), reason=reason) for t, reason in result.skipped],
        expected_failures=[t.id() for t, _ in result.expectedFailures])
    report['status'] = ('PASS' if result.testsRun and result.wasSuccessful()
                        and not result.skipped and not result.expectedFailures else 'FAIL')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=CAMPAIGNS, default='cpu')
    parser.add_argument('--module', action='append')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = run(args.campaign, modules=args.module)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'modules'}, indent=2))
    if report['status'] == 'BLOCKED':
        for name, row in report['modules'].items():
            if row['status'] == 'BLOCKED': print(name + ': ' + row['reason'])
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__': raise SystemExit(main())
