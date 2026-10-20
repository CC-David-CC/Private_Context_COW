"""Compile actual instrumented host targets and execute real CUDA diagnosis."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from lifecycle_evidence import sanitizer_status
from run_compute_sanitizer import run as run_memcheck

REQUIRED=('asan-ubsan','tsan','cuda-memcheck')


def suite_status(tools):
    if any(row.get('status')=='FAIL' for row in tools.values()):return 'FAIL'
    if any(name not in tools or tools[name].get('status')!='PASS' or tools[name].get('instrumented') is not True for name in REQUIRED):return 'BLOCKED'
    return 'PASS'


def run_suite(profile_path,native_root,directory,tool):
    raw=Path(profile_path).read_bytes();profile=json.loads(raw)
    if profile.get('backend')!='strata' or profile.get('calibration_status')!='FROZEN':
        raise ValueError('Pinned frozen Strata base required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    native_root=Path(native_root).resolve()
    receipt=dict(schema=1,status='RUNNING',scope='Bounded host metadata and real CUDA branching sanitizer coverage',
        profile_sha256=hashlib.sha256(raw).hexdigest(),execution_profile_id=profile['execution_profile_id'],
        tools={},source_sha256={},limitations=['Host targets cover handle ownership metadata, not full host inference or expert-worker races',
        'CUDA diagnosis uses a separately bound unqualified instrumentation profile','No long-soak or cumulative gate qualification'])
    def save():(directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def execute(argv,label,env=None):
        start=time.monotonic();p=subprocess.run(argv,capture_output=True,env=env)
        (directory/(label+'.stdout')).write_bytes(p.stdout);(directory/(label+'.stderr')).write_bytes(p.stderr)
        return p,dict(command=argv,exit_code=p.returncode,seconds=time.monotonic()-start)
    save()
    try:
        for name in ('tests/core/branch_handles_test.cpp','include/strata/core/branch_handles.hpp'):
            digest=hashlib.sha256((native_root/name).read_bytes()).hexdigest()
            if profile['native_build']['source_sha256'].get(name)!=digest:raise ValueError('Native source fingerprint mismatch: '+name)
            receipt['source_sha256'][name]=digest
        compiler=shutil.which('g++')
        if compiler is None:
            receipt.update(status='BLOCKED',reason='g++ sanitizer compiler unavailable');return receipt
        version,version_run=execute([compiler,'--version'],'compiler-version')
        if version.returncode:raise RuntimeError('Cannot identify compiler')
        receipt['compiler']=dict(path=compiler,sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),version=version.stdout.decode(errors='replace'))
        for name,flags,symbols in [('asan-ubsan','address,undefined',('__asan_','__ubsan_')),('tsan','thread',('__tsan_',))]:
            binary=directory/name
            p,build=execute([compiler,'-std=c++20','-O1','-g','-fsanitize='+flags,'-fno-omit-frame-pointer',
                '-I',str(native_root/'include'),str(native_root/'tests/core/branch_handles_test.cpp'),'-o',str(binary)],name+'-build')
            row=dict(status='FAIL',instrumented=False,build=build,scope='Production branch handle metadata; no model/GPU simulated')
            receipt['tools'][name]=row
            if p.returncode:save();continue
            nm,nm_run=execute(['nm','-D',str(binary)],name+'-symbols')
            row['instrumented']=nm.returncode==0 and all(s.encode() in nm.stdout for s in symbols)
            row['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
            if not row['instrumented']:row.update(status='BLOCKED',reason='Expected instrumented symbols absent');save();continue
            env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',TSAN_OPTIONS='halt_on_error=1')
            p,execution=execute([str(binary)],name+'-run',env);row['execution']=execution
            diagnostics=p.stderr.decode(errors='replace')
            if name=='tsan' and 'FATAL: ThreadSanitizer: unexpected memory mapping' in diagnostics:
                row['initial_execution']=execution
                wrapper=shutil.which('setarch')
                if wrapper is not None:
                    row['layout_workaround']=dict(command=[wrapper,platform.machine(),'-R',str(binary)],scope='ASLR disabled for this test process only',wrapper_sha256=hashlib.sha256(Path(wrapper).read_bytes()).hexdigest())
                    p,execution=execute(row['layout_workaround']['command'],name+'-layout-run',env)
                    row['execution']=execution;diagnostics=p.stderr.decode(errors='replace')
                if wrapper is None or 'FATAL: ThreadSanitizer: unexpected memory mapping' in diagnostics:
                    row.update(status='BLOCKED',reason='Executed TSan runtime cannot initialize with this address-space layout');save();continue
            if row.get('status')!='BLOCKED':
                row['status']=sanitizer_status(p.returncode,diagnostics,instrumented=row['instrumented'])
                if row['status']=='PASS':
                    value=json.loads(p.stdout.decode().strip().splitlines()[-1]);row['target_result']=value
                    if value['status']!='PASS' or value['failures'] or value['checks']<1 or value['allocations']!=value['releases'] or value['live_bytes']:
                        row['status']='FAIL'
            save()
        # Do not launch a second expensive GPU check when a required host tool is blocked.
        if any(row['status']!='PASS' for row in receipt['tools'].values()):
            receipt['tools']['cuda-memcheck']=dict(status='BLOCKED',instrumented=False,reason='Host sanitizer prerequisites incomplete')
            receipt['status']=suite_status(receipt['tools']);return receipt
        probe=Path(__file__).with_name('sanitizer_branching_probe.py')
        cuda=run_memcheck(tool,[sys.executable,str(probe),'--profile',str(Path(profile_path).resolve()),
            '--tool',str(Path(tool).resolve()),'--output',str(directory/'probe')],directory/'cuda')
        receipt['tools']['cuda-memcheck']=dict(cuda,instrumented=cuda['status']=='PASS')
        target=directory/'probe/report.json'
        if cuda['status']=='PASS':
            report=json.loads(target.read_text());receipt['cuda_target']=report
            if report['status']!='DIAGNOSTIC_COMPLETE' or len(report['checks'])!=3 or report['oracle_requests']!=3:
                receipt['tools']['cuda-memcheck']['status']='FAIL'
        receipt['status']=suite_status(receipt['tools']);return receipt
    except BaseException as error:receipt.update(status='FAIL',error=repr(error));raise
    finally:save()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--native-root',type=Path,required=True)
    ap.add_argument('--tool',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=run_suite(args.profile,args.native_root,args.output,args.tool)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)

if __name__=='__main__':main()
