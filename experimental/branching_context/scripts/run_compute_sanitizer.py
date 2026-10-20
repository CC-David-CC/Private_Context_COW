"""Run actual CUDA memcheck and retain commands, tool hashes and diagnostics.

A clean tool receipt is bounded memory-check evidence, not numerical or release
qualification. The target must provide its own correctness/oracle evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def classify(returncode, diagnostics):
    if type(returncode) is not int:raise ValueError('Actual exit code required')
    if returncode!=0 or re.search(r'ERROR SUMMARY: [1-9][0-9]* errors|(?:^|\n).*Error:',diagnostics):return 'FAIL'
    if 'COMPUTE-SANITIZER' not in diagnostics or 'ERROR SUMMARY: 0 errors' not in diagnostics:return 'BLOCKED'
    return 'PASS'


def run(tool, command, directory):
    if not command or any(not isinstance(arg,str) or not arg for arg in command):
        raise ValueError('Explicit target argv required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema=1,status='RUNNING',scope='Executed CUDA memcheck only; target correctness is separate',target=command)
    def save():
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    tool=Path(tool).resolve()
    try:
        if not tool.is_file():
            receipt.update(status='BLOCKED',reason='Compute Sanitizer executable absent');return receipt
        receipt['tool']=dict(path=str(tool),sha256=hashlib.sha256(tool.read_bytes()).hexdigest())
        version=subprocess.run([str(tool),'--version'],capture_output=True)
        (directory/'version.stdout').write_bytes(version.stdout);(directory/'version.stderr').write_bytes(version.stderr)
        if version.returncode!=0:raise RuntimeError('Could not verify actual tool version')
        receipt['tool']['version']=(version.stdout+version.stderr).decode(errors='replace')
        receipt['target_file_sha256']={arg:hashlib.sha256(Path(arg).read_bytes()).hexdigest()
            for arg in command if Path(arg).is_file()}
        executable=shutil.which(command[0])
        if executable:receipt['target_file_sha256'][executable]=hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        argv=[str(tool),'--tool','memcheck','--error-exitcode','86','--require-cuda-init','yes',
              '--target-processes','all','--log-file',str(directory/'compute-%p.log'),*command]
        receipt['command']=argv;save()
        with (directory/'stdout').open('wb') as stdout,(directory/'stderr').open('wb') as stderr:
            start=time.monotonic();process=subprocess.Popen(argv,stdout=stdout,stderr=stderr)
            receipt['pid']=process.pid;save();code=process.wait()
        logs=sorted(directory.glob('compute-*.log'))
        diagnostics='\n'.join(p.read_text(errors='replace') for p in logs)
        receipt.update(status=classify(code,diagnostics),exit_code=code,seconds=time.monotonic()-start,
            diagnostic_logs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in logs},
            zero_error_summaries=diagnostics.count('ERROR SUMMARY: 0 errors'))
        if receipt['status']=='BLOCKED':receipt['reason']='No complete CUDA instrumentation summary observed'
        return receipt
    except BaseException as error:
        receipt.update(status='FAIL',error=repr(error));raise
    finally:save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command[:1]==['--']:command=command[1:]
    result=run(args.tool,command,args.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)


if __name__=='__main__':main()
