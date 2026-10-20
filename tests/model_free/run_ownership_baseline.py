"""Compile and execute the real ownership witness, preserving failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nvcc',default='/usr/bin/nvcc')
    p.add_argument('--cxx',default='/usr/bin/g++-12')
    p.add_argument('--devices',nargs='+',default=['0','1'])
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    binary=a.output/'device_ownership_test'
    command=[a.nvcc,'-std=c++17','-ccbin',a.cxx,'-I'+str(a.source/'include'),
             str(a.source/'tests/model_free/device_ownership_test.cpp'),
             str(a.source/'src/core/vmm.cpp'),'-lcuda','-o',str(binary)]
    report={'scope':'actual CUDA VMM ownership; no model','started':time.time(),
            'source_sha256':{str(x.relative_to(a.source)):hashlib.sha256(x.read_bytes()).hexdigest()
            for x in [a.source/'tests/model_free/device_ownership_test.cpp',a.source/'src/core/vmm.cpp',a.source/'include/strata/core/vmm.hpp']},'commands':[]}
    for stage,argv in [('build',command),('run',[str(binary),*a.devices])]:
        with (a.output/(stage+'.stdout')).open('wb') as out,(a.output/(stage+'.stderr')).open('wb') as err:
            r=subprocess.run(argv,stdout=out,stderr=err)
        report['commands'].append({'stage':stage,'argv':argv,'exit_code':r.returncode})
        if stage=='build' and r.returncode: report['status']='BUILD_FAIL';break
        if stage=='run': report['status']='PASS' if r.returncode==0 else 'BLOCKED' if r.returncode==2 else 'FAIL'
    if binary.exists(): report['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report)); return 0 if report['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
