"""Build the actual COW fixture daemon on the selected Linux GPU host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--compiler',required=True)
    p.add_argument('--arch',required=True);p.add_argument('--cxx');p.add_argument('--hip',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    files=['tests/model_free/fixture_backend.cu','src/core/branch_pages.cpp','src/core/branch_digest.cu',
           'src/core/vmm_hip.cpp' if a.hip else 'src/core/vmm.cpp']
    headers=['include/strata/core/'+x for x in ['vmm.hpp','branch_pages.hpp','branch_digest.hpp','branch_sha256.hpp']]+['tests/model_free/fixture_data.hpp']
    if a.hip:headers.append('include/strata/core/branch_gpu_compat.hpp')
    binary=a.output/'fixture_backend'
    command=[a.compiler,'-std=c++17','-O2','-I'+str(a.source/'include'),'-I'+str(a.source/'tests/model_free')]
    if a.hip:command+=['--offload-arch='+a.arch,'-DSTRATA_USE_HIP']
    else:
        command+=['-arch='+a.arch]
        if a.cxx:command+=['-ccbin',a.cxx]
    command += [str(a.source/x) for x in files]+['-o',str(binary)]
    with (a.output/'build.stdout').open('wb') as out,(a.output/'build.stderr').open('wb') as err:
        r=subprocess.run(command,stdout=out,stderr=err)
    report=dict(status='BUILT_NOT_EXECUTED' if r.returncode==0 else 'BUILD_FAIL',command=command,
        exit_code=r.returncode,source_sha256={x:hashlib.sha256((a.source/x).read_bytes()).hexdigest() for x in files+headers})
    if binary.exists():report['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (a.output/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return r.returncode
if __name__=='__main__':raise SystemExit(main())
