"""Build an isolated malicious GPU-page fixture; never alter production binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ANCHOR='                if (!session.begin(position, 1, error) ||'
HOOK=r'''                // Test fixture deliberately bypasses immutable-page protection.
                static bool injected = false;
                if (!injected && position > 0 && handle_work.live_entries > 0) {
                    const auto pools = session.pools();
                    for (const auto& page : session.ownership()) {
                        if (std::string(page.kind) != "k" || page.owners < 2 || page.pool < 0) continue;
                        const auto& p = pools.at((size_t)page.pool);
                        if (!p.bytes_per_cell || !page.bytes) continue;
                        const uint64_t offset = (position - 1) * p.bytes_per_cell;
                        // Short prefix must occupy a partial first allocation page.
                        if (offset + p.bytes_per_cell >= page.bytes || offset >= p.bytes) continue;
                        int device = -1;
                        CUmemAccessDesc access{};
                        if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("M05 device query failed");
                        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        access.location.id = device;
                        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                        if (cuMemSetAccess((CUdeviceptr)p.address, page.bytes, &access, 1) != CUDA_SUCCESS)
                            throw std::runtime_error("M05 writable mapping failed");
                        auto* target = static_cast<unsigned char*>(p.address) + offset;
                        unsigned char before = 0;
                        if (cudaMemcpy(&before, target, 1, cudaMemcpyDeviceToHost) != cudaSuccess)
                            throw std::runtime_error("M05 read failed");
                        unsigned char after = before ^ 1;
                        if (cudaMemcpy(target, &after, 1, cudaMemcpyHostToDevice) != cudaSuccess ||
                            cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("M05 write failed");
                        std::cerr << "M05_SHARED_PAGE_WRITE " << page.allocation_id << ' ' << page.driver_handle
                                  << ' ' << page.owners << ' ' << page.pool << ' ' << offset << ' '
                                  << (unsigned)before << ' ' << (unsigned)after << std::endl;
                        injected = true;
                        break;
                    }
                    if (!injected) throw std::runtime_error("M05 no shared partial page found");
                }
'''

def mutate_source(source):
    if source.count(ANCHOR)!=1 or 'M05_SHARED_PAGE_WRITE' in source:
        raise ValueError('Expected unchanged unique feed boundary')
    return '#include <cuda.h>\n#include <cuda_runtime.h>\n#include <stdexcept>\n'+source.replace(ANCHOR,HOOK+ANCHOR)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    source=root/'src/core/branch_driver.cpp'
    generated=output/'branch_driver_mutant.cpp';generated.write_text(mutate_source(source.read_text()))
    flags_path=root/'build/CMakeFiles/strata-branch.dir/flags.make'
    flags={}
    for line in flags_path.read_text().splitlines():
        if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
            name,value=line.split('=',1);flags[name.strip()]=shlex.split(value)
    obj=output/'branch_driver_mutant.o';binary=output/'strata-branch-mutant'
    compile_cmd=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(generated),'-o',str(obj)]
    link_path=root/'build/CMakeFiles/strata-branch.dir/link.txt'
    link=shlex.split(link_path.read_text())
    original_obj='CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o'
    if link.count(original_obj)!=1 or link.count('-o')!=1:raise ValueError('Unexpected pinned link recipe')
    link[link.index(original_obj)]=str(obj);link[link.index('-o')+1]=str(binary)
    report=dict(status='RUNNING',original_source_sha256=sha(source),generated_source_sha256=sha(generated),flags_sha256=sha(flags_path),link_recipe_sha256=sha(link_path),commands=[])
    try:
        for i,cmd in enumerate([compile_cmd,link]):
            with (output/(str(i)+'.stdout')).open('wb') as out,(output/(str(i)+'.stderr')).open('wb') as err:
                result=subprocess.run(cmd,cwd=root/'build',stdout=out,stderr=err)
            report['commands'].append(dict(argv=cmd,exit_code=result.returncode))
            if result.returncode:raise RuntimeError('Native fixture build failed')
        report.update(status='BUILT_NOT_EXECUTED',binary=str(binary),binary_sha256=sha(binary),production_source_unchanged=sha(source)==report['original_source_sha256'])
        if not report['production_source_unchanged']:raise RuntimeError('Production source changed')
        return report
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:(output/'build.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.native_root,args.output),indent=2))
