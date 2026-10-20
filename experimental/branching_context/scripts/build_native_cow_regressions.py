"""Compile unchanged COW regressions against the actual rebuilt core objects."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from build_fast_prefill_probe import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--tests',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root,probe,tests,output = [path.resolve() for path in (args.root,args.probe,args.tests,args.output)]
    core = probe.with_name(probe.name+'-core')
    objects = [core/'branch_digest.o',core/'branch_pages.o',probe/'branch_session.o',
               probe/'conversation_snapshot.o',probe/'conversation_state.o']
    if any(not path.is_file() for path in objects): raise ValueError('Missing rebuilt core object')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='BUILDING',sources={},objects={str(path):sha(path) for path in objects},commands=[])
    for name in ('branch_prepare_test','branch_session_test','branch_integrity_test','branch_lifecycle_test'):
        directory = root/'build/CMakeFiles'/(name+'.dir')
        flags = {}
        for line in (directory/'flags.make').read_text().splitlines():
            if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
                key,value = line.split('=',1);flags[key.strip()] = shlex.split(value)
        source,obj = tests/(name+'.cpp'),output/(name+'.o')
        report['sources'][str(source)] = sha(source)
        compile_command = ['/usr/bin/c++','-I'+str(root/'include'),*flags['CXX_DEFINES'],
            *flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(source),'-o',str(obj)]
        link = shlex.split((directory/'link.txt').read_text())
        old = 'CMakeFiles/'+name+'.dir/tests/core/'+name+'.cpp.o'
        if link.count(old) != 1 or link.count(name) != 1: raise ValueError('Unexpected existing test link recipe')
        link[link.index(old)] = str(obj);link[link.index(name)] = str(output/name)
        link[1:1] = [str(path) for path in objects]
        for command in (compile_command,link):
            index = len(report['commands'])
            with (output/(str(index)+'.stdout')).open('wb') as out,(output/(str(index)+'.stderr')).open('wb') as err:
                process = subprocess.run(command,cwd=root/'build',stdout=out,stderr=err)
            report['commands'].append(dict(argv=command,exit_code=process.returncode))
            (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
            if process.returncode: raise RuntimeError('Regression build failed: '+str(output/(str(index)+'.stderr')))
    report.update(status='BUILT_NOT_EXECUTED',binaries={path.name:sha(path) for path in output.iterdir()
        if path.is_file() and path.suffix == ''})
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],binaries=report['binaries'])))


if __name__ == '__main__': main()
