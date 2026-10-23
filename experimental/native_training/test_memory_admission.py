"""Real native CPU admission/refusal and subsequent training on an owned fixture.

Run as python -m experimental.native_training.test_memory_admission
--executable /path/to/pcc-native-train --output /new/private/evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
from .make_cpu_fixture import make_fixture


def run(executable, output):
    os.umask(0o077)
    output = Path(output); output.mkdir(mode=0o700, parents=True, exist_ok=False)
    model = make_fixture(output/'fixture.gguf', context_length=512)
    original = hashlib.sha256(model.read_bytes()).hexdigest()
    rows = []
    for mode in ('lora', 'full'):
        for name, command, budget, expected in [
                ('accept', '--admit-training', 2<<30, 'MEMORY_ADMISSION_PASS'),
                ('refuse', '--admit-training', 1, 'MEMORY_ADMISSION_REFUSED'),
                ('train-refused', '--train', 1, None),
                ('train', '--train', 2<<30, 'MECHANICS_EXECUTED')]:
            label = mode+'-'+name
            candidate = output/(label+'.gguf')
            request = dict(model=str(model), candidate=str(candidate), base_sha256=original,
                mode=mode, rank=2, devices=['CPU'], device_budget_bytes={'CPU':budget},
                reserve_bytes=128<<20, batch_size=1, steps=1, learning_rate=.0001,
                examples=[dict(id='owned-fixture',tokens=[1,2,3],targets=[2,3,4],
                               loss_mask=[False,True,True],verified=True,provenance_sha256='a'*64)])
            path=output/(label+'.json');path.write_text(json.dumps(request)+'\n')
            result=subprocess.run([str(executable),command,str(path)],capture_output=True,text=True,timeout=60)
            (output/(label+'.stdout')).write_text(result.stdout)
            (output/(label+'.stderr')).write_text(result.stderr)
            if expected is None:
                assert result.returncode != 0 and 'memory admission refused' in result.stderr.lower()
            else:
                assert result.returncode == 0, result.stderr
                report=json.loads(result.stdout);assert report['status']==expected
                if command=='--admit-training':
                    assert not report['weights_loaded'] and not report['weights_updated']
                    assert not report['graph_allocation_qualified']
            assert candidate.exists() == (name=='train')
            assert hashlib.sha256(model.read_bytes()).hexdigest()==original
            rows.append(dict(mode=mode,case=name,status='PASS'))
    result=dict(status='PASS',checks=rows,base_unchanged=True,gpu_used=False,
                scope='actual CPU native memory planner, refusal and one ordinary update; tiny fixture only')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.executable,args.output)))
