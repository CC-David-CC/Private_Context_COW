"""Seeded real programs with retained attempts and failure-preserving reduction.

Helper executors are for orchestration unit tests only; the CLI always runs the
real immutable API and independent fresh oracle. No campaign certifies a gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from state_machine import make_program, minimize, run_program, OPERATIONS


def valid_program(program):
    known={'root'}
    for row in program:
        if row['source'] not in known or row['out'] in known or row['op'] not in OPERATIONS:
            return False
        known.add(row['out'])
    return True


def run_campaign(profile, directory, seeds, steps, *, _execute=None):
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Distinct integer seeds required')
    if type(steps) is not int or steps<1:raise ValueError('Positive step count required')
    if profile.get('calibration_status')!='FROZEN':raise ValueError('Frozen profile required')
    tokens=profile['fixtures']['suffix_a']+profile['fixtures']['suffix_b']
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    execute=run_program if _execute is None else _execute
    report=dict(schema=1,status='RUNNING',seeds=seeds,steps=steps,attempts=0,
                completed_programs=0,programs=[],oracle_requests=0,oracle_tokens_computed=0,
                max_absolute_error=0,executor='real' if _execute is None else 'orchestration-test')
    def save():
        (directory/'campaign.json').write_text(json.dumps(report,indent=2)+'\n')
    def attempt(program, seed, purpose):
        index=report['attempts'];report['attempts']+=1
        path=directory/('attempt-'+str(index).zfill(5));path.mkdir()
        (path/'program.json').write_text(json.dumps(dict(seed=seed,purpose=purpose,program=program),indent=2)+'\n')
        result=execute(profile,path,program,seed=seed)
        (path/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
        report['oracle_requests']+=result.get('oracle_requests',0)
        report['oracle_tokens_computed']+=result.get('oracle_tokens_computed',0)
        report['max_absolute_error']=max(report['max_absolute_error'],result.get('max_absolute_error',0))
        save()
        if result['status']=='PASS':
            if result.get('completed_operations')!=len(program) or result.get('oracle_requests',0)<1 or not result.get('boundaries'):
                raise RuntimeError('Incomplete execution cannot qualify as a passing program')
            return None
        failure=result.get('failure',{})
        if result['status']!='FAIL' or failure.get('type')!='ProgramFailure' or not failure.get('signature'):
            raise RuntimeError('Unexpected execution error: '+repr(failure))
        return (failure['type'],failure['signature'])
    save()
    try:
        for seed in seeds:
            program=make_program(seed,steps,tokens)
            signature=attempt(program,seed,'original')
            report['programs'].append(dict(seed=seed,program=program,status='FAIL' if signature else 'PASS'))
            if signature is None:
                report['completed_programs']+=1;save();continue
            report['status']='FAIL'
            report['failure']=dict(seed=seed,signature=list(signature),original_program=program,reproduced=False)
            save()
            for _ in range(2):
                if attempt(program,seed,'replay')!=signature:return report
            report['failure']['reproduced']=True;save()
            def reproduce(trial):
                if not valid_program(trial):return ('InvalidProgram','dependency')
                return attempt(trial,seed,'shrink')
            reduced=minimize(program,reproduce)
            for _ in range(2):
                if attempt(reduced,seed,'minimal-replay')!=signature:
                    report['failure']['minimal_replay_stable']=False
                    return report
            report['failure'].update(minimal_program=reduced,minimal_replay_stable=True)
            return report
        report['status']='PASS'
        return report
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    raw=args.profile.read_bytes();profile=json.loads(raw)
    result=run_campaign(profile,args.output,args.seeds,args.steps)
    result['profile_sha256']=hashlib.sha256(raw).hexdigest()
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','completed_programs','attempts','oracle_requests','max_absolute_error']},indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)


if __name__=='__main__':main()
