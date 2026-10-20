"""Private greedy resident-request/disk-restart campaign with matching native operators."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from native_request_probe import reference, publish, frontier, _record_native_shutdown
from durable_resume import open_model, capture, sha


def require_native_shutdown(record, phase):
    codes = record.get('native_exit_codes')
    success = record.get('native_exit_success')
    if (type(success) is not bool or not success or not isinstance(codes, dict) or not codes
            or any(type(code) is not int or code != 0 for code in codes.values())):
        raise AssertionError(phase + ' did not report successful numeric exit codes for every native worker')
    return True


def native_phase_summary(phase, python_pid, exit_code, child):
    return dict(phase=phase, python_pid=python_pid, exit_code=exit_code,
        native_exit_observed=child.get('native_exit_observed'),
        native_exit_success=child.get('native_exit_success'),
        native_exit_codes=child.get('native_exit_codes'))


def require_phase_success(phase, process_exit_code, child):
    if process_exit_code != 0 or child.get('status') != 'PASS':
        raise AssertionError('Process-separated stage failed: ' + phase)
    require_native_shutdown(child, phase)
    return True


def save(model, name, sid):
    head = model.store.head(name)
    digest = hashlib.sha256(('resident:'+name+':'+sid).encode()).digest()
    operation = 'r-'+base64.urlsafe_b64encode(digest).decode().rstrip('=')
    return model.save_branch(name,head.revision,sid,operation)


def checked_restore(model, sid, digest):
    receipt = model.restore_checkpoint(sid,expected_schedule_sha256=digest)
    delta = receipt['counter_deltas']
    if receipt['forward_tokens'] != 0 or delta['forward_tokens'] != 0:
        raise AssertionError('Saved prefix was recomputed')
    if delta['loaded_file_bytes'] <= 0:
        raise AssertionError('Restart did not load actual disk bytes')
    return receipt


def same_state(actual, expected, output):
    for key in ('history','controller','consumed'):
        if actual[key] != expected[key]:
            raise AssertionError('Continuation differs: '+key)
    if (output/actual['frontier']).read_bytes() != (output/expected['frontier']).read_bytes():
        raise AssertionError('Continuation frontier differs')


def execute(args):
    record = dict(status='RUNNING',python_pid=os.getpid(),phase=args.phase)
    destination = args.output/(args.phase+'.json')
    publish(destination,record)
    model = None
    try:
        model, startup = open_model(args)
        record.update(startup_seconds=startup,native_pid=model.backend.process.pid)
        fixture = json.loads((args.output/'fixture.json').read_text())
        if args.phase == 'prepare':
            began = time.monotonic()
            root = model.create(fixture['prefix'])
            record['prefix_seconds'] = time.monotonic()-began
            model.store.fork('prefix',root)
            record['root_save'] = save(model,'prefix',root)
            record['root'] = capture(model,root,args.output,'prepared-root')
            manifest = json.loads((model.checkpoint_directory/(root+'.json')).read_text())['manifest']
            payload = model.checkpoint_directory/(manifest['payload_sha256']+'.state')
            record.update(parent_payload=str(payload),parent_payload_sha256=sha(payload),branches={})
            for name in ('A','B'):
                result = model.run_greedy_request(root,fixture[name],args.generated,stream='resident-'+name)
                sid = result['snapshot']
                model.store.fork('prepared-'+name,sid)
                began = time.monotonic()
                publication = save(model,'prepared-'+name,sid)
                state = capture(model,sid,args.output,'prepared-'+name)
                record['branches'][name] = dict(result=result,state=state,publication=publication,
                    save_and_audit_seconds=time.monotonic()-began)
                publish(destination,record)
            record['boundary_checks'] = {}
            for name in ('A','B'):
                total = len(record['branches'][name]['result']['tokens'])
                counts = list(range(1,total+1)) if len(fixture['prefix']) < 8192 else sorted({1,total,*[n for n in (16,64) if n <= total]})
                record['boundary_checks'][name] = {}
                for count in counts:
                    result = model.run_greedy_request(root,fixture[name],count,stream='resident-'+name)
                    if result['tokens'] != record['branches'][name]['result']['tokens'][:count]:
                        raise AssertionError('Boundary request changed generated tokens')
                    state = capture(model,result['snapshot'],args.output,'prepared-'+name+'-'+str(count))
                    record['boundary_checks'][name][str(count)] = dict(tokens=result['tokens'],state=state)
                    publish(destination,record)
            if len(record['branches']['A']['result']['tokens']) <= 4:
                raise AssertionError('Fixture stopped before a resumable four-output boundary')
            mid = model.run_greedy_request(root,fixture['A'],4,stream='resident-A')
            if mid['tokens'] != record['branches']['A']['result']['tokens'][:4]:
                raise AssertionError('Generated checkpoint request differs from uninterrupted prefix')
            model.store.fork('midpoint',mid['snapshot'])
            record['mid_save'] = save(model,'midpoint',mid['snapshot'])
            record['midpoint'] = capture(model,mid['snapshot'],args.output,'prepared-midpoint')
            if sha(payload) != record['parent_payload_sha256']:
                raise AssertionError('Prefix disk payload was modified')
        else:
            prepared = json.loads((args.output/'prepare.json').read_text())
            if prepared['status'] != 'PASS':
                raise AssertionError('Producer did not finish successfully')
            require_native_shutdown(prepared, 'prepare producer')
            if (record['python_pid'] == prepared['python_pid'] or record['native_pid'] == prepared['native_pid']):
                raise AssertionError('Restart reused producer processes')
            root = model.store.head('prefix').tip
            record['restore'] = checked_restore(model,root,prepared['root']['schedule_sha256'])
            state = capture(model,root,args.output,'restored-root')
            same_state(state,prepared['root'],args.output)
            record['branches'] = {}
            for name in ('A','B'):
                result = model.run_greedy_request(root,fixture[name],args.generated,stream='resident-'+name)
                state = capture(model,result['snapshot'],args.output,'restored-'+name)
                if result['tokens'] != prepared['branches'][name]['result']['tokens']:
                    raise AssertionError('Restored generated tokens differ: '+name)
                same_state(state,prepared['branches'][name]['state'],args.output)
                record['branches'][name] = dict(result=result,state=state)
                publish(destination,record)
            record['boundary_checks'] = {}
            for name,checks in prepared['boundary_checks'].items():
                record['boundary_checks'][name] = {}
                for count,expected in checks.items():
                    result = model.run_greedy_request(root,fixture[name],int(count),stream='resident-'+name)
                    state = capture(model,result['snapshot'],args.output,'restored-'+name+'-'+count)
                    if result['tokens'] != expected['tokens']:
                        raise AssertionError('Restored boundary request changed generated tokens')
                    same_state(state,expected['state'],args.output)
                    record['boundary_checks'][name][count] = dict(tokens=result['tokens'],state=state)
                    publish(destination,record)
            mid = model.store.head('midpoint').tip
            record['mid_restore'] = checked_restore(model,mid,prepared['midpoint']['schedule_sha256'])
            result = model.run_greedy_request(mid,[],args.generated-4,stream='must-not-reset-controller')
            if result['tokens'] != prepared['branches']['A']['result']['tokens'][4:]:
                raise AssertionError('Generated restart duplicated or omitted a token')
            state = capture(model,result['snapshot'],args.output,'restored-final')
            same_state(state,prepared['branches']['A']['state'],args.output)
            record['generated_resume'] = dict(result=result,state=state)
            if sha(prepared['parent_payload']) != prepared['parent_payload_sha256']:
                raise AssertionError('Prefix disk payload was modified across restart')
        record.update(status='PASS',counters=model.backend.work_counters())
    except BaseException as error:
        record.update(status='FAIL',error=str(error))
        raise
    finally:
        if model is not None:
            _record_native_shutdown(record, {model.backend.gpu_id: model.backend.process},
                                    model.backend.close)
        publish(destination,record)
    if record.get('status') != 'PASS':
        raise RuntimeError(record.get('cleanup_error', record.get('native_exit_error',
                            'Native phase did not complete successfully')))


def campaign(args):
    profile = json.loads(args.profile.read_text())
    fixture = read_fixture(args.fixture,profile,args.generated) if args.fixture else dict(
        prefix=[100+i%17 for i in range(args.prefix)],
        A=[117+i%13 for i in range(args.suffix)],B=[150+i%11 for i in range(args.suffix)])
    args.output.mkdir(parents=True,exist_ok=False)
    publish(args.output/'fixture.json',fixture)
    record = dict(status='RUNNING',scope='greedy matching-native-operator resident/disk resume; not gate certification',
        profile_id=profile['execution_profile_id'],fixture_sizes=dict(prefix=len(fixture['prefix']),
        suffixes={name:len(fixture[name]) for name in ('A','B')},
        output_limit=args.generated),comparisons=0,numerical_max_error=0.,
        fixture_scope=fixture.get('scope','synthetic deterministic token workload'),
        harness_sha256=sha(__file__),fixture_sha256=sha(args.output/'fixture.json'),phases=[])
    publish(args.output/'result.json',record)
    try:
        for phase in ('prepare','resume'):
            began = time.monotonic()
            with (args.output/(phase+'.stdout')).open('wb') as out,(args.output/(phase+'.stderr')).open('wb') as err:
                process = subprocess.Popen([sys.executable,__file__,phase,'--profile',str(args.profile),
                    '--output',str(args.output),'--generated',str(args.generated)],stdout=out,stderr=err)
                while True:
                    try: code = process.wait(timeout=30); break
                    except subprocess.TimeoutExpired:
                        record.update(phase=phase,child_pid=process.pid,elapsed_seconds=time.monotonic()-began)
                        publish(args.output/'result.json',record)
                        print(json.dumps(dict(phase=phase,pid=process.pid,elapsed=record['elapsed_seconds'])),flush=True)
            child = json.loads((args.output/(phase+'.json')).read_text())
            record['phases'].append(native_phase_summary(phase, process.pid, code, child))
            require_phase_success(phase, code, child)
            publish(args.output/'result.json',record)
        # The direct fresh references import no checkpoint or BranchSession.
        for key in list(os.environ):
            if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
        os.environ.update(profile['execution']['environment'])
        refs = {}
        for name in ('A','B','A-repeat'):
            suffix = fixture['A' if name == 'A-repeat' else name]
            refs[name] = reference(profile,args.output,'fresh-'+name,fixture['prefix']+suffix,
                                   [len(fixture['prefix']),len(suffix)],args.generated)
            publish(args.output/'result.json',record)
        for key in ('tokens','prompt_frontiers','generated_frontiers'):
            if refs['A'][key] != refs['A-repeat'][key]:
                raise AssertionError('Independent native repeat is not exact')
        prepared = json.loads((args.output/'prepare.json').read_text())
        for name in ('A','B'):
            actual = prepared['branches'][name]
            if actual['result']['tokens'] != refs[name]['tokens']:
                raise AssertionError('Canonical request differs from independent fresh tokens')
            if (args.output/actual['state']['frontier']).read_bytes() != refs[name]['generated_frontiers'][-1]:
                raise AssertionError('Canonical request differs from independent fresh frontier')
            if (args.output/prepared['root']['frontier']).read_bytes() != refs[name]['prompt_frontiers'][0]:
                raise AssertionError('Saved prefix differs from independent fresh frontier')
            record['comparisons'] += 2
            for count,checked in prepared['boundary_checks'][name].items():
                if (args.output/checked['state']['frontier']).read_bytes() != refs[name]['generated_frontiers'][int(count)-1]:
                    raise AssertionError('Canonical boundary differs from independent fresh frontier')
                record['comparisons'] += 1
        if (args.output/prepared['midpoint']['frontier']).read_bytes() != refs['A']['generated_frontiers'][3]:
            raise AssertionError('Generated checkpoint differs from independent fresh frontier')
        record.update(status='PASS',comparisons=record['comparisons']+1,
            checked_boundaries='every output for prefixes below 8K; otherwise 1/16/64/final; prefix and generated restart at A token 4')
    except BaseException as error:
        record.update(status='FAIL',error=str(error))
        raise
    finally:
        publish(args.output/'result.json',record)


def read_fixture(path, profile, generated):
    fixture = json.loads(path.read_text())
    if fixture.get('model_contract_id') != profile['model_contract_id']:
        raise ValueError('Fixture model/tokenizer contract mismatch')
    for name in ('prefix','A','B'):
        ids = fixture.get(name)
        if (not isinstance(ids,list) or not ids
                or any(type(t) is not int or not 0 <= t < 248320 for t in ids)):
            raise ValueError('Invalid exact-token fixture: '+name)
    if fixture['A'] == fixture['B']:
        raise ValueError('Require two different suffixes')
    if any(len(fixture['prefix'])+len(fixture[name])+generated > profile['execution']['capacity']
           for name in ('A','B')):
        raise ValueError('Fixture exceeds pinned context budget')
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','resume','campaign'])
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix',type=int,default=128)
    parser.add_argument('--suffix',type=int,default=4)
    parser.add_argument('--generated',type=int,default=8)
    parser.add_argument('--fixture',type=Path,help='Fully encoded source/chat prompts bound to this model contract')
    args = parser.parse_args()
    if min(args.prefix,args.suffix) <= 0 or args.generated <= 4:
        parser.error('Positive prefix/suffix and at least five output slots required')
    if args.phase == 'campaign': campaign(args)
    else: execute(args)


if __name__ == '__main__': main()
