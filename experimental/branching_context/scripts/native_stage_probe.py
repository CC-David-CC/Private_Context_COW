"""Attribute real consumed-request stages, then check independent native references."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from durable_resume import open_model, publish, sha
from native_request_probe import reference, frontier
from resident_resume import read_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--prefix',type=int,default=128)
    parser.add_argument('--suffix',type=int,default=4)
    parser.add_argument('--generated',type=int,default=8)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    fixture = read_fixture(args.fixture,profile,args.generated) if args.fixture else dict(
        prefix=[100+n%17 for n in range(args.prefix)],A=[117+n%13 for n in range(args.suffix)],
        B=[150+n%11 for n in range(args.suffix)])
    args.output.mkdir(parents=True,exist_ok=False)
    publish(args.output/'profile.json',profile)
    publish(args.output/'fixture.json',fixture)
    report = dict(status='RUNNING',profile_id=profile['execution_profile_id'],harness_sha256=sha(__file__),
        scope='prepare/compute/finish blocking native wall time; not GPU-event durations; one run per branch',
        branches={},numerical_max_error=0.,checks=[])
    model = None
    try:
        model,startup = open_model(SimpleNamespace(profile=args.profile,output=args.output,phase='native-stages'))
        report['startup_seconds'] = startup
        root = model.create(fixture['prefix'])
        root_hash = hashlib.sha256(frontier(model.next_logits(root))).hexdigest()
        for name in ('A','B'):
            before = model.backend.work_counters()
            answer = model.run_greedy_request(root,fixture[name],args.generated,stream='stage-'+name)
            after = model.backend.work_counters()
            count = len(answer['tokens'])
            stages = {key:after[key]-before[key] for key in
                ('decode_prepare_ns','decode_compute_ns','decode_finish_ns','decode_timed_tokens')}
            if (count <= 0 or stages['decode_timed_tokens'] != count
                    or any(stages[key] <= 0 for key in ('decode_prepare_ns','decode_compute_ns','decode_finish_ns'))
                    or after['forward_tokens']-before['forward_tokens'] != len(fixture[name])+count):
                raise AssertionError('Missing real stage timing or incorrect native work')
            total = sum(stages[key] for key in ('decode_prepare_ns','decode_compute_ns','decode_finish_ns'))/1e9
            if total > answer['generation_seconds']:
                raise AssertionError('Native stages exceed the whole measured generation interval')
            report['branches'][name] = dict(outputs=count,tokens=answer['tokens'],timings=stages,
                integrity_request_counters={key:after[key]-before[key] for key in
                    ('integrity_plan_upload_ns','integrity_page_hash_ns','integrity_page_digest_download_ns',
                     'integrity_host_workspace_allocations') if key in after and key in before},
                ms_per_output={key:(stages[key]/count/1e6) for key in
                    ('decode_prepare_ns','decode_compute_ns','decode_finish_ns')},
                unassigned_generation_seconds=answer['generation_seconds']-total,
                generation_seconds=answer['generation_seconds'],request_seconds=answer['request_seconds'],
                frontier_sha256=hashlib.sha256(frontier(model.next_logits(answer['snapshot']))).hexdigest(),
                hashed_bytes=after['integrity_device_hashed_bytes']-before['integrity_device_hashed_bytes'],
                digest_copy_bytes=after['integrity_digest_copy_bytes']-before['integrity_digest_copy_bytes'])
            publish(args.output/'result.json',report)
        if hashlib.sha256(frontier(model.next_logits(root))).hexdigest() != root_hash:
            raise AssertionError('Parent frontier changed')
        process = model.backend.process
        model.backend.close()
        report['native_exit_observed'] = process.poll() is not None
        if not report['native_exit_observed']: raise AssertionError('Native worker remains live')
        model = None
        refs = {}
        for name in ('A','B','A-repeat'):
            branch = 'A' if name == 'A-repeat' else name
            refs[name] = reference(profile,args.output,'fresh-'+name,fixture['prefix']+fixture[branch],
                [len(fixture['prefix']),len(fixture[branch])],args.generated)
            actual = report['branches'][branch]
            if (refs[name]['tokens'] != actual['tokens']
                    or hashlib.sha256(refs[name]['prompt_frontiers'][0]).hexdigest() != root_hash
                    or hashlib.sha256(refs[name]['generated_frontiers'][-1]).hexdigest() != actual['frontier_sha256']):
                raise AssertionError('Instrumented request differs from independent reference')
            report['checks'].append(dict(reference=name,status='PASS',native_work=refs[name]['work']))
            publish(args.output/'result.json',report)
        for key in ('tokens','prompt_frontiers','generated_frontiers'):
            if refs['A'][key] != refs['A-repeat'][key]: raise AssertionError('Fresh repeat is not exact')
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL',error=str(error))
        raise
    finally:
        if model is not None: model.backend.close()
        publish(args.output/'result.json',report)


if __name__ == '__main__': main()
