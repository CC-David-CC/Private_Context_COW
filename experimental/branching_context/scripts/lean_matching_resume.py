"""Time the real API without optional per-token audit dumps or test checkpoints.

Run only after the source campaign passes. Canonical consumed-state publication,
native COW checks and strict restore remain enabled. Startup is reported separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['fresh','resident','disk'],required=True)
    parser.add_argument('--generated',type=int,default=16,
                        help='Use an endpoint already checked in the source campaign')
    args = parser.parse_args()
    sys.path.insert(0,str(args.api/'src'))
    from context_time_travel.real import BranchingModel
    from context_time_travel.store import IntegrityError
    campaign = args.campaign
    def read(name): return json.loads((campaign/name).read_text())
    if read('result.json')['status'] != 'PASS' or read('compare.json')['status'] != 'PASS':
        raise IntegrityError('Lean measurement requires successful matching-schedule campaign')
    fixture,prepared,resumed,profile = [read(n+'.json') for n in ('fixture','prepare','resume','profile')]
    if args.generated <= 0 or args.generated > read('result.json')['output_bound']:
        raise IntegrityError('Output request exceeds the validated source campaign')
    expected_states = {}
    for name in ('A','B'):
        count = min(args.generated,len(resumed['branches'][name]['generated']))
        consumed = len(fixture['prefix'])+len(fixture[name])+count
        found = [state for state in resumed['branches'][name]['states'] if state['consumed'] == consumed]
        if not found: raise IntegrityError('Requested output endpoint has no validated frontier/controller')
        expected_states[name] = found[-1]
    if len(fixture['prefix']) >= 8192:
        for repetition in (2,3):
            if read('restore-repeat-'+str(repetition)+'.json')['status'] != 'PASS':
                raise IntegrityError('Required separate-process restore repetition did not pass')
    args.output.mkdir(parents=True,exist_ok=False)
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_','CTT_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    result = dict(status='RUNNING',mode=args.mode,python_pid=os.getpid(),branches={},
        execution_profile_id=profile['execution_profile_id'],prefix_tokens=len(fixture['prefix']),
        requested_output_limit=args.generated,source_validation_output_limit=read('result.json')['output_bound'],
        scope='real BranchingModel API; canonical fsync/COW/strict restore retained; optional audits excluded',
        filesystem_cache='uncontrolled warm filesystem; no cold-disk claim')
    path = args.output/'result.json'
    def publish():
        stage = path.with_suffix('.tmp')
        with stage.open('w') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        stage.replace(path)
    backend = None
    publish()
    try:
        started = time.monotonic()
        backend = BranchingModel.load_backend(profile)
        result.update(startup_seconds=time.monotonic()-started,native_pid=backend.process.pid)
        model = BranchingModel(args.output/'store',backend,profile)
        root = resident_image = None
        if args.mode != 'fresh':
            root = model.store.create_root(profile['model_contract_id'],fixture['prefix'],{'status':'idle'})
            if root != prepared['root']['snapshot']: raise IntegrityError('Canonical root identity changed')
            source_manifest = campaign/'store/checkpoints'/(root+'.json')
            manifest = json.loads(source_manifest.read_text())['manifest']
            source_payload = source_manifest.parent/(manifest['payload_sha256']+'.state')
            destination = model.checkpoint_directory/source_payload.name
            # Immutable payload setup is outside restoration timing. Hard links
            # preserve identical saved bytes without a synthetic transfer baseline.
            os.link(source_payload,destination)
            shutil.copyfile(source_manifest,model.checkpoint_directory/source_manifest.name)
            if args.mode == 'resident':
                result['resident_setup_restore'] = model.restore_checkpoint(root,
                    expected_schedule_sha256=prepared['root']['schedule_sha256'])
                resident_image = model.cache[root]
                result['resident_retention'] = 'Explicit strong reference to the same native checkpoint image'
        publish()
        for name in ('A','B'):
            measurements = dict(mode=args.mode,branch=name)
            before = backend.work_counters()
            started = time.monotonic()
            if args.mode == 'fresh':
                root = model.create(fixture['prefix'])
            elif args.mode == 'disk':
                measurements['restore_receipt'] = model.restore_checkpoint(root,
                    expected_schedule_sha256=prepared['root']['schedule_sha256'])
            else:
                model.cache[root] = resident_image
                with model.materialize(root): pass
            measurements['prefix_ready_seconds'] = time.monotonic()-started
            ready = backend.work_counters()
            measurements['prefix_forward_tokens'] = ready['consumed_tokens']-before['consumed_tokens']
            measurements['prefix_loaded_file_bytes'] = ready['loaded_file_bytes']-before['loaded_file_bytes']
            measurements['prefix_cold_restore_copy_bytes'] = ready['cold_restore_copy_bytes']-before['cold_restore_copy_bytes']
            if args.mode == 'resident' and measurements['prefix_loaded_file_bytes'] != 0:
                raise IntegrityError('Resident measurement silently loaded disk state')
            expected_work = len(fixture['prefix']) if args.mode == 'fresh' else 0
            if measurements['prefix_forward_tokens'] != expected_work:
                raise IntegrityError('Lean prefix path performed unexpected forward work')
            started = time.monotonic()
            tip = model.feed(root,fixture[name])
            measurements['suffix_seconds'] = time.monotonic()-started
            suffix_work = backend.work_counters()['consumed_tokens']-ready['consumed_tokens']
            if suffix_work != len(fixture[name]): raise IntegrityError('Lean suffix replayed/omitted tokens')
            measurements['suffix_forward_tokens'] = suffix_work
            started = time.monotonic()
            tip = model.start_generation(tip,dict(temperature=0),'durable-greedy')
            measurements['controller_start_seconds'] = time.monotonic()-started
            before_generation = backend.work_counters()
            store_before_generation = model.store.work_counters()
            started = time.monotonic()
            tokens=[]
            first=None
            for tip,token in model.stream(tip,args.generated):
                tokens.append(token)
                if first is None: first = time.monotonic()-started
            elapsed = time.monotonic()-started
            after_generation = backend.work_counters()
            store_after_generation = model.store.work_counters()
            measurements.update(generation_seconds=elapsed,generated_tokens=len(tokens),
                first_token_seconds=first,tokens_per_second=len(tokens)/elapsed if elapsed else None,
                generated_forward_tokens=after_generation['consumed_tokens']-before_generation['consumed_tokens'],
                generated_prefill_tokens=after_generation['prefill_tokens']-before_generation['prefill_tokens'],
                verifier_decode_tokens=after_generation['decode_timed_tokens']-before_generation['decode_timed_tokens'])
            measurements['generation_store_work'] = {key:value-store_before_generation[key]
                for key,value in store_after_generation.items()}
            if (measurements['generated_forward_tokens'] != len(tokens)
                    or measurements['generated_prefill_tokens'] != len(tokens)
                    or measurements['verifier_decode_tokens'] != 0):
                raise IntegrityError('Lean generation changed operators or work accounting')
            # Exact validation is deliberately outside all timed sections.
            expected = resumed['branches'][name]
            frontier = model.next_logits(tip)
            digest = hashlib.sha256(struct.pack('<'+'f'*len(frontier),*frontier)).hexdigest()
            expected_state = expected_states[name]
            if (tokens != expected['generated'][:args.generated] or digest != expected_state['frontier_sha256']
                    or model.controller(tip) != expected_state['controller']):
                raise IntegrityError('Lean continuation differs from validated campaign')
            measurements.update(tokens=tokens,final_frontier_sha256=digest,exact_validation='PASS',
                timed_request_seconds=sum(measurements[key] for key in ('prefix_ready_seconds','suffix_seconds',
                    'controller_start_seconds','generation_seconds')))
            result['branches'][name] = measurements;publish()
        result.update(status='PASS',final_counters=backend.work_counters(),
                      final_store_counters=model.store.work_counters())
    except BaseException as error:
        result.update(status='FAIL',error=str(error));raise
    finally:
        if backend is not None:
            backend.close()
            result.update(native_exit_observed=backend.process.poll() is not None,
                          native_exit_code=backend.process.returncode)
        publish()


if __name__ == '__main__': main()
