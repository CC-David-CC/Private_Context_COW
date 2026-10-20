"""Measure another native-process restore using the campaign's frozen API/profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetition',type=int,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.api/'src'))
    from context_time_travel.real import BranchingModel
    from context_time_travel.store import IntegrityError
    profile = json.loads((args.output/'profile.json').read_text())
    for name in list(os.environ):
        if name.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[name]
    os.environ.update(profile['execution']['environment'])
    label = 'restore-repeat-'+str(args.repetition)
    os.environ['CTT_NATIVE_TRACE'] = str(args.output/(label+'-native.jsonl'))
    result = dict(status='RUNNING',python_pid=os.getpid(),repetition=args.repetition,
                  execution_profile_id=profile['execution_profile_id'])
    backend = None
    path = args.output/(label+'.json')
    def publish():
        stage = path.with_suffix('.tmp')
        with stage.open('w') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        stage.replace(path)
    publish()
    try:
        started = time.monotonic()
        backend = BranchingModel.load_backend(profile)
        result.update(startup_seconds=time.monotonic()-started,native_pid=backend.process.pid);publish()
        model = BranchingModel(args.output/'store',backend,profile)
        prepared = json.loads((args.output/'prepare.json').read_text())
        root = model.store.head('saved').tip
        if root != prepared['root']['snapshot']: raise IntegrityError('Retained branch moved')
        receipt = model.restore_checkpoint(root,expected_schedule_sha256=prepared['root']['schedule_sha256'])
        logits = model.next_logits(root)
        frontier = hashlib.sha256(struct.pack('<'+'f'*len(logits),*logits)).hexdigest()
        if (frontier != prepared['root']['frontier_sha256']
                or model.controller(root) != prepared['root']['controller']
                or receipt['forward_tokens'] != 0
                or receipt['counter_deltas'].get('loaded_file_bytes',0) <= 0
                or receipt['counter_deltas'].get('cold_restore_copy_bytes',0) <= 0):
            raise IntegrityError('Repeated cold restore failed frontier/controller/work/byte checks')
        result.update(status='PASS',receipt=receipt,frontier_sha256=frontier)
    except BaseException as error:
        result.update(status='FAIL',error=str(error));raise
    finally:
        if backend is not None:
            backend.close()
            result.update(native_exit_code=backend.process.returncode,
                          native_exit_observed=backend.process.poll() is not None)
        publish()


if __name__ == '__main__': main()
