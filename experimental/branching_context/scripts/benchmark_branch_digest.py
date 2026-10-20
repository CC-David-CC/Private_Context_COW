"""Measure the real CUDA page-digest stages, independently checking every digest.

This isolated hashing benchmark is not inference throughput or a COW gate.
The native witness warms once then measures three full-byte passes per process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--mib',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 1 <= args.mib <= 1024: parser.error('Use 1 through 1024 MiB')
    if args.output.exists(): raise FileExistsError(args.output)
    page = random.Random(1938001962).randbytes(65536)
    data = page*(args.mib*16)
    expected_page = [hashlib.sha256(page[i:i+4096]).hexdigest() for i in range(0,len(page),4096)]
    expected = expected_page*(args.mib*16)
    report = dict(status='RUNNING',scope='isolated CUDA SHA-256; not inference; three passes after warm-up per process',
        bytes_per_pass=len(data),fixture_sha256=hashlib.sha256(data).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),repetitions=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        for index in range(3):
            start = time.monotonic()
            process = subprocess.run([str(args.binary.resolve())],input=data,capture_output=True,check=True)
            measured = json.loads(process.stdout)
            if measured.pop('digests') != expected: raise AssertionError('CUDA SHA differs from independent hashlib')
            stages = [measured[key] for key in ('plan_upload_ns','page_hash_ns','page_digest_download_ns')]
            if (not all(value > 0 for value in stages) or sum(stages) > measured['wall_ns']
                    or measured['device_hashed_bytes'] != 3*len(data) or measured['kernel_calls'] != 3
                    or measured['workspace_live_bytes'] != 0):
                raise AssertionError('Missing stage measurement, incorrect work or leaked workspace')
            measured.update(repetition=index,process_seconds=time.monotonic()-start,
                ms_per_pass={key:measured[key]/3e6 for key in
                    ('wall_ns','plan_upload_ns','page_hash_ns','page_digest_download_ns')},
                ordered_digest_sha256=hashlib.sha256(''.join(expected).encode('ascii')).hexdigest())
            report['repetitions'].append(measured)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL',error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
