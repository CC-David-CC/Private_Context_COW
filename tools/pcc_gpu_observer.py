"""Bounded read-only NVIDIA telemetry for an owner-run training campaign.

No GPU allocation, job control, admission or scheduling occurs here. Samples
are observations, not reservations or per-process compute measurements.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time

QUERY = ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits']


def parse_sample(text):
    rows = []
    indices, identities = set(), set()
    for values in csv.reader(io.StringIO(text)):
        if not values:
            continue
        if len(values) != 7:
            raise ValueError('Incomplete NVIDIA telemetry row')
        index, identity, total, used, free, activity, bandwidth = [v.strip() for v in values]
        index, total, used, free, activity, bandwidth = map(int, [index, total, used, free, activity, bandwidth])
        if (index < 0 or not identity.startswith('GPU-') or index in indices
                or identity in identities or min(total, used, free) < 0
                or total == 0 or used > total or free > total
                or not 0 <= activity <= 100 or not 0 <= bandwidth <= 100):
            raise ValueError('Invalid or duplicate NVIDIA telemetry')
        indices.add(index); identities.add(identity)
        rows.append(dict(index=index, uuid=identity, total_mib=total,
                         used_mib=used, free_mib=free,
                         gpu_percent=activity, memory_percent=bandwidth))
    if not rows:
        raise ValueError('No GPUs observed')
    return sorted(rows, key=lambda row: row['index'])


def sample():
    started = time.monotonic()
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 monotonic_seconds=started)
    try:
        result = subprocess.run(QUERY, capture_output=True, text=True, timeout=15, check=True)
        value.update(status='observed', gpus=parse_sample(result.stdout))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        value.update(status='unavailable', error_type=type(error).__name__)
    value['query_seconds'] = time.monotonic() - started
    return value


def write_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def observe(output, *, interval_seconds=30, duration_seconds=3600):
    if (type(interval_seconds) is not int or not 1 <= interval_seconds <= 300
            or type(duration_seconds) is not int or not 1 <= duration_seconds <= 259200):
        raise ValueError('Interval 1..300 seconds and duration 1..259200 seconds required')
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    old = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, stop)
    started = time.monotonic()
    identity = dict(schema=1, pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    interval_seconds=interval_seconds, duration_seconds=duration_seconds,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    read_only=True, allocates_gpus=False, controls_jobs=False,
                    measures_per_process_compute=False)
    if Path('/proc/self/stat').exists():
        identity['start_ticks'] = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]
    (output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
    count = unavailable = 0
    try:
        with (output/'samples-private.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
            while not stopped and time.monotonic()-started < duration_seconds:
                row = sample()
                stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                count += 1; unavailable += row['status'] != 'observed'
                write_status(output/'status.json', dict(state='running', samples=count,
                             unavailable=unavailable, latest=row))
                deadline = min(started+duration_seconds, time.monotonic()+interval_seconds)
                while not stopped and time.monotonic() < deadline:
                    time.sleep(max(0, min(1, deadline-time.monotonic())))
        result = dict(state='stopped' if stopped else 'complete', samples=count,
                      unavailable=unavailable, elapsed_seconds=time.monotonic()-started,
                      allocates_gpus=False, controls_jobs=False)
        write_status(output/'status.json', result)
        return result
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=int, default=30)
    parser.add_argument('--duration-seconds', type=int, default=3600)
    args = parser.parse_args(argv)
    print(json.dumps(observe(args.output, interval_seconds=args.interval_seconds, duration_seconds=args.duration_seconds)))


if __name__ == '__main__':
    main()
