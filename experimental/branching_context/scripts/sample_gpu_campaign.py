"""Read-only Linux resource samples while an identified campaign worker exists.

Two-second samples are observations, not exact allocation peaks. No process is
stopped. Chat-server restoration is labelled separately from inference work.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def process_record(pid):
    root = Path('/proc')/str(pid)
    try:
        status = {}
        for line in (root/'status').read_text().splitlines():
            key,_,value = line.partition(':')
            if key in ('VmRSS','VmHWM','Threads','Cpus_allowed_list'): status[key] = value.strip()
        status['pid'] = pid
        status['executable'] = os.readlink(root/'exe')
        fields = (root/'stat').read_text().rsplit(')',1)[1].split()
        status['cpu_seconds'] = (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        return status
    except (FileNotFoundError,ProcessLookupError): return dict(pid=pid,exited_during_sample=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--pid',type=int)
    parser.add_argument('--queue-pid',type=int,help='Wait for this identified queue to create the worker')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if (args.pid is None) == (args.queue_pid is None): parser.error('Specify either --pid or --queue-pid')
    if args.queue_pid is not None:
        queued = Path('/proc')/str(args.queue_pid)/'stat'
        queue_start = queued.read_text().rsplit(')',1)[1].split()[19]
        while not args.worker.exists():
            try: current = queued.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError: raise RuntimeError('Queue exited before creating the campaign') from None
            if current != queue_start: raise RuntimeError('Queue process identity changed')
            time.sleep(2)
        args.pid = json.loads(args.worker.read_text())['pid']
    identity = Path('/proc')/str(args.pid)/'stat'
    began = identity.read_text().rsplit(')',1)[1].split()[19]
    with args.output.open('x') as log:
        while identity.exists():
            try:
                if identity.read_text().rsplit(')',1)[1].split()[19] != began: break
            except FileNotFoundError: break
            worker = json.loads(args.worker.read_text())
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits'],text=True).strip()
            apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'],text=True).strip()
            memory = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key,_,value = line.partition(':')
                if key in ('MemTotal','MemAvailable','SwapTotal','SwapFree'): memory[key] = value.strip()
            pids = [int(line.split(',')[0]) for line in apps.splitlines() if line.strip()]
            # Include Python/controller workers as well as native GPU owners.
            # The process tree is transient; an exiting child is an observation.
            seen = set(pids)
            pending = [args.pid]
            while pending and len(seen) < 1024:
                pid = pending.pop()
                if pid not in seen: seen.add(pid)
                try:
                    children = (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
                except (FileNotFoundError,ProcessLookupError): continue
                pending.extend(int(child) for child in children if int(child) not in seen)
            record = dict(unix_seconds=time.time(),monotonic_seconds=time.monotonic(),hostname=socket.gethostname(),
                logical_cpus=os.cpu_count(),phase=worker.get('phase'),worker_status=worker['status'],
                sample_scope='2-second observations; not exact peak; executable distinguishes restored server',
                gpu_csv=gpu,compute_apps_csv=apps,system_memory=memory,
                processes=[process_record(pid) for pid in sorted(seen)])
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            time.sleep(2)


if __name__ == '__main__': main()
