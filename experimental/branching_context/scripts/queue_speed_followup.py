"""Serialize a bounded follow-up after an identified maintenance worker finishes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--web',type=Path,required=True)
    p.add_argument('--config-sha256',required=True)
    p.add_argument('--benchmark-case',type=Path)
    p.add_argument('--stream-regressions',type=Path)
    p.add_argument('--matrix',type=Path)
    p.add_argument('--api',type=Path)
    a = p.parse_args()
    started = time.monotonic()
    while True:
        record = json.loads(a.after.read_text())
        if 'server_http_status' in record and not Path('/proc/'+str(record['pid'])).exists(): break
        if time.monotonic()-started > 7200: raise TimeoutError('Previous worker has not completed')
        print(json.dumps(dict(waiting_for=record['pid'],phase=record.get('phase'),elapsed=time.monotonic()-started)),flush=True)
        time.sleep(30)
    if record['status'] != 'PASS' or record['server_http_status'] != 200:
        raise RuntimeError('Previous campaign did not pass and restore serving')
    server = record['restored_server_pid']
    children = Path('/proc/'+str(server)+'/task/'+str(server)+'/children').read_text().split()
    native = [pid for pid in children if '--serve' in Path('/proc/'+pid+'/cmdline').read_bytes().decode()]
    if len(native) != 1: raise RuntimeError('Ambiguous native serving process')
    a.root.mkdir(parents=True,exist_ok=False)
    if a.api is None and a.benchmark_case is None:
        raise ValueError('Require an explicit API for matrix execution')
    (a.root/'api').symlink_to(a.api or a.benchmark_case.parent/'api',target_is_directory=True)
    command = [sys.executable,str(Path(__file__).parent/'run_matching_restart_campaign.py'),
        '--root',str(a.root),'--web',str(a.web),'--config-sha256',a.config_sha256,
        '--server-pid',str(server),'--native-pid',native[0]]
    if a.matrix:
        command += ['--matrix',str(a.matrix)]
    else:
        if not a.benchmark_case or not a.stream_regressions: raise ValueError('Missing baseline arguments')
        command += ['--benchmark-case',str(a.benchmark_case),
            '--benchmark-runner','profile','--benchmark-modes','resident',
            '--stream-regressions',str(a.stream_regressions),'--expect-stream-red']
    return subprocess.call(command)


if __name__ == '__main__': sys.exit(main())
