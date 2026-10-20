"""Isolated Linux campaign; restore the explicitly selected chat server in finally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--web',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--server-pid',type=int,required=True)
    parser.add_argument('--native-pid',type=int,required=True)
    parser.add_argument('--through',choices=['tiny','medium','target'],default='tiny')
    parser.add_argument('--start',choices=['tiny','medium','target'],default='tiny')
    parser.add_argument('--benchmark-case',type=Path,
                        help='After validation, time the lean API against this passed case')
    parser.add_argument('--benchmark-runner',choices=['lean','profile'],default='lean')
    parser.add_argument('--benchmark-modes',nargs='+',choices=['resident','disk','fresh'],
                        default=['resident','disk','fresh'])
    parser.add_argument('--benchmark-repetitions',type=int,default=1)
    parser.add_argument('--benchmark-warmup',action='store_true')
    parser.add_argument('--stream-regressions',type=Path,help='Directory containing real stream regression tests')
    parser.add_argument('--expect-stream-red',action='store_true',help='Require the baseline optimization regression to fail')
    parser.add_argument('--tests-only',action='store_true')
    parser.add_argument('--stream-profile',type=Path)
    parser.add_argument('--matrix',type=Path,help='Explicit sequential diagnostic jobs; stop on first unexpected exit')
    args = parser.parse_args()
    if not 1 <= args.benchmark_repetitions <= 3:
        parser.error('Use one to three measured repetitions')
    root,api = args.root,args.root/'api'
    config = args.web/'config.json'
    if hashlib.sha256(config.read_bytes()).hexdigest() != args.config_sha256:
        raise RuntimeError('Server configuration changed; refusing maintenance')
    settings = json.loads(config.read_text())
    for pid,marker in [(args.server_pid,'serve.server'),(args.native_pid,'--serve')]:
        command = Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().replace('\0',' ')
        if marker not in command: raise RuntimeError('Selected process identity mismatch')
    record = dict(status='STARTING',pid=os.getpid(),phases=[],server_config_sha256=args.config_sha256)
    def write():
        stage = root/'worker.json.tmp'
        stage.write_text(json.dumps(record,indent=2)+'\n')
        stage.replace(root/'worker.json')
    def phase(name,argv,expected_exit=0,environment=None):
        started = time.monotonic()
        with (root/(name+'.stdout')).open('wb') as out,(root/(name+'.stderr')).open('wb') as err:
            child = subprocess.Popen(argv,cwd=api,stdout=out,stderr=err,start_new_session=True,
                                     env=dict(os.environ,**(environment or {})))
            record.update(status='RUNNING',phase=name,child_pid=child.pid);write()
            while True:
                try: code = child.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    record['elapsed_seconds'] = time.monotonic()-started;write()
                    print(json.dumps(dict(phase=name,pid=child.pid,elapsed=record['elapsed_seconds'])),flush=True)
                    if time.monotonic()-started > 7200:
                        os.killpg(child.pid,signal.SIGKILL);child.wait();raise TimeoutError(name)
        record['phases'].append(dict(phase=name,exit_code=code,expected_exit_code=expected_exit,
            seconds=time.monotonic()-started));write()
        if code != expected_exit: raise RuntimeError(name+' unexpected exit; inspect retained logs')
    write()
    try:
        for pid in (args.server_pid,args.native_pid):
            try: os.kill(pid,signal.SIGTERM)
            except ProcessLookupError: pass
        for _ in range(30):
            if not any(Path('/proc/'+str(pid)).exists() for pid in (args.server_pid,args.native_pid)):break
            time.sleep(.5)
        for pid in (args.server_pid,args.native_pid):
            if Path('/proc/'+str(pid)).exists():
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass
        time.sleep(2)
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied: raise RuntimeError('GPU occupied by another process: '+occupied)
        if args.matrix:
            jobs = json.loads(args.matrix.read_text())['jobs']
            for job in jobs:
                name = job['name']
                if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
                    raise ValueError('Invalid matrix job name')
                phase(name,job['argv'],expected_exit=job.get('expected_exit',0),environment=job.get('environment'))
        elif args.tests_only:
            if not args.stream_regressions or not args.benchmark_case:
                raise ValueError('Tests-only requires stream tests and independent reference case')
        elif args.benchmark_case:
            script = 'profile_matching_resume.py' if args.benchmark_runner == 'profile' else 'lean_matching_resume.py'
            repetitions = ([0] if args.benchmark_warmup else [])+list(range(1,args.benchmark_repetitions+1))
            for repeat in repetitions:
                for mode in args.benchmark_modes:
                    name = args.benchmark_runner+'-'+mode+'-'+str(repeat)
                    phase(name,[sys.executable,str(Path(__file__).parent/script),
                        '--api',str(api),'--campaign',str(args.benchmark_case),
                        '--output',str(root/name),'--mode',mode,'--generated','16'])
        else:
            fixtures = [('tiny',128,4,8),('medium',1024,64,16),('target',8192,512,128)]
            names = [item[0] for item in fixtures]
            if names.index(args.start) > names.index(args.through): raise ValueError('Invalid campaign stage range')
            for name,prefix,suffix,generated in fixtures[names.index(args.start):]:
                phase(name,[sys.executable,'scripts/durable_resume.py','campaign','--profile',str(root/'profile.json'),
                    '--output',str(root/name),'--prefix',str(prefix),'--suffix',str(suffix),'--generated',str(generated)])
                if name == 'target':
                    for repetition in (2,3):
                        phase('restore-repeat-'+str(repetition),[sys.executable,str(Path(__file__).parent/'matching_restore_repeat.py'),
                            '--api',str(api),'--output',str(root/name),'--repetition',str(repetition)])
                if name == args.through:break
        if args.stream_regressions:
            os.environ['PYTHONPATH'] = str(api/'src')
            os.environ['CTT_MATCHING_RESUME_RESULTS'] = str(args.benchmark_case)
            if args.stream_profile: os.environ['CTT_PROFILE'] = str(args.stream_profile)
            phase('stream-regressions',[sys.executable,'-m','unittest','discover',
                '-s',str(args.stream_regressions),'-p','test_stream_restore_work.py','-v'],
                expected_exit=1 if args.expect_stream_red else 0)
            if args.expect_stream_red:
                log = (root/'stream-regressions.stderr').read_text()
                if 'Selection and consumption must share one materialization' not in log or 'FAILED (failures=1)' not in log:
                    raise RuntimeError('Baseline red must be the one expected restore regression only')
        record['status'] = 'PASS'
    except BaseException as error:
        record.update(status='FAIL',error=str(error))
    finally:
        write()
        with (args.web/'server-matching-resume-restored.log').open('ab') as log:
            server = subprocess.Popen([str(args.web/'.venv/bin/python'),'-m','serve.server','--engine','strata',
                '--config',str(config),'--host','127.0.0.1','--port','8097'],cwd=args.web,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['restored_server_pid'] = server.pid;write()
        for _ in range(48):
            try:
                req = urllib.request.Request('http://127.0.0.1:8097/workspace/chats',
                    headers={'Authorization':'Bearer '+settings['api_key']})
                with urllib.request.urlopen(req,timeout=3) as response:
                    record['server_http_status'] = response.status
                break
            except Exception:
                if server.poll() is not None:
                    record['server_exit_code'] = server.returncode;break
                time.sleep(5)
        write()
        print(json.dumps(record),flush=True)
    return 0 if record['status'] == 'PASS' and record.get('server_http_status') == 200 else 1


if __name__ == '__main__': sys.exit(main())
