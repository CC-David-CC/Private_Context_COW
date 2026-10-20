"""Run one real COW browser campaign and verify native shutdown, under external flock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    fixtures = parser.add_mutually_exclusive_group(required=True)
    fixtures.add_argument('--fixtures', type=Path)
    fixtures.add_argument('--single-fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser-script', type=Path,
        help='Versioned browser-only harness; server/profile sources are unchanged')
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--phase', choices=('baseline', 'pressure'), default='baseline')
    args = parser.parse_args()
    if args.ram_gib <= 0 or (args.phase == 'baseline' and args.ram_gib != 96):
        parser.error('Baseline requires normal 96GiB pool; pressure must be explicitly labelled')
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    profile = json.loads(args.profile.read_bytes())
    fixture_path = args.fixtures if args.fixtures is not None else args.single_fixture
    record = dict(status='RUNNING', phase='startup', pid=os.getpid(), campaign_kind=args.phase,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        execution_profile_id=profile['execution_profile_id'],
        fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        context_mode='three-contexts' if args.fixtures is not None else 'single-context-smoke',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ram_gib=args.ram_gib, disk_gib=128, numerical_audit='NOT_RUN',
        scope='Real Chromium client/store flow and shutdown only; fresh numerical audit is separate')
    started = time.monotonic()
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            print(json.dumps(dict(event='heartbeat', pid=os.getpid(), phase=record['phase'],
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    server = browser = None
    stdout_queue = queue.Queue()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))}
    env.update(profile['execution']['environment'])
    env['PYTHONPATH'] = str(repo / 'experimental/branching_context/src') + ':' + str(repo)
    env['CTT_NATIVE_TRACE'] = str(output / 'native.jsonl')
    command = [sys.executable, '-m', 'serve.cow_http', '--profile', str(args.profile.resolve()),
        '--store', str(output / 'store'), '--host', '127.0.0.1', '--port', '0', '--text',
        '--ram-gib', str(args.ram_gib), '--disk-gib', '128',
        '--pool-trace-output', str(output / 'pool-trace.json')]
    record['server_command'] = command
    publish(output / 'result.json', record)
    try:
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True, timeout=15).strip()
        record['gpu_processes_before_start'] = occupied.splitlines() if occupied else []
        if occupied:
            raise RuntimeError('GPU already has compute processes; refusing to interfere')
        with (output / 'server.stderr').open('wb') as err:
            server = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=err, text=True, start_new_session=True)
            record['server_pid'] = server.pid
            def read_stdout():
                with (output / 'server.stdout').open('w', encoding='utf-8', newline='\n') as log:
                    for line in server.stdout:
                        log.write(line); log.flush(); stdout_queue.put(line)
                stdout_queue.put(None)
            reader = threading.Thread(target=read_stdout, daemon=True)
            reader.start()
            deadline = time.monotonic() + 300
            ready = None
            while ready is None:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Server startup exceeded 300 seconds')
                try: line = stdout_queue.get(timeout=min(1, remaining))
                except queue.Empty:
                    if server.poll() is not None: raise RuntimeError('Server exited during startup')
                    continue
                if line is None: raise RuntimeError('Server stdout closed before ready')
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if event.get('event') == 'ready': ready = event
            if ready.get('execution_profile_id') != profile['execution_profile_id']:
                raise RuntimeError('Ready server profile differs')
            record.update(phase='browser', startup_seconds=time.monotonic()-started, ready=ready)
            browser_script = args.browser_script.resolve() if args.browser_script else repo / 'serve/web/cow_native_smoke.py'
            if browser_script.parent != repo / 'serve/web':
                raise ValueError('Versioned browser harness must remain beside its real UI source')
            record['browser_harness_sha256'] = hashlib.sha256(browser_script.read_bytes()).hexdigest()
            browser_command = [sys.executable, str(browser_script),
                '--base-url', 'http://127.0.0.1:' + str(ready['address'][1]),
                '--output-dir', str(output / 'browser'),
                '--server-profile-id', profile['execution_profile_id'], '--max-resumes', '0',
                '--output-limit', '128', '--inference-kind', 'real']
            if args.fixtures is not None:
                browser_command += ['--reusable-context-set', str(args.fixtures.resolve())]
            else:
                fixture = json.loads(args.single_fixture.read_bytes())
                browser_command += ['--reusable-context', '--reusable-fixture', str(args.single_fixture.resolve()),
                    '--context-name', fixture['attachment_name'],
                    '--context-file', str(args.single_fixture.resolve().parent / fixture['artifacts']['context_file'])]
            record['browser_command'] = browser_command
            publish(output / 'result.json', record)
            with (output / 'browser.stdout').open('wb') as out, (output / 'browser.stderr').open('wb') as browser_err:
                browser = subprocess.Popen(browser_command, cwd=repo, env=env, stdout=out, stderr=browser_err)
                record['browser_pid'] = browser.pid
                record['browser_exit_code'] = browser.wait(timeout=900)
            if record['browser_exit_code'] != 0:
                raise RuntimeError('Browser campaign failed; inspect retained report')
            report = json.loads((output / 'browser/report.json').read_bytes())
            if report.get('status') != 'PASS': raise RuntimeError('Browser report is not PASS')
            record['browser_checks'] = len(report.get('checks', []))
    except BaseException as error:
        record.update(status='FAIL', error=str(error))
    finally:
        record['phase'] = 'shutdown'
        if browser is not None and browser.poll() is None:
            browser.terminate()
            try: browser.wait(timeout=30)
            except subprocess.TimeoutExpired: browser.kill(); browser.wait()
            record['browser_exit_code'] = browser.returncode
        if server is not None:
            if server.poll() is None: server.send_signal(signal.SIGINT)
            try: server.wait(timeout=180)
            except subprocess.TimeoutExpired:
                record.update(status='FAIL', cleanup_error='Server did not close within 180 seconds')
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            record['server_exit_code'] = server.returncode
            reader.join(timeout=5)
        try:
            trace = json.loads((output / 'pool-trace.json').read_bytes())
            closed = [row for row in trace['events'] if row.get('event') == 'native_worker_closed']
            if (trace.get('overflowed') or len(closed) != 1 or closed[0].get('gpu_id') != 0
                    or type(closed[0].get('pid')) is not int or closed[0]['pid'] <= 0
                    or type(closed[0].get('returncode')) is not int or closed[0]['returncode'] != 0
                    or closed[0].get('close_succeeded') is not True):
                raise RuntimeError('Native worker shutdown is absent, incomplete or nonzero')
            record['native_worker_shutdown'] = closed
            if record.get('status') != 'FAIL':
                if record.get('server_exit_code') != 0 or record.get('browser_exit_code') != 0:
                    raise RuntimeError('Server/browser exit is not zero')
                record['status'] = 'PASS_CLIENT_FLOW_ONLY'
        except BaseException as error:
            record.update(status='FAIL', shutdown_verification_error=str(error))
        record.update(phase='terminal', elapsed_seconds=time.monotonic()-started)
        publish(output / 'result.json', record)
        stop.set(); beat.join(timeout=1)
        print(json.dumps({key: record.get(key) for key in ('status', 'elapsed_seconds',
            'browser_checks', 'browser_exit_code', 'server_exit_code', 'error',
            'shutdown_verification_error')}), flush=True)
    return 0 if record['status'] == 'PASS_CLIENT_FLOW_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
