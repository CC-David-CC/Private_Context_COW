"""Archive terminal campaigns and verify a private box-git copy after timing ends.

This tool never stops inference or changes Git remotes. It waits for the selected
client worker to exit and restore serving. Models and web configuration are not
selected by the archive plan. A verified copy is not a gate or power-loss claim.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from archive_resident_evidence import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--queue-pid',type=int,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--status',type=Path,required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    queue_stat = Path('/proc')/str(args.queue_pid)/'stat'
    queue_start = queue_stat.read_text().rsplit(')',1)[1].split()[19]
    record = dict(status='WAITING',pid=os.getpid(),plan_sha256=sha(args.plan))
    def publish():
        temporary = args.status.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(args.status)
    def remote(argv):
        return subprocess.check_output(['ssh','box-git',shlex.join(argv)],text=True).strip()
    try:
        while True:
            if args.after.exists():
                worker = json.loads(args.after.read_text())
                record.update(waiting_for=worker['pid'],phase=worker.get('phase'))
                if not (Path('/proc')/str(worker['pid'])).exists():
                    if worker['status'] not in ('PASS','FAIL') or worker.get('server_http_status') != 200:
                        raise RuntimeError('Client worker did not finish and restore serving')
                    break
            else:
                try: current = queue_stat.read_text().rsplit(')',1)[1].split()[19]
                except FileNotFoundError: raise RuntimeError('Client queue exited before creating a worker') from None
                if current != queue_start: raise RuntimeError('Client queue identity changed')
            publish();print(json.dumps(record),flush=True);time.sleep(30)
        output = Path(plan['output'])
        receipt_path = output.with_suffix('.receipt.json')
        record['status'] = 'ARCHIVING';publish()
        command = [sys.executable,str(Path(__file__).parent/'archive_resident_evidence.py'),
            '--probe',plan['probe'],'--output',str(output)]
        for flag,key in [('--campaign','campaigns'),('--probe-part','probe_parts'),('--api-name','api_versions')]:
            for value in plan[key]: command += [flag,value]
        if output.exists():
            receipt = json.loads(receipt_path.read_text())
            if sha(output) != receipt['sha256']: raise RuntimeError('Existing local archive checksum failed')
        else:
            subprocess.run(command,check=True)
            receipt = json.loads(receipt_path.read_text())
        destination = plan['remote_directory']
        if (not destination.startswith('/srv/data/git/private_context_cow.evidence/')
                or '..' in Path(destination).parts or not output.name.endswith('.zip')):
            raise ValueError('Expected private evidence destination')
        record.update(status='COPYING',archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes']);publish()
        remote(['mkdir','-p','-m','700','--',destination])
        for path in (output,receipt_path):
            subprocess.run(['scp','-p',str(path),'box-git:'+destination+'/'],check=True)
        remote_path = destination+'/'+output.name
        actual = remote(['sha256sum','--',remote_path]).split()[0]
        size = int(remote(['stat','-c','%s','--',remote_path]))
        if actual != receipt['sha256'] or size != receipt['bytes']: raise RuntimeError('Remote archive verification failed')
        if remote(['sha256sum','--',destination+'/'+receipt_path.name]).split()[0] != sha(receipt_path):
            raise RuntimeError('Remote receipt verification failed')
        record.update(status='VERIFIED_BYTES',remote_host='box-git',remote_path=remote_path,
            scope='matching source/destination checksums and byte counts; not power-loss qualification')
    except BaseException as error:
        record.update(status='FAIL',error=str(error));raise
    finally: publish()
    print(json.dumps(record),flush=True)


if __name__ == '__main__': main()
