"""Stop these specifically identified idle follow-ups for the user's review."""
import json
import os
from pathlib import Path
import signal

base = Path('/home/dflanag3/branching-g3/native-request-probe-20261014')
targets = [(1580633,'queue_speed_followup.py'),(1581825,'sample_gpu_campaign.py'),
           (1583635,'queue_evidence_backup.py')]
record = dict(status='PAUSED_BY_USER',processes=[])
client = Path('/home/dflanag3/branching-g3/digest-client-v1-20261014/worker.json')
if client.exists():
    worker = json.loads(client.read_text())
    child = worker.get('child_pid')
    if child and (Path('/proc')/str(child)).exists():
        command = (Path('/proc')/str(child)/'cmdline').read_bytes().decode()
        if 'client_stream_speed.py' not in command or 'digest-reuse-v1-20261014' not in command or os.getpgid(child) != child:
            raise RuntimeError('Unexpected client process group')
        os.killpg(child,signal.SIGTERM)
        record['processes'].append(dict(pid=child,status='INTERRUPTED_CLIENT_MEASUREMENT_BY_USER'))
for pid,marker in targets:
    root = Path('/proc')/str(pid)
    try: command = (root/'cmdline').read_bytes().decode()
    except FileNotFoundError:
        record['processes'].append(dict(pid=pid,status='ALREADY_EXITED'));continue
    if marker not in command or 'branching-g3' not in command:
        raise RuntimeError('Process identity mismatch: '+str(pid))
    # Freeze the queue before inspecting children, so it cannot launch a new job
    # between that inspection and shutdown. Never kill a live inference child.
    os.kill(pid,signal.SIGSTOP)
    children = (root/'task'/str(pid)/'children').read_text().split()
    if children:
        if marker != 'queue_speed_followup.py': raise RuntimeError('Unexpected child: '+str(pid))
        for child in children:
            command = (Path('/proc')/child/'cmdline').read_bytes().decode()
            if 'run_matching_restart_campaign.py' not in command or 'digest-client-v1-20261014' not in command:
                raise RuntimeError('Unexpected queue child')
        # Its child remains alive solely to restore the original server in finally.
    os.kill(pid,signal.SIGKILL)
    record['processes'].append(dict(pid=pid,marker=marker,status='STOPPED_IDLE_FOLLOWUP'))
status = base/'post-v3-backup-worker.json'
if status.exists():
    prior = json.loads(status.read_text());prior.update(status='PAUSED_BY_USER',reason='User requested review before further work')
    status.write_text(json.dumps(prior,indent=2)+'\n')
(base/'pause-followups-20261014.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
