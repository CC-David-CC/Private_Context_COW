"""Sequential private GPU campaign; stop at first failure, leave chat off."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from native_request_probe import publish


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    script=Path(__file__).with_name('pool_campaign.py')
    plans=script.parent.parent/'plans'
    jobs=[('q4-tiny','pools-llm60-q4-pressure.json',128,4,8),
          ('q4-8k','pools-llm60-q4-pressure.json',8192,512,128),
          ('q8-tiny','pools-llm60-q8.json',128,4,8),
          ('q8-8k','pools-llm60-q8.json',8192,512,128)]
    result=dict(status='RUNNING',GPU=[0],jobs=[],chat_server_started=False)
    destination=args.output/'matrix.json';publish(destination,result)
    for name,config,prefix,suffix,generated in jobs:
        output=args.output/name
        command=[sys.executable,str(script),'campaign','--config',str(plans/config),
            '--output',str(output),'--prefix',str(prefix),'--suffix',str(suffix),
            '--generated',str(generated),'--trees','8','--roots','2']
        began=time.monotonic()
        print(json.dumps(dict(event='phase_start',name=name,command=command)),flush=True)
        with (args.output/(name+'.stdout')).open('wb') as out,(args.output/(name+'.stderr')).open('wb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err)
            while child.poll() is None:
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(event='heartbeat',name=name,pid=child.pid,
                        elapsed_seconds=time.monotonic()-began)),flush=True)
        receipt=json.loads((output/'result.json').read_text())
        result['jobs'].append(dict(name=name,status=receipt['status'],pid=child.pid,
            observed_exit=True,exit_code=child.returncode,seconds=time.monotonic()-began,
            output=str(output),max_error=receipt['numerical_max_error']))
        publish(destination,result)
        print(json.dumps(dict(event='phase_complete',**result['jobs'][-1])),flush=True)
        if child.returncode or receipt['status']!='PASS':
            result['status']='FAIL';publish(destination,result);return 1
    result['status']='PASS';publish(destination,result);return 0


if __name__=='__main__': raise SystemExit(main())
