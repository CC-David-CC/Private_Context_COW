"""Exercise the existing HTTP/native/reload/continuation controls on the opt-in profile."""
from pathlib import Path
import argparse,json
from tools.test_native_action_jobs import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['dense-llama-action-selected-head-8k-v1','dense-llama-action-recompute-8k-v1'],default='dense-llama-action-selected-head-8k-v1')
    a=p.parse_args();print(json.dumps(run(a.executable,a.output,profile=a.profile)))
